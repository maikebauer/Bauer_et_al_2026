import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from astropy.io import fits
import math
import os
import json
import matplotlib.dates as mdates
from matplotlib.patches import Circle, Patch
from matplotlib.colors import to_rgba, LinearSegmentedColormap, LogNorm
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, mannwhitneyu, fisher_exact, chi2_contingency, spearmanr
from scipy.stats.contingency import odds_ratio as conditional_odds_ratio
from matplotlib.gridspec import GridSpec
from sunpy.coordinates import frames, get_horizons_coord
import astropy.units as u
import seaborn as sns
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ELEvoHI'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'STRUDL_Tracking_Analysis'))
import helper_functions
import functions as elevohi_functions

def convert_time(time_data, format='%Y-%m-%dT%H:%MZ'):
    try:
        converted_time = datetime.strptime(time_data, format)
    except (ValueError, TypeError):
        converted_time = pd.NaT
    
    return converted_time

def load_donki_data(file_path):

    donki_df = pd.read_json(file_path)

    time_columns = ['time21_5', 'submissionTime', 'earthArrivalTime', 'strudlFirstHItime']

    for time_col in time_columns:
        donki_df[time_col] = donki_df[time_col].apply(lambda x: convert_time(x, '%Y-%m-%dT%H:%MZ'))

    time_range = [datetime(2024,5,1,0,0,0), datetime(2025,4,30,23,59,59)]

    donki_df_filtered = donki_df[(donki_df['time21_5'] >= time_range[0]) & (donki_df['time21_5'] <= time_range[1])]
    donki_df_filtered = donki_df_filtered[donki_df_filtered['featureCode'] != 'TE']
    donki_df_dedupe = donki_df_filtered.sort_values(by='featureCode', axis=0).drop_duplicates(subset='associatedActivityID')

    return donki_df_dedupe, donki_df_filtered


# ---------------------------------------------------------------------------
# Venn diagram of the DONKI CME categories
# ---------------------------------------------------------------------------

# Set key -> (column of DONKI_cme_analyses_matched.json, how a CME qualifies).
VENN_SETS = {
    "V": ("couldBeInHI1AFov", "true"),
    "E": ("rinitCorrected", "notnull"),
    "A": ("arrivalEllipsePossibleCorrected", "true"),
    "S": ("associatedStrudlID", "notnull"),
    "I": ("hasEarthArrival", "true"),
}

VENN_SET_LABELS = {
    "D": "DONKI: unique CMEs with SH or LE entry",
    "V": "VISIBLE-HI: visible in STEREO-A HI1",
    "E": "IN-ECLIPTIC: propagating in the ecliptic",
    "A": "ARRIVAL-ELLIPSE-POSSIBLE: ELEvo arrival possible",
    "S": "STRUDL-MATCH: matched to STRUDL detection",
    "I": "HAS-IN-SITU-ARRIVAL: in situ arrival at L1",
}

# color_dict keys, one per set. geosphere_yellow and geosphere_orange are deliberately
# not used here: yellow has a WCAG contrast of 1.2 against the white page, which makes
# an outline invisible, and orange is indistinguishable from geosphere_green for
# deuteranopic readers (OKLab dE 3.3, well under the 8 that is needed). That matters
# more than in the other figures because here the circles are identified by colour
# alone. venn_blue and venn_red replace them; every pair of the resulting five is at
# least 9.7 apart under simulated protanopia and deuteranopia.
VENN_SET_COLOR_KEYS = {
    "V": "geosphere_purple",
    "E": "venn_blue",
    "A": "venn_red",
    "S": "geosphere_green",
    "I": "my_pink",
}

VENN_V_RADIUS = 3.0           # fixes the overall scale of the layout
VENN_LABEL_CLEARANCE = 0.24   # data units a region needs to hold its count label

# Starting point for the layout solver, verified for the counts of the 2024-05-01 to
# 2025-04-30 analysis period. The solver refines it, and falls back to cold restarts if
# the counts ever change enough to make one of the regions disappear.
VENN_REFERENCE_LAYOUT = {
    "V": (2.423, -0.216, 3.000),
    "E": (0.505, -0.028, 2.574),
    "A": (0.883, -0.404, 1.308),
    "S": (2.569, -0.636, 1.717),
    "I": (1.247, -1.606, 1.633),
}

def get_venn_set_masks(donki_df):
    """Boolean mask per Venn set, keyed as in VENN_SETS."""

    masks = {}

    for key, (column, qualifies) in VENN_SETS.items():
        if qualifies == "true":
            masks[key] = donki_df[column].fillna(False).astype(bool)
        else:
            masks[key] = donki_df[column].notna()

    return masks

def get_venn_region_counts(donki_df):
    """Exclusive region counts of the DONKI CME categories.

    Keys are the sets a CME belongs to, in VEASI order (e.g. 'VEI'); the empty key
    holds the CMEs that fall into none of the sub-categories. Returns
    (regions, totals), where totals holds the per-set totals plus 'D' for all CMEs.
    """

    masks = get_venn_set_masks(donki_df)

    # The layout solver places A inside V and E and S inside V, so fail loudly rather
    # than draw a diagram that contradicts the data if that ever stops holding.
    outside = int((masks["A"] & ~(masks["V"] & masks["E"])).sum())
    if outside:
        raise ValueError(f"{outside} CMEs have arrivalEllipsePossibleCorrected but are not "
                         "both visible in HI1A and propagating in the ecliptic")

    outside = int((masks["S"] & ~masks["V"]).sum())
    if outside:
        raise ValueError(f"{outside} CMEs have a STRUDL match but are not visible in HI1A")

    region_key = pd.Series([""] * len(donki_df), index=donki_df.index)

    for key in VENN_SETS:
        region_key = region_key + np.where(masks[key], key, "")

    regions = {region: int(count) for region, count in region_key.value_counts().items()}

    totals = {key: int(mask.sum()) for key, mask in masks.items()}
    totals["D"] = len(donki_df)

    return regions, totals

def _venn_layout_from_params(params):
    """Unpack the solver's parameter vector into {set key: (x, y, r)}."""

    layout = {key: (float(params[3 * i]), float(params[3 * i + 1]), abs(float(params[3 * i + 2])))
              for i, key in enumerate(VENN_SETS)}
    layout["V"] = (layout["V"][0], layout["V"][1], VENN_V_RADIUS)

    return layout

def _venn_params_from_layout(layout):

    return np.array([value for key in VENN_SETS for value in layout[key]], dtype=float)

def _venn_grid(n_points, extent):

    g = np.linspace(-extent, extent, n_points, dtype=np.float32)
    X, Y = np.meshgrid(g, g)

    return X, Y, float((g[1] - g[0]) ** 2)

def _venn_signed_distances(layout, grid):
    """Distance inside (positive) or outside (negative) each circle, per grid cell."""

    X, Y = grid[0], grid[1]

    return {key: r - np.hypot(X - x, Y - y) for key, (x, y, r) in layout.items()}

def _venn_measure_regions(layout, grid):
    """Area and label clearance of every region the five circles actually create."""

    signed_distance = _venn_signed_distances(layout, grid)
    keys = list(VENN_SETS)

    code = np.zeros(grid[0].shape, dtype=np.int32)

    for i, key in enumerate(keys):
        code |= (signed_distance[key] > 0).astype(np.int32) << i

    areas, clearance = {}, {}

    for value in np.unique(code):
        if value == 0:
            continue

        region = "".join(key for i, key in enumerate(keys) if value >> i & 1)
        inside = code == value
        areas[region] = float(inside.sum()) * grid[2]

        # distance from the closest boundary of the region, i.e. room for a label
        margin = np.minimum.reduce([signed_distance[key] if value >> i & 1 else -signed_distance[key]
                                    for i, key in enumerate(keys)])
        clearance[region] = float(margin[inside].max())

    return areas, clearance

def _venn_layout_cost(params, grid, log_targets):
    """Cost of one candidate layout, together with everything measured on the way.

    Region areas are matched to sqrt(N) up to a common factor, so that small and large
    regions both stay readable while keeping their size order.
    """

    layout = _venn_layout_from_params(params)
    vx, vy, vr = layout["V"]
    ex, ey, er = layout["E"]
    ax, ay, ar = layout["A"]
    sx, sy, sr = layout["S"]

    containment = max(0.0, float(np.hypot(ax - vx, ay - vy)) + ar - vr + 0.08)
    containment += max(0.0, float(np.hypot(ax - ex, ay - ey)) + ar - er + 0.08)
    containment += max(0.0, float(np.hypot(sx - vx, sy - vy)) + sr - vr + 0.08)

    areas, clearance = _venn_measure_regions(layout, grid)

    missing = [region for region in log_targets if region not in areas]
    if missing:
        return 1e3 + 1e2 * len(missing) + 10.0 * containment, layout, areas, clearance, missing

    log_areas = {region: math.log(areas[region]) for region in log_targets}
    offset = np.mean([log_areas[region] - log_targets[region] for region in log_targets])

    area_err = sum((log_areas[region] - log_targets[region] - offset) ** 2 for region in log_targets)
    label_pen = sum(max(0.0, VENN_LABEL_CLEARANCE - clearance[region]) ** 2 * 50 for region in log_targets)
    extra_area = sum(area for region, area in areas.items() if region not in log_targets)

    cost = area_err + label_pen + 20.0 * extra_area + 50.0 * containment

    return cost, layout, areas, clearance, missing

def _venn_anneal(params, grid, log_targets, seed, n_iter, step, temperature):
    """Simulated annealing on the five circle centres and radii."""

    rng = np.random.default_rng(seed)

    current = np.asarray(params, dtype=float)
    current_cost = _venn_layout_cost(current, grid, log_targets)[0]
    best_cost, best = current_cost, current.copy()

    for i in range(n_iter):
        scale = step if i < n_iter // 2 else step / 3.0
        trial = current + rng.normal(0, scale, current.size) * (rng.random(current.size) < 0.45)
        cost = _venn_layout_cost(trial, grid, log_targets)[0]

        if cost < current_cost or rng.random() < np.exp(-(cost - current_cost) / max(temperature, 1e-6)):
            current, current_cost = trial, cost

            if cost < best_cost:
                best_cost, best = cost, trial.copy()

        temperature *= 0.9985

    return best_cost, best

def solve_venn_layout(regions, n_restarts=6, verbose=True):
    """Circle centres and radii that create exactly the regions in `regions`.

    Returns (layout, areas, clearance). VENN_REFERENCE_LAYOUT is refined first, which
    is enough as long as the counts resemble the ones it was found for; if a region is
    missing or an unwanted one appears, the search restarts from a wider spread. Every
    step is seeded, so the same counts always give the same figure.
    """

    log_targets = {region: 0.5 * math.log(count) for region, count in regions.items() if region}
    search_grid = _venn_grid(160, 6.5)
    check_grid = _venn_grid(400, 6.5)
    reference = _venn_params_from_layout(VENN_REFERENCE_LAYOUT)

    attempts = [("reference layout", reference, 0, 600, 0.04, 0.05)]

    for seed in range(n_restarts):
        start = reference + np.random.default_rng(1000 + seed).normal(0, 0.35, reference.size)
        attempts.append((f"restart {seed}", start, seed, 2500, 0.14, 0.5))

    problems = []

    for name, start, seed, n_iter, step, temperature in attempts:
        _, params = _venn_anneal(start, search_grid, log_targets, seed, n_iter, step, temperature)
        cost, layout, areas, clearance, missing = _venn_layout_cost(params, check_grid, log_targets)
        unwanted = [region for region, area in areas.items() if region not in log_targets and area > 0.02]
        problems = missing + unwanted

        if not problems:
            if verbose:
                print(f"Venn layout from {name}: cost {cost:.3f}, smallest label clearance "
                      f"{min(clearance[region] for region in log_targets):.2f}")

            return layout, areas, clearance

    raise RuntimeError("no Venn layout found; regions wrong in the best attempt: " + ", ".join(problems))

def make_donki_venn_plot(donki_df, color_dict, font_params, SAVE_PLOT_PAPER_PATH):
    """Venn diagram of the DONKI CME categories, region areas approx. prop. to sqrt(N)."""

    regions, totals = get_venn_region_counts(donki_df)
    layout, areas, clearance = solve_venn_layout(regions)

    # DONKI circle: centred on the smallest circle enclosing the five sets. Its radius
    # would follow the same sqrt(N) scaling as the inner regions, but with the measured
    # counts it has to be grown to enclose them, so the DONKI-only region is the one
    # region whose area is not proportional to sqrt(N).
    enclosing_radius = lambda c: max(np.hypot(c[0] - x, c[1] - y) + r for x, y, r in layout.values())
    donki_center = tuple(minimize(enclosing_radius, [0.0, 0.0], method="Nelder-Mead").x)

    area_per_sqrt_n = np.median([areas[region] / np.sqrt(regions[region])
                                 for region in areas if region in regions])
    donki_radius = np.sqrt((sum(areas.values()) + area_per_sqrt_n * np.sqrt(regions.get("", 1))) / np.pi)
    donki_radius = max(donki_radius, enclosing_radius(donki_center) + 0.3)

    # Label each region at its interior point farthest from every boundary.
    label_extent = max(abs(donki_center[0]), abs(donki_center[1])) + donki_radius + 0.2
    label_grid = _venn_grid(1300, label_extent)
    X, Y = label_grid[0], label_grid[1]

    signed_distance = _venn_signed_distances(layout, label_grid)
    signed_distance["D"] = donki_radius - np.hypot(X - donki_center[0], Y - donki_center[1])

    label_pos = {}

    for region in regions:
        margin = np.minimum.reduce([signed_distance["D"]]
                                   + [signed_distance[key] if key in region else -signed_distance[key]
                                      for key in layout])
        flat = int(np.argmax(margin))

        if margin.flat[flat] <= 0:
            raise RuntimeError(f"region {region!r} is missing from the solved layout")

        label_pos[region] = (float(X.flat[flat]), float(Y.flat[flat]))

    set_colors = {key: color_dict[name] for key, name in VENN_SET_COLOR_KEYS.items()}

    # Outlines are darkened so every circle keeps a WCAG contrast above 3 against the
    # white page, while the fills stay light enough to read five overlapping circles.
    edge_colors = {key: tuple(0.8 * channel for channel in to_rgba(color)[:3])
                   for key, color in set_colors.items()}

    fig, ax = plt.subplots(figsize=(9, 10.5))

    ax.add_patch(Circle(donki_center, donki_radius, fc=to_rgba("0.6", 0.06), ec="0.6", lw=2))

    for key, (x, y, r) in layout.items():
        ax.add_patch(Circle((x, y), r, fc=to_rgba(set_colors[key], 0.10), ec=edge_colors[key], lw=2.2))

    for region, (x, y) in label_pos.items():
        # tight regions get a smaller number so that it stays clear of the circle edges
        fontsize = 11 if not region or clearance[region] > 0.33 else 9
        ax.text(x, y, regions[region], ha="center", va="center", fontsize=fontsize, color="0.15")

    handles = [Patch(fc=to_rgba("0.6", 0.2), ec="0.6",
                     label=f"{VENN_SET_LABELS['D']} (N = {totals['D']})")]
    handles += [Patch(fc=to_rgba(set_colors[key], 0.3), ec=edge_colors[key],
                      label=f"{VENN_SET_LABELS[key]} (N = {totals[key]})") for key in layout]

    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False,
              fontsize=font_params['legend_fontsize'])
    ax.text(0.99, 0.01, "Region areas approx. $\\propto \\sqrt{N}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=font_params['ticks_fontsize'] - 4, color="0.4")

    ax.set_xlim(donki_center[0] - donki_radius - 0.2, donki_center[0] + donki_radius + 0.2)
    ax.set_ylim(donki_center[1] - donki_radius - 0.2, donki_center[1] + donki_radius + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    plt.savefig(SAVE_PLOT_PAPER_PATH + 'donki_categories_venn.pdf', bbox_inches='tight')
    plt.savefig(SAVE_PLOT_PAPER_PATH + 'donki_categories_venn.png', bbox_inches='tight')
    plt.close()

    print(f"Venn set totals: {totals}")

    for region in sorted(regions, key=regions.get):
        if region:
            print(f"  {region:11s} N={regions[region]:5d}  area={areas[region]:6.2f}  "
                  f"area/sqrt(N)={areas[region] / np.sqrt(regions[region]):5.2f}  "
                  f"clearance={clearance[region]:.2f}")
        else:
            print(f"  {'DONKI only':11s} N={regions[region]:5d}")

    return regions, totals

def cliffs_delta_with_ci(sample_tp, sample_fn, u_statistic, rng, n_bootstrap=4000):
    """
    Cliff's delta for TP against FN, with a percentile bootstrap confidence interval.

    delta runs from -1 to +1 and is zero when neither group tends to exceed the other; positive
    means a randomly drawn TP exceeds a randomly drawn FN more often than the reverse. Reporting
    it alongside the p-value matters here because the groups are very unevenly sized, so a p-value
    reflects how many CMEs there are as much as how far apart the two populations lie.

    It is read off the Mann-Whitney statistic the panel already computes, so the effect size and
    the test cannot disagree, and like that test it assumes nothing about the shape of the
    distributions and handles the heavy ties in the integer-degree quantities through midranks. A
    mean-and-standard-deviation measure such as Cohen's d would misrepresent these panels, where
    the speeds are strongly skewed and the angles are tied at whole degrees.

    `u_statistic` must come from `mannwhitneyu(sample_fn, sample_tp)`, with FN first.
    """
    pair_count = len(sample_fn) * len(sample_tp)
    delta = 1.0 - 2.0 * u_statistic / pair_count

    bootstrap_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resampled_tp = rng.choice(sample_tp, len(sample_tp), replace=True)
        resampled_fn = rng.choice(sample_fn, len(sample_fn), replace=True)
        bootstrap_u, _ = mannwhitneyu(resampled_fn, resampled_tp, alternative='two-sided')
        bootstrap_deltas[i] = 1.0 - 2.0 * bootstrap_u / pair_count

    return (float(delta), float(np.percentile(bootstrap_deltas, 2.5)),
            float(np.percentile(bootstrap_deltas, 97.5)))


def _fn_analysis_populations(donki_df, strudl_cmes_path, overlaps_path,
                             overlap_coefficient=0.1, overlap_minutes=80,
                             elongation_separation=helper_functions.ELONGATION_SEPARATION_DEG):
    """
    The two populations the false-negative analysis compares, and the overlap flag they are judged on.

    Both versions of the figure draw from this, so a change of criterion cannot leave them
    disagreeing about which CMEs were missed. `get_stereo_heeq_lon` goes to JPL Horizons, so the
    result is meant to be computed once and handed to both plotting functions through `prepared`.

    Everything is measured on the raw DONKI parameters rather than the ecliptic-corrected ones,
    since the question is what the instrument and the detector saw, not what the CME did in the
    ecliptic.
    """

    visible = donki_df[donki_df['couldBeInHI1AFov'].fillna(False).astype(bool)].copy()

    sta_lon = helper_functions.get_stereo_heeq_lon(list(visible['time21_5']), sc='A')
    # the signed separation is the second return value: positive means the CME apex sits at a
    # lower HEEQ longitude than the spacecraft. ELCon needs the magnitude, the polar panels need
    # the side, so both are kept.
    visible['phi'], visible['phiSigned'] = elevohi_functions.phi_from_longitudes(
        sta_lon, visible['longitude'].to_numpy(float))
    visible['absLatitude'] = visible['latitude'].abs()

    overlaps_df = pd.read_json(overlaps_path)
    if 'medianElongationSeparation' not in overlaps_df.columns:
        raise RuntimeError(f"{overlaps_path} predates the elongation separation criterion; "
                           "re-run get_hi1a_overlaps in run_pre_fitting_pipeline.py")

    # coerce in case every pair came back null, which would leave an object dtype that cannot
    # be compared against a float
    separation = pd.to_numeric(overlaps_df['medianElongationSeparation'], errors='coerce')

    # a pair obscures one another when it shares the field for long enough, covers the same
    # position angles, and its two fronts stay closer in elongation than the width of one
    # STRUDL annotation, which is the point below which the two fronts fall in the same mask.
    # Note the last test runs the other way round: less separation is more obscuring. A pair
    # that never shared a position angle has no separation, and NaN fails the test, which is
    # the wanted answer since such a pair has no position angle overlap either.
    flagged = ((overlaps_df['medianOverlapCoefficient'] >= overlap_coefficient)
               & (overlaps_df['sharedWindowMinutes'] >= overlap_minutes)
               & (separation <= elongation_separation))
    overlapping_ids = set(overlaps_df.loc[flagged, 'cmeIdA']) | set(overlaps_df.loc[flagged, 'cmeIdB'])
    visible['overlapping'] = visible['associatedActivityID'].isin(overlapping_ids)

    strudl_df = pd.read_json(strudl_cmes_path)
    false_positives = set(strudl_df['track_key']) - set(donki_df['associatedStrudlID'].dropna())

    is_true_positive = visible['associatedStrudlID'].notna()

    return {'visible': visible,
            'populations': {'TP': visible[is_true_positive], 'FN': visible[~is_true_positive]},
            'false_positives': false_positives,
            'n_strudl_tracks': int(len(strudl_df)),
            'overlap_coefficient': overlap_coefficient,
            'overlap_minutes': overlap_minutes,
            'elongation_separation': elongation_separation,
            'n_overlapping_pairs': int(flagged.sum()),
            'n_overlapping_cmes': len(overlapping_ids)}


def _fn_analysis_summary(prepared):
    """The part of the figure summary that describes the sample rather than any one panel."""

    populations = prepared['populations']

    return {'n_tp': int(len(populations['TP'])), 'n_fn': int(len(populations['FN'])),
            'n_fp': int(len(prepared['false_positives'])),
            'n_strudl_tracks': prepared['n_strudl_tracks'],
            'overlap_coefficient': prepared['overlap_coefficient'],
            'overlap_minutes': prepared['overlap_minutes'],
            'elongation_separation': float(prepared['elongation_separation']),
            'n_overlapping_pairs': prepared['n_overlapping_pairs'],
            'n_overlapping_cmes': prepared['n_overlapping_cmes'],
            'panels': {}}


def _plot_kde_panel(ax, populations, colors, key, xlabel, unit, clip_display, font_params, rng):
    """
    The TP and FN distributions of one quantity, with the test of the difference between them.

    The curves carry no label: the two populations are named once for the whole figure, so the
    panel legend is left to the medians, which differ from panel to panel. The test is named in
    the caption rather than in every box, so only its numbers appear here.

    Each panel is scaled to its own taller peak. The absolute density runs to about 0.0018 per km/s
    in the speed panel, which costs a seven-character tick label to say something no reader uses,
    while the comparison the panel exists for survives the rescaling intact.

    `rng` is shared across panels so that the bootstrap intervals are reproducible, which means
    the panels must be drawn in a fixed order.
    """

    values = {cls: populations[cls][key].to_numpy(dtype=float) for cls in ('TP', 'FN')}

    low = min(v.min() for v in values.values())
    high = max(v.max() for v in values.values())
    pad = 0.1 * (high - low)
    # the padding is kept out of the negatives for a quantity that cannot go there, but a signed
    # one is padded like any other: clamping it would cut off half its distribution
    lower = max(low - pad, 0.0) if low >= 0.0 else low - pad
    x_grid = np.linspace(lower, high + pad, 400)

    # both curves are divided by the same number, the taller of the two peaks, so the panel keeps
    # the height of one population against the other and loses only the units, which are a density
    # per km/s or per degree that nobody reads off a panel like this
    curves = {cls: gaussian_kde(values[cls])(x_grid) for cls in ('TP', 'FN')}
    peak = max(curve.max() for curve in curves.values())

    for cls in ('TP', 'FN'):
        v = values[cls]
        ax.plot(x_grid, curves[cls] / peak, color=colors[cls], lw=2)
        ax.axvline(np.median(v), color=colors[cls], ls='--', lw=1.5,
                   label=f'Median {cls}: {np.median(v):.0f}{unit}')

    u_statistic, p_value = mannwhitneyu(values['FN'], values['TP'], alternative='two-sided')
    delta, delta_low, delta_high = cliffs_delta_with_ci(values['TP'], values['FN'],
                                                        u_statistic, rng)
    ax.text(0.97, 0.04,
            f'$\\delta$ = {delta:+.2f} [{delta_low:+.2f}, {delta_high:+.2f}]'
            f'\n$p$ = {p_value:.2g}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=font_params['legend_fontsize'] - 3,
            bbox=dict(boxstyle='round', fc=(1.0, 1.0, 1.0, 0.65), ec=(0.7, 0.7, 0.7, 0.65)))

    ax.set_xlabel(xlabel, fontsize=font_params['label_fontsize'])
    ax.set_ylabel('Normalised density', fontsize=font_params['label_fontsize'])
    ax.tick_params(labelsize=font_params['ticks_fontsize'])
    ax.set_ylim(0.0, 1.05)
    upper = high + pad
    if clip_display:
        upper = min(upper, np.percentile(np.concatenate(list(values.values())), 99))
    ax.set_xlim(lower, upper)
    ax.legend(fontsize=font_params['legend_fontsize'] - 4, loc='upper right', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    return {'median_tp': float(np.median(values['TP'])), 'median_fn': float(np.median(values['FN'])),
            'u': float(u_statistic), 'p': float(p_value),
            'delta': delta, 'delta_ci_low': delta_low, 'delta_ci_high': delta_high}


def _plot_overlap_panel(ax, populations, colors, font_params):
    """
    How often each population shared the field with a CME close enough to obscure it.

    The bars go unlabelled and the panel carries no legend of its own: the two populations are
    named once for the whole figure, and the colours are the same ones used throughout it.
    """

    groups = ['Overlapping', 'Not overlapping']
    positions = np.arange(len(groups))
    width = 0.35
    contingency = {}

    for i, cls in enumerate(('FN', 'TP')):
        flags = populations[cls]['overlapping'].to_numpy()
        counts = np.array([int(flags.sum()), int((~flags).sum())])
        contingency[cls] = counts
        bars = ax.bar(positions + (i - 0.5) * width, 100 * counts / counts.sum(), width,
                      color=colors[cls])
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'n={count}',
                    ha='center', va='bottom', fontsize=font_params['ticks_fontsize'] - 2)

    table = [list(contingency['TP']), list(contingency['FN'])]
    _, p_value = fisher_exact(table)
    # the conditional estimate and its exact interval are the pair that matches Fisher's exact p
    odds_ratio_result = conditional_odds_ratio(table)
    odds_ratio_ci = odds_ratio_result.confidence_interval(0.95)
    ax.text(0.97, 0.97,
            f'OR = {odds_ratio_result.statistic:.2f} '
            f'[{odds_ratio_ci.low:.2f}, {odds_ratio_ci.high:.2f}]\n$p$ = {p_value:.2g}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=font_params['legend_fontsize'] - 3,
            bbox=dict(boxstyle='round', fc=(1.0, 1.0, 1.0, 0.65), ec=(0.7, 0.7, 0.7, 0.65)))

    ax.set_xticks(positions)
    ax.set_xticklabels(groups, fontsize=font_params['label_fontsize'])
    ax.set_ylabel('Fraction of events [%]', fontsize=font_params['label_fontsize'])
    ax.tick_params(labelsize=font_params['ticks_fontsize'])
    ax.set_ylim(0, 122)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.spines[['top', 'right']].set_visible(False)

    return {'tp_overlapping': int(contingency['TP'][0]), 'tp_not_overlapping': int(contingency['TP'][1]),
            'fn_overlapping': int(contingency['FN'][0]), 'fn_not_overlapping': int(contingency['FN'][1]),
            'odds_ratio': float(odds_ratio_result.statistic),
            'odds_ratio_ci_low': float(odds_ratio_ci.low),
            'odds_ratio_ci_high': float(odds_ratio_ci.high), 'p': float(p_value)}


def _enlarged_fonts(font_params, by=2):
    """
    The figure-wide font sizes, one step up.

    Applied inside the two false-negative figures rather than to the shared `font_params` built in
    __main__, so that the other figures in this module keep the sizes they were tuned at.
    """

    return {name: size + by for name, size in font_params.items()}


def _population_legend(fig, populations, colors, font_params, y=0.99):
    """
    The one key both figures carry: which colour is which population, and how many of each.

    Drawn once for the whole figure rather than once per panel, since the colours and the counts
    are the same everywhere in it. The panels are then free to use their legends for what does
    differ between them, which is the median of each distribution.
    """

    handles = [Patch(facecolor=colors[cls], label=f'{cls}, N={len(populations[cls])}')
               for cls in ('TP', 'FN')]

    return fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, y), ncol=2,
                      frameon=False, fontsize=font_params['legend_fontsize'])


def _place_panel_letters(fig, panels, font_params, pad=0.012):
    """
    Label the panels in figure coordinates, aligned by row and column.

    Each panel labelling itself in its own axes coordinates leaves the letters wherever the axes
    happen to sit, and a square polar axes inset in a wider cell starts further right than a
    rectangular one above it. Grouping the panels first and taking one anchor per group puts the
    letters on a common line instead.

    `panels` pairs each axes with its letter. Positions are read straight from the axes, which is
    sound because neither figure uses a layout engine that would move them afterwards; saving with
    bbox_inches='tight' crops the result but does not shift anything within it.
    """

    boxes = [(ax.get_position(), letter) for ax, letter in panels]

    def column(box):
        centre = 0.5 * (box.x0 + box.x1)
        return 'left' if centre < 0.45 else ('right' if centre > 0.55 else 'middle')

    # the left edge of each column and the top of each row, shared by everything in them
    left_edge = {}
    for box, _ in boxes:
        key = column(box)
        left_edge[key] = min(left_edge.get(key, box.x0), box.x0)

    top_edge = {}
    for box, _ in boxes:
        key = round(box.y1, 2)
        top_edge[key] = max(top_edge.get(key, box.y1), box.y1)

    for box, letter in boxes:
        fig.text(left_edge[column(box)], top_edge[round(box.y1, 2)] + pad, f'({letter})',
                 va='bottom', ha='left', fontweight='bold',
                 fontsize=font_params['label_fontsize'])


# the axis labels carry mathmode, which has no place in a text file, so the record keeps plain
# forms of its own
_PANEL_LABELS = {'speed': 'CME speed at 21.5 Rsun [km/s]',
                 'halfAngle': 'CME half-width [deg]',
                 'phi': 'CME-STEREO-A separation, unsigned [deg]',
                 'phiSigned': 'CME-STEREO-A separation, signed [deg]',
                 'absLatitude': '|CME latitude| [deg]',
                 'latitude': 'CME latitude [deg]',
                 'overlap': 'Shared field with an obscuring CME'}


def _odds_ratio_text(record):
    """
    One bin's odds ratio and interval, or dashes where it has none.

    An empty bin has no odds either way, and a bin that caught no detection has a conditional
    estimate of exactly zero, which restates the empty nTP column rather than measuring anything.
    Both are written as dashes instead of as 0.00, which would read as a measured value. The
    interval is kept wherever it is finite, since in those bins it carries what there is to say.
    """

    populated = record['nTp'] + record['nFn'] > 0
    estimate = record['oddsRatio']
    point = (f'{estimate:12.2f}' if populated and np.isfinite(estimate) and estimate > 0.0
             else f"{'-':>12s}")

    low, high = record['ciLow'], record['ciHigh']
    interval = (f'[{low:.2f}, {high:.2f}]'
                if populated and np.isfinite(low) and np.isfinite(high) else '-')

    return f'{point}  {interval:<16s}'


def _format_panel_block(key, panel):
    """
    One panel's statistics, recognised by the keys it carries rather than by the figure it is from.

    The three kinds of panel differ in shape but not in kind, so one formatter serves every version
    of the figure: a KDE panel reports a rank test, a binned panel a homogeneity test and its bins,
    the overlap panel a two-by-two table.
    """

    label = _PANEL_LABELS.get(key, key)
    lines = []

    if 'delta' in panel:
        lines.append(f'PANEL  {label}')
        lines.append(f"  median TP{'':22s}{panel['median_tp']:10.0f}")
        lines.append(f"  median FN{'':22s}{panel['median_fn']:10.0f}")
        lines.append(f"  Mann-Whitney U{'':17s}{panel['u']:10.0f}")
        lines.append(f"  p{'':30s}{panel['p']:>10.2g}")
        lines.append(f"  Cliff's delta{'':18s}{panel['delta']:+10.2f}"
                     f"  95% CI [{panel['delta_ci_low']:+.2f}, {panel['delta_ci_high']:+.2f}]")

    elif 'bins' in panel:
        lines.append(f"PANEL  {label}   ({panel['bin_width_deg']:.0f} deg bins)")
        lines.append(f"  homogeneity chi2{'':15s}{panel['chi2']:10.2f}")
        lines.append(f"  dof{'':28s}{panel['dof']:10d}")
        lines.append(f"  p{'':30s}{panel['p']:>10.2g}")
        lines.append('')
        lines.append(f"  {'bin':<14s}{'nTP':>5s}{'nFN':>6s}{'n':>6s}{'det%':>7s}"
                     f"{'odds ratio':>12s}  {'95% CI':<16s}{'p':>8s}")
        for b in panel['bins']:
            n = b['nTp'] + b['nFn']
            rate = f'{100 * b["nTp"] / n:7.1f}' if n else f'{"-":>7s}'
            p_text = f"{b['p']:8.2g}" if n else f'{"-":>8s}'
            flag = '   sparse' if b['sparse'] else ''
            lines.append(f"  [{b['low']:+4.0f},{b['high']:+5.0f})  {b['nTp']:5d}{b['nFn']:6d}"
                         f"{n:6d}{rate}{_odds_ratio_text(b)}{p_text}{flag}")

    elif 'odds_ratio' in panel:
        lines.append(f'PANEL  {label}')
        lines.append(f"  TP overlapping / not{'':11s}{panel['tp_overlapping']:5d} /"
                     f"{panel['tp_not_overlapping']:5d}")
        lines.append(f"  FN overlapping / not{'':11s}{panel['fn_overlapping']:5d} /"
                     f"{panel['fn_not_overlapping']:5d}")
        lines.append(f"  odds ratio{'':21s}{panel['odds_ratio']:10.2f}"
                     f"  95% CI [{panel['odds_ratio_ci_low']:.2f}, "
                     f"{panel['odds_ratio_ci_high']:.2f}]")
        lines.append(f"  Fisher exact p{'':17s}{panel['p']:>10.2g}")

    return lines


def _format_fn_analysis_report(summary, stem):
    """
    The figure's numbers as a readable record, for quoting in the paper text.

    Written from the same summary the figure is drawn from, at the same moment, so the two cannot
    disagree. The format specifiers are those the panels themselves are labelled with, which is why
    a median reads the same here as it does in the legend it was taken from.
    """

    n_visible = summary['n_tp'] + summary['n_fn']
    rate = 100.0 * summary['n_tp'] / n_visible

    lines = [stem, '=' * len(stem),
             f'written {datetime.now():%Y-%m-%d %H:%M}, beside {stem}.pdf and {stem}.png', '']

    if 'signed_separation' in summary:
        lines += [f"separation drawn as: "
                  f"{'signed' if summary['signed_separation'] else 'unsigned magnitude'}", '']

    lines += ['SAMPLE',
              f"  DONKI CMEs that could have been seen in HI-1A{'':6s}{n_visible:6d}",
              f"    detected by STRUDL (TP){'':25s}{summary['n_tp']:6d}{rate:9.1f} %",
              f"    missed by STRUDL (FN){'':27s}{summary['n_fn']:6d}{100.0 - rate:9.1f} %",
              f"  STRUDL tracks, total{'':30s}{summary['n_strudl_tracks']:6d}",
              f"  STRUDL tracks with no DONKI counterpart (FP){'':7s}{summary['n_fp']:6d}",
              '',
              'OVERLAP CRITERION',
              f"  median overlap coefficient{'':20s}>= {summary['overlap_coefficient']:.2f}",
              f"  shared window{'':33s}>= {summary['overlap_minutes']:.0f} min",
              f"  median elongation separation{'':18s}<= "
              f"{summary['elongation_separation']:.3f} deg",
              f"  CME pairs flagged as overlapping{'':15s}{summary['n_overlapping_pairs']:6d}",
              f"  CMEs involved{'':33s}{summary['n_overlapping_cmes']:6d}",
              '']

    for key, panel in summary['panels'].items():
        lines += _format_panel_block(key, panel) + ['']

    scan = summary.get('bin_width_sensitivity')
    if scan:
        lines += ['BIN WIDTH SENSITIVITY',
                  f"  {'quantity':11s} {'width':>6s} {'bins':>5s} {'sparse':>7s} {'dropped':>8s} "
                  f"{'chi2':>7s} {'dof':>4s} {'p':>8s}"]
        for key, records in scan.items():
            for r in records:
                lines.append(f"  {key:11s} {r['bin_width_deg']:6.0f} {r['n_bins']:5d} "
                             f"{r['n_sparse']:7d} {r['n_discarded']:8d} {r['chi2']:7.2f} "
                             f"{r['dof']:4d} {r['p']:8.4f}")
        lines.append('')
        for key, records in scan.items():
            ps = [r['p'] for r in records if np.isfinite(r['p'])]
            lines.append(f"  {key}: p ranges {min(ps):.4f} to {max(ps):.4f} across "
                         f"{min(r['bin_width_deg'] for r in records):.0f}-"
                         f"{max(r['bin_width_deg'] for r in records):.0f} deg bins")
        lines.append('')

    lines += ['NOTES',
              f"  A bin is sparse when it would be expected to hold fewer than "
              f"{summary.get('min_expected_detections', 5.0):.1f} detections",
              '  under the overall detection rate. Sparse bins are left uncoloured in the figure',
              '  and excluded from the homogeneity test; the threshold is applied to the expected',
              '  count rather than the observed one, so the rule cannot see the result it decides on.',
              '  Odds ratios are conditional estimates with exact intervals, matching Fisher p.',
              '']

    return '\n'.join(lines)


def _latex_number(text):
    """
    A formatted number made safe to typeset, whatever notation Python wrote it in.

    A small p-value comes out of the general format as 7.7e-07, which LaTeX would set literally,
    exponent letter and all. Such a value is rewritten as a product with a power of ten, wrapped so
    that it typesets in text and in maths alike -- these macros are cited in running prose as often
    as in equations. A plain decimal needs neither and is left as it is.
    """

    if 'e' not in text:
        return text

    mantissa, exponent = text.split('e')

    return f'\\ensuremath{{{mantissa}\\times10^{{{int(exponent)}}}}}'


def _format_fn_analysis_macros(summary, prefix):
    """
    The same numbers as LaTeX definitions, in the style of `make_results_table`.

    A macro name cannot contain a digit, which is why the existing definitions read `tabOne` rather
    than `tabOne1`; the names built here obey the same rule. `prefix` must differ between figures:
    all of them describe the same populations, so the sample-level names would otherwise collide and
    a manuscript inputting two of these files would fail on the redefinition.
    """

    n_visible = summary['n_tp'] + summary['n_fn']
    macros = [('% sample', None),
              (f'{prefix}NVisible', f"{n_visible}"),
              (f'{prefix}NTp', f"{summary['n_tp']}"),
              (f'{prefix}NFn', f"{summary['n_fn']}"),
              (f'{prefix}NFp', f"{summary['n_fp']}"),
              (f'{prefix}NStrudlTracks', f"{summary['n_strudl_tracks']}"),
              (f'{prefix}DetectionRate', f"{100.0 * summary['n_tp'] / n_visible:.1f}"),
              ('% overlap criterion', None),
              (f'{prefix}OverlapCoefficient', f"{summary['overlap_coefficient']:.2f}"),
              (f'{prefix}OverlapMinutes', f"{summary['overlap_minutes']:.0f}"),
              (f'{prefix}ElongationSeparation', f"{summary['elongation_separation']:.3f}"),
              (f'{prefix}NOverlappingPairs', f"{summary['n_overlapping_pairs']}"),
              (f'{prefix}NOverlappingCmes', f"{summary['n_overlapping_cmes']}")]

    for key, panel in summary['panels'].items():
        name = key[0].upper() + key[1:]
        macros.append((f'% panel: {_PANEL_LABELS.get(key, key)}', None))

        if 'delta' in panel:
            macros += [(f'{prefix}{name}MedianTp', f"{panel['median_tp']:.0f}"),
                       (f'{prefix}{name}MedianFn', f"{panel['median_fn']:.0f}"),
                       (f'{prefix}{name}P', _latex_number(f"{panel['p']:.2g}")),
                       (f'{prefix}{name}Delta', f"{panel['delta']:+.2f}"),
                       (f'{prefix}{name}DeltaCiLow', f"{panel['delta_ci_low']:+.2f}"),
                       (f'{prefix}{name}DeltaCiHigh', f"{panel['delta_ci_high']:+.2f}")]

        elif 'bins' in panel:
            macros += [(f'{prefix}{name}BinWidth', f"{panel['bin_width_deg']:.0f}"),
                       (f'{prefix}{name}ChiSq', f"{panel['chi2']:.2f}"),
                       (f'{prefix}{name}Dof', f"{panel['dof']}"),
                       (f'{prefix}{name}P', _latex_number(f"{panel['p']:.2g}"))]

        elif 'odds_ratio' in panel:
            macros += [(f'{prefix}{name}TpOverlapping', f"{panel['tp_overlapping']}"),
                       (f'{prefix}{name}TpNotOverlapping', f"{panel['tp_not_overlapping']}"),
                       (f'{prefix}{name}FnOverlapping', f"{panel['fn_overlapping']}"),
                       (f'{prefix}{name}FnNotOverlapping', f"{panel['fn_not_overlapping']}"),
                       (f'{prefix}{name}OddsRatio', f"{panel['odds_ratio']:.2f}"),
                       (f'{prefix}{name}OrCiLow', f"{panel['odds_ratio_ci_low']:.2f}"),
                       (f'{prefix}{name}OrCiHigh', f"{panel['odds_ratio_ci_high']:.2f}"),
                       (f'{prefix}{name}P', _latex_number(f"{panel['p']:.2g}"))]

    scan = summary.get('bin_width_sensitivity')
    if scan:
        macros.append(('% bin width sensitivity', None))
        for key, records in scan.items():
            name = key[0].upper() + key[1:]
            ps = [r['p'] for r in records if np.isfinite(r['p'])]
            macros += [(f'{prefix}Sens{name}PMin', _latex_number(f"{min(ps):.4f}")),
                       (f'{prefix}Sens{name}PMax', _latex_number(f"{max(ps):.4f}"))]

    lines = []
    for name, value in macros:
        if value is None:
            lines += ['', name]
        else:
            lines.append(f'\\newcommand\\{name}{{{value}}}')

    return '\n'.join(lines)


def write_fn_analysis_results(summary, save_path, stem, macro_prefix):
    """
    The figure's numbers beside the figure: a readable record and a set of LaTeX definitions.

    Written from inside the plotting function rather than from __main__, so that a figure cannot be
    saved without its numbers and the two can never describe different runs. The `_latex.txt`
    suffix follows `results_latex.txt`, the file `make_results_table` already writes.
    """

    with open(f'{save_path}{stem}.txt', 'w') as handle:
        handle.write(_format_fn_analysis_report(summary, stem) + '\n')

    with open(f'{save_path}{stem}_latex.txt', 'w') as handle:
        handle.write(f'% {stem} -- written by make_paper_plots.py, do not edit by hand\n')
        handle.write(_format_fn_analysis_macros(summary, macro_prefix) + '\n')


def make_strudl_fn_analysis_plot(donki_df, strudl_cmes_path, overlaps_path, color_dict,
                                 font_params, SAVE_PLOT_PAPER_PATH,
                                 overlap_coefficient=0.1, overlap_minutes=80,
                                 elongation_separation=helper_functions.ELONGATION_SEPARATION_DEG,
                                 signed_separation=False, prepared=None):
    """
    Characteristics of the CMEs STRUDL missed against the ones it found.

    Both populations are DONKI CMEs that could have been seen in HI-1: a true positive carries a
    STRUDL match, a false negative does not. False positives are STRUDL tracks with no DONKI
    counterpart; they have no DONKI parameters, so they appear only in the returned summary.

    Everything is measured on the raw DONKI parameters rather than the ecliptic-corrected ones,
    since the question is what the instrument and the detector saw, not what the CME did in the
    ecliptic.

    Panels (a) to (d) each separate the two populations over one DONKI parameter, as densities
    with the Mann-Whitney test of the difference between them. Panel (e) instead compares how
    often each population shared the field with a CME close enough to obscure it.

    `signed_separation` draws panel (c) on the signed separation phi = wrap180(sta_lon - cme_lon),
    positive where the CME apex sits at a lower HEEQ longitude than the spacecraft, which is the
    convention `plot_separation_polar_panel` is drawn in; the default takes its magnitude instead.
    The choice is not cosmetic: on the magnitude the Mann-Whitney test and Cliff's delta ask whether
    the missed CMEs lie further from the spacecraft than the detected ones, and on the signed value
    they ask whether they lie to one side of it, where a delta near zero means the two sides balance
    rather than that the separations agree. The signed figure is written to its own files, so the two
    can be compared without re-running.

    `prepared` takes the result of `_fn_analysis_populations` when the populations have already
    been built, which also means the overlap criteria come from there rather than from the
    keywords here.
    """

    if prepared is None:
        prepared = _fn_analysis_populations(donki_df, strudl_cmes_path, overlaps_path,
                                            overlap_coefficient, overlap_minutes,
                                            elongation_separation)

    populations = prepared['populations']
    colors = {'TP': color_dict['geosphere_green'], 'FN': color_dict['geosphere_purple']}
    font_params = _enlarged_fonts(font_params)

    separation_key = 'phiSigned' if signed_separation else 'phi'
    separation_label = ('Signed CME-STEREO-A separation [deg]' if signed_separation
                        else 'CME-STEREO-A separation [deg]')

    # the last entry cuts the *view* at the 99th percentile where a long tail would otherwise
    # squash the informative part of the axis; the density is still estimated from every event
    panels = [('speed', 'CME speed at 21.5 R$_\\odot$ [km s$^{-1}$]', ' km s$^{-1}$', True),
              ('halfAngle', 'CME half-width [deg]', '°', False),
              (separation_key, separation_label, '°', False),
              ('absLatitude', '|CME latitude| [deg]', '°', False)]

    # rows of two, two and one; the 4-column grid lets the last panel span the middle two
    # columns so it sits centred under the pair above it
    fig = plt.figure(figsize=(13, 14))
    grid = GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.65, top=0.92)
    axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[1, 0:2]), fig.add_subplot(grid[1, 2:4]),
            fig.add_subplot(grid[2, 1:3])]

    summary = _fn_analysis_summary(prepared)
    summary['signed_separation'] = bool(signed_separation)

    # one stream shared by the four panels, so the intervals are reproducible
    bootstrap_rng = np.random.default_rng(0)

    for ax, (key, xlabel, unit, clip_display) in zip(axes[:4], panels):
        summary['panels'][key] = _plot_kde_panel(ax, populations, colors, key, xlabel, unit,
                                                 clip_display, font_params, bootstrap_rng)

    # the two panels on the right of each row share their neighbour's axis
    for ax in (axes[1], axes[3]):
        ax.set_ylabel('')

    summary['panels']['overlap'] = _plot_overlap_panel(axes[4], populations, colors, font_params)

    _population_legend(fig, populations, colors, font_params)
    _place_panel_letters(fig, list(zip(axes, 'abcde')), font_params)

    suffix = '_signed' if signed_separation else ''
    plt.savefig(SAVE_PLOT_PAPER_PATH + f'strudl_fn_analysis{suffix}.pdf', bbox_inches='tight')
    plt.savefig(SAVE_PLOT_PAPER_PATH + f'strudl_fn_analysis{suffix}.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)

    write_fn_analysis_results(summary, SAVE_PLOT_PAPER_PATH, f'strudl_fn_analysis{suffix}',
                              'FNsigned' if signed_separation else 'FN')

    return summary


# the geometry of the two polar panels, in units of the radius of the wedge ring's outer edge
_R_SUN = 0.30
_R_RING_INNER = 0.42
_R_RING_WIDTH = 0.58
_R_SPACECRAFT = 1.09
_R_LIMIT = 1.15


def detection_odds_colormap(color_dict):
    """
    The diverging scale the wedges are painted on: two hues about a neutral midpoint.

    Purple is the false-negative colour and green the true-positive colour everywhere else in the
    figure, so a bin where STRUDL detected worse than the rest of the sample reads purple and one
    where it detected better reads green. The midpoint is grey rather than a third hue, so that an
    odds ratio of one shows up as the absence of a difference instead of as a value of its own.
    """

    return LinearSegmentedColormap.from_list(
        'detection_odds',
        [color_dict['geosphere_purple'], '#EEEEEE', color_dict['geosphere_green']])


def binned_detection_odds(populations, key, edges, min_expected_detections=5.0):
    """
    The odds that STRUDL detected a CME in each bin of `key`, against the rest of the sample.

    Each bin gives the table [[tp_in, fn_in], [tp_out, fn_out]], so an odds ratio above one means
    the detected fraction inside the bin exceeds the detected fraction everywhere else. The
    conditional estimate and its exact interval are the pair that matches Fisher's exact p, as in
    the overlap panel, but it collapses to zero in a bin holding no true positive at all, and zero
    has no logarithm and so no colour. `oddsRatioPlot` therefore carries the sample odds ratio with
    the Haldane-Anscombe half added to every cell, which stays finite, and that is the number the
    wedges are painted with; the reported estimate is the conditional one.

    A bin is flagged `sparse`, and left uncoloured by the panels, when it would be expected to hold
    fewer than `min_expected_detections` detections. What limits one of these ratios is the count in
    its smallest cell, and with detections the minority class by some margin that cell is the one
    counting them: a bin expecting one detection moves by a factor of two on a single event, however
    many CMEs went into it. The threshold is applied to the expected count under the overall
    detection rate rather than to the observed one, so that the rule cannot see the result it
    decides on. Gating on the detections a bin actually got would strip out the bins where few were
    found, which are the very bins the figure is drawn to show.

    The default of five is a convention.

    It is also the expected cell count that `detection_homogeneity_test` is subject to, where the
    requirement is real rather than cosmetic -- that test is asymptotic and misstates its p-value
    on thin bins, which Cochran's rule of thumb puts at an expected count of five. Sharing the one
    threshold keeps the coloured bins and the tested bins the same set, but the rule is cited for
    the test; the wedges borrow the number.
    """

    values = {cls: populations[cls][key].to_numpy(dtype=float) for cls in ('TP', 'FN')}
    counts = {cls: np.histogram(values[cls], bins=edges)[0] for cls in ('TP', 'FN')}

    for cls in ('TP', 'FN'):
        if int(counts[cls].sum()) != len(values[cls]):
            raise ValueError(f"{len(values[cls]) - int(counts[cls].sum())} {cls} values of "
                             f"'{key}' fall outside [{edges[0]}, {edges[-1]}]")

    totals = {cls: int(counts[cls].sum()) for cls in ('TP', 'FN')}
    detection_rate = totals['TP'] / (totals['TP'] + totals['FN'])
    bins = []

    for i in range(len(edges) - 1):
        tp_in, fn_in = int(counts['TP'][i]), int(counts['FN'][i])
        tp_out, fn_out = totals['TP'] - tp_in, totals['FN'] - fn_in

        table = [[tp_in, fn_in], [tp_out, fn_out]]
        _, p_value = fisher_exact(table)
        odds_ratio_result = conditional_odds_ratio(table)
        odds_ratio_ci = odds_ratio_result.confidence_interval(0.95)

        # an empty bin has no odds either way, and the correction alone would hand it the ratio of
        # the two population sizes, which says nothing about that bin
        corrected = (float('nan') if tp_in + fn_in == 0 else
                     ((tp_in + 0.5) * (fn_out + 0.5)) / ((fn_in + 0.5) * (tp_out + 0.5)))

        expected_tp = (tp_in + fn_in) * detection_rate

        bins.append({'low': float(edges[i]), 'high': float(edges[i + 1]),
                     'nTp': tp_in, 'nFn': fn_in, 'expectedTp': float(expected_tp),
                     'oddsRatio': float(odds_ratio_result.statistic),
                     'oddsRatioPlot': float(corrected),
                     'ciLow': float(odds_ratio_ci.low), 'ciHigh': float(odds_ratio_ci.high),
                     'p': float(p_value),
                     'sparse': bool(expected_tp < min_expected_detections)})

    return bins


def detection_homogeneity_test(bins):
    """
    Whether the detected fraction depends on the binned quantity at all.

    Taken over the counts of every bin at once rather than bin by bin, so the quantity gets one
    statement instead of one borderline p-value per wedge, and over the bins that are not sparse,
    which is the same expected-count cut the test itself wants: unlike the exact statistics on each
    bin, this one is asymptotic and goes wrong where the expected counts are tiny. The polar panels
    show no statistics themselves, so this is reported through the figure summary. Returns NaN where
    fewer than two bins survive that cut.
    """

    table = np.array([[b['nTp'], b['nFn']] for b in bins if not b['sparse']], dtype=float)

    # a bin holding nothing at all is an empty row, and its expected frequencies come out zero,
    # which the test cannot divide by. No positive threshold admits such a bin, since it expects
    # no detections either, but the test must not depend on the threshold to stay valid.
    if table.size:
        table = table[table.sum(axis=1) > 0]

    if table.shape[0] < 2 or not table.sum(axis=0).all():
        return float('nan'), float('nan'), 0

    chi2, p_value, dof, _ = chi2_contingency(table)

    return float(chi2), float(p_value), int(dof)


def _check_bin_width(bin_width_deg):
    """
    Reject a bin width that would put a bin across the end of either range.

    A bin straddling +-180 deg of separation or a pole is not a bin of anything: the quantity is
    continuous across that edge but the histogram is not, so the events either side of it would be
    counted apart. Shared by the figure and by the sensitivity scan, so that no width can be
    admitted to one and refused by the other.
    """

    for span in (360.0, 180.0):
        if abs(round(span / bin_width_deg) - span / bin_width_deg) > 1e-9:
            raise ValueError(f'bin_width_deg={bin_width_deg} does not divide {span:.0f} deg '
                             'exactly, so a bin would straddle the end of the range')


def _detection_bin_edges(bin_width_deg):
    """
    The separation and latitude edges of one width, as the pair the two polar panels are drawn on.

    The scan has to bin exactly as the figure does, or it would report the sensitivity of a
    different analysis, so both take their edges from here.
    """

    return (np.linspace(-180.0, 180.0, int(round(360.0 / bin_width_deg)) + 1),
            np.linspace(-90.0, 90.0, int(round(180.0 / bin_width_deg)) + 1))


def detection_bin_width_sensitivity(populations, widths=(15.0, 20.0, 30.0, 45.0, 60.0),
                                    min_expected_detections=5.0, verbose=True):
    """
    The homogeneity test repeated across bin widths, so the choice of width is visible.

    The width of the wedges is a decision, and the test is not indifferent to it. A narrow binning
    resolves more structure but sends more bins under the expected-count threshold, where they are
    dropped from the test and left uncoloured; a wide one tests nearly everything at coarser
    resolution. Both the p-value and which CMEs it was computed from move accordingly.

    Reporting the test at a single width would leave that invisible, and choosing the width that
    minimises p would make the p-value a statement about the choice rather than about the CMEs. The
    scan is therefore computed whenever the figure is, and the claim to quote from it is whether the
    structure holds across the row rather than what it reaches at any one width. `n_discarded`
    counts the CMEs the threshold removed from the test, which is the cost of a narrow binning and
    the quantity a width should be argued from, since it can be read without looking at any p-value.
    """

    scan = {'phiSigned': [], 'latitude': []}

    for width in widths:
        _check_bin_width(width)
        edges = dict(zip(('phiSigned', 'latitude'), _detection_bin_edges(width)))

        for key in scan:
            bins = binned_detection_odds(populations, key, edges[key], min_expected_detections)
            chi2, p_value, dof = detection_homogeneity_test(bins)
            sparse = [b for b in bins if b['sparse']]

            scan[key].append({'bin_width_deg': float(width), 'chi2': chi2, 'dof': dof,
                              'p': p_value, 'n_bins': len(bins), 'n_sparse': len(sparse),
                              'n_discarded': int(sum(b['nTp'] + b['nFn'] for b in sparse)),
                              'discarded_deg': float(sum(b['high'] - b['low'] for b in sparse))})

    if verbose:
        print('\nbin width sensitivity of the detection homogeneity test:')
        print(f"  {'quantity':11s} {'width':>6s} {'bins':>5s} {'sparse':>7s} {'dropped':>8s} "
              f"{'chi2':>7s} {'dof':>4s} {'p':>8s}")
        for key, records in scan.items():
            for r in records:
                print(f"  {key:11s} {r['bin_width_deg']:6.0f} {r['n_bins']:5d} {r['n_sparse']:7d} "
                      f"{r['n_discarded']:8d} {r['chi2']:7.2f} {r['dof']:4d} {r['p']:8.4f}")

    return scan


def _draw_sun_disc(ax, color_dict):
    """The Sun at the centre of a polar panel, the body both views are drawn around."""

    theta = np.linspace(-np.pi, np.pi, 181)
    # the fill is drawn without an edge: a polygon closed over a full turn would draw its closing
    # edge as a seam straight across the disc, so the rim is a constant-radius arc of its own
    ax.fill_between(theta, 0.0, _R_SUN, color=color_dict['geosphere_yellow'], lw=0.0, zorder=3)
    ax.plot(theta, np.full_like(theta, _R_SUN), color=color_dict['geosphere_orange'], lw=1.2,
            zorder=4)


def _draw_odds_wedges(ax, bins, cmap, norm):
    """
    One wedge per bin, as a ring around the Sun, coloured by the bin's odds of detection.

    `ax.bar` on a polar axes already draws a wedge per entry, so the ring needs no patches of its
    own.
    """

    centres = np.deg2rad([0.5 * (b['low'] + b['high']) for b in bins])
    widths = np.deg2rad([b['high'] - b['low'] for b in bins])
    facecolors = ['0.85' if b['sparse'] else cmap(norm(np.log2(b['oddsRatioPlot'])))
                  for b in bins]

    wedges = ax.bar(centres, height=_R_RING_WIDTH, width=widths, bottom=_R_RING_INNER,
                    color=facecolors, edgecolor='white', lw=0.6, zorder=2)

    # a hatch takes the edge colour, so the sparse wedges need a darker edge of their own
    for wedge, record in zip(wedges, bins):
        if record['sparse']:
            wedge.set_hatch('///')
            wedge.set_edgecolor('0.45')


# past this many bin boundaries on one panel, only every second one is labelled
_MAX_POLAR_TICK_LABELS = 12


def _bin_edge_ticks(bins, full_turn):
    """
    The angular ticks of a polar panel: the boundaries of the bins it draws.

    Taken from the records the wedges themselves are drawn from, so that every label sits on a
    wedge boundary whatever width the bins were built at. A full turn leaves out the last
    boundary, which is the first one seen again from the other side.

    Returns the boundaries together with a flag each saying whether it carries a label. Past
    `_MAX_POLAR_TICK_LABELS` boundaries only every second one does, counted out from the boundary
    nearest zero: that is the reference direction of either panel, the Sun-STEREO-A line or the
    equator, and counting from the end of the range instead would sometimes drop it.
    """

    ticks = np.array([b['low'] for b in bins] + ([] if full_turn else [bins[-1]['high']]))
    step = 2 if len(ticks) > _MAX_POLAR_TICK_LABELS else 1
    anchor = int(np.argmin(np.abs(ticks)))

    return ticks, [(i - anchor) % step == 0 for i in range(len(ticks))]


def _finish_polar_panel(ax, font_params):
    """The radial axis and the spines, shared by both polar panels."""

    ax.set_ylim(0.0, _R_LIMIT)
    ax.set_yticks([])
    # a sector axes carries straight spines along its cut as well as the polar one, and none of
    # them are wanted: the ring itself is the only outline the panels need. They are cleared by
    # colour rather than by visibility, which a polar axes recomputes for itself as it draws.
    for spine in ax.spines.values():
        spine.set_color('none')
    ax.grid(False)
    ax.tick_params(axis='x', labelsize=font_params['ticks_fontsize'] - 2, pad=2)


def plot_separation_polar_panel(ax, bins, cmap, norm, color_dict, font_params):
    """
    Where around STEREO-A the CMEs sit, seen from solar north.

    The polar angle is the signed separation phi = wrap180(sta_lon - cme_lon), zero along the
    Sun-STEREO-A line and growing clockwise on the page. A CME at a lower HEEQ longitude than the
    spacecraft therefore falls on the left half and one at a higher longitude on the right, which
    puts HEEQ longitude growing counter-clockwise: the sense of a view down from solar north.
    """

    ax.set_theta_zero_location('S')
    ax.set_theta_direction(-1)

    _draw_odds_wedges(ax, bins, cmap, norm)
    _draw_sun_disc(ax, color_dict)

    ax.plot([0.0], [_R_SPACECRAFT], marker='^', ms=14, color='black', clip_on=False, zorder=4)

    # the zero tick is the Sun-STEREO-A line, so the spacecraft names that tick instead of
    # carrying a label of its own next to the marker. The angles are wrapped onto [0, 360) before
    # they are handed over, since a negative tick widens the axis past a full turn and the ring
    # then wraps over itself.
    ticks, labelled = _bin_edge_ticks(bins, full_turn=True)
    labels = [('STEREO-A' if t == 0.0 else
               ('±180°' if t == -180.0 else f'{t:+.0f}°')) if show else ''
              for t, show in zip(ticks, labelled)]
    ax.set_thetagrids(ticks % 360.0, labels)

    _finish_polar_panel(ax, font_params)


def plot_latitude_polar_panel(ax, bins, cmap, norm, color_dict, font_params):
    """
    The latitudes of the CMEs, seen edge-on.

    The polar angle is the DONKI latitude, so north is up and south is down. Only one limb is
    drawn: a band of constant latitude falls on both limbs of an edge-on view and the two are
    mirror images, so the second carries nothing the first does not.
    """

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_thetamin(-90.0)
    ax.set_thetamax(90.0)

    ax.plot([0.0, 0.0], [0.0, _R_LIMIT], color='0.4', ls='--', lw=1.2, zorder=1)

    _draw_odds_wedges(ax, bins, cmap, norm)
    _draw_sun_disc(ax, color_dict)

    # the ticks keep their signs here, unlike the separation panel: inside a sector the wrapped
    # angles would fall outside the limits and widen it again
    ticks, labelled = _bin_edge_ticks(bins, full_turn=False)
    labels = [('0°' if t == 0.0 else f'{t:+.0f}°') if show else ''
              for t, show in zip(ticks, labelled)]
    ax.set_thetagrids(ticks, labels)

    _finish_polar_panel(ax, font_params)


def make_strudl_fn_analysis_plot_v2(donki_df, strudl_cmes_path, overlaps_path, color_dict,
                                    font_params, SAVE_PLOT_PAPER_PATH,
                                    overlap_coefficient=0.1, overlap_minutes=80,
                                    elongation_separation=helper_functions.ELONGATION_SEPARATION_DEG,
                                    bin_width_deg=45.0, min_expected_detections=5.0,
                                    log2_odds_cap=3.0, prepared=None):
    """
    The false-negative analysis with the two geometric quantities drawn as the geometry they are.

    Panels (a), (b) and (e) are those of `make_strudl_fn_analysis_plot`. Panels (c) and (d) replace
    the distributions of the unsigned separation and the unsigned latitude with two polar views:
    the separation seen from solar north, the latitude seen edge-on. Taking the magnitude, as the
    first version does, puts a CME east of the spacecraft in the same bin as one west of it, and a
    southern CME in the same bin as a northern one, which is what these panels are asked about.

    Each bin is painted with its odds of detection against the rest of the sample, so the panels
    answer where STRUDL detected well rather than only where the CMEs were. Both panels share one
    colour scale and one colourbar, which is what makes them comparable. Neither carries a
    statistic of its own; the test of each is returned in the summary instead.

    `bin_width_deg` sets the width of both sets of bins and must divide 360 and 180 exactly, so
    that no bin straddles +-180 deg of separation or a pole. `min_expected_detections` is the point
    below which a bin is left uncoloured, in detections it would be expected to hold rather than in
    CMEs. `log2_odds_cap` bounds the colour scale at a factor of 2**cap either way, so that one
    extreme bin cannot flatten the rest.
    """

    _check_bin_width(bin_width_deg)

    if prepared is None:
        prepared = _fn_analysis_populations(donki_df, strudl_cmes_path, overlaps_path,
                                            overlap_coefficient, overlap_minutes,
                                            elongation_separation)

    populations = prepared['populations']
    colors = {'TP': color_dict['geosphere_green'], 'FN': color_dict['geosphere_purple']}
    font_params = _enlarged_fonts(font_params)

    separation_edges, latitude_edges = _detection_bin_edges(bin_width_deg)

    separation_bins = binned_detection_odds(populations, 'phiSigned', separation_edges,
                                            min_expected_detections)
    latitude_bins = binned_detection_odds(populations, 'latitude', latitude_edges,
                                          min_expected_detections)

    # one scale across both panels, from the bins that are actually coloured, so that the same
    # colour means the same odds ratio in either view
    spread = [abs(np.log2(b['oddsRatioPlot'])) for b in separation_bins + latitude_bins
              if not b['sparse']]
    limit = min(float(np.ceil(max(spread))), float(log2_odds_cap)) if spread else 1.0
    cmap = detection_odds_colormap(color_dict)
    norm = plt.Normalize(vmin=-limit, vmax=limit)

    # the polar row is given the most height; its colourbar hangs off the latitude panel, which
    # shows only half a disc and so leaves the room for it
    fig = plt.figure(figsize=(13, 16))
    grid = GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.25, 1.0], hspace=0.38, wspace=0.65,
                    top=0.92)
    axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[1, 0:2], projection='polar'),
            fig.add_subplot(grid[1, 2:4], projection='polar'),
            fig.add_subplot(grid[2, 1:3])]

    summary = _fn_analysis_summary(prepared)
    summary['bin_width_deg'] = float(bin_width_deg)
    summary['min_expected_detections'] = float(min_expected_detections)

    bootstrap_rng = np.random.default_rng(0)

    panels = [('speed', 'CME speed at 21.5 R$_\\odot$ [km s$^{-1}$]', ' km s$^{-1}$', True),
              ('halfAngle', 'CME half-width [deg]', '°', False)]

    for ax, (key, xlabel, unit, clip_display) in zip(axes[:2], panels):
        summary['panels'][key] = _plot_kde_panel(ax, populations, colors, key, xlabel, unit,
                                                 clip_display, font_params, bootstrap_rng)

    axes[1].set_ylabel('')

    plot_separation_polar_panel(axes[2], separation_bins, cmap, norm, color_dict, font_params)
    plot_latitude_polar_panel(axes[3], latitude_bins, cmap, norm, color_dict, font_params)

    # the panels carry no statistics of their own, so the tests are reported through the summary
    for key, panel_bins in (('phiSigned', separation_bins), ('latitude', latitude_bins)):
        chi2, p_value, dof = detection_homogeneity_test(panel_bins)
        summary['panels'][key] = {'bin_width_deg': float(bin_width_deg), 'chi2': chi2, 'dof': dof,
                                  'p': p_value, 'bins': panel_bins}

    # the width the figure is drawn at is scanned alongside the standard set, so the record always
    # contains the test as this figure ran it
    summary['bin_width_sensitivity'] = detection_bin_width_sensitivity(
        populations, widths=tuple(sorted({15.0, 20.0, 30.0, 45.0, 60.0, float(bin_width_deg)})),
        min_expected_detections=min_expected_detections)

    scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_map.set_array([])
    # anchored to the latitude panel's box, so it follows that panel wherever the grid puts it
    cbar = fig.colorbar(scalar_map, cax=axes[3].inset_axes([1.06, 0.15, 0.035, 0.70]),
                        orientation='vertical')
    # the odds ratio is multiplicative, so the scale is linear in log2 but labelled in odds ratios
    powers = np.arange(-np.floor(limit), np.floor(limit) + 1)
    cbar.set_ticks(powers)
    cbar.set_ticklabels(['1' if p == 0 else (f'{2 ** int(p):g}' if p > 0 else f'1/{2 ** int(-p):g}')
                         for p in powers])
    cbar.set_label('Detection Odds Ratio', fontsize=font_params['label_fontsize'])
    cbar.ax.tick_params(labelsize=font_params['ticks_fontsize'])

    summary['panels']['overlap'] = _plot_overlap_panel(axes[4], populations, colors, font_params)

    _population_legend(fig, populations, colors, font_params)
    _place_panel_letters(fig, list(zip(axes, 'abcde')), font_params)

    plt.savefig(SAVE_PLOT_PAPER_PATH + 'strudl_fn_analysis_v2.pdf', bbox_inches='tight')
    plt.savefig(SAVE_PLOT_PAPER_PATH + 'strudl_fn_analysis_v2.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    write_fn_analysis_results(summary, SAVE_PLOT_PAPER_PATH, 'strudl_fn_analysis_v2', 'FNvTwo')

    return summary


def _read_paired_tracks(paired_tracks_path):
    """
    The paired tracks that carry an automatic counterpart to compare against.

    That is narrower than STRUDL having detected the CME: `strudlTrackAvailable` is the flag
    `build_paired_track_records` sets for the tracks that survive the ecliptic-correction and
    Earth-position-angle cuts, 80 of the 166 hand-corrected tracks.
    """

    paired_df = pd.read_json(paired_tracks_path)

    return paired_df[paired_df['strudlTrackAvailable'].astype(bool)]


def _paired_track_overlap(row):
    """
    The two elongation profiles of one paired track, over the time range they share.

    The manual profile is interpolated onto the STRUDL timestamps, which is why the range matters:
    most STRUDL tracks begin before the first manual point, and `np.interp` holds the manual
    elongation flat outside its range rather than refusing, so a comparison taken over the whole
    STRUDL track would be measuring that flat extension rather than the disagreement between two
    tracks. The STRUDL timestamps are the ones kept because STRUDL is the thing being judged and
    samples the more densely of the two.
    """

    time_manual = pd.to_datetime(pd.Series(row['time_manual'])).to_numpy()
    time_strudl = pd.to_datetime(pd.Series(row['time_strudl'])).to_numpy()
    elongation_manual = np.asarray(row['elongation_manual'], dtype=float)
    elongation_strudl = np.asarray(row['elongation_strudl'], dtype=float)

    shared = ((time_strudl >= max(time_manual.min(), time_strudl.min()))
              & (time_strudl <= min(time_manual.max(), time_strudl.max())))

    # date2num rather than the raw timestamps, matching how the corrected and uncorrected
    # tracks are compared in get_results_table_one
    manual_interp = np.interp(mdates.date2num(time_strudl[shared]),
                              mdates.date2num(time_manual), elongation_manual)

    return manual_interp, elongation_strudl[shared]


def _paired_track_metrics(paired_tracks_path):
    """
    How far each STRUDL track got along its manual counterpart, and how closely it followed it.

    The reach ratio divides the two maximum elongations and so needs no common time base. The mean
    absolute difference does, and takes the shared window `_paired_track_overlap` returns.
    """

    records = []

    for _, row in _read_paired_tracks(paired_tracks_path).iterrows():
        manual, strudl = _paired_track_overlap(row)

        records.append({'cmeID_elevo': row['cmeID_elevo'],
                        'reach_ratio': float(np.max(row['elongation_strudl'])
                                             / np.max(row['elongation_manual'])),
                        'mean_abs_difference': float(np.mean(np.abs(manual - strudl))),
                        'n_shared_points': int(manual.size)})

    return pd.DataFrame(records)


def _paired_track_profiles(paired_tracks_path):
    """
    The paired tracks as pooled point pairs and as per-event elongation spans.

    The point pairs are every shared-window sample of every track, pooled: the agreement and the
    residual panels are built from them and neither asks which track a point came from. The spans
    are per event and taken over the whole track rather than the shared window, because the
    retention panel asks at each elongation which tracks reach it, which is a question about how
    far a track got and not about where the two overlap in time.
    """

    manual, strudl, spans = [], [], []

    for _, row in _read_paired_tracks(paired_tracks_path).iterrows():
        manual_shared, strudl_shared = _paired_track_overlap(row)

        manual.append(manual_shared)
        strudl.append(strudl_shared)
        spans.append((np.min(row['elongation_manual']), np.max(row['elongation_manual']),
                      np.min(row['elongation_strudl']), np.max(row['elongation_strudl'])))

    return (np.concatenate(manual), np.concatenate(strudl),
            np.asarray(spans, dtype=float), len(spans))


def _plot_performance_panel(ax, groups, colors, x_key, y_key, font_params):
    """
    One tracking-performance measure against one CME parameter, split by the overlap flag.

    The points carry no label: the two groups are named once for the whole figure. The box instead
    reports Spearman's rank correlation within each group and over all events, which is the test the
    panel exists for. It is rank based, so it neither assumes a straight-line relation nor cares that
    the speeds are skewed and the angles tied at whole degrees.
    """

    lines = []

    for name, mask in groups.items():
        ax.scatter(mask[x_key], mask[y_key], s=45, alpha=0.75, color=colors[name],
                   edgecolor='none')
        rho, p_value = spearmanr(mask[x_key], mask[y_key])
        lines.append(f'{name}: $\\rho$ = {rho:.2f}, $p$ = {p_value:.2g}')

    combined = pd.concat(list(groups.values()))
    rho_all, p_all = spearmanr(combined[x_key], combined[y_key])
    lines.append(f'All: $\\rho$ = {rho_all:.2f}, $p$ = {p_all:.2g}')

    ax.text(0.97, 0.97, '\n'.join(lines), transform=ax.transAxes, ha='right', va='top',
            fontsize=font_params['legend_fontsize'] - 6,
            bbox=dict(boxstyle='round', fc=(1.0, 1.0, 1.0, 0.65), ec=(0.7, 0.7, 0.7, 0.65)))

    ax.tick_params(labelsize=font_params['ticks_fontsize'])
    ax.spines[['top', 'right']].set_visible(False)

    return {name: dict(zip(('rho', 'p'), spearmanr(mask[x_key], mask[y_key])))
            for name, mask in groups.items()} | {'all': {'rho': float(rho_all), 'p': float(p_all)}}


def make_strudl_tracking_performance_plot_scatter(donki_df, paired_tracks_path, strudl_cmes_path,
                                                  overlaps_path, color_dict, font_params,
                                                  SAVE_PLOT_PAPER_PATH, prepared=None):
    """
    How well STRUDL reproduced a manual track, for the CMEs it did track.

    The false-negative figures ask whether STRUDL found a CME at all; this one takes the events it
    found and asks how far along the manual track it got and how closely it followed it. The two
    rows are those two questions: the reach ratio of the maximum elongations, and the mean absolute
    difference between the profiles over the time they share.

    The columns are the same four DONKI parameters the false-negative figure separates its
    populations over, measured the same way, on the raw values rather than the ecliptic-corrected
    ones. The colours split the events by the same overlap flag, so a reader can carry the criterion
    from one figure to the other.

    `prepared` takes the result of `_fn_analysis_populations`, which is where the separation angle
    and the overlap flag come from; it is meant to be passed in, since building it goes to JPL
    Horizons.
    """

    if prepared is None:
        prepared = _fn_analysis_populations(donki_df, strudl_cmes_path, overlaps_path)

    metrics = _paired_track_metrics(paired_tracks_path)
    events = metrics.merge(prepared['visible'], how='inner',
                           left_on='cmeID_elevo', right_on='associatedActivityID')

    # the paired tracks are drawn from the CMEs that could have been seen in HI-1, so every one of
    # them should survive the join; losing any means the two files have drifted apart
    if len(events) != len(metrics):
        missing = set(metrics['cmeID_elevo']) - set(events['cmeID_elevo'])
        raise RuntimeError(f'{len(missing)} paired tracks have no counterpart among the CMEs '
                           f'visible in HI-1A, for instance {sorted(missing)[:3]}; '
                           f'{paired_tracks_path} and the DONKI file disagree')

    font_params = _enlarged_fonts(font_params)
    colors = {'Overlapping': color_dict['geosphere_purple'],
              'Not overlapping': color_dict['geosphere_green']}
    overlapping = events['overlapping'].to_numpy(dtype=bool)
    groups = {'Overlapping': events[overlapping], 'Not overlapping': events[~overlapping]}

    # the same quantities and labels as the false-negative figure, so the two agree
    columns = [('speed', 'CME speed at 21.5 R$_\\odot$ [km s$^{-1}$]'),
               ('halfAngle', 'CME half-width [deg]'),
               ('phi', 'CME-STEREO-A separation [deg]'),
               ('absLatitude', '|CME latitude| [deg]')]
    # the reference is the line a track reaching exactly as far as its manual counterpart sits on,
    # and the scale is logarithmic where the quantity spans decades
    rows = [('reach_ratio',
             'Reach ratio\n$\\varepsilon_\\mathrm{max,STRUDL}$ / $\\varepsilon_\\mathrm{max,manual}$',
             1.0, 'linear'),
            ('mean_abs_difference', 'Mean absolute difference\nmanual vs. STRUDL [deg]',
             None, 'log')]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex='col', sharey='row')
    summary = {'n_events': int(len(events)),
               'n_overlapping': int(overlapping.sum()),
               'n_not_overlapping': int((~overlapping).sum()),
               'panels': {}}
    letters = iter('abcdefgh')

    for r, (y_key, ylabel, reference, _) in enumerate(rows):
        for c, (x_key, xlabel) in enumerate(columns):
            ax = axes[r, c]
            if reference is not None:
                ax.axhline(reference, color='0.5', lw=0.8, ls=':', zorder=0)

            summary['panels'][f'{y_key}_vs_{x_key}'] = _plot_performance_panel(
                ax, groups, colors, x_key, y_key, font_params)

            # labelled in axes coordinates rather than through _place_panel_letters, which bins
            # panels into three columns and would put two of these four letters on top of their
            # neighbours; a regular grid aligns them without help. They sit above the axes because
            # the correlation box already occupies the top of every panel.
            ax.text(0.0, 1.01, f'({next(letters)})', transform=ax.transAxes, va='bottom',
                    fontweight='bold', fontsize=font_params['label_fontsize'])

            if r == len(rows) - 1:
                ax.set_xlabel(xlabel, fontsize=font_params['label_fontsize'])
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=font_params['label_fontsize'])

    # the correlation box takes the top of each panel, so each row is given room above its
    # highest point rather than letting the box land on the data. The headroom is a fraction of the
    # span the axis actually shows, which on a log row is the span of the logarithms: the mean
    # absolute difference runs over two decades, and a linear row 2 would press three quarters of
    # the events into the bottom tenth of the panel.
    for r, (y_key, _, _, scale) in enumerate(rows):
        values = events[y_key].to_numpy(dtype=float)
        axes[r, 0].set_yscale(scale)
        if scale == 'log':
            low, high = np.log10(values.min()), np.log10(values.max())
            span = high - low
            axes[r, 0].set_ylim(10 ** (low - 0.08 * span), 10 ** (high + 0.42 * span))
        else:
            span = values.max() - values.min()
            axes[r, 0].set_ylim(values.min() - 0.08 * span, values.max() + 0.42 * span)

    handles = [Patch(facecolor=colors[name], label=f'{name}, N={len(group)}')
               for name, group in groups.items()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2,
               frameon=False, fontsize=font_params['legend_fontsize'])

    fig.subplots_adjust(hspace=0.22, wspace=0.12, top=0.90)

    plt.savefig(SAVE_PLOT_PAPER_PATH + 'strudl_tracking_performance_scatter.pdf',
                bbox_inches='tight')
    plt.savefig(SAVE_PLOT_PAPER_PATH + 'strudl_tracking_performance_scatter.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)

    return summary



# the elongations the retention curve is evaluated at: fine enough that the grid itself adds no
# visible steps to the curve, and stopping at 24 deg because HI-1A's field of view does
_RETENTION_GRID = np.arange(5.0, 24.01, 0.25)


def midnight_green_colormap(color_dict):
    """
    The sequential ramp the tracking-profile figure is drawn in.

    Purple is the false-negative colour everywhere else in this module, so a figure comparing two
    tracks cannot borrow it without reading as a statement about detection. The anchors are
    GeoSphere's Midnight Green and the three tints the corporate design document gives for it,
    which are its opacity over white; white to 15 to 40 to 70 to 100 per cent is an even
    progression, so the ramp is smooth and runs light to dark.
    """

    return LinearSegmentedColormap.from_list(
        'midnight_green',
        ['#FFFFFF',
         color_dict['geosphere_midnight_green_15'],
         color_dict['geosphere_midnight_green_40'],
         color_dict['geosphere_midnight_green_70'],
         color_dict['geosphere_midnight_green']])


def _tracking_retention(spans, grid):
    """
    At each elongation, the fraction of the tracks that reach it manually which STRUDL reaches too.

    A track counts at an elongation when its span covers it, so one STRUDL dropped early leaves the
    numerator while staying in the denominator. The denominator is returned alongside: it falls
    away at both ends of the grid, and it is what decides where the curve is worth showing.
    """

    manual = (spans[:, None, 0] <= grid) & (spans[:, None, 1] >= grid)
    both = manual & (spans[:, None, 2] <= grid) & (spans[:, None, 3] >= grid)

    with np.errstate(invalid='ignore'):
        return both.sum(0) / manual.sum(0), manual.sum(0)


def _retention_with_band(spans, grid, n_bootstrap=2000, min_tracks=10, seed=0):
    """
    The retention curve, its bootstrap band, and where there are enough tracks to show them.

    The resampling is over events rather than over track points, because the question is how the
    curve would move with a different sample of 80 CMEs; resampling points would instead give the
    far narrower uncertainty of having measured these CMEs densely. The seed is fixed so the band
    is the same on every run of the figure.
    """

    retention, n_manual = _tracking_retention(spans, grid)
    valid = n_manual >= min_tracks

    rng = np.random.default_rng(seed)
    boot = np.array([_tracking_retention(spans[rng.integers(0, len(spans), len(spans))], grid)[0]
                     for _ in range(n_bootstrap)])

    # only the shown part of the band is taken, which also keeps np.nanpercentile away from the
    # all-NaN columns beyond the ends of the grid
    low, high = np.full(grid.shape, np.nan), np.full(grid.shape, np.nan)
    low[valid], high[valid] = np.nanpercentile(boot[:, valid], [2.5, 97.5], axis=0)

    return retention, low, high, n_manual, valid


def make_strudl_tracking_performance_plot(paired_tracks_path, color_dict, font_params,
                                          SAVE_PLOT_PAPER_PATH):
    """
    How closely STRUDL's time-elongation profiles follow the hand-corrected ones.

    Three questions about the same paired tracks, one per panel. (a) whether the two agree where
    both exist, as the density of matched points about the 1:1 line. (b) how far along the manual
    track STRUDL stays, as the fraction of the tracks reaching each elongation that STRUDL still
    reaches. (c) whether the disagreement in (a) is a bias and whether it grows, as the
    distribution of the residual against elongation.

    The figure is conditioned on an automatic track existing, which is the cut `_read_paired_tracks`
    describes. Panel (b) is therefore how much of a track STRUDL keeps once it has one, not how
    often it finds one at all; the false-negative figures are what answer the second question.
    """

    manual, strudl, spans, n_events = _paired_track_profiles(paired_tracks_path)
    grid = _RETENTION_GRID

    font_params = _enlarged_fonts(font_params)
    annotation_fontsize = font_params['legend_fontsize'] - 6
    base = color_dict['geosphere_midnight_green']
    tint_40 = color_dict['geosphere_midnight_green_40']
    tint_15 = color_dict['geosphere_midnight_green_15']

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6))
    summary = {'n_events': int(n_events), 'n_matched_points': int(manual.size)}

    # (a) agreement between the two profiles
    ax = axes[0]
    hexes = ax.hexbin(manual, strudl, gridsize=40, extent=(0, 26, 0, 26),
                      cmap=midnight_green_colormap(color_dict), norm=LogNorm(), mincnt=1,
                      linewidths=0)
    ax.plot([0, 26], [0, 26], color='0.3', lw=1, ls='--', label='1:1')
    ax.set(xlim=(0, 26), ylim=(0, 26), aspect='equal',
           xticks=np.arange(0, 26, 5), yticks=np.arange(0, 26, 5))
    ax.set_xlabel('Manual elongation [deg]', fontsize=font_params['label_fontsize'])
    ax.set_ylabel('STRUDL elongation [deg]', fontsize=font_params['label_fontsize'])
    # the colourbar goes in an inset rather than being taken out of the axes, which would shrink
    # panel (a) below the other two and defeat the square box set at the end of the function
    colorbar = fig.colorbar(hexes, cax=ax.inset_axes([1.04, 0.0, 0.045, 1.0]))
    # a step below the panel's own label and ticks: the colourbar is a key to the panel rather
    # than a fourth axis, and the smaller type also narrows the gap it needs to panel (b)
    colorbar.set_label('Number of track points', fontsize=font_params['label_fontsize'] - 3)
    colorbar.ax.tick_params(labelsize=font_params['ticks_fontsize'] - 3)
    ax.legend(loc='upper left', frameon=False, fontsize=font_params['legend_fontsize'])

    # (b) how far along the manual track STRUDL stays
    ax = axes[1]
    retention, low, high, n_manual, valid = _retention_with_band(spans, grid)
    ax.fill_between(grid[valid], low[valid] * 100, high[valid] * 100, color=tint_40, lw=0)
    ax.plot(grid[valid], retention[valid] * 100, color=base, lw=2.5)

    summary['retention'] = {}
    for elongation in (10, 15, 20):
        at = np.argmin(np.abs(grid - elongation))
        ax.plot(elongation, retention[at] * 100, 'o', color=base, ms=6)
        ax.annotate(f'{retention[at] * 100:.0f} %', (elongation, retention[at] * 100),
                    textcoords='offset points', xytext=(6, 6), fontsize=annotation_fontsize,
                    color=base)
        summary['retention'][elongation] = {'retention': float(retention[at]),
                                            'ci_low': float(low[at]), 'ci_high': float(high[at]),
                                            'n_manual': int(n_manual[at])}

    ax.set(xlim=(5, 24), ylim=(0, 105))
    ax.set_xlabel('Elongation [deg]', fontsize=font_params['label_fontsize'])
    ax.set_ylabel('CMEs still tracked by STRUDL [%]', fontsize=font_params['label_fontsize'])

    # how many manual tracks the curve rests on, under the elongation they rest on, since a
    # retention of 40 per cent means something different over 60 tracks than over 12
    for elongation in np.arange(5, 25, 5):
        ax.text(elongation, -0.22, f'{n_manual[np.argmin(np.abs(grid - elongation))]}',
                transform=ax.get_xaxis_transform(), ha='center',
                fontsize=font_params['ticks_fontsize'] - 4, color='0.35')
    ax.text(4.4, -0.22, 'N manual:', transform=ax.get_xaxis_transform(), ha='right',
            fontsize=font_params['ticks_fontsize'] - 4, color='0.35')

    # (c) the residual against elongation
    ax = axes[2]
    residual = strudl - manual
    edges = np.arange(4, 25, 2)
    centres = 0.5 * (edges[1:] + edges[:-1])
    quantiles = []
    for low_edge, high_edge in zip(edges[:-1], edges[1:]):
        in_bin = (manual >= low_edge) & (manual < high_edge)
        # a bin holding a handful of points would put a 5th and a 95th percentile on the figure
        # that are really just its two extreme points
        quantiles.append(np.percentile(residual[in_bin], [5, 25, 50, 75, 95])
                         if in_bin.sum() >= 20 else [np.nan] * 5)
    quantiles = np.asarray(quantiles)

    ax.axhline(0, color='0.3', lw=1, ls='--')
    ax.fill_between(centres, quantiles[:, 0], quantiles[:, 4], color=tint_15, lw=0,
                    label='5–95 %')
    ax.fill_between(centres, quantiles[:, 1], quantiles[:, 3], color=tint_40, lw=0,
                    label='25–75 %')
    ax.plot(centres, quantiles[:, 2], color=base, lw=2.5, marker='o', ms=5, label='Median')
    ax.set(xlim=(4, 24))
    ax.set_xlabel('Manual elongation [deg]', fontsize=font_params['label_fontsize'])
    ax.set_ylabel('STRUDL − manual elongation [deg]', fontsize=font_params['label_fontsize'])
    ax.legend(frameon=False, loc='lower left', fontsize=font_params['legend_fontsize'])

    summary['median_residual'] = float(np.median(residual))
    summary['residual_quantiles'] = {float(centre): dict(zip(('p5', 'p25', 'p50', 'p75', 'p95'),
                                                             row.tolist()))
                                     for centre, row in zip(centres, quantiles)}

    for ax, letter in zip(axes, 'abc'):
        ax.text(0.0, 1.01, f'({letter})', transform=ax.transAxes, va='bottom', fontweight='bold',
                fontsize=font_params['label_fontsize'])
        ax.tick_params(labelsize=font_params['ticks_fontsize'])
        ax.spines[['top', 'right']].set_visible(False)
        # a square box for all three, so the panels read as equals and (a), whose two axes carry
        # the same quantity, puts its 1:1 line on the diagonal. Without it (a) is the only panel
        # constrained by aspect='equal' and comes out narrower than its neighbours.
        ax.set_box_aspect(1)

    fig.tight_layout()
    # room under the panels for the numbers-at-risk row, and between them for the colourbar and
    # its label, which tight_layout does not see because the colourbar is an inset
    fig.subplots_adjust(bottom=0.2, wspace=0.48)

    plt.savefig(SAVE_PLOT_PAPER_PATH + 'strudl_tracking_profiles.pdf', bbox_inches='tight')
    plt.savefig(SAVE_PLOT_PAPER_PATH + 'strudl_tracking_profiles.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)

    return summary


def make_figure_donki_stats(donki_df_dedupe, donki_df_filtered, color_dict, label_dict, font_params, SAVE_PLOT_PAPER_PATH):

    purple_palette = sns.light_palette(color_dict['geosphere_purple'], n_colors=3)
    green_palette = sns.light_palette(color_dict['geosphere_green'], n_colors=3)


    color_map = {
        (label_dict['cat_features'],      "LE"): purple_palette[1],
        (label_dict['cat_features'],      "SH"): green_palette[1],
        (label_dict['cat_features_arrival'], "LE"): purple_palette[2],
        (label_dict['cat_features_arrival'], "SH"): green_palette[2],
        (label_dict['cat_unique_cmes'], "total"): '#CCCCCC',
        (label_dict['cat_unique_cmes_arrival'], "total"): '#666666'
    }

    # --- Prep ---
    sns.set_theme(context="paper", style="whitegrid")

    donki_df_filtered["month"] = donki_df_filtered["time21_5"].dt.to_period("M")
    donki_df_dedupe["month"] = donki_df_dedupe["time21_5"].dt.to_period("M")

    months = donki_df_filtered["month"].sort_values().unique()
    month_strs = [str(m) for m in months]



    all_cmes = (
        donki_df_filtered.groupby(["month", "featureCode"])
        .size()
        .reset_index(name="count")
        .assign(category=label_dict['cat_features'])
    )
    earth_cmes = (
        donki_df_filtered[donki_df_filtered["hasEarthArrival"]]
        .groupby(["month", "featureCode"])
        .size()
        .reset_index(name="count")
        .assign(category=label_dict['cat_features_arrival'])
    )

    unique_all = (
        donki_df_dedupe.groupby("month")
        .size()
        .reset_index(name="count")
        .assign(category=label_dict['cat_unique_cmes'], featureCode="total")
    )
    unique_earth = (
        donki_df_dedupe[donki_df_dedupe["hasEarthArrival"]]
        .groupby("month")
        .size()
        .reset_index(name="count")
        .assign(category=label_dict['cat_unique_cmes_arrival'], featureCode="total")
    )

    plot_df = pd.concat([all_cmes, earth_cmes, unique_all, unique_earth], ignore_index=True)
    plot_df["month_str"] = plot_df["month"].astype(str)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(18.5, 6))

    x = range(len(month_strs))
    bar_width = 0.2
    offsets = {
        label_dict['cat_unique_cmes']:              -0.3,
        label_dict['cat_unique_cmes_arrival']:   -0.1,
        label_dict['cat_features']:                   0.1,
        label_dict['cat_features_arrival']:     0.3,
    }
    bar_bottoms = {(cat, m): 0 for cat in offsets for m in month_strs}

    for (category, feature), group in plot_df.groupby(["category", "featureCode"]):
        counts = dict(zip(group["month_str"], group["count"]))
        values = [counts.get(m, 0) for m in month_strs]
        bottoms = [bar_bottoms[(category, m)] for m in month_strs]
        x_positions = [i + offsets[category] for i in x]

        if feature == "total":
            lb = f"{category}"
        else:
            lb = f"{category} – {feature}"

        ax.bar(
            x_positions,
            values,
            width=bar_width,
            bottom=bottoms,
            color=color_map[(category, feature)],
            label=lb,
        )

        for i, m in enumerate(month_strs):
            bar_bottoms[(category, m)] += values[i]

    ax.set_xticks(list(x))
    ax.set_xticklabels(month_strs, rotation=0, ha="center")
    ax.tick_params(axis='both', which='major', labelsize=font_params['ticks_fontsize'])
    ax.set_ylabel("No. of CMEs", fontsize=font_params['label_fontsize'])
    ax.legend(title="", bbox_to_anchor=(0.5, 1.2), loc="upper center", ncols=3, frameon=False, fontsize=font_params['legend_fontsize'])
    plt.tight_layout()

    plt.savefig(SAVE_PLOT_PAPER_PATH+'donki_stats.pdf', bbox_inches='tight')
    plt.close()

    return plot_df

def get_corresponding_rundifs(segmentation_times, rundif_path):

    # get times for this CME
    image_list = [f"{rundif_path}{tim.strftime('%Y%m%d_%H%M%S')[:8]}/science/hi_1/{tim.strftime('%Y%m%d_%H%M%S')}_1bh1A.fts" for tim in segmentation_times]
    
    # load the fts files if they exist, otherwise skip
    image_data = []
    header_data = []
    for image_path in image_list:
        try:
            with fits.open(image_path) as hdul:
                image_data.append(hdul[0].data)
                header_data.append(hdul[0].header)
        except FileNotFoundError:
            image_data.append(np.zeros((1024, 1024)))  # placeholder for missing image
            header_data.append(None)  # placeholder for missing header
            print(f"File not found: {image_path}")
            continue
    
    return image_data, header_data

def make_strudl_segmentation_plot(strudl_seg, rundif_path, chosen_rundif_times, color_dict, SAVE_PLOT_PAPER_PATH, rdif_freq=1):

    if not isinstance(rdif_freq, int) or rdif_freq < 1:
        raise ValueError(f"rdif_freq must be an integer greater than zero, got {rdif_freq}")

    for plot_time in chosen_rundif_times:
        for key in strudl_seg.keys():

            if plot_time not in strudl_seg[key]['times']:
                continue

            else:
                image_data, header_data = get_corresponding_rundifs(strudl_seg[key]['times'], rundif_path)
                total_images = len(strudl_seg[key]['times'])
                num_images_per_fig = 8

                if total_images < num_images_per_fig * rdif_freq:
                    raise ValueError(
                        f"rdif_freq={rdif_freq} is too large: only {total_images} images are available for "
                        f"'{key}', but {num_images_per_fig * rdif_freq} are needed to plot "
                        f"{num_images_per_fig} images at this interval."
                    )

                num_figs = math.floor(total_images / (num_images_per_fig * rdif_freq))

                num_cols = 4
                num_rows = 2
                figsize = (num_cols * 2, num_rows * 2)

                fac_contrast = 0.25


                for i in range(num_figs):
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
                    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
                    
                    for j in range(num_images_per_fig):
                        img_index = i * num_images_per_fig * rdif_freq + j * rdif_freq

                        if img_index >= total_images:
                            break
                        
                        ax = axes[j // num_cols, j % num_cols]

                        ax.imshow(np.flipud(image_data[img_index]),
                                aspect='equal',
                                cmap='gray',
                                vmin=np.nanmedian(image_data[img_index])-fac_contrast*np.nanstd(image_data[img_index]),
                                vmax=np.nanmedian(image_data[img_index])+fac_contrast*np.nanstd(image_data[img_index]),
                                extent=(0, image_data[img_index].shape[1], image_data[img_index].shape[0], 0),
                                interpolation='none')
                        
                        ax.axis('off')

                        xcoords = strudl_seg[key]['cme_front_pixels'][img_index][1]
                        ycoords = strudl_seg[key]['cme_front_pixels'][img_index][0]

                        if image_data[img_index].shape[0] != 128:
                            xcoords = np.clip(xcoords*8.0, 0, image_data[img_index].shape[0]-1).astype(int)
                            ycoords = np.clip(ycoords*8.0, 0, image_data[img_index].shape[0]-1).astype(int)

                        ax.scatter(xcoords, image_data[img_index].shape[0]-ycoords, s=10, color=color_dict['geosphere_yellow'], alpha=0.8, marker='x')
                        ax.set_frame_on(False)

                        if header_data[img_index] is not None and 'DATE-OBS' in header_data[img_index]:
                            ax.text(0.5, 0.98, header_data[img_index]['DATE-OBS'][:-4],
                                    transform=ax.transAxes, ha='center', va='top',
                                    color='black', fontsize=6,
                                    bbox=dict(facecolor='white', alpha=0.2, edgecolor='none', pad=1))
                    
                    #if strudl_seg[key]['times'][img_index-num_images_per_fig*rdif_freq] == plot_time:
                    plt.savefig(SAVE_PLOT_PAPER_PATH+'strudl_' + datetime.strftime(strudl_seg[key]['times'][img_index-num_images_per_fig*rdif_freq], '%Y%m%d_%H%M%S') + '.pdf', dpi=300, bbox_inches='tight',pad_inches=0.0)
                    plt.close()

def get_results_table_one(elevo_results, corrected_results, uncorrected_results, elevohi_results_uncorrected, elevohi_results_corrected):

    elevo_results_df = pd.read_json(elevo_results)
    elevo_results_df.rename(columns={'L1_ist_obs': 'L1_ist_obs [UT]'}, inplace=True)

    # convert all times to datetime objects
    time_columns = ['L1_ist_obs [UT]', 'tinit [UT]','arrival_time_min [UT]', 'arrival_time_max [UT]', 'arrival_time [UT]']
    for time_col in time_columns:
        elevo_results_df[time_col] = elevo_results_df[time_col].apply(lambda x: convert_time(x, '%Y-%m-%dT%H:%MZ'))

    elevo_results_df['time_axis [UT]'] = elevo_results_df['time_axis [UT]'].apply(lambda x: [convert_time(y, '%Y-%m-%dT%H:%MZ') for y in x])
    corrected_results_df = pd.read_json(corrected_results)
    corrected_results_df['time'] = corrected_results_df['time'].apply(lambda x: [convert_time(y, '%Y-%m-%dT%H:%M:%S.%f') for y in x])

    uncorrected_results_df = pd.read_json(uncorrected_results)
    uncorrected_results_df['time'] = uncorrected_results_df['time'].apply(lambda x: [convert_time(y, '%Y-%m-%dT%H:%M:%S.000') for y in x])

    elevohi_results_uncorrected_df = pd.read_json(elevohi_results_uncorrected)

    elevohi_results_corrected_df = pd.read_json(elevohi_results_corrected)

    for time_col in time_columns:
        elevohi_results_uncorrected_df[time_col] = elevohi_results_uncorrected_df[time_col].apply(lambda x: convert_time(x, '%Y-%m-%dT%H:%MZ'))
        elevohi_results_corrected_df[time_col] = elevohi_results_corrected_df[time_col].apply(lambda x: convert_time(x, '%Y-%m-%dT%H:%MZ'))


    dbm_status = []
    dbm_status_corrected = []
    for row_num, row_results in uncorrected_results_df.iterrows():

        if row_results['cmeID_strudl'] is not None:
            corresponding_data = elevohi_results_uncorrected_df[(elevohi_results_uncorrected_df['cmeID_elevo'] == row_results['cmeID_elevo']) &
                                                                (elevohi_results_uncorrected_df['cmeID_strudl'] == row_results['cmeID_strudl']) &
                                                                (elevohi_results_uncorrected_df['featureCode'] == row_results['featureCode'])]
            
            corresponding_data_corrected = elevohi_results_corrected_df[(elevohi_results_corrected_df['cmeID_elevo'] == row_results['cmeID_elevo']) &
                                                                (elevohi_results_corrected_df['cmeID_strudl'] == row_results['cmeID_strudl']) &
                                                                (elevohi_results_corrected_df['featureCode'] == row_results['featureCode'])]
        else:
            corresponding_data = elevohi_results_uncorrected_df[(elevohi_results_uncorrected_df['cmeID_elevo'] == row_results['cmeID_elevo']) &
                                                                (pd.isna(elevohi_results_uncorrected_df['cmeID_strudl'])) &
                                                                (elevohi_results_uncorrected_df['featureCode'] == row_results['featureCode'])]

            corresponding_data_corrected = elevohi_results_corrected_df[(elevohi_results_corrected_df['cmeID_elevo'] == row_results['cmeID_elevo']) &
                                                                (pd.isna(elevohi_results_corrected_df['cmeID_strudl'])) &
                                                                (elevohi_results_corrected_df['featureCode'] == row_results['featureCode'])]
            
        if corresponding_data.empty:
            dbm_status.append('Failure')
        else:
            dbm_status.append('Success')
        
        if corresponding_data_corrected.empty:
            dbm_status_corrected.append('Failure')
        else:
            dbm_status_corrected.append('Success')

    uncorrected_results_df['DBM_status'] = dbm_status
    uncorrected_results_df['DBM_status_corrected'] = dbm_status_corrected

    dbm_status_corrected = []

    for row_num, row_results in corrected_results_df.iterrows():

        if row_results['cmeID_strudl'] is not None:
            corresponding_data = elevohi_results_corrected_df[(elevohi_results_corrected_df['cmeID_elevo'] == row_results['cmeID_elevo']) &
                                                                (elevohi_results_corrected_df['cmeID_strudl'] == row_results['cmeID_strudl']) &
                                                                (elevohi_results_corrected_df['featureCode'] == row_results['featureCode'])]
        else:
            corresponding_data = elevohi_results_corrected_df[(elevohi_results_corrected_df['cmeID_elevo'] == row_results['cmeID_elevo']) &
                                                                (pd.isna(elevohi_results_corrected_df['cmeID_strudl'])) &
                                                                (elevohi_results_corrected_df['featureCode'] == row_results['featureCode'])]
        
        if corresponding_data.empty:
            dbm_status_corrected.append('Failure')
        else:
            dbm_status_corrected.append('Success')

    corrected_results_df['DBM_status'] = dbm_status_corrected

    # load uncorrected tracks and corrected tracks
    # for each feature/CME combination, interpolate the uncorrected tarcks to be on the ame timesteps as the corrected tracks and then calculate the mean absolute difference in elongation between the two tracks

    # only consider feature/CME combinations with a possible arrival at L1

    last_points = 3
    difference_elongation_list_abs = []
    difference_elongation_list_low_abs = []
    difference_elongation_list_high_abs = []
    difference_elongation_list = []
    normalized_time = []
    difference_whole_track_abs = []
    difference_whole_track = []

    for row_num, row in uncorrected_results_df.iterrows():
        elevo_id = row['cmeID_elevo']
        feature_code = row['featureCode']

        corresponding_corrected = corrected_results_df[(corrected_results_df['cmeID_elevo'] == elevo_id) & (corrected_results_df['featureCode'] == feature_code)]
        time_corrected = corresponding_corrected['time'].values[0]
        time_uncorrected = row['time']

        elon_corrected = corresponding_corrected['elongation'].values[0]
        elon_uncorrected = row['elongation']

        # interpolate uncorrected elongation to corrected timesteps
        elon_corrected_interp = np.interp(
            mdates.date2num(time_uncorrected),
            mdates.date2num(time_corrected),
            elon_corrected
        )

        difference_elongation_list_abs.append(np.mean(np.abs(elon_corrected_interp - elon_uncorrected)))
        difference_elongation_list.append(np.mean(elon_corrected_interp - elon_uncorrected))
        
        difference_whole_track_abs.append(np.abs(elon_corrected_interp - elon_uncorrected))
        difference_whole_track.append(elon_corrected_interp - elon_uncorrected)

        normalized_time.append([(tim_u - time_uncorrected[0]).total_seconds()/3600 for tim_u in time_uncorrected])
        difference_elongation_list_high_abs.append(np.mean(np.abs(elon_corrected_interp[-last_points:] - elon_uncorrected[-last_points:])))
        difference_elongation_list_low_abs.append(np.mean(np.abs(elon_corrected_interp[:-last_points] - elon_uncorrected[:-last_points])))


    uncorrected_results_df['mean_abs_difference_elongation'] = difference_elongation_list_abs
    uncorrected_results_df['abs_difference_elongation_whole_track'] = difference_whole_track_abs
    uncorrected_results_df['normalized_time'] = normalized_time

    uncorrected_results_df['mean_difference_elongation'] = difference_elongation_list
    uncorrected_results_df['difference_elongation_whole_track'] = difference_whole_track

    return elevo_results_df, elevohi_results_uncorrected_df, elevohi_results_corrected_df, uncorrected_results_df, corrected_results_df

def prepare(df, name):
    df = df.copy()
    df["label"] = (
        df["cmeID_elevo"].astype(str) + " / " +
        df["featureCode"].astype(str)
    )
    df["err_mean"] = (df["L1_ist_obs [UT]"] - df["arrival_time [UT]"]).dt.total_seconds() / 3600
    df["err_min"]  = (df["L1_ist_obs [UT]"] - df["arrival_time_min [UT]"]).dt.total_seconds() / 3600
    df["err_max"]  = (df["L1_ist_obs [UT]"] - df["arrival_time_max [UT]"]).dt.total_seconds() / 3600
    df = df.dropna(subset=["err_mean", "err_min", "err_max"])
    df["source"] = name
    df["tinit [UT]"] = df["tinit [UT]"].dt.strftime('%Y-%m-%dT%H:%MZ')
    return df[["label", "err_mean", "err_min", "err_max", "tinit [UT]", "source"]]
    
def make_pipeline_comparison_plot(elevo_results_df, elevohi_results_uncorrected_df, elevohi_results_corrected_df, pipeline_label_dict, color_dict, font_params, SAVE_PLOT_PAPER_PATH):

    # --- Colors and styles ---
    marker_styles = {
        pipeline_label_dict['elevohi_uncorrected_label']: dict(marker="s", color=color_dict['geosphere_yellow'],  label=pipeline_label_dict['elevohi_uncorrected_label']),
        pipeline_label_dict['elevohi_corrected_label']:   dict(marker="X", color=color_dict['geosphere_orange'],  label=pipeline_label_dict['elevohi_corrected_label']),
        pipeline_label_dict['elevo_label']:               dict(marker="^", color=color_dict['my_pink'],           label=pipeline_label_dict['elevo_label']),
    }
    offsets = {
        pipeline_label_dict['elevohi_uncorrected_label']: -0.15,
        pipeline_label_dict['elevohi_corrected_label']:    0.0,
        pipeline_label_dict['elevo_label']:                0.15,
    }

    # --- Prep ---
    sns.set_theme(context="paper", style="whitegrid")

    dfs = pd.concat([
        prepare(elevohi_results_uncorrected_df, pipeline_label_dict['elevohi_uncorrected_label']),
        prepare(elevohi_results_corrected_df,   pipeline_label_dict['elevohi_corrected_label']),
        prepare(elevo_results_df,               pipeline_label_dict['elevo_label']),
    ], ignore_index=True)

    dfs = dfs.sort_values(by=["tinit [UT]"], axis=0).reset_index(drop=True)
    # All unique labels in order of first appearance
    all_labels = list(dict.fromkeys(dfs["label"]))
    label_to_x = {l: i for i, l in enumerate(all_labels)}

    label_table = pd.DataFrame({
        "index": range(1, len(all_labels) + 1),
        "label": all_labels,
    })

    # --- Split evenly across 3 subplots ---
    n = len(all_labels)
    split1 = n // 3 + (1 if n % 3 > 0 else 0)   # 24
    split2 = split1 + n // 3 + (1 if n % 3 > 1 else 0)  # 48

    subsets = [
        all_labels[:split1],
        all_labels[split1:split2],
        all_labels[split2:],
    ]

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for ax, subset in zip(axes, subsets):
        subset_dfs = dfs[dfs["label"].isin(subset)]

        for source, group in subset_dfs.groupby("source"):
            style = marker_styles[source]
            offset = offsets[source]
            x_pos = [label_to_x[l] + offset for l in group["label"]]

            ax.errorbar(
                x_pos, group["err_mean"],
                yerr=[np.abs(group["err_mean"] - group["err_min"]),
                    np.abs(group["err_max"] - group["err_mean"])],
                fmt="none", color=style["color"], capsize=3, linewidth=1,capthick=1.5
            )

            ax.errorbar(
                x_pos, group["err_mean"],
                yerr=[np.abs(group["err_mean"] - group["err_min"]),
                    np.abs(group["err_max"] - group["err_mean"])],
                fmt="none", color='black', capsize=3.5, linewidth=1.5,capthick=2,
                zorder=0
            )
            ax.scatter(
                x_pos, group["err_mean"],
                marker=style["marker"], color=style["color"],
                label=style["label"], zorder=3, s=80,
                edgecolors='black', linewidth=0.5
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks([label_to_x[l] for l in subset])
        ax.set_xticklabels(
            [label_to_x[l] + 1 for l in subset],
            rotation=0, ha="center", fontsize=font_params['ticks_fontsize']+2,
        )
        ax.tick_params(axis='y', labelsize=font_params['ticks_fontsize']+2)
        ax.set_ylabel("Arrival Time Error [h]", fontsize=font_params['label_fontsize']+2)

    axes[2].set_xlabel("Event Index", fontsize=font_params['label_fontsize']+2)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, title="", bbox_to_anchor=(0.5, 1.03),
            loc="upper center", ncols=3, frameon=True, fontsize=font_params['legend_fontsize'], markerscale=2)

    plt.tight_layout()
    plt.savefig(SAVE_PLOT_PAPER_PATH+'comparison_all_pipelines.pdf', bbox_inches='tight')
    plt.close()

def plot_jplot_on_axis(
        ax,
        start,
        end,
        time_elon_elevo,
        time_elon_elevohi,
        elevo_elon_key='elongation_apex [°]',
        elevohi_elon_key='elongation_apex_tracked [°]',
    ):
        
        print(f'Loading J-plot for {start} to {end}...')
        try:
            jplot_test_path = f'HI_Data_Processing/stereo_processed/jplots_pickles/{start}_hi_1.p'
            jplot_test = pd.read_pickle(jplot_test_path)

            t_beg = jplot_test['dates'][0]
            t_end = jplot_test['dates'][-1]
            e_beg = jplot_test['elongations'][0]
            e_end = jplot_test['elongations'][-1]
            jplot_nonmedian = jplot_test['data']
            img_rescale = np.median(jplot_nonmedian,1).T
            orig = jplot_test['origin']
            
        except Exception:
            ax.text(0.5, 0.5, 'No J-plot found',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return

        # ---- prep image
        img = img_rescale.astype(float)
        vmin = np.nanmedian(img) - 0.25 * np.nanstd(img)
        vmax = np.nanmedian(img) + 0.25 * np.nanstd(img)

        cadence = 40.0  # minutes (HI-1)

        extent = [
            mdates.date2num(t_beg - timedelta(minutes=cadence / 2)),
            mdates.date2num(t_end + timedelta(minutes=cadence / 2)),
            e_beg,
            e_end
        ]

        ax.imshow(
            img,
            cmap='gray',
            aspect='auto',
            vmin=vmin,
            vmax=vmax,
            interpolation='none',
            origin=orig,
            extent=extent
        )

        # ---- overlay tracks

        alpha_vals_elevo = [0.9, 0.7, 0.7]

            
        for i in range(np.shape(time_elon_elevo[elevo_elon_key])[1]):
            ax.plot(
                time_elon_elevo['time_axis [UT]'],
                np.array(time_elon_elevo[elevo_elon_key])[:, i],
                color=my_pink,
                alpha=alpha_vals_elevo[i],
                lw=4
            )

        ax.scatter(
            time_elon_elevohi['time_apex_tracked [UT]'],
            time_elon_elevohi[elevohi_elon_key],
            color=geosphere_orange,
            marker='X',
            s=95,
            alpha=0.9,
            zorder=100,
            edgecolors='black',
            linewidths=0.5
        )

        ax.set_xlim(
            np.array(time_elon_elevohi['time_apex_tracked [UT]']).min() - timedelta(minutes=6*60),
            np.array(time_elon_elevohi['time_apex_tracked [UT]']).max() + timedelta(minutes=6*60)
        )

        # ---- axis formatting (compact)
        ax.set_ylim(e_beg, 20)

        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))

        ax.tick_params(axis='both', labelsize=ticks_fontsize)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(False)

def convert_time_columns_to_datetime(params_array, time_columns):
        
    for param in params_array:
        for col in time_columns:
            if col in param:
                if type(param[col]) is list or type(param[col]) is np.ndarray:
                    param[col] = np.array([datetime.strptime(t, '%Y-%m-%dT%H:%MZ') for t in param[col]])
                else:
                    try:
                        param[col] = datetime.strptime(param[col], '%Y-%m-%dT%H:%MZ')
                    except Exception:
                        param[col] = np.nan
                            
    return params_array

def make_dbm_comparison_plot(elevohi_path, elevo_path, chosen_events, color_dict, font_params, SAVE_PLOT_PAPER_PATH):
    
    AU_in_km = u.au.to(u.km)
    RSun_in_km = u.R_sun.to(u.km)

    elevohi_all_params = np.array(json.load(open(elevohi_path, 'r')))
    elevo_all_params = np.array(json.load(open(elevo_path, 'r')))

    list_of_cme_ids = set([elevohi_all_params[i]['cmeID_elevo'] for i in range(len(elevohi_all_params))])

    # convert all times to datetime objects
    time_columns = ['L1_ist_obs [UT]','tinit [UT]','arrival_time [UT]','time_axis [UT]','time_apex_tracked [UT]', 'arrival_time_min [UT]', 'arrival_time_max [UT]']

    elevohi_all_params = convert_time_columns_to_datetime(elevohi_all_params, time_columns)
    elevo_all_params = convert_time_columns_to_datetime(elevo_all_params, time_columns)

    for sub_num, elevo_sub in enumerate(elevo_all_params):

        corresponding_elevohi = elevohi_all_params[[(elevohi_all_params[i]['cmeID_elevo'] == elevo_sub['cmeID_elevo']) and (elevohi_all_params[i]['featureCode'] == elevo_sub['featureCode']) for i in range(len(elevohi_all_params))]]

        if len(corresponding_elevohi) == 0:
            is_arrival = np.nan
            dt = np.nan

        else:
            is_arrival = corresponding_elevohi[0]['L1_ist_obs [UT]']
            dt = (elevo_sub['arrival_time [UT]'] - is_arrival).total_seconds()/3600
        
        elevo_all_params[sub_num]['L1_ist_obs [UT]'] = is_arrival
        elevo_all_params[sub_num]['dt [h]'] = dt

    color_elevo = color_dict['my_pink']
    color_hi = 'black'
    marker_hi = 'o'
    rsun_lim = 220

    cmap_res = plt.get_cmap("jet", 15)
    cmap_linsp = np.round(np.linspace(0, 1.5, 16),1)

    residual_to_color = {res: cmap_res(i) for i, res in enumerate(cmap_linsp)}
    
    for cme_id in list_of_cme_ids:
        elevohi_cmes_fcode = elevohi_all_params[[(elevohi_all_params[i]['cmeID_elevo'] == cme_id) for i in range(len(elevohi_all_params))]]

        fcodes = set([elevohi_cmes_fcode[i]['featureCode'] for i in range(len(elevohi_cmes_fcode))])

        for fcode in fcodes:
            if cme_id + '_' + fcode not in chosen_events:
                print(cme_id + '_' + fcode + ' not in chosen events, skipping...')
                continue

            elevohi_cmes = elevohi_cmes_fcode[[elevohi_cmes_fcode[i]['featureCode'] == fcode for i in range(len(elevohi_cmes_fcode))]]

            elevo_cmes = elevo_all_params[[(elevo_all_params[i]['cmeID_elevo'] == cme_id) and (elevo_all_params[i]['featureCode'] == fcode) for i in range(len(elevo_all_params))]][0]

            fig = plt.figure(figsize=(18.5, 6), constrained_layout=False)

            gs = fig.add_gridspec(
                2, 3,
                height_ratios=[1.0, 0.05],
                width_ratios=[1.5, 1.5, 2],
                hspace=0.35,
                wspace=0.3,
            )

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax_cbar = fig.add_subplot(gs[1, :])

            pos2 = ax2.get_position()
            fig.text(
                pos2.x0 + pos2.width / 2,
                pos2.y1 + 0.03,
                f'{cme_id}_{fcode}',
                ha='center',
                va='bottom',
                fontsize=font_params['label_fontsize'] + 2
            )

            elevohi_dbm_residuals = [np.round(elevohi_cmes[i]['dbm_residual [R_sun]'],5) for i in range(len(elevohi_cmes))]

            for i in range(len(elevohi_cmes)):
                ax1.plot(
                    np.array(elevohi_cmes[i]['r_at_target_longitude [km]'])/RSun_in_km,
                    elevohi_cmes[i]['speed [km/s]'],
                    color=residual_to_color[cmap_linsp[np.searchsorted(cmap_linsp, elevohi_dbm_residuals[i])]],
                    lw=2,
                    alpha=0.85
                )

                ax2.plot(
                    elevohi_cmes[i]['time_axis [UT]'],
                    np.array(elevohi_cmes[i]['r_at_target_longitude [km]'])/ RSun_in_km,
                    color=residual_to_color[cmap_linsp[np.searchsorted(cmap_linsp, elevohi_dbm_residuals[i])]],
                    lw=2,
                    alpha=0.85
                    )


            ax2.plot(
                elevohi_cmes[0]['time_apex_tracked [UT]'],
                np.array(elevohi_cmes[0]['r_apex_tracked [km]']) / RSun_in_km,
                marker=marker_hi,
                color=color_hi,
                zorder=100
            )

            alpha_vals_elevo = [0.9, 0.7, 0.7]
            for num in range(np.shape(elevo_cmes['r_at_apex [km]'])[1]):
                ax1.plot(
                    np.array(elevo_cmes['r_at_target_longitude [km]'])[:,num] / RSun_in_km,
                    np.array(elevo_cmes['speed [km/s]'])[:,num],
                    color=color_elevo,
                    alpha=alpha_vals_elevo[num],
                    lw=4
                )

                ax2.plot(
                    elevo_cmes['time_axis [UT]'],
                    np.array(elevo_cmes['r_at_target_longitude [km]'])[:,num] / RSun_in_km,
                    color=color_elevo,
                    alpha=alpha_vals_elevo[num],
                    lw=4
                ) 

            ax1.scatter(
                np.array(elevohi_cmes[0]['r_apex_tracked [km]'])/RSun_in_km,
                elevohi_cmes[0]['speed_apex_tracked [km/s]'],
                marker=marker_hi,
                color=color_hi,
                s=25,
                zorder=100
            )


            ax1.set_xlabel('Distance from Sun [R$_\\odot$]',fontsize=font_params['label_fontsize']+2)
            ax1.set_ylabel('Speed [km/s]',fontsize=font_params['label_fontsize']+2)
            ax1.set_xlim(0, rsun_lim)
            ax1.set_ylim(0, 1400)
            ax1.grid(alpha=0.4)
            ax1.tick_params(axis='both', which='major', labelsize=font_params['ticks_fontsize'])

            r_l1 = get_horizons_coord('SEMB-L1', elevohi_cmes[0]['L1_ist_obs [UT]']).transform_to(frames.HeliographicStonyhurst).radius.value*AU_in_km/RSun_in_km

            ax1.vlines(x=r_l1,ymin=0,ymax=1400,colors='blue',linestyles='dashed',alpha=0.7, linewidth=1)


            ax2.set_xlabel('Time',fontsize=font_params['label_fontsize']+2)
            ax2.set_ylabel('Distance from Sun [R$_\\odot$]',fontsize=font_params['label_fontsize']+2)
            ax2.set_ylim(0, rsun_lim)

            ax2.set_xlim(
                min(
                    np.array(elevohi_cmes[0]['time_axis [UT]']).min(),
                    np.array(elevo_cmes['time_axis [UT]']).min()
                )-timedelta(hours=6),
                max(
                    np.array(elevohi_cmes[0]['arrival_time [UT]']),
                    np.array(elevo_cmes['arrival_time [UT]']),
                    np.array(elevohi_cmes[0]['L1_ist_obs [UT]'])
                )+timedelta(hours=6)
            )

            ax2.grid(alpha=0.4)
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))

            ax2.vlines(elevo_cmes['arrival_time [UT]'], ymin=0, ymax=r_l1, colors=color_dict['my_pink'], linestyles='--', linewidth=2)
            ax2.vlines(elevohi_cmes[0]['arrival_time [UT]'], ymin=0, ymax=r_l1, colors='black', linestyles='--', linewidth=2)

            ax2.vlines(x=elevohi_cmes[0]['L1_ist_obs [UT]'], ymin=0, ymax=r_l1, colors='blue', linestyles='--', linewidth=1, alpha=0.7)
            ax2.hlines(y=r_l1, xmin=ax2.get_xlim()[0], xmax=ax2.get_xlim()[1], colors='blue', linestyles='--', linewidth=1, alpha=0.7)
            ax2.tick_params(axis='both', which='major', labelsize=font_params['ticks_fontsize'])


            starttime_jplot = (elevo_cmes['tinit [UT]']-timedelta(hours=24)).strftime('%Y%m%d')
            endtime_jplot = (elevo_cmes['tinit [UT]']+timedelta(days=4)).strftime('%Y%m%d')

            plot_jplot_on_axis(
            ax=ax3,
            start=starttime_jplot,
            end=endtime_jplot,
            time_elon_elevo=elevo_cmes,
            time_elon_elevohi=elevohi_cmes[0]
            )

            norm = plt.Normalize(
                vmin=0,
                vmax=1.5
            )
            sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
            sm.set_array([])

            cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
            cbar.set_label("DBM Fitting Residual [R$_\\odot$]", fontsize=font_params['label_fontsize']+2)
            cbar.ax.tick_params(labelsize=font_params['ticks_fontsize'])

            fsuffix = 'vs_elevo'

            plt.savefig(SAVE_PLOT_PAPER_PATH +f'{fsuffix}_jplot_{cme_id}_{fcode}.png',dpi=300)
            plt.close()

def make_plot_elongation_difference(uncorrected_results_df, color_dict, font_params, SAVE_PLOT_PAPER_PATH):

    purple_palette = sns.light_palette(color_dict['geosphere_purple'], n_colors=3)
    green_palette = sns.light_palette(color_dict['geosphere_green'], n_colors=3)

    # Create a figure and plot the line on it
    mean_diff_fail = uncorrected_results_df[(uncorrected_results_df['DBM_status'] == 'Failure') & (uncorrected_results_df['DBM_status_corrected'] == 'Success')]['mean_difference_elongation'].abs().mean()
    mean_diff_success = uncorrected_results_df[uncorrected_results_df['DBM_status'] == 'Success']['mean_difference_elongation'].abs().mean()

    perc_track = 0.25
    mean_abs_diff_fail_last_points = []
    mean_abs_diff_success_last_points = []

    mean_diff_fail_last_points = []
    mean_diff_success_last_points = []

    for row_num, row in uncorrected_results_df.iterrows():
        num_pts = int(len(row['abs_difference_elongation_whole_track'])*perc_track)
        if row['DBM_status'] == 'Failure':
            mean_abs_diff_fail_last_points.append(np.mean(row['abs_difference_elongation_whole_track'][-num_pts:]))
            mean_diff_fail_last_points.append(np.abs(np.mean(row['difference_elongation_whole_track'][-num_pts:])))
        else:
            mean_abs_diff_success_last_points.append(np.mean(row['abs_difference_elongation_whole_track'][-num_pts:]))
            mean_diff_success_last_points.append(np.abs(np.mean(row['difference_elongation_whole_track'][-num_pts:])))

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    chosen_difference = uncorrected_results_df['difference_elongation_whole_track']

    for num, entry in enumerate(chosen_difference):
        if uncorrected_results_df.iloc[num]['DBM_status_corrected'] == 'Failure':
            continue
        else:
            ax1.plot(uncorrected_results_df['normalized_time'] [num], entry, alpha=0.5, color=purple_palette[2] if (uncorrected_results_df.iloc[num]['DBM_status'] == 'Failure' and uncorrected_results_df.iloc[num]['DBM_status_corrected'] == 'Success') else green_palette[2], ls='-', lw=2)

    ax1.set_xlabel('Normalized Time [h]', fontsize=font_params['label_fontsize'])

    ax1.set_ylabel('$\Delta \epsilon$ [°]', fontsize=font_params['label_fontsize'])

    ax1.axhline(mean_diff_fail, color=purple_palette[2], linestyle='-', label=f'$\Delta \overline{{\epsilon}}$ (DBM Fit Failure) = {mean_diff_fail:.1f}°')
    ax1.axhline(mean_diff_success, color=green_palette[2], linestyle='-', label=f'$\Delta \overline{{\epsilon}}$ (DBM Fit Success) =  {mean_diff_success:.1f}°')

    ax1.axhline(np.mean(mean_diff_fail_last_points), color=purple_palette[2], linestyle=':', label=f'$\Delta \overline{{\epsilon}}_{{0.25}}$ (DBM Fit Failure) = {np.mean(mean_diff_fail_last_points):.1f}°')
    ax1.axhline(np.mean(mean_diff_success_last_points), color=green_palette[2], linestyle=':', label=f'$\Delta \overline{{\epsilon}}_{{0.25}}$ (DBM Fit Success) = {np.mean(mean_diff_success_last_points):.1f}°')
    
    ax1.legend(fontsize=font_params['ticks_fontsize'], bbox_to_anchor=(0.5, 1.25),
            loc="upper center", ncols=2, frameon=True)
    
    # change font size of x and y tick labels
    ax1.tick_params(axis='both', which='major', labelsize=font_params['ticks_fontsize'])
    plt.savefig(f"{SAVE_PLOT_PAPER_PATH}elongation_difference_tracks_abs_False.pdf", dpi=300, bbox_inches='tight')

def make_results_table(plot_df, elevo_results_df, uncorrected_results_df, corrected_results_df, elevohi_results_uncorrected_df, elevohi_results_corrected_df, pipeline_label_dict, save_path):

    uncorrected_max_el = np.nanmean(uncorrected_results_df['elongation'].apply(lambda x: max(x) if isinstance(x, list) else np.nan))
    corrected_max_el = np.nanmean(corrected_results_df['elongation'].apply(lambda x: max(x) if isinstance(x, list) else np.nan))

    uncorrected_delta_time_215 = np.nanmean(uncorrected_results_df.apply(lambda row: (row['time'][0] - datetime.strptime(row['time_21_5Rs'], '%Y-%m-%dT%H:%MZ')).total_seconds() / 3600.0 if isinstance(row['time'], list) and len(row['time']) > 0 and pd.notnull(datetime.strptime(row['time_21_5Rs'], '%Y-%m-%dT%H:%MZ')) else np.nan, axis=1))
    corrected_delta_time_215 = np.nanmean(corrected_results_df.apply(lambda row: (row['time'][0] - datetime.strptime(row['time_21_5Rs'], '%Y-%m-%dT%H:%MZ')).total_seconds() / 3600.0 if isinstance(row['time'], list) and len(row['time']) > 0 and pd.notnull(datetime.strptime(row['time_21_5Rs'], '%Y-%m-%dT%H:%MZ')) else np.nan, axis=1))

    max_fitted_uncorrected_failed = []
    max_fitted_uncorrected_succeeded = []
    for row_num, row_results in uncorrected_results_df.iterrows():

        if row_results['cmeID_strudl'] is not None:
            corresponding_data = elevohi_results_uncorrected_df[(elevohi_results_uncorrected_df['cmeID_elevo'] == row_results['cmeID_elevo']) &
                                                                (elevohi_results_uncorrected_df['cmeID_strudl'] == row_results['cmeID_strudl']) &
                                                                (elevohi_results_uncorrected_df['featureCode'] == row_results['featureCode'])]
        else:
            corresponding_data = elevohi_results_uncorrected_df[(elevohi_results_uncorrected_df['cmeID_elevo'] == row_results['cmeID_elevo']) &
                                                                (pd.isna(elevohi_results_uncorrected_df['cmeID_strudl'])) &
                                                                (elevohi_results_uncorrected_df['featureCode'] == row_results['featureCode'])]
        
        if corresponding_data.empty:
            max_fitted_uncorrected_failed.append(np.nanmax(row_results['elongation']))
        else:
            max_fitted_uncorrected_succeeded.append(np.nanmax(row_results['elongation']))

    max_fitted_corrected_failed = []
    max_fitted_corrected_succeeded = []

    for row_num, row_results in corrected_results_df.iterrows():
        
        if row_results['cmeID_strudl'] is not None:
            corresponding_data = elevohi_results_corrected_df[(elevohi_results_corrected_df['cmeID_elevo'] == row_results['cmeID_elevo']) &
                                                                (elevohi_results_corrected_df['cmeID_strudl'] == row_results['cmeID_strudl']) &
                                                                (elevohi_results_corrected_df['featureCode'] == row_results['featureCode'])]
        else:
            corresponding_data = elevohi_results_corrected_df[(elevohi_results_corrected_df['cmeID_elevo'] == row_results['cmeID_elevo']) &
                                                                (pd.isna(elevohi_results_corrected_df['cmeID_strudl'])) &
                                                                (elevohi_results_corrected_df['featureCode'] == row_results['featureCode'])]
        
        if corresponding_data.empty:
            max_fitted_corrected_failed.append(np.nanmax(row_results['elongation']))
        else:
            max_fitted_corrected_succeeded.append(np.nanmax(row_results['elongation']))

    mean_abs_dt_elevo = np.nanmean(np.abs(elevo_results_df["arrival_time [UT]"] - elevo_results_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)
    mean_abs_dt_elevohi_uncorrected = np.nanmean(np.abs(elevohi_results_uncorrected_df["arrival_time [UT]"] - elevohi_results_uncorrected_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)
    mean_abs_dt_elevohi_corrected = np.nanmean(np.abs(elevohi_results_corrected_df["arrival_time [UT]"] - elevohi_results_corrected_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)

    mean_abs_dt_elevo_max = np.nanmean(np.abs(elevo_results_df["arrival_time_max [UT]"] - elevo_results_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)
    mean_abs_dt_elevo_min = np.nanmean(np.abs(elevo_results_df["arrival_time_min [UT]"] - elevo_results_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)

    mean_abs_dt_elevohi_uncorrected_max = np.nanmean(np.abs(elevohi_results_uncorrected_df["arrival_time_max [UT]"] - elevohi_results_uncorrected_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)
    mean_abs_dt_elevohi_uncorrected_min = np.nanmean(np.abs(elevohi_results_uncorrected_df["arrival_time_min [UT]"] - elevohi_results_uncorrected_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)

    mean_abs_dt_elevohi_corrected_max = np.nanmean(np.abs(elevohi_results_corrected_df["arrival_time_max [UT]"] - elevohi_results_corrected_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)
    mean_abs_dt_elevohi_corrected_min = np.nanmean(np.abs(elevohi_results_corrected_df["arrival_time_min [UT]"] - elevohi_results_corrected_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)

    std_mean_abs_dt_elevo = np.nanstd(np.abs(elevo_results_df["arrival_time [UT]"] - elevo_results_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)
    std_mean_abs_dt_elevohi_uncorrected = np.nanstd(np.abs(elevohi_results_uncorrected_df["arrival_time [UT]"] - elevohi_results_uncorrected_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)
    std_mean_abs_dt_elevohi_corrected = np.nanstd(np.abs(elevohi_results_corrected_df["arrival_time [UT]"] - elevohi_results_corrected_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)

    std_mean_abs_dt_elevo_max = np.nanstd(np.abs(elevo_results_df["arrival_time_max [UT]"] - elevo_results_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)
    std_mean_abs_dt_elevo_min = np.nanstd(np.abs(elevo_results_df["arrival_time_min [UT]"] - elevo_results_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)
    std_mean_abs_dt_elevohi_uncorrected_max = np.nanstd(np.abs(elevohi_results_uncorrected_df["arrival_time_max [UT]"] - elevohi_results_uncorrected_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)

    std_mean_abs_dt_elevohi_uncorrected_min = np.nanstd(np.abs(elevohi_results_uncorrected_df["arrival_time_min [UT]"] - elevohi_results_uncorrected_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)
    std_mean_abs_dt_elevohi_corrected_max = np.nanstd(np.abs(elevohi_results_corrected_df["arrival_time_max [UT]"] - elevohi_results_corrected_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)
    std_mean_abs_dt_elevohi_corrected_min = np.nanstd(np.abs(elevohi_results_corrected_df["arrival_time_min [UT]"] - elevohi_results_corrected_df["L1_ist_obs [UT]"]).dt.total_seconds() / 3600)

    dfs = pd.concat([
        prepare(elevohi_results_uncorrected_df, pipeline_label_dict['elevohi_uncorrected_label']),
        prepare(elevohi_results_corrected_df,   pipeline_label_dict['elevohi_corrected_label']),
        prepare(elevo_results_df,               pipeline_label_dict['elevo_label']),
    ], ignore_index=True)

    # get only errors of of events where predicitons are present from all three sources

    unique_labels = dfs['label'].unique()
    error_common_elevo = []
    error_common_elevohi_uncorrected = []
    error_common_elevohi_corrected = []

    for label in unique_labels:
        subset = dfs[dfs['label'] == label]

        if len(subset) != 3:
            continue
        else:
            subset_elevo = subset[subset['source'] == 'ELEvo']
            error_common_elevo.append([subset_elevo['err_mean'].values[0], subset_elevo['err_min'].values[0], subset_elevo['err_max'].values[0]])

            subset_elevohi_uncorrected = subset[subset['source'] == pipeline_label_dict['elevohi_uncorrected_label']]
            error_common_elevohi_uncorrected.append([subset_elevohi_uncorrected['err_mean'].values[0], subset_elevohi_uncorrected['err_min'].values[0], subset_elevohi_uncorrected['err_max'].values[0]])

            subset_elevohi_corrected = subset[subset['source'] == pipeline_label_dict['elevohi_corrected_label']]
            error_common_elevohi_corrected.append([subset_elevohi_corrected['err_mean'].values[0], subset_elevohi_corrected['err_min'].values[0], subset_elevohi_corrected['err_max'].values[0]])

    # Create a figure and plot the line on it
    mean_diff_fail = uncorrected_results_df[(uncorrected_results_df['DBM_status'] == 'Failure') & (uncorrected_results_df['DBM_status_corrected'] == 'Success')]['mean_difference_elongation'].abs().mean()
    mean_diff_success = uncorrected_results_df[uncorrected_results_df['DBM_status'] == 'Success']['mean_difference_elongation'].abs().mean()

    perc_track = 0.25
    mean_abs_diff_fail_last_points = []
    mean_abs_diff_success_last_points = []

    mean_diff_fail_last_points = []
    mean_diff_success_last_points = []

    for row_num, row in uncorrected_results_df.iterrows():
        num_pts = int(len(row['abs_difference_elongation_whole_track'])*perc_track)
        if row['DBM_status'] == 'Failure':
            mean_abs_diff_fail_last_points.append(np.mean(row['abs_difference_elongation_whole_track'][-num_pts:]))
            mean_diff_fail_last_points.append(np.abs(np.mean(row['difference_elongation_whole_track'][-num_pts:])))
        else:
            mean_abs_diff_success_last_points.append(np.mean(row['abs_difference_elongation_whole_track'][-num_pts:]))
            mean_diff_success_last_points.append(np.abs(np.mean(row['difference_elongation_whole_track'][-num_pts:])))

    vinit_detected = []
    vinit_not_detected = []

    detected_results_elevohi = uncorrected_results_df[uncorrected_results_df['L1_ist_obs'].notna()]
    detected_results_elevohi_list = [row['cmeID_elevo']+'_'+row['featureCode'] for row_num, row in detected_results_elevohi.iterrows()]

    for row_num, row in elevo_results_df[elevo_results_df['L1_ist_obs [UT]'].notna()].iterrows():
        if row['cmeID_elevo']+ '_' + row['featureCode'] in detected_results_elevohi_list:
            vinit_detected.append(row['vinit [km/s]'])
        else:
            vinit_not_detected.append(row['vinit [km/s]'])

    with open(f"{save_path}/results_latex.txt", "w") as text_file:
        print("% TABLE 1 %", file=text_file)

        total_sum = plot_df[(plot_df['featureCode'] == 'SH') & (plot_df['category'] == cat_features)]['count'].sum() + plot_df[(plot_df['featureCode'] == 'LE') & (plot_df['category'] == cat_features)]['count'].sum()

        print(f"\\newcommand\\tabOneTotalSH{{{plot_df[(plot_df['featureCode'] == 'SH') & (plot_df['category'] == cat_features)]['count'].sum()}}}", file=text_file)
        print(f"\\newcommand\\tabOneTotalLE{{{plot_df[(plot_df['featureCode'] == 'LE') & (plot_df['category'] == cat_features)]['count'].sum()}}}", file=text_file)
        print(f"\\newcommand\\tabOneTotalSum{{{total_sum}}}", file=text_file)
        print(f"\\newcommand\\tabOneTotalUnique{{{plot_df[(plot_df['featureCode'] == 'total') & (plot_df['category'] == cat_unique_cmes)]['count'].sum()}}}", file=text_file)
        print("\n", file=text_file)
        print(f"\\newcommand\\tabOneHasArrivalSH{{{plot_df[(plot_df['featureCode'] == 'SH') & (plot_df['category'] == cat_features_arrival)]['count'].sum()}}}", file=text_file)
        print(f"\\newcommand\\tabOneHasArrivalLE{{{plot_df[(plot_df['featureCode'] == 'LE') & (plot_df['category'] == cat_features_arrival)]['count'].sum()}}}", file=text_file)
        print(f"\\newcommand\\tabOneHasArrivalSum{{{plot_df[(plot_df['featureCode'] == 'SH') & (plot_df['category'] == cat_features_arrival)]['count'].sum() + plot_df[(plot_df['featureCode'] == 'LE') & (plot_df['category'] == cat_features_arrival)]['count'].sum()}}}", file=text_file)
        print(f"\\newcommand\\tabOneHasArrivalUnique{{{plot_df[(plot_df['featureCode'] == 'total') & (plot_df['category'] == cat_unique_cmes_arrival)]['count'].sum()}}}", file=text_file)
        print("\n", file=text_file)
        print(f"\\newcommand\\tabOneArrivalPossibleSH{{{len(elevo_results_df[elevo_results_df['featureCode'] == 'SH'])}}}", file=text_file)
        print(f"\\newcommand\\tabOneArrivalPossibleLE{{{len(elevo_results_df[elevo_results_df['featureCode'] == 'LE'])}}}", file=text_file)
        print(f"\\newcommand\\tabOneArrivalPossibleSum{{{len(elevo_results_df)}}}", file=text_file)
        print(f"\\newcommand\\tabOneArrivalPossibleUnique{{{len(elevo_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False))}}}", file=text_file)
        print("\n", file=text_file)
        print(f"\\newcommand\\tabOneArrivalPossibleAndHasArrivalSH{{{len(elevo_results_df[(elevo_results_df['featureCode'] == 'SH') & (~pd.isna(elevo_results_df['L1_ist_obs [UT]']))])}}}", file=text_file)
        print(f"\\newcommand\\tabOneArrivalPossibleAndHasArrivalLE{{{len(elevo_results_df[(elevo_results_df['featureCode'] == 'LE') & (~pd.isna(elevo_results_df['L1_ist_obs [UT]']))])}}}", file=text_file)
        print(f"\\newcommand\\tabOneArrivalPossibleAndHasArrivalSum{{{len(elevo_results_df[~pd.isna(elevo_results_df['L1_ist_obs [UT]'])])}}}", file=text_file)
        print(f"\\newcommand\\tabOneArrivalPossibleAndHasArrivalUnique{{{len(elevo_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)[~pd.isna(elevo_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)['L1_ist_obs [UT]'])])}}}", file=text_file)
        print("\n", file=text_file)
        print("% Table 2%", file=text_file)
        print(f"\\newcommand\\tabTwoStrudlNoTracked{{{len(uncorrected_results_df)}}}", file=text_file)
        print(f"\\newcommand\\tabTwoStrudlAvgMaxE{{{uncorrected_max_el:.1f}}}", file=text_file)
        print(f"\\newcommand\\tabTwoStrudlAvgDeltaTStart{{{uncorrected_delta_time_215:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabTwoStrudlNoFailedFits{{{len(uncorrected_results_df)-len(elevohi_results_uncorrected_df)}}}", file=text_file)
        print(f"\\newcommand\\tabTwoStrudlAvgMaxEFailedFits{{{np.nanmean(max_fitted_uncorrected_failed):.1f}}}", file=text_file)
        print(f"\\newcommand\\tabTwoStrudlAvgMaxESuccessfulFits{{{np.nanmean(max_fitted_uncorrected_succeeded):.1f}}}", file=text_file)
        print("\n", file=text_file)
        print(f"\\newcommand\\tabTwoManualNoTracked{{{len(corrected_results_df)}}}", file=text_file)
        print(f"\\newcommand\\tabTwoManualAvgMaxE{{{corrected_max_el:.1f}}}", file=text_file)
        print(f"\\newcommand\\tabTwoManualAvgDeltaTStart{{{corrected_delta_time_215:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabTwoManualNoFailedFits{{{len(corrected_results_df)-len(elevohi_results_corrected_df)}}}", file=text_file)
        print(f"\\newcommand\\tabTwoManualAvgMaxEFailedFits{{{np.nanmean(max_fitted_corrected_failed):.1f}}}", file=text_file)
        print(f"\\newcommand\\tabTwoManualAvgMaxESuccessfulFits{{{np.nanmean(max_fitted_corrected_succeeded):.1f}}}", file=text_file)
        print("\n", file=text_file)
        print("% TABLE 3 %", file=text_file)
        print(f"\\newcommand\\tabThreeELEvoNoValid{{{len(elevo_results_df[~pd.isna(elevo_results_df['L1_ist_obs [UT]'])])}}}", file=text_file)
        print(f"\\newcommand\\tabThreeELEvoAvgMinDeltaT{{{mean_abs_dt_elevo_min:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeELEvoAvgMaxDeltaT{{{mean_abs_dt_elevo_max:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeELEvoAvgMeanDeltaT{{{mean_abs_dt_elevo:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeELEvoAvgMinDeltaTStd{{{std_mean_abs_dt_elevo_min:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeELEvoAvgMaxDeltaTStd{{{std_mean_abs_dt_elevo_max:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeELEvoAvgMeanDeltaTStd{{{std_mean_abs_dt_elevo:.2f}}}", file=text_file)
        print("\n", file=text_file)
        print(f"\\newcommand\\tabThreeStrudlNoValid{{{len(elevohi_results_uncorrected_df)}}}", file=text_file)
        print(f"\\newcommand\\tabThreeStrudlAvgMinDeltaT{{{mean_abs_dt_elevohi_uncorrected_min:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeStrudlAvgMaxDeltaT{{{mean_abs_dt_elevohi_uncorrected_max:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeStrudlAvgMeanDeltaT{{{mean_abs_dt_elevohi_uncorrected:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeStrudlAvgMinDeltaTStd{{{std_mean_abs_dt_elevohi_uncorrected_min:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeStrudlAvgMaxDeltaTStd{{{std_mean_abs_dt_elevohi_uncorrected_max:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeStrudlAvgMeanDeltaTStd{{{std_mean_abs_dt_elevohi_uncorrected:.2f}}}", file=text_file)
        print("\n", file=text_file)
        print(f"\\newcommand\\tabThreeManualNoValid{{{len(elevohi_results_corrected_df)}}}", file=text_file)
        print(f"\\newcommand\\tabThreeManualAvgMinDeltaT{{{mean_abs_dt_elevohi_corrected_min:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeManualAvgMaxDeltaT{{{mean_abs_dt_elevohi_corrected_max:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeManualAvgMeanDeltaT{{{mean_abs_dt_elevohi_corrected:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeManualAvgMinDeltaTStd{{{std_mean_abs_dt_elevohi_corrected_min:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeManualAvgMaxDeltaTStd{{{std_mean_abs_dt_elevohi_corrected_max:.2f}}}", file=text_file)
        print(f"\\newcommand\\tabThreeManualAvgMeanDeltaTStd{{{std_mean_abs_dt_elevohi_corrected:.2f}}}", file=text_file)
        print("\n", file=text_file)
        print("% OTHER %", file=text_file)
        print(f"\\newcommand\\deltaTMeanAllValidELEvo{{{np.mean([abs(x[0]) for x in error_common_elevo]):.2f}}}", file=text_file)
        print(f"\\newcommand\\deltaTMeanAllValidStrudl{{{np.mean([abs(x[0]) for x in error_common_elevohi_uncorrected]):.2f}}}", file=text_file)
        print(f"\\newcommand\\deltaTMeanAllValidManual{{{np.mean([abs(x[0]) for x in error_common_elevohi_corrected]):.2f}}}", file=text_file)
        print("\n", file=text_file)
        print(f"\\newcommand\\deltaTMinAllValidELEvo{{{np.mean([abs(x[1]) for x in error_common_elevo]):.2f}}}", file=text_file)
        print(f"\\newcommand\\deltaTMinAllValidStrudl{{{np.mean([abs(x[1]) for x in error_common_elevohi_uncorrected]):.2f}}}", file=text_file)
        print(f"\\newcommand\\deltaTMinAllValidManual{{{np.mean([abs(x[1]) for x in error_common_elevohi_corrected]):.2f}}}", file=text_file)
        print("\n", file=text_file)
        print(f"\\newcommand\\deltaTMaxAllValidELEvo{{{np.mean([abs(x[2]) for x in error_common_elevo]):.2f}}}", file=text_file)
        print(f"\\newcommand\\deltaTMaxAllValidStrudl{{{np.mean([abs(x[2]) for x in error_common_elevohi_uncorrected]):.2f}}}", file=text_file)
        print(f"\\newcommand\\deltaTMaxAllValidManual{{{np.mean([abs(x[2]) for x in error_common_elevohi_corrected]):.2f}}}", file=text_file)
        print("\n", file=text_file)
        print(f"\\newcommand\\deltaEDBMFailure{{{np.mean(mean_diff_fail):.1f}}}", file=text_file)
        print(f"\\newcommand\\deltaEDBMSuccess{{{np.mean(mean_diff_success):.1f}}}", file=text_file)
        print(f"\\newcommand\\deltaEDBMFailEnd{{{np.mean(mean_diff_fail_last_points):.1f}}}", file=text_file)
        print(f"\\newcommand\\deltaEDBMSuccessEnd{{{np.mean(mean_diff_success_last_points):.1f}}}", file=text_file)
        print("\n", file=text_file)
        print(f"\\newcommand\\StrudlNoTrackedUnique{{{len(uncorrected_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False))}}}", file=text_file)
        print(f"\\newcommand\\StrudlNoTrackedHasArrival{{{len(uncorrected_results_df[uncorrected_results_df['L1_ist_obs'].notna()])}}}", file=text_file)
        print(f"\\newcommand\\StrudlNoTrackedUniqueHasArrival{{{len(uncorrected_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)[uncorrected_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)['L1_ist_obs'].notna()])}}}", file=text_file)
        print("\n", file=text_file)
        print(f"\\newcommand\\StrudlMiss{{{len(elevo_results_df[~pd.isna(elevo_results_df['L1_ist_obs [UT]'])]) - len(uncorrected_results_df[uncorrected_results_df['L1_ist_obs'].notna()])}}}", file=text_file)
        print(f"\\newcommand\\StrudlMissUnique{{{len(elevo_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)[~pd.isna(elevo_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)['L1_ist_obs [UT]'])]) - len(uncorrected_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)[uncorrected_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)['L1_ist_obs'].notna()])}}}", file=text_file)
        print("\n", file=text_file)

        false_positives_strudl_unqiue = len(uncorrected_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)[uncorrected_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)['L1_ist_obs'].isna()])
        true_positives_strudl_unique = len(uncorrected_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)[uncorrected_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)['L1_ist_obs'].notna()])
        fdr_strudl = false_positives_strudl_unqiue / (false_positives_strudl_unqiue + true_positives_strudl_unique)
        print(f"\\newcommand\\StrudlFalseDetectionRate{{{fdr_strudl*100:.1f}}}", file=text_file)

        false_positives_elevo_unique = len(elevo_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)[elevo_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)['L1_ist_obs [UT]'].isna()])
        true_positives_elevo_unique = len(elevo_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)[elevo_results_df.drop_duplicates(subset=['cmeID_elevo'], inplace=False)['L1_ist_obs [UT]'].notna()])
        fdr_elevo = false_positives_elevo_unique / (false_positives_elevo_unique + true_positives_elevo_unique)
        print(f"\\newcommand\\ELEvoFalseDetectionRate{{{fdr_elevo*100:.1f}}}", file=text_file)
        print("\n", file=text_file)
        print(f"\\newcommand\\MeanVinitStrudlDetected{{{np.mean(vinit_detected):.0f}}}", file=text_file)
        print(f"\\newcommand\\MeanVinitStrudlNotDetected{{{np.nanmean(vinit_not_detected):.0f}}}", file=text_file)
        print("\n", file=text_file)
        print(f"\\newcommand\\CommonFits{{{len(error_common_elevo)}}}", file=text_file)

if __name__ == '__main__':

    SAVE_PLOTS = True
    SAVE_PLOT_PAPER_PATH = 'paper_plots/'

    geosphere_purple = '#701F80'
    geosphere_green = '#75CC41'
    geosphere_yellow = '#F9F200'
    geosphere_orange = '#F29707'
    my_pink = '#FF69B4'

    # Midnight Green and its 70 / 40 / 15 % tints, from the corporate design document; the tints
    # are the base colour's opacity over white. Purple means false negative everywhere else in
    # this module, so the tracking-profile figure is drawn in this ramp instead, anchored on
    # these four by midnight_green_colormap.
    geosphere_midnight_green = '#052E37'
    geosphere_midnight_green_70 = '#546D73'
    geosphere_midnight_green_40 = '#9DABAE'
    geosphere_midnight_green_15 = '#DAE0E1'

    # only used by the Venn diagram, see VENN_SET_COLOR_KEYS
    venn_blue = '#2A78D6'
    venn_red = '#B22222'

    color_dict = {
        "geosphere_purple": geosphere_purple,
        "geosphere_green": geosphere_green,
        "geosphere_yellow": geosphere_yellow,
        "geosphere_orange": geosphere_orange,
        "my_pink": my_pink,
        "geosphere_midnight_green": geosphere_midnight_green,
        "geosphere_midnight_green_70": geosphere_midnight_green_70,
        "geosphere_midnight_green_40": geosphere_midnight_green_40,
        "geosphere_midnight_green_15": geosphere_midnight_green_15,
        "venn_blue": venn_blue,
        "venn_red": venn_red
    }


    cat_features = "Features"
    cat_features_arrival = "Features w. L1 Arrival"
    cat_unique_cmes = "Unique CMEs"
    cat_unique_cmes_arrival = "Unique CMEs w. L1 Arrival"

    label_dict = {
        "cat_features": cat_features,
        "cat_features_arrival": cat_features_arrival,
        "cat_unique_cmes": cat_unique_cmes,
        "cat_unique_cmes_arrival": cat_unique_cmes_arrival
    }

    legend_fontsize = 15
    ticks_fontsize = 14
    label_fontsize = 15

    font_params = {'legend_fontsize': legend_fontsize,
                   'label_fontsize': label_fontsize,
                   'ticks_fontsize': ticks_fontsize}
    
    if not os.path.exists(SAVE_PLOT_PAPER_PATH):
        os.makedirs(SAVE_PLOT_PAPER_PATH)

    DATA_DONKI_PATH = 'STRUDL_Tracking_Analysis/data/DONKI_cme_analyses_matched.json'

    donki_df_dedupe, donki_df_filtered = load_donki_data(DATA_DONKI_PATH)

    DATA_STRUDL_CMES_PATH = 'STRUDL_Tracking_Analysis/data/STRUDL_cmes.json'
    DATA_OVERLAPS_PATH = 'STRUDL_Tracking_Analysis/data/DONKI_cme_hi1_overlaps.json'

    # built once and shared, since it queries JPL Horizons for the STEREO-A position
    fn_analysis_prepared = _fn_analysis_populations(donki_df_dedupe,
                                                    strudl_cmes_path=DATA_STRUDL_CMES_PATH,
                                                    overlaps_path=DATA_OVERLAPS_PATH)

    # fn_analysis_summary = make_strudl_fn_analysis_plot(donki_df_dedupe,
    #                                                    strudl_cmes_path=DATA_STRUDL_CMES_PATH,
    #                                                    overlaps_path=DATA_OVERLAPS_PATH,
    #                                                    color_dict=color_dict,
    #                                                    font_params=font_params,
    #                                                    SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH,
    #                                                    signed_separation=False,
    #                                                    prepared=fn_analysis_prepared)

    fn_analysis_summary_v2 = make_strudl_fn_analysis_plot_v2(donki_df_dedupe,
                                                             strudl_cmes_path=DATA_STRUDL_CMES_PATH,
                                                             overlaps_path=DATA_OVERLAPS_PATH,
                                                             color_dict=color_dict,
                                                             font_params=font_params,
                                                             SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH,
                                                             bin_width_deg=45.0,
                                                             prepared=fn_analysis_prepared)

    DATA_PAIRED_TRACKS_PATH = 'STRUDL_Tracking_Analysis/data/manual_vs_strudl_tracks.json'

    tracking_profiles_summary = make_strudl_tracking_performance_plot(
        paired_tracks_path=DATA_PAIRED_TRACKS_PATH,
        color_dict=color_dict,
        font_params=font_params,
        SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

    # tracking_performance_summary = make_strudl_tracking_performance_plot_scatter(
    #     donki_df_dedupe,
    #     paired_tracks_path=DATA_PAIRED_TRACKS_PATH,
    #     strudl_cmes_path=DATA_STRUDL_CMES_PATH,
    #     overlaps_path=DATA_OVERLAPS_PATH,
    #     color_dict=color_dict,
    #     font_params=font_params,
    #     SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH,
    #     prepared=fn_analysis_prepared)

    # # make Figure 1
    # plot_df = make_figure_donki_stats(donki_df_dedupe, donki_df_filtered, color_dict=color_dict, label_dict=label_dict, font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

    # make Figure 1
    venn_regions, venn_totals = make_donki_venn_plot(donki_df_dedupe, color_dict=color_dict, font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

    # STRUDL_SEGMENTATION_PATH = 'STRUDL/segmentation_dictionary_mean_45_2024_05_01_2025_05_01.npy'
    # strudl_seg = np.load(STRUDL_SEGMENTATION_PATH, allow_pickle=True).item()

    # RUNDIF_PATH = 'HI_Data_Processing/stereo_processed/running_difference/data/A/'
    # CHOSEN_RUNDIF_TIMES = [datetime(2024,8,20,15,58,11), datetime(2024,7,23,11,18,11)]

    # # make Figure 4
    # make_strudl_segmentation_plot(strudl_seg, RUNDIF_PATH, CHOSEN_RUNDIF_TIMES, color_dict=color_dict, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH, rdif_freq=4)

    # ELEVO_RESULTS_PATH = 'ELEvo/data/elevo_results_20240501_20250501.json'
    # STRUDL_CORRECTED_RESULTS_PATH = 'STRUDL_Tracking_Analysis/data/mean_45_2024_05_01_2025_05_01_strudl_tracks_corrected.json'
    # STRUDL_UNCORRECTED_RESULTS_PATH = 'STRUDL_Tracking_Analysis/data/mean_45_2024_05_01_2025_05_01_strudl_tracks.json'
    # ELEVOHI_CORRECTED_RESULTS_PATH = 'ELEvoHI/results/elevohi_ensembles_results_cleaned.json'
    # ELEVOHI_UNCORRECTED_RESULTS_PATH = 'ELEvoHI/results/elevohi_ensembles_results_raw.json'

    # # get results Table 1
    # elevo_results_df, elevohi_results_uncorrected_df, elevohi_results_corrected_df, uncorrected_results_df, corrected_results_df =  get_results_table_one(ELEVO_RESULTS_PATH, STRUDL_CORRECTED_RESULTS_PATH, STRUDL_UNCORRECTED_RESULTS_PATH, ELEVOHI_UNCORRECTED_RESULTS_PATH, ELEVOHI_CORRECTED_RESULTS_PATH)

    # STRUDL_ELEVOHI_LABEL = "STRUDL-ELEvoHI"
    # MANUAL_ELEVOHI_LABEL = "Manual-ELEvoHI"
    # ELEVO_LABEL = "ELEvo"

    # pipeline_label_dict = {
    #     "elevohi_uncorrected_label": STRUDL_ELEVOHI_LABEL,
    #     "elevohi_corrected_label": MANUAL_ELEVOHI_LABEL,
    #     "elevo_label": ELEVO_LABEL
    # }

    # # Make Figure 7
    # make_pipeline_comparison_plot(elevohi_results_uncorrected_df, elevohi_results_corrected_df, elevo_results_df, pipeline_label_dict, color_dict=color_dict, font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

    # CHOSEN_DBM_EVENTS = ['2024-07-28T02:24:00-CME-001_SH', '2024-05-08T22:24:00-CME-001_SH', '2024-12-29T18:24:00-CME-001_LE', '2024-05-08T22:24:00-CME-001_LE']

    # # Make Figure 8
    # ELEVOHI_CORRECTED_PATH = 'ELEvoHI/results/elevohi_results_cleaned.json'
    # make_dbm_comparison_plot(elevohi_path=ELEVOHI_CORRECTED_PATH, elevo_path=ELEVO_RESULTS_PATH, chosen_events=CHOSEN_DBM_EVENTS, color_dict=color_dict, font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)


    # # Make Figure 6
    # make_plot_elongation_difference(uncorrected_results_df=uncorrected_results_df, color_dict=color_dict, font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

    # # Make results
    # make_results_table(plot_df=plot_df, elevo_results_df=elevo_results_df, uncorrected_results_df=uncorrected_results_df, corrected_results_df=corrected_results_df, elevohi_results_uncorrected_df=elevohi_results_uncorrected_df, elevohi_results_corrected_df=elevohi_results_corrected_df, pipeline_label_dict=pipeline_label_dict, save_path=SAVE_PLOT_PAPER_PATH)


