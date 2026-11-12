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
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Patch
from matplotlib.colors import to_rgba, to_rgb, to_hex, LinearSegmentedColormap, LogNorm
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, mannwhitneyu, fisher_exact, chi2_contingency, spearmanr, wilcoxon, t as student_t
from scipy.stats.contingency import odds_ratio as conditional_odds_ratio
from matplotlib.gridspec import GridSpec
from sunpy.coordinates import frames, get_horizons_coord
import astropy.units as u
import seaborn as sns
import sys
import re

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

    time_range = [datetime(2024,5,1,0,0,0), datetime(2026,4,30,23,59,59)]

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
# 2026-04-30 analysis period. The solver refines it, and falls back to cold restarts if
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

def solve_venn_layout(regions, n_restarts=6, verbose=False):
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


def _overlapping_cme_ids(overlaps_path, overlap_coefficient=0.1, overlap_minutes=80,
                         elongation_separation=helper_functions.ELONGATION_SEPARATION_DEG):
    """
    The CMEs that shared the HI-1A field with another one closely enough to obscure it.

    Kept apart from `_fn_analysis_populations` because the forecast-error figure asks the same
    question of a different population, and a second copy of the criterion would let the two
    figures disagree about which CMEs were overlapped.

    Returns the flagged ids and the counts the reports quote.
    """

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

    return {'overlapping_ids': overlapping_ids,
            'overlap_coefficient': overlap_coefficient,
            'overlap_minutes': overlap_minutes,
            'elongation_separation': elongation_separation,
            'n_overlapping_pairs': int(flagged.sum()),
            'n_overlapping_cmes': len(overlapping_ids)}


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

    overlaps = _overlapping_cme_ids(overlaps_path, overlap_coefficient=overlap_coefficient,
                                    overlap_minutes=overlap_minutes,
                                    elongation_separation=elongation_separation)
    visible['overlapping'] = visible['associatedActivityID'].isin(overlaps['overlapping_ids'])

    strudl_df = pd.read_json(strudl_cmes_path)
    false_positives = set(strudl_df['track_key']) - set(donki_df['associatedStrudlID'].dropna())

    is_true_positive = visible['associatedStrudlID'].notna()

    return {'visible': visible,
            'populations': {'TP': visible[is_true_positive], 'FN': visible[~is_true_positive]},
            'false_positives': false_positives,
            'n_strudl_tracks': int(len(strudl_df)),
            'overlap_coefficient': overlaps['overlap_coefficient'],
            'overlap_minutes': overlaps['overlap_minutes'],
            'elongation_separation': overlaps['elongation_separation'],
            'n_overlapping_pairs': overlaps['n_overlapping_pairs'],
            'n_overlapping_cmes': overlaps['n_overlapping_cmes']}


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


def _outline(stroke, lw):
    """A line of width `lw` laid over a wider stroke of `stroke`, or no effect when it is None."""

    return None if stroke is None else [pe.Stroke(linewidth=lw + 1.5, foreground=stroke), pe.Normal()]


def _plot_kde_panel(ax, populations, colors, strokes, key, xlabel, unit, clip_display, font_params,
                    rng, annotate=True):
    """
    The TP and FN distributions of one quantity, with the test of the difference between them.

    The curves carry no label: the two populations are named once for the whole figure, so the
    panel legend is left to the medians, which differ from panel to panel. The test is named in
    the caption rather than in every box, so only its numbers appear here.

    `strokes` gives each population an outline colour, or None to draw it plainly. Lime over the
    white page has a contrast of about 1.8, which a two-point curve disappears at, so the true
    positives are stroked in a darker lime and the near-black false negatives are not.

    `annotate` False drops the median legend and the statistics box, for the copy of the figure
    whose numbers are carried by the surrounding text instead.

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
        ax.plot(x_grid, curves[cls] / peak, color=colors[cls], lw=2,
                path_effects=_outline(strokes[cls], 2.0))
        ax.axvline(np.median(v), color=colors[cls], ls='--', lw=1.5,
                   label=f'Median {cls}: {np.median(v):.0f}{unit}',
                   path_effects=_outline(strokes[cls], 1.5))

    u_statistic, p_value = mannwhitneyu(values['FN'], values['TP'], alternative='two-sided')
    delta, delta_low, delta_high = cliffs_delta_with_ci(values['TP'], values['FN'],
                                                        u_statistic, rng)
    if annotate:
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
    if annotate:
        ax.legend(fontsize=font_params['legend_fontsize'] - 4, loc='upper right', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    return {'median_tp': float(np.median(values['TP'])), 'median_fn': float(np.median(values['FN'])),
            'u': float(u_statistic), 'p': float(p_value),
            'delta': delta, 'delta_ci_low': delta_low, 'delta_ci_high': delta_high}


def _plot_overlap_panel(ax, populations, colors, strokes, font_params, annotate=True):
    """
    How often each population shared the field with a CME close enough to obscure it.

    The bars go unlabelled and the panel carries no legend of its own: the two populations are
    named once for the whole figure, and the colours are the same ones used throughout it. The
    `strokes` outline keeps the pale true-positive fill from running into the page; `annotate`
    False drops the statistics box, as in the distribution panels.
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
                      color=colors[cls], edgecolor=strokes[cls], lw=0.0 if strokes[cls] is None else 1.0)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'n={count}',
                    ha='center', va='bottom', fontsize=font_params['ticks_fontsize'] - 2)

    table = [list(contingency['TP']), list(contingency['FN'])]
    _, p_value = fisher_exact(table)
    # the conditional estimate and its exact interval are the pair that matches Fisher's exact p
    odds_ratio_result = conditional_odds_ratio(table)
    odds_ratio_ci = odds_ratio_result.confidence_interval(0.95)
    if annotate:
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


def _population_legend(fig, populations, colors, strokes, font_params, y=0.99):
    """
    The one key both figures carry: which colour is which population, and how many of each.

    Drawn once for the whole figure rather than once per panel, since the colours and the counts
    are the same everywhere in it. The panels are then free to use their legends for what does
    differ between them, which is the median of each distribution.
    """

    handles = [Patch(facecolor=colors[cls], edgecolor=strokes[cls],
                     lw=0.0 if strokes[cls] is None else 1.0,
                     label=f'{cls}, N={len(populations[cls])}')
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
    The same numbers as LaTeX definitions, to be cited from the manuscript by name.

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
    saved without its numbers and the two can never describe different runs.
    """

    with open(f'{save_path}{stem}.txt', 'w') as handle:
        handle.write(_format_fn_analysis_report(summary, stem) + '\n')

    with open(f'{save_path}{stem}_latex.txt', 'w') as handle:
        handle.write(f'% {stem} -- written by make_paper_plots.py, do not edit by hand\n')
        handle.write(_format_fn_analysis_macros(summary, macro_prefix) + '\n')


# the geometry of the two polar panels, in units of the radius of the wedge ring's outer edge
_R_SUN = 0.30
_R_RING_INNER = 0.42
_R_RING_WIDTH = 0.58
_R_SPACECRAFT = 1.09
_R_LIMIT = 1.15


def detection_odds_colormap(color_dict):
    """
    The diverging scale the wedges are painted on: two hues about a neutral midpoint.

    Midnight Green is the false-negative colour and Lime the true-positive colour everywhere else in
    the figure, so a bin where STRUDL detected worse than the rest of the sample reads Midnight Green
    and one where it detected better reads Lime. The midpoint is grey rather than a third hue, so
    that an odds ratio of one shows up as the absence of a difference instead of as a value of its
    own.

    The false-negative end is the 70 per cent tint rather than the base colour, which the two flat
    populations are drawn in. Base Midnight Green is nearly black and Lime is pale, so anchoring the
    ramp on the pair would give the two halves unequal lightness and make a bin STRUDL missed look
    stronger than an equally extreme bin it caught. The tint puts both ends a comparable distance
    from the midpoint.
    """

    return LinearSegmentedColormap.from_list(
        'detection_odds',
        [color_dict['geosphere_midnight_green_70'], '#EEEEEE', color_dict['geosphere_lime']])


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
                                    log2_odds_cap=3.0, prepared=None, annotate=True):
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

    `annotate` False draws the same figure without the key, the panel letters, the median legends
    and the statistics boxes, and writes it under its own name. The colourbar stays: it is the scale
    the wedges are read against rather than a key to them. Nothing else differs, since the bootstrap
    is seeded, so the two files carry the same numbers.

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
    colors = {'TP': color_dict['geosphere_lime'], 'FN': color_dict['geosphere_midnight_green']}
    strokes = {'TP': color_dict['geosphere_lime_shade'], 'FN': None}
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
        summary['panels'][key] = _plot_kde_panel(ax, populations, colors, strokes, key, xlabel,
                                                 unit, clip_display, font_params, bootstrap_rng,
                                                 annotate)

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

    summary['panels']['overlap'] = _plot_overlap_panel(axes[4], populations, colors, strokes,
                                                       font_params, annotate)

    if annotate:
        _population_legend(fig, populations, colors, strokes, font_params)
        _place_panel_letters(fig, list(zip(axes, 'abcde')), font_params)

    name = 'strudl_fn_analysis_v2' if annotate else 'strudl_fn_analysis_v2_bare'
    plt.savefig(SAVE_PLOT_PAPER_PATH + name + '.pdf', bbox_inches='tight')
    plt.savefig(SAVE_PLOT_PAPER_PATH + name + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # the two runs produce the same numbers, so only the annotated one files them
    if annotate:
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

    # date2num rather than the raw timestamps, so that np.interp gets a monotonic float
    # abscissa rather than datetime64, which it cannot interpolate over
    manual_interp = np.interp(mdates.date2num(time_strudl[shared]),
                              mdates.date2num(time_manual), elongation_manual)

    return manual_interp, elongation_strudl[shared]


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



# the elongations the retention curve is evaluated at: fine enough that the grid itself adds no
# visible steps to the curve, and stopping at 24 deg because HI-1A's field of view does
_RETENTION_GRID = np.arange(5.0, 24.01, 0.25)


def midnight_green_colormap(color_dict):
    """
    The sequential ramp the tracking-profile figure is drawn in.

    The figure compares two tracks of the same CME and says nothing about whether a CME was found,
    so it is drawn on one ramp rather than in the Midnight Green and Lime pair that means missed
    against detected in the false-negative figure. The anchors are GeoSphere's Midnight Green and
    the three tints the corporate design document gives for it,
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
    # ax.text(4.4, -0.22, 'N manual:', transform=ax.get_xaxis_transform(), ha='right',
    #         fontsize=font_params['ticks_fontsize'] - 4, color='0.35')

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

# ---------------------------------------------------------------------------------------------
# Contingency-table verification, after Verbeke et al. (2019), doi:10.1029/2018SW002046
# ---------------------------------------------------------------------------------------------

# forecast_outcome values that count as a predicted arrival; everything else is a predicted
# non-arrival, except 'no_fit', which is not a forecast at all
PREDICTED_ARRIVAL = 'arrival'
NOT_A_FORECAST = 'no_fit'


def _ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else np.nan


def contingency_scores(hits, misses, false_alarms, correct_rejections):
    """
    The eight skill scores of Verbeke et al. (2019), Table 9.

    POD, SR, FAR, Bias and CSI carry no correct-rejection term and so do not depend on where the
    population boundary is drawn. Accuracy, POFD and HK do, and with CR an order of magnitude
    larger than FA the last two collapse onto POD, so they have to be read against the stated
    population rather than on their own.
    """

    pod = _ratio(hits, hits + misses)
    sr = _ratio(hits, hits + false_alarms)
    pofd = _ratio(false_alarms, correct_rejections + false_alarms)

    return {'H': int(hits),
            'M': int(misses),
            'FA': int(false_alarms),
            'CR': int(correct_rejections),
            'total': int(hits + misses + false_alarms + correct_rejections),
            'POD': pod,
            'SR': sr,
            'FAR': np.nan if np.isnan(sr) else 1.0 - sr,
            'bias': _ratio(hits + false_alarms, hits + misses),
            'CSI': _ratio(hits, hits + misses + false_alarms),
            'accuracy': _ratio(hits + correct_rejections,
                               hits + misses + false_alarms + correct_rejections),
            'POFD': pofd,
            'HK': np.nan if (np.isnan(pod) or np.isnan(pofd)) else pod - pofd}


# Set membership that defines each model's "an arrival is predicted", in VENN_SETS keys. The
# masks come from get_venn_set_masks, so the tables cannot drift from the Venn figure.
GEOMETRY_MODELS = {
    'Ellipse': ('A',),
    'Ellipse + STRUDL': ('A', 'S'),
    'STRUDL': ('S',),
}

GEOMETRY_POPULATION = 'V'
GEOMETRY_OBSERVED = 'I'

# the reference model, scored to show where the combined model's precision comes from rather
# than as a pipeline anyone would run
GEOMETRY_REFERENCE = 'STRUDL'


def geometry_contingency(population, predicted, observed):
    """
    Verbeke scores for one binary predictor, from three boolean masks on a common index.

    There is no hit window here: the prediction is a yes or no about whether the CME can reach
    Earth at all, so an event falls in exactly one cell and the four cells sum to the
    population.
    """

    return contingency_scores(hits=int((population & predicted & observed).sum()),
                              misses=int((population & ~predicted & observed).sum()),
                              false_alarms=int((population & predicted & ~observed).sum()),
                              correct_rejections=int((population & ~predicted & ~observed).sum()))


def geometry_skill_summary(donki_df):
    """
    How much the DONKI arrival ellipse, and then STRUDL on top of it, tell us about an arrival.

    The population is every CME that could have appeared in the HI-1A field of view, and the
    observation is a DONKI in-situ link at Earth. Requiring a STRUDL detection can only shrink
    the set of predicted arrivals, so it trades recall for precision and the question is only
    which way the trade goes.

    get_venn_set_masks is the single source of both this and the Venn figure, and
    get_venn_region_counts asserts A within V and E and S within V, so each model's cells are
    Venn regions by construction.
    """

    masks = get_venn_set_masks(donki_df)

    population = masks[GEOMETRY_POPULATION]
    observed = masks[GEOMETRY_OBSERVED]

    models = {}
    for label, keys in GEOMETRY_MODELS.items():
        predicted = masks[keys[0]]
        for key in keys[1:]:
            predicted = predicted & masks[key]

        models[label] = geometry_contingency(population, predicted, observed)

    return {'models': models,
            'reference': GEOMETRY_REFERENCE,
            'population': {'n_visible': int(population.sum()),
                           'n_arrived': int((population & observed).sum()),
                           'set_totals': {key: int(mask.sum()) for key, mask in masks.items()}}}

def forecast_events(results_df):
    """
    One verification event per forecast in an ELEvoHI or ELEvo results table.

    Rows whose DBM fit was rejected are not forecasts and are excluded; they are reported as
    forecast yield instead, because scoring them as predicted non-arrivals would credit the
    model for having said nothing.
    """

    events = []
    not_forecast = 0

    for _, row in results_df.iterrows():
        outcome = row.get('forecast_outcome')

        if outcome is None or outcome == NOT_A_FORECAST:
            not_forecast += 1
            continue

        arrival_time = pd.to_datetime(row.get('arrival_time [UT]'))
        observed_time = pd.to_datetime(row.get('L1_ist_obs [UT]'))

        if pd.notna(arrival_time) and pd.notna(observed_time):
            dt_h = (arrival_time - observed_time).total_seconds() / 3600.0
        else:
            dt_h = np.nan

        events.append({'cmeID_elevo': row['cmeID_elevo'],
                       'cmeID_strudl': row.get('cmeID_strudl'),
                       'featureCode': row.get('featureCode'),
                       'predicted': outcome == PREDICTED_ARRIVAL,
                       'observed': pd.notna(observed_time),
                       'dt_h': dt_h})

    return events, not_forecast


def hit_error_statistics(errors, n_bootstrap=10000, seed=42):
    """
    Verbeke's hit-only metrics on the signed arrival time error, with confidence intervals.

    ME is the bias and takes the Student-t interval; MAE, RMSE and the standard deviation are
    not means of the quantity whose spread is being estimated, so they take a percentile
    bootstrap instead.
    """

    errors = np.asarray([e for e in errors if np.isfinite(e)], dtype=float)
    n = len(errors)

    if n == 0:
        return {'n': 0, 'ME': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'sd': np.nan}

    me = float(np.mean(errors))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    sd = float(np.std(errors, ddof=1)) if n > 1 else np.nan

    summary = {'n': n, 'ME': me, 'MAE': mae, 'RMSE': rmse, 'sd': sd,
               'median_abs': float(np.median(np.abs(errors)))}

    if n > 1:
        half_width = student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        summary['ME_ci'] = (me - half_width, me + half_width)

        rng = np.random.default_rng(seed)
        draws = rng.choice(errors, size=(n_bootstrap, n), replace=True)
        summary['MAE_ci'] = tuple(np.percentile(np.mean(np.abs(draws), axis=1), [2.5, 97.5]))
        summary['RMSE_ci'] = tuple(np.percentile(np.sqrt(np.mean(draws ** 2, axis=1)), [2.5, 97.5]))

    else:
        summary['ME_ci'] = (np.nan, np.nan)
        summary['MAE_ci'] = (np.nan, np.nan)
        summary['RMSE_ci'] = (np.nan, np.nan)

    return summary



_CELL_ORDER = ['H', 'M', 'FA', 'CR']
_METRIC_ORDER = ['POD', 'SR', 'FAR', 'bias', 'CSI', 'accuracy', 'POFD', 'HK']

_METRIC_TITLES = {'POD': 'POD', 'SR': 'SR', 'FAR': 'FAR', 'bias': 'Bias', 'CSI': 'CSI',
                  'accuracy': 'Accuracy', 'POFD': 'POFD', 'HK': 'HK'}

# what each model requires before it predicts an arrival, for the record beside the numbers
_MODEL_CRITERIA = {
    'Ellipse': 'arrivalEllipsePossibleCorrected',
    'Ellipse + STRUDL': 'arrivalEllipsePossibleCorrected and a matched STRUDL detection',
    'STRUDL': 'a matched STRUDL detection, ignoring the ellipse',
}


def _format_contingency_report(summary):
    """The contingency tables as plain text, to be read beside the performance diagram."""

    population = summary['population']
    totals = population['set_totals']

    lines = ['Geometry-gate contingency tables, after Verbeke et al. (2019)', '',
             f'Population: {population["n_visible"]} CMEs that could have appeared in the '
             'STEREO-A HI-1A field of',
             f'view, of which {population["n_arrived"]} have a DONKI in-situ arrival link at '
             'Earth. An arrival is predicted',
             'or it is not, so each CME falls in exactly one cell and the four cells sum to the',
             'population.', '',
             'Venn set totals: ' + '  '.join(f'{key} {totals[key]}' for key in totals), '']

    for label, scores in summary['models'].items():
        note = ' (reference)' if label == summary['reference'] else ''
        lines += [f'== {label}{note} ==',
                  f'  predicts an arrival when: {_MODEL_CRITERIA.get(label, "see GEOMETRY_MODELS")}',
                  '',
                  '                    arrived   did not arrive',
                  f'    predicted      {scores["H"]:9d} {scores["FA"]:16d}',
                  f'    not predicted  {scores["M"]:9d} {scores["CR"]:16d}',
                  f'    (H, M, FA, CR = {scores["H"]}, {scores["M"]}, {scores["FA"]}, '
                  f'{scores["CR"]}; total {scores["total"]})',
                  '']

    width = max(len(label) for label in summary['models'])
    lines += [f'{"model":<{width}} ' + ' '.join(f'{_METRIC_TITLES[key]:>8}'
                                                for key in _METRIC_ORDER)]

    for label, scores in summary['models'].items():
        lines.append(f'{label:<{width}} ' + ' '.join(f'{scores[key]:>8.3f}'
                                                     for key in _METRIC_ORDER))

    lines += ['',
              'POD, SR, FAR, Bias and CSI carry no correct-rejection term and so do not move when',
              'the population boundary does. Accuracy, POFD and HK do, and here the correct',
              'rejections outnumber the false alarms by more than an order of magnitude, so those',
              'three have to be read against the stated population rather than on their own.', '']

    return '\n'.join(lines)


def _format_contingency_tex(summary):
    """
    The same tables as two LaTeX tabulars, ready to \\input inside a table float.

    No caption, label or float of their own, so the manuscript keeps control of placement and
    numbering; and no \\newcommand macros, since at three models by twelve numbers the tabulars
    carry everything.
    """

    population = summary['population']

    def row(label, cells):
        name = label if label != summary['reference'] else f'{label}$^{{a}}$'
        return f'{name} & ' + ' & '.join(cells) + r' \\'

    lines = [r'% contingency tables -- written by make_paper_plots.py, do not edit by hand',
             f'% population: {population["n_visible"]} CMEs visible in HI-1A, '
             f'{population["n_arrived"]} with an in-situ arrival',
             '',
             r'% the four contingency cells',
             r'\begin{tabular}{lrrrr}',
             r'\hline',
             r'Model & Hits & Misses & False alarms & Correct rejections \\',
             r'\hline']

    for label, scores in summary['models'].items():
        lines.append(row(label, [f'{scores[key]}' for key in _CELL_ORDER]))

    lines += [r'\hline', r'\end{tabular}', '',
              r'% the skill scores of Verbeke et al. (2019), Table 9',
              r'\begin{tabular}{lrrrrrrrr}',
              r'\hline',
              'Model & ' + ' & '.join(_METRIC_TITLES[key] for key in _METRIC_ORDER) + r' \\',
              r'\hline']

    for label, scores in summary['models'].items():
        lines.append(row(label, [f'{scores[key]:.3f}' for key in _METRIC_ORDER]))

    lines += [r'\hline', r'\end{tabular}', '',
              r'% $^{a}$ reference only: STRUDL detection without the ellipse test.']

    return '\n'.join(lines)


def write_contingency_results(summary, save_path, stem):
    """The contingency numbers as a readable record and as LaTeX tabulars."""

    with open(f'{save_path}{stem}.txt', 'w') as handle:
        handle.write(_format_contingency_report(summary) + '\n')

    with open(f'{save_path}{stem}.tex', 'w') as handle:
        handle.write(_format_contingency_tex(summary) + '\n')

# base colour and marker per pipeline, read only by the arrival-time error figure. Lime is the
# automatic arm and Midnight Green the manual one, the pair the rest of the paper is drawn in.
# ELEvo keeps the triangle it had, which collides with the automatic arm's; nothing plots it,
# since the only call site supplies the two ELEvoHI arms alone.
VERIFICATION_COLOR_KEYS = {'ELEvo': 'verification_elevo',
                           'STRUDL-ELEvoHI': 'geosphere_lime',
                           'Manual-ELEvoHI': 'geosphere_midnight_green'}

VERIFICATION_MARKERS = {'ELEvo': '^', 'STRUDL-ELEvoHI': '^', 'Manual-ELEvoHI': 'o'}

# the stroke a pale fill needs to hold its edge against the page. Lime sits at a contrast of 1.8
# and cannot; Midnight Green needs nothing and is left out.
VERIFICATION_STROKE_KEYS = {'STRUDL-ELEvoHI': 'geosphere_lime_shade'}


# colour and marker per model. Midnight Green is the ellipse test and Lime is STRUDL, as in the
# rest of the paper, and the model that requires both takes the colour the two make where their
# discs cross in the Venn figure, so the diagram says without a caption that the combined model is
# the conjunction of the other two rather than a third thing.
GEOMETRY_STYLES = {
    'Ellipse': ('geosphere_midnight_green', 'o', True),
    'Ellipse + STRUDL': ('geosphere_lime_midnight_mix', 's', True),
    'STRUDL': ('geosphere_lime', '^', True),
}

# the rim each marker is drawn with. Lime sits near a contrast of 1.8 against the page and cannot
# hold an edge of its own, so it is ringed in the dark shade of its own palette; the other two are
# dark enough to carry themselves and keep the white rim that lifts them off the CSI contours
# behind them, which is what None means here.
GEOMETRY_EDGE_KEYS = {
    'Ellipse': None,
    'Ellipse + STRUDL': None,
    'STRUDL': 'geosphere_lime_shade',
}

# the fallback for a model with no style of its own; every model in GEOMETRY_MODELS has one
GEOMETRY_REFERENCE_COLOR = '0.45'

# offset in points for each model's annotation, chosen so the three labels clear each other and
# the markers at the positions the current data put them in
GEOMETRY_LABEL_OFFSETS = {
    'Ellipse': (14, 8),
    'Ellipse + STRUDL': (14, 8),
    'STRUDL': (14, -18),
}


def make_performance_diagram(summary, color_dict, font_params, SAVE_PLOT_PAPER_PATH,
                             stem='performance_diagram'):
    """
    POD against the success ratio, with CSI and frequency-bias contours.

    Verbeke note that POD, SR, FAR, bias and CSI can all be read off one diagram; none of them
    carries a correct-rejection term, so this is the comparison that does not move when the
    population boundary does, which is what makes it the right figure for a population whose
    correct rejections outnumber everything else.

    Each model is a single point, because the prediction is untimed: either the geometry admits
    an arrival or it does not. Requiring a STRUDL detection can only remove predicted arrivals,
    so the step from one marker to the other is a pure recall-for-precision trade and the
    diagram shows which way it goes. The triangle is STRUDL without the ellipse test, named a
    reference in the legend and carried to show where the combined model's precision comes from.
    """

    grid = np.linspace(0.001, 1.0, 400)
    sr_grid, pod_grid = np.meshgrid(grid, grid)
    csi_grid = 1.0 / (1.0 / sr_grid + 1.0 / pod_grid - 1.0)
    bias_grid = pod_grid / sr_grid

    fig, ax = plt.subplots(figsize=(7.5, 7))

    csi_contours = ax.contourf(sr_grid, pod_grid, csi_grid, levels=np.arange(0.0, 1.1, 0.1),
                               cmap='Greys', alpha=0.35)
    bias_lines = ax.contour(sr_grid, pod_grid, bias_grid, levels=[0.25, 0.5, 1.0, 1.5, 2.0, 4.0],
                            colors='0.4', linestyles='--', linewidths=0.8)
    ax.clabel(bias_lines, fmt='%.2g', fontsize=font_params['ticks_fontsize'] - 3)

    annotation_size = font_params['ticks_fontsize'] - 3

    for label, scores in summary['models'].items():
        color_key, marker, filled = GEOMETRY_STYLES.get(label, (None, 'o', True))
        color = color_dict[color_key] if color_key else GEOMETRY_REFERENCE_COLOR

        edge_key = GEOMETRY_EDGE_KEYS.get(label)
        edge = color_dict[edge_key] if edge_key else 'white'

        legend_label = f'{label} (reference)' if label == GEOMETRY_REFERENCE else label

        ax.plot(scores['SR'], scores['POD'], marker=marker, color=color, markersize=16,
                markerfacecolor=color if filled else 'white',
                markeredgecolor=edge,
                markeredgewidth=1.6 if filled else 2.4,
                linestyle='none', label=legend_label, zorder=4)

        ax.annotate(f'CSI {scores["CSI"]:.2f}\nbias {scores["bias"]:.2f}',
                    (scores['SR'], scores['POD']), textcoords='offset points',
                    xytext=GEOMETRY_LABEL_OFFSETS.get(label, (14, 8)),
                    fontsize=annotation_size, color='0.2',
                    bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85),
                    zorder=5)

    colorbar = fig.colorbar(csi_contours, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label('Critical Success Index', fontsize=font_params['label_fontsize'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Success Ratio (1 - FAR)', fontsize=font_params['label_fontsize'])
    ax.set_ylabel('Probability of Detection', fontsize=font_params['label_fontsize'])
    ax.tick_params(labelsize=font_params['ticks_fontsize'])
    # below the panel rather than inside it, where the three markers sat over the CSI contours and
    # the frame had to be opaque to stay readable. Anchored to the axes, so the row centres under
    # the plot box rather than under the plot and its colourbar together.
    ax.legend(fontsize=font_params['legend_fontsize'] - 2, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    fig.savefig(f'{SAVE_PLOT_PAPER_PATH}{stem}.pdf', bbox_inches='tight')
    fig.savefig(f'{SAVE_PLOT_PAPER_PATH}{stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


# a half violin is a kernel density estimate wearing a shape, and nine points -- four or five
# once the overlap split is applied -- do not carry one. Groups below this draw their
# members instead, so the reader counts what is there rather than reading a smooth curve
MIN_VIOLIN_N = 10

# what the scatter panels plot the error against, with the plain forms the text report uses
FORECAST_ERROR_AXES = (('speed', 'Initial CME speed [km s$^{-1}$]',
                        'fitted initial CME speed [km/s]'),
                       ('separation', 'CME-STEREO-A separation $\\phi$ [deg]',
                        'CME-STEREO-A separation, unsigned [deg]'))

FORECAST_ERROR_GROUPS = (('Overlapping', True), ('Not overlapping', False))


def forecast_error_table(results_df, overlapping_ids):
    """
    One row per scoreable forecast: its signed arrival time error and what that is plotted against.

    The error is taken from `forecast_events` rather than recomputed, so that the figure and the
    contingency tables cannot end up with different conventions for it; there it is formed as
    forecast minus observation, in hours. Only events with both an arrival forecast and an
    observed arrival have an error at all, which is what cuts the automatic arm to nine.

    `separation` is the CME apex to STEREO-A angle `phi`, already unsigned, and not
    `delta_target`, which measures the apex against L1 instead of against the imager.
    """

    events, _ = forecast_events(results_df)
    events = pd.DataFrame(events)

    scoreable = events[events['predicted'] & events['observed']
                       & np.isfinite(events['dt_h'])]

    columns = ['cmeID_elevo', 'vinit [km/s]', 'phi [°]',
               'arrival_time [UT]', 'arrival_time_min [UT]', 'arrival_time_max [UT]']
    table = scoreable.merge(results_df[columns], on='cmeID_elevo', how='left')

    table['error'] = table['dt_h']
    table['speed'] = table['vinit [km/s]'].astype(float)
    table['separation'] = table['phi [°]'].astype(float)
    table['overlapping'] = table['cmeID_elevo'].isin(overlapping_ids)

    # the error is forecast minus observation and the observation is one fixed instant per CME,
    # so the ensemble's own bounds carry over to the error axis unshifted
    forecast = pd.to_datetime(table['arrival_time [UT]'])
    table['error_low'] = (forecast - pd.to_datetime(table['arrival_time_min [UT]'])
                          ).dt.total_seconds() / 3600.0
    table['error_high'] = (pd.to_datetime(table['arrival_time_max [UT]']) - forecast
                           ).dt.total_seconds() / 3600.0

    return table


def _forecast_error_correlations(table):
    """Spearman's rho against each scatter panel's abscissa, for one arm."""

    correlations = {}
    for key, _, _ in FORECAST_ERROR_AXES:
        if len(table) < 3:
            correlations[key] = {'rho': np.nan, 'p': np.nan, 'n': len(table)}
            continue

        rho, p_value = spearmanr(table[key], table['error'])
        correlations[key] = {'rho': float(rho), 'p': float(p_value), 'n': len(table)}

    return correlations


def _forecast_error_overlap(table):
    """
    The error on either side of the overlap split, and the test of the difference.

    The flag is `_overlapping_cme_ids`, the same criterion the false-negative figure judges
    detection on, so a CME that counts as obscured there counts as obscured here.
    """

    groups = {}
    for name, flag in FORECAST_ERROR_GROUPS:
        errors = table.loc[table['overlapping'] == flag, 'error'].to_numpy(float)
        groups[name] = {'n': len(errors),
                        'median': float(np.median(errors)) if len(errors) else np.nan,
                        'ME': float(np.mean(errors)) if len(errors) else np.nan,
                        'MAE': float(np.mean(np.abs(errors))) if len(errors) else np.nan}

    both = [table.loc[table['overlapping'] == flag, 'error'].to_numpy(float)
            for _, flag in FORECAST_ERROR_GROUPS]

    if all(len(side) for side in both):
        _, p_value = mannwhitneyu(*both, alternative='two-sided')
        groups['p'] = float(p_value)
    else:
        groups['p'] = np.nan

    return groups


def _forecast_error_paired(tables, labels):
    """
    The two arms on the events both of them forecast, which is the only unconfounded comparison.

    The arms do not cover the same CMEs -- the automatic one reaches nine of the manual one's
    forty-three -- so their headline errors differ partly because the event mixes differ. On the
    shared events that cancels, and what is left is the cost of the tracking. The pairing is by
    CME, so the test is a signed-rank test rather than a rank-sum one.

    `labels` is (manual, automatic); the difference is reported automatic minus manual.
    """

    manual, automatic = (tables[label].set_index('cmeID_elevo')['error'] for label in labels)
    shared = automatic.index.intersection(manual.index)

    if len(shared) == 0:
        return {'n': 0}

    difference = (automatic[shared] - manual[shared]).to_numpy(float)

    paired = {'n': int(len(shared)),
              'labels': list(labels),
              'mean_difference': float(np.mean(difference)),
              'median_difference': float(np.median(difference)),
              'MAE': {labels[0]: float(np.mean(np.abs(manual[shared]))),
                      labels[1]: float(np.mean(np.abs(automatic[shared])))}}

    # a signed-rank test needs a spread to rank; one pair gives it nothing to work with
    paired['p'] = float(wilcoxon(difference).pvalue) if len(difference) > 1 else np.nan

    return paired


def forecast_error_summary(tables, labels):
    """Everything the figure shows and the paired comparison it cannot show, for one report."""

    summary = {'arms': {}, 'labels': list(labels)}

    for label, table in tables.items():
        summary['arms'][label] = {
            'n': int(len(table)),
            'errors': hit_error_statistics(table['error'].to_numpy(float)),
            'correlations': _forecast_error_correlations(table),
            'overlap': _forecast_error_overlap(table)}

    summary['paired'] = _forecast_error_paired(tables, labels)

    return summary


def _format_forecast_error_report(summary):
    """The arrival time error numbers as plain text, to be read beside the figure."""

    lines = ['Arrival time error (forecast - observed, hours), by track type', '']

    for label, arm in summary['arms'].items():
        stats_ = arm['errors']
        lines += [f'== {label} ==',
                  f'  n = {arm["n"]}',
                  f'    ME   {stats_["ME"]:+7.2f}  [{stats_["ME_ci"][0]:+.2f}, {stats_["ME_ci"][1]:+.2f}]',
                  f'    MAE  {stats_["MAE"]:7.2f}  [{stats_["MAE_ci"][0]:.2f}, {stats_["MAE_ci"][1]:.2f}]',
                  f'    RMSE {stats_["RMSE"]:7.2f}  [{stats_["RMSE_ci"][0]:.2f}, {stats_["RMSE_ci"][1]:.2f}]',
                  f'    sd   {stats_["sd"]:7.2f}',
                  f'    median |error| {stats_["median_abs"]:.2f}',
                  '', '  Spearman correlation of the error with']

        for key, _, plain in FORECAST_ERROR_AXES:
            correlation = arm['correlations'][key]
            lines.append(f'    {plain}: rho = {correlation["rho"]:+.2f}, '
                         f'p = {correlation["p"]:.3g}')

        overlap = arm['overlap']
        lines += ['', '  split by whether the CME shared the HI-1A field with an obscuring one']

        for name, _ in FORECAST_ERROR_GROUPS:
            group = overlap[name]
            lines.append(f'    {name:<15} n = {group["n"]:>2}  '
                         f'median {group["median"]:+7.2f}  MAE {group["MAE"]:7.2f}')

        lines += [f'    Mann-Whitney p = {overlap["p"]:.3g}', '']

    paired = summary['paired']
    if paired['n']:
        manual_label, automatic_label = paired['labels']
        lines += ['== paired comparison ==',
                  f'  the {paired["n"]} CMEs both arms forecast; difference is '
                  f'{automatic_label} minus {manual_label}',
                  f'    mean difference   {paired["mean_difference"]:+7.2f} h',
                  f'    median difference {paired["median_difference"]:+7.2f} h',
                  f'    MAE {manual_label} {paired["MAE"][manual_label]:.2f} h, '
                  f'{automatic_label} {paired["MAE"][automatic_label]:.2f} h',
                  f'    Wilcoxon signed-rank p = {paired["p"]:.3g}', '']

    return '\n'.join(lines)


def _format_forecast_error_macros(summary, prefix):
    """The same numbers as LaTeX definitions, in the style of `_format_verification_macros`."""

    macros = []

    for label, arm in summary['arms'].items():
        name = ''.join(part.capitalize() for part in re.split(r'[^0-9A-Za-z]+', label) if part)
        name = re.sub(r'\d+', '', name)

        stats_ = arm['errors']
        macros += [(f'% {label}', None),
                   (f'{prefix}{name}N', f'{arm["n"]}'),
                   (f'{prefix}{name}Me', f'{stats_["ME"]:+.2f}'),
                   (f'{prefix}{name}MeCiLow', f'{stats_["ME_ci"][0]:+.2f}'),
                   (f'{prefix}{name}MeCiHigh', f'{stats_["ME_ci"][1]:+.2f}'),
                   (f'{prefix}{name}Mae', f'{stats_["MAE"]:.2f}'),
                   (f'{prefix}{name}MaeCiLow', f'{stats_["MAE_ci"][0]:.2f}'),
                   (f'{prefix}{name}MaeCiHigh', f'{stats_["MAE_ci"][1]:.2f}'),
                   (f'{prefix}{name}Rmse', f'{stats_["RMSE"]:.2f}'),
                   (f'{prefix}{name}Sd', f'{stats_["sd"]:.2f}'),
                   (f'{prefix}{name}MedianAbs', f'{stats_["median_abs"]:.2f}')]

        for key, _, _ in FORECAST_ERROR_AXES:
            correlation = arm['correlations'][key]
            macros += [(f'{prefix}{name}{key.capitalize()}Rho', f'{correlation["rho"]:+.2f}'),
                       (f'{prefix}{name}{key.capitalize()}P', f'{correlation["p"]:.3g}')]

        overlap = arm['overlap']
        for group_name, flag in FORECAST_ERROR_GROUPS:
            group = overlap[group_name]
            suffix = 'Overlapping' if flag else 'NotOverlapping'
            macros += [(f'{prefix}{name}{suffix}N', f'{group["n"]}'),
                       (f'{prefix}{name}{suffix}Median', f'{group["median"]:+.2f}'),
                       (f'{prefix}{name}{suffix}Mae', f'{group["MAE"]:.2f}')]

        macros.append((f'{prefix}{name}OverlapP', f'{overlap["p"]:.3g}'))

    paired = summary['paired']
    if paired['n']:
        manual_label, automatic_label = paired['labels']
        macros += [('% paired comparison', None),
                   (f'{prefix}PairedN', f'{paired["n"]}'),
                   (f'{prefix}PairedMeanDifference', f'{paired["mean_difference"]:+.2f}'),
                   (f'{prefix}PairedMedianDifference', f'{paired["median_difference"]:+.2f}'),
                   (f'{prefix}PairedP', f'{paired["p"]:.3g}')]

        for label in (manual_label, automatic_label):
            name = ''.join(part.capitalize() for part in re.split(r'[^0-9A-Za-z]+', label) if part)
            name = re.sub(r'\d+', '', name)
            macros.append((f'{prefix}Paired{name}Mae', f'{paired["MAE"][label]:.2f}'))

    lines = []
    for macro, value in macros:
        if value is None:
            lines += ['', macro]
        else:
            lines.append(f'\\newcommand\\{macro}{{{_latex_number(value)}}}')

    return '\n'.join(lines)


def write_forecast_error_results(summary, save_path, stem, macro_prefix):
    """The numbers beside the figure, as a readable record and a set of LaTeX definitions."""

    with open(f'{save_path}{stem}.txt', 'w') as handle:
        handle.write(_format_forecast_error_report(summary) + '\n')

    with open(f'{save_path}{stem}_latex.txt', 'w') as handle:
        handle.write(f'% {stem} -- written by make_paper_plots.py, do not edit by hand\n')
        handle.write(_format_forecast_error_macros(summary, macro_prefix) + '\n')


def _forecast_error_scatter_panel(ax, tables, colors, strokes, key, xlabel, font_params,
                                  annotate_bars=False, annotate=True):
    """
    One arm against the other on a quantity the forecast might depend on.

    `annotate` False drops the correlation boxes, which leaves the arms identified by their colour
    and marker shape alone, for the copy of the figure whose numbers are carried by the text.
    """

    for row, (label, table) in enumerate(tables.items()):
        # a Lime bar at this alpha is not there at all, so the pale arm's bars take its dark
        # shade outright rather than a halo, which at this width would only thicken them
        ax.errorbar(table[key], table['error'],
                    yerr=[table['error_low'], table['error_high']],
                    fmt='none', ecolor=strokes[label] or colors[label], elinewidth=0.9, alpha=0.45,
                    capsize=2, capthick=0.9, zorder=1)

        ax.scatter(table[key], table['error'], s=42, alpha=0.8, color=colors[label],
                   marker=VERIFICATION_MARKERS.get(label, 'o'),
                   edgecolor=strokes[label] or 'white',
                   linewidth=0.6, label=f'{label} (N={len(table)})', zorder=2)

        rho, p_value = spearmanr(table[key], table['error'])
        # each line is drawn in its arm's colour, which saves the panel a legend of its own; the
        # same coefficient means something different at n=9 than at n=43, and a reader who sees
        # only the number will not make that distinction, so the n travels with it. Set in the
        # dark shade for the pale arm, since a line of Lime text is not readable at this size
        if annotate:
            ax.text(0.97, 0.97 - 0.07 * row,
                    f'{label}: $\\rho$ = {rho:+.2f}, $p$ = {p_value:.2g} ($n$ = {len(table)})',
                    transform=ax.transAxes, ha='right', va='top',
                    color=strokes[label] or colors[label],
                    fontsize=font_params['legend_fontsize'] - 4)

    # # the panel identifies its arms by colour through the correlation boxes rather than a
    # # legend, so the bars are named once for the pair of scatter panels
    # if annotate_bars:
    #     ax.text(0.03, 0.03, 'bars: full ensemble range over the DBM wind sweep',
    #             transform=ax.transAxes, ha='left', va='bottom', color='0.45',
    #             fontsize=font_params['legend_fontsize'] - 4)

    ax.axhline(0, color='0.5', lw=0.8, zorder=0)
    ax.set_xlabel(xlabel, fontsize=font_params['label_fontsize'])
    ax.tick_params(labelsize=font_params['ticks_fontsize'])
    ax.spines[['top', 'right']].set_visible(False)


def _half_violin(ax, position, errors, side, color, width=0.4):
    """One side of a split violin, as in the prototype this figure started from."""

    grid = np.linspace(errors.min(), errors.max(), 300)
    density = gaussian_kde(errors)(grid)
    density = density / density.max() * width

    edge = position - density if side == 'left' else position + density

    # light enough that the overlaid points and their bars stay legible against it, with the
    # profile redrawn as an outline so the shape survives the lower fill opacity
    ax.fill_betweenx(grid, np.minimum(edge, position), np.maximum(edge, position),
                     color=color, alpha=0.25, lw=0, zorder=1)
    ax.plot(edge, grid, color=color, lw=1.2, alpha=0.9, zorder=1)


def _half_points(ax, position, errors, low, high, side, color, rng, stroke=None, width=0.4):
    """The members themselves, each with the range its DBM ensemble spanned."""

    offset = rng.uniform(0.08 * width, 0.75 * width, len(errors))
    x = position + (-offset if side == 'left' else offset)

    ax.errorbar(x, errors, yerr=[low, high], fmt='none', ecolor=stroke or color,
                elinewidth=0.9, alpha=0.45, capsize=2, capthick=0.9, zorder=2)
    ax.scatter(x, errors, s=42, color=color, alpha=0.9, edgecolor=stroke or 'white',
               linewidth=0.6, zorder=3)


def _half_strip(ax, position, errors, side, color, stroke=None, width=0.4):
    """The median of a group too small to carry a density, beside its own points."""

    ax.hlines(np.median(errors),
              position - 0.75 * width if side == 'left' else position,
              position if side == 'left' else position + 0.75 * width,
              color=color, lw=2.0, zorder=4, path_effects=_outline(stroke, 2.0))


def _forecast_error_overlap_panel(ax, tables, colors, strokes, labels, font_params, seed=4,
                                  annotate=True):
    """
    The error split by whether the CME shared the field with an obscuring one.

    Every forecast is drawn as its own point with the range its ensemble spanned. The manual
    arm has twenty-odd events per group and carries a density behind them; the automatic arm has
    four and five, too few for a shape the sample could support, so it carries a median instead.
    Mixing the two backdrops in one panel is the honest option, and dropping the smaller arm
    would hide the comparison entirely.
    """

    rng = np.random.default_rng(seed)
    manual_label, automatic_label = labels

    for position, (group_name, flag) in enumerate(FORECAST_ERROR_GROUPS):
        for label, side in ((manual_label, 'left'), (automatic_label, 'right')):
            group = tables[label].loc[tables[label]['overlapping'] == flag]
            errors = group['error'].to_numpy(float)

            if len(errors) == 0:
                continue

            if len(errors) >= MIN_VIOLIN_N:
                _half_violin(ax, position, errors, side, colors[label])
            else:
                _half_strip(ax, position, errors, side, colors[label], strokes[label])

            _half_points(ax, position, errors, group['error_low'].to_numpy(float),
                         group['error_high'].to_numpy(float), side, colors[label], rng,
                         strokes[label])

            if annotate:
                ax.text(position + (-0.2 if side == 'left' else 0.2), 0.02, f'n={len(errors)}',
                        transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                        fontsize=font_params['ticks_fontsize'] - 2)

    ax.axhline(0, color='0.5', lw=0.8, zorder=0)
    ax.set_xticks(range(len(FORECAST_ERROR_GROUPS)),
                  [name for name, _ in FORECAST_ERROR_GROUPS])
    ax.set_xlim(-0.6, len(FORECAST_ERROR_GROUPS) - 0.4)
    ax.tick_params(labelsize=font_params['ticks_fontsize'])
    ax.spines[['top', 'right']].set_visible(False)


def _forecast_error_ecdf_panel(ax, tables, colors, strokes, font_params, annotate=True):
    """
    The full distribution of each arm, without smoothing.

    A kernel density of nine points says more about the bandwidth than about the forecast. The
    empirical distribution function is exact at any sample size and puts both arms on one axis,
    where the stacked densities of the prototype put them on two and made the shift between
    them a matter of comparing peak positions by eye.
    """

    for label, table in tables.items():
        errors = np.sort(table['error'].to_numpy(float))
        # the curve is closed at both ends so it starts at zero and finishes at one, rather
        # than appearing to begin partway up
        ax.step(np.concatenate(([errors[0]], errors, [errors[-1]])),
                np.concatenate(([0.0], np.arange(1, len(errors) + 1) / len(errors), [1.0])),
                where='post', color=colors[label], lw=2.2,
                path_effects=_outline(strokes[label], 2.2),
                label=f'{label} (N={len(errors)}, ME = {errors.mean():+.1f} h)')

        # the two means sit six hours apart, which is close enough that labelling each line
        # where it stands would overprint one on the other, so the legend carries them
        ax.axvline(errors.mean(), color=colors[label], ls='--', lw=1.4, alpha=0.8,
                   path_effects=_outline(strokes[label], 1.4))

    ax.axvline(0, color='0.5', lw=0.8, zorder=0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Arrival time error [h] (predicted $-$ in situ)',
                  fontsize=font_params['label_fontsize'])
    ax.set_ylabel('Cumulative fraction', fontsize=font_params['label_fontsize'])
    ax.tick_params(labelsize=font_params['ticks_fontsize'])
    if annotate:
        ax.legend(fontsize=font_params['legend_fontsize'] - 3, loc='upper left', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)


def make_forecast_error_plot(tables, labels, color_dict, font_params, SAVE_PLOT_PAPER_PATH,
                             stem='forecast_error', annotate=True):
    """
    What the arrival time error depends on, for manually corrected and automatic tracks.

    The contingency tables say how often each arm was right; this says by how much it was wrong
    and against what. The automatic arm reaches nine scoreable forecasts, so every panel is
    drawn so that nine points are legible as nine points: the scatter panels carry the n beside
    each correlation, the overlap panel falls back to drawing the members, and the
    distribution panel is an ECDF rather than a density.
    """

    colors = {label: color_dict[VERIFICATION_COLOR_KEYS[label]] for label in tables}
    strokes = {label: color_dict[VERIFICATION_STROKE_KEYS[label]]
                      if label in VERIFICATION_STROKE_KEYS else None
               for label in tables}

    fig = plt.figure(figsize=(13, 11))
    grid = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
    ax_speed = fig.add_subplot(grid[0, 0])
    ax_separation = fig.add_subplot(grid[0, 1], sharey=ax_speed)
    ax_overlap = fig.add_subplot(grid[1, 0])
    ax_ecdf = fig.add_subplot(grid[1, 1])

    for ax, (key, xlabel, _) in zip((ax_speed, ax_separation), FORECAST_ERROR_AXES):
        _forecast_error_scatter_panel(ax, tables, colors, strokes, key, xlabel, font_params,
                                      annotate_bars=ax is ax_speed, annotate=annotate)

    _forecast_error_overlap_panel(ax_overlap, tables, colors, strokes, labels, font_params,
                                  annotate=annotate)
    _forecast_error_ecdf_panel(ax_ecdf, tables, colors, strokes, font_params, annotate=annotate)

    error_label = 'Arrival time error [h]\n(predicted $-$ in situ)'
    for ax in (ax_speed, ax_overlap):
        ax.set_ylabel(error_label, fontsize=font_params['label_fontsize'])

    # the error runs from about -25 to +88 h and is nothing like symmetric about zero, so the
    # limits are taken from the data rather than from a round number either side of it. The top
    # margin is the wider one because the correlation boxes of the scatter panels sit there, and
    # at the narrower margin they print over the largest errors instead of above them. Without
    # the boxes that headroom is empty, so the unannotated figure is padded like any other panel
    errors = np.concatenate([table['error'].to_numpy(float) for table in tables.values()])
    low = np.concatenate([(table['error'] - table['error_low']).to_numpy(float)
                          for table in tables.values()])
    high = np.concatenate([(table['error'] + table['error_high']).to_numpy(float)
                           for table in tables.values()])

    span = errors.max() - errors.min()
    ax_speed.set_ylim(low.min() - 0.12 * span, high.max() + (0.30 if annotate else 0.12) * span)

    # panel d plots the distribution of the errors and carries no bars, so it keeps the bare
    # extent; stretching it to cover them would only add empty margin
    ax_ecdf.set_xlim(errors.min() - 0.12 * span, errors.max() + 0.12 * span)

    # the overlap panel is left off that shared axis: it carries no correlation box and the
    # headroom the scatter panels need for theirs would be empty space here. Its violins already
    # run past the largest error, so it is padded on its own drawn extent rather than on the data
    ax_overlap.set_ylim(min(ax_overlap.get_ylim()[0], low.min() - 0.12 * span),
                            max(ax_overlap.get_ylim()[1], high.max() + 0.08 * span))

    plt.setp(ax_separation.get_yticklabels(), visible=False)

    if annotate:
        _place_panel_letters(fig, [(ax_speed, 'a'), (ax_separation, 'b'),
                                   (ax_overlap, 'c'), (ax_ecdf, 'd')], font_params)

    fig.savefig(f'{SAVE_PLOT_PAPER_PATH}{stem}.pdf', bbox_inches='tight')
    fig.savefig(f'{SAVE_PLOT_PAPER_PATH}{stem}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_forecast_error_figure(manual_df, strudl_df, overlaps_path, pipeline_label_dict,
                               color_dict, font_params, SAVE_PLOT_PAPER_PATH,
                               stem='forecast_error', annotate=True):
    """
    Assemble both arms, draw the figure and write the numbers beside it.

    `annotate` False draws the same figure without the correlation boxes, the group counts, the
    legend or the panel letters, under its own name. The numbers are the same either way, so only
    the annotated run files them.
    """

    overlapping_ids = _overlapping_cme_ids(overlaps_path)['overlapping_ids']

    labels = (pipeline_label_dict['elevohi_corrected_label'],
              pipeline_label_dict['elevohi_uncorrected_label'])

    tables = {labels[0]: forecast_error_table(manual_df, overlapping_ids),
              labels[1]: forecast_error_table(strudl_df, overlapping_ids)}

    summary = forecast_error_summary(tables, labels)

    make_forecast_error_plot(tables, labels, color_dict, font_params, SAVE_PLOT_PAPER_PATH,
                             stem=stem if annotate else f'{stem}_bare', annotate=annotate)

    if annotate:
        write_forecast_error_results(summary, SAVE_PLOT_PAPER_PATH, stem, 'fer')

    return summary


def load_forecast_results(path):
    """A results table from either forecast code, with the column names the scoring code uses."""

    df = pd.read_json(path)

    if 'L1_ist_obs' in df.columns:
        df = df.rename(columns={'L1_ist_obs': 'L1_ist_obs [UT]'})

    if 'forecast_outcome' not in df.columns:
        raise ValueError(f'{path} predates the forecast_outcome field; re-run the forecast')

    return df


if __name__ == '__main__':

    SAVE_PLOTS = True
    SAVE_PLOT_PAPER_PATH = 'paper_plots/'

    geosphere_purple = '#701F80'
    geosphere_green = '#75CC41'
    geosphere_yellow = '#F9F200'
    geosphere_orange = '#F29707'
    my_pink = '#FF69B4'

    # Midnight Green and Lime with their 70 / 40 / 15 % tints, the two palettes of the corporate
    # design document; the tints are each base colour's opacity over white. The false-negative
    # figure is drawn in the pair, Midnight Green for the missed CMEs and Lime for the detected
    # ones, and the tracking-profile figure uses the Midnight Green ramp through
    # midnight_green_colormap.
    geosphere_midnight_green = '#052E37'
    geosphere_midnight_green_70 = '#546D73'
    geosphere_midnight_green_40 = '#9DABAE'
    geosphere_midnight_green_15 = '#DAE0E1'

    geosphere_lime = '#BFCE40'
    geosphere_lime_70 = '#D2DD65'
    geosphere_lime_40 = '#E5ECA3'
    geosphere_lime_15 = '#F6F8DC'
    # not in the design document, which gives tints and no shades: 40 % of Lime over black. Lime
    # has a contrast of 1.8 against the white page, enough for a filled bar and not for a
    # two-point curve, so the curves are stroked in this and nothing else uses it.
    geosphere_lime_shade = '#4C521A'

    # only used by the Venn diagram, see VENN_SET_COLOR_KEYS
    venn_blue = '#2A78D6'
    venn_red = '#B22222'

    # Lime and Midnight Green mixed in equal parts, the colour the two make where they meet. It
    # means "both criteria" in the performance diagram, where the combined model is the
    # conjunction of the Midnight Green one and the Lime one. The Venn draws its discs
    # translucent over a pale backdrop, which gives the same overlap a much lighter cast; taken
    # from there the marker washed out against the white page, so the mix is taken at full
    # strength instead.
    geosphere_lime_midnight_mix = '#627E3C'

    # the palette the verification figures were drawn in before they moved onto Midnight Green
    # and Lime. Nothing reads brick red or mint any more and indigo survives only as the unused
    # 'ELEvo' slot of VERIFICATION_COLOR_KEYS; they are kept against a third pipeline being
    # scored, which is what that slot is there for.
    verification_elevo = '#CC2C01'
    verification_strudl = '#5833FE'
    verification_manual = '#5AE291'

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
        "geosphere_lime": geosphere_lime,
        "geosphere_lime_70": geosphere_lime_70,
        "geosphere_lime_40": geosphere_lime_40,
        "geosphere_lime_15": geosphere_lime_15,
        "geosphere_lime_shade": geosphere_lime_shade,
        "venn_blue": venn_blue,
        "venn_red": venn_red,
        "geosphere_lime_midnight_mix": geosphere_lime_midnight_mix,
        "verification_elevo": verification_elevo,
        "verification_strudl": verification_strudl,
        "verification_manual": verification_manual
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

    fn_analysis_summary_v2 = make_strudl_fn_analysis_plot_v2(donki_df_dedupe,
                                                             strudl_cmes_path=DATA_STRUDL_CMES_PATH,
                                                             overlaps_path=DATA_OVERLAPS_PATH,
                                                             color_dict=color_dict,
                                                             font_params=font_params,
                                                             SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH,
                                                             bin_width_deg=45.0,
                                                             prepared=fn_analysis_prepared)

    # the same figure without the key, the panel letters or the statistics boxes
    make_strudl_fn_analysis_plot_v2(donki_df_dedupe,
                                    strudl_cmes_path=DATA_STRUDL_CMES_PATH,
                                    overlaps_path=DATA_OVERLAPS_PATH,
                                    color_dict=color_dict,
                                    font_params=font_params,
                                    SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH,
                                    bin_width_deg=45.0,
                                    prepared=fn_analysis_prepared,
                                    annotate=False)

    DATA_PAIRED_TRACKS_PATH = 'STRUDL_Tracking_Analysis/data/manual_vs_strudl_tracks.json'

    tracking_profiles_summary = make_strudl_tracking_performance_plot(
        paired_tracks_path=DATA_PAIRED_TRACKS_PATH,
        color_dict=color_dict,
        font_params=font_params,
        SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

    # make Figure 1
    venn_regions, venn_totals = make_donki_venn_plot(donki_df_dedupe, color_dict=color_dict, font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

    # STRUDL_SEGMENTATION_PATH = 'STRUDL/segmentation_dictionary_mean_45_2024_05_01_2026_05_01.npy'
    # strudl_seg = np.load(STRUDL_SEGMENTATION_PATH, allow_pickle=True).item()

    # RUNDIF_PATH = 'HI_Data_Processing/stereo_processed/running_difference/data/A/'
    # CHOSEN_RUNDIF_TIMES = [datetime(2024,8,20,15,58,11), datetime(2024,7,23,11,18,11)]

    # # make Figure 4
    # make_strudl_segmentation_plot(strudl_seg, RUNDIF_PATH, CHOSEN_RUNDIF_TIMES, color_dict=color_dict, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH, rdif_freq=4)

    # What the arrival ellipse, and then STRUDL on top of it, tell us about an Earth arrival.
    # Scored on the Venn masks, so these tables and the Venn figure cannot disagree.
    contingency = geometry_skill_summary(donki_df_dedupe)

    write_contingency_results(contingency, SAVE_PLOT_PAPER_PATH, 'contingency')

    make_performance_diagram(contingency, color_dict=color_dict, font_params=font_params,
                             SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

    # how far wrong each pipeline's arrival time was, rather than how often it called the
    # arrival at all, which is the one part of the forecast comparison this paper still carries
    pipeline_label_dict = {
        "elevohi_uncorrected_label": "STRUDL-ELEvoHI",
        "elevohi_corrected_label": "Manual-ELEvoHI"
    }

    strudl_df = load_forecast_results('ELEvoHI/results/elevohi_ensembles_results_raw.json')
    manual_df = load_forecast_results('ELEvoHI/results/elevohi_ensembles_results_cleaned.json')

    forecast_error = make_forecast_error_figure(
        manual_df, strudl_df, overlaps_path=DATA_OVERLAPS_PATH,
        pipeline_label_dict=pipeline_label_dict, color_dict=color_dict,
        font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

    # the same figure without the correlation boxes, the counts, the legend or the panel letters
    make_forecast_error_figure(
        manual_df, strudl_df, overlaps_path=DATA_OVERLAPS_PATH,
        pipeline_label_dict=pipeline_label_dict, color_dict=color_dict,
        font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH,
        annotate=False)


