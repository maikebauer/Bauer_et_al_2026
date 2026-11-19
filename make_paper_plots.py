import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import math
import os
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Patch
from matplotlib.colors import to_rgba, LinearSegmentedColormap, LogNorm
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, mannwhitneyu, fisher_exact, chi2_contingency, spearmanr, wilcoxon, t as student_t
from scipy.stats.contingency import odds_ratio as conditional_odds_ratio
from matplotlib.gridspec import GridSpec
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ELEvoHI'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'STRUDL_Tracking_Analysis'))
import helper_functions
import functions as elevohi_functions

def convert_time(time_data, format='%Y-%m-%dT%H:%MZ'):
    """Parse `time_data` for `load_donki_data`; pd.NaT when it does not parse."""

    try:
        converted_time = datetime.strptime(time_data, format)
    except (ValueError, TypeError):
        converted_time = pd.NaT
    
    return converted_time

def load_donki_data(file_path):
    """
    Read the matched DONKI CME analyses and reduce them to the analysis period.

    Parameters
    ----------
    file_path : str
        Path to DONKI_cme_analyses_matched.json.

    Returns
    -------
    tuple of pandas.DataFrame
        (deduplicated, filtered). Both cover 2024-05-01 to 2026-04-30 on `time21_5` and
        exclude the 'TE' feature code, which is a time extrapolation rather than a
        measurement. The first keeps one analysis per `associatedActivityID`, the 'LE'
        one where a CME has both, since featureCode sorts ahead of 'SH'.
    """

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

# color_dict keys, one per set
VENN_SET_COLOR_KEYS = {
    "V": "geosphere_purple",
    "E": "venn_blue",
    "A": "venn_red",
    "S": "geosphere_green",
    "I": "my_pink",
}

VENN_V_RADIUS = 3.0           # fixes the overall scale of the layout
VENN_LABEL_CLEARANCE = 0.24   # data units a region needs to hold its count label

# solver starting point, found for the counts of the 2024-05-01 to 2026-04-30 period
VENN_REFERENCE_LAYOUT = {
    "V": (2.423, -0.216, 3.000),
    "E": (0.505, -0.028, 2.574),
    "A": (0.883, -0.404, 1.308),
    "S": (2.569, -0.636, 1.717),
    "I": (1.247, -1.606, 1.633),
}

def get_venn_set_masks(donki_df):
    """
    Membership of each Venn set, one boolean mask per set.

    Parameters
    ----------
    donki_df : pandas.DataFrame
        Deduplicated analyses from `load_donki_data`.

    Returns
    -------
    dict of pandas.Series
        One mask per VENN_SETS key. Shared by the Venn figure and
        `geometry_skill_summary`, so the two cannot disagree on who is in which set.
    """

    masks = {}

    for key, (column, qualifies) in VENN_SETS.items():
        if qualifies == "true":
            masks[key] = donki_df[column].fillna(False).astype(bool)
        else:
            masks[key] = donki_df[column].notna()

    return masks

def get_venn_region_counts(donki_df):
    """
    Count the CMEs in every exclusive region of the five Venn sets.

    Parameters
    ----------
    donki_df : pandas.DataFrame
        Deduplicated analyses from `load_donki_data`.

    Returns
    -------
    tuple of dict
        (regions, totals). `regions` is keyed by the sets a CME belongs to, in VEASI
        order (e.g. 'VEI'), the empty key holding the CMEs in none of them; `totals`
        gives the per-set totals plus 'D' for all CMEs.

    Raises
    ------
    ValueError
        If A is not contained in V and E, or S not in V. The layout solver assumes
        both, so a diagram that contradicts the data must fail rather than be drawn.
    """

    masks = get_venn_set_masks(donki_df)

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
    """Unpack the solver's parameter vector into {set key: (x, y, r)} for
    `_venn_layout_cost`; V's radius is pinned to VENN_V_RADIUS."""

    layout = {key: (float(params[3 * i]), float(params[3 * i + 1]), abs(float(params[3 * i + 2])))
              for i, key in enumerate(VENN_SETS)}
    layout["V"] = (layout["V"][0], layout["V"][1], VENN_V_RADIUS)

    return layout

def _venn_params_from_layout(layout):
    """Flatten a layout into the solver's parameter vector, inverting
    `_venn_layout_from_params` for `solve_venn_layout`."""

    return np.array([value for key in VENN_SETS for value in layout[key]], dtype=float)

def _venn_grid(n_points, extent):
    """Square sampling grid on [-`extent`, `extent`] for `_venn_measure_regions`, as
    (X, Y, cell area)."""

    g = np.linspace(-extent, extent, n_points, dtype=np.float32)
    X, Y = np.meshgrid(g, g)

    return X, Y, float((g[1] - g[0]) ** 2)

def _venn_signed_distances(layout, grid):
    """Per grid cell, the distance inside (positive) or outside (negative) each circle.
    Serves `_venn_measure_regions`."""

    X, Y = grid[0], grid[1]

    return {key: r - np.hypot(X - x, Y - y) for key, (x, y, r) in layout.items()}

def _venn_measure_regions(layout, grid):
    """
    Measure every region a candidate layout actually creates. Serves `_venn_layout_cost`.

    Parameters
    ----------
    layout : dict
        {set key: (x, y, r)}.
    grid : tuple
        Sampling grid from `_venn_grid`.

    Returns
    -------
    tuple of dict
        (areas, clearance), both keyed by region as in `get_venn_region_counts`.
        Clearance is the room a region has for its count label.
    """

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

        # room for a label is the distance to the region's closest boundary
        margin = np.minimum.reduce([signed_distance[key] if value >> i & 1 else -signed_distance[key]
                                    for i, key in enumerate(keys)])
        clearance[region] = float(margin[inside].max())

    return areas, clearance

def _venn_layout_cost(params, grid, log_targets):
    """
    Score one candidate layout. Serves `_venn_anneal` and `solve_venn_layout`.

    Region areas are matched to sqrt(N) up to a common factor, which keeps the small
    regions readable without losing the size order. The cost also penalises missing
    regions, regions the counts do not call for, labels without clearance, and A or S
    escaping the circles that must contain them.

    Parameters
    ----------
    params : array-like
        Parameter vector as built by `_venn_params_from_layout`.
    grid : tuple
        Sampling grid from `_venn_grid`.
    log_targets : dict
        Target log area per region, 0.5 * log(count).

    Returns
    -------
    tuple
        (cost, layout, areas, clearance, missing). A layout missing one of the wanted
        regions is scored above 1e3 and its areas are not comparable.
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
    """
    Anneal the five circle centres and radii. Serves `solve_venn_layout`.

    Parameters
    ----------
    params : array-like
        Starting parameter vector.
    grid, log_targets : tuple, dict
        Passed to `_venn_layout_cost`.
    seed : int
        Seeds the proposal stream, so a given start always anneals the same way.
    n_iter : int
        Proposals; the step is cut to a third after half of them.
    step : float
        Standard deviation of a proposal, in data units.
    temperature : float
        Initial acceptance temperature, decayed by 0.9985 per iteration.

    Returns
    -------
    tuple
        (best cost, best parameter vector).
    """

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
    """
    Find circle centres and radii that create exactly the regions in `regions`.

    VENN_REFERENCE_LAYOUT is refined first, which suffices while the counts resemble
    the ones it was found for; if a region is missing or an unwanted one appears, the
    search restarts from a wider spread. Every step is seeded, so the same counts
    always give the same figure.

    Parameters
    ----------
    regions : dict
        Region counts from `get_venn_region_counts`.
    n_restarts : int, default=6
        Cold restarts attempted after the reference layout.
    verbose : bool, default=False
        Print the accepted layout's cost and smallest label clearance.

    Returns
    -------
    tuple
        (layout, areas, clearance) as in `_venn_measure_regions`.

    Raises
    ------
    RuntimeError
        If no attempt reproduces the wanted regions.
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
    """
    Draw the Venn diagram of the DONKI CME categories, with areas approx. prop. to sqrt(N).

    Saves donki_categories_venn.pdf and .png.

    Parameters
    ----------
    donki_df : pandas.DataFrame
        Deduplicated analyses from `load_donki_data`.
    color_dict, font_params : dict
        Palette and font sizes built in __main__.
    SAVE_PLOT_PAPER_PATH : str
        Directory to write into, trailing separator included.

    Returns
    -------
    tuple of dict
        (regions, totals) as returned by `get_venn_region_counts`.

    Raises
    ------
    RuntimeError
        If a region of `regions` has no interior point in the solved layout.
    """

    regions, totals = get_venn_region_counts(donki_df)
    layout, areas, clearance = solve_venn_layout(regions)

    # the DONKI circle has to be grown to enclose the five sets, so it is the one region
    # whose area is not proportional to sqrt(N)
    enclosing_radius = lambda c: max(np.hypot(c[0] - x, c[1] - y) + r for x, y, r in layout.values())
    donki_center = tuple(minimize(enclosing_radius, [0.0, 0.0], method="Nelder-Mead").x)

    area_per_sqrt_n = np.median([areas[region] / np.sqrt(regions[region])
                                 for region in areas if region in regions])
    donki_radius = np.sqrt((sum(areas.values()) + area_per_sqrt_n * np.sqrt(regions.get("", 1))) / np.pi)
    donki_radius = max(donki_radius, enclosing_radius(donki_center) + 0.3)

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

    edge_colors = {key: tuple(0.8 * channel for channel in to_rgba(color)[:3])
                   for key, color in set_colors.items()}

    fig, ax = plt.subplots(figsize=(9, 10.5))

    ax.add_patch(Circle(donki_center, donki_radius, fc=to_rgba("0.6", 0.06), ec="0.6", lw=2))

    for key, (x, y, r) in layout.items():
        ax.add_patch(Circle((x, y), r, fc=to_rgba(set_colors[key], 0.10), ec=edge_colors[key], lw=2.2))

    for region, (x, y) in label_pos.items():
        # a tight region takes a smaller number, to keep it clear of the circle edges
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
    Cliff's delta for TP against FN, with a percentile bootstrap interval. Serves `_plot_kde_panel`.

    delta runs from -1 to +1, zero when neither group tends to exceed the other, positive when a
    randomly drawn TP exceeds a randomly drawn FN more often than the reverse. It is reported
    beside the p-value because the two groups are very unevenly sized, which makes a p-value
    reflect the sample sizes as much as the distance between the populations.

    Derived from the Mann-Whitney statistic rather than recomputed, so the effect size and the
    test cannot disagree. Like that test it is distribution-free and handles the heavy ties of the
    integer-degree quantities through midranks, which a mean-based measure such as Cohen's d would
    not on these skewed, tied samples.

    Parameters
    ----------
    sample_tp, sample_fn : array-like
        The detected and the missed CMEs' values of one quantity.
    u_statistic : float
        Must come from `mannwhitneyu(sample_fn, sample_tp)`, FN first, or the sign flips.
    rng : numpy.random.Generator
        Resampling stream; shared across panels, so the intervals are reproducible.
    n_bootstrap : int, default=4000
        Resamples behind the interval.

    Returns
    -------
    tuple of float
        (delta, 2.5th percentile, 97.5th percentile).
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
    Flag the CMEs that shared the HI-1A field with another one closely enough to obscure it.

    Kept apart from `_fn_analysis_populations` because the forecast-error figure asks the same
    question of a different population, and a second copy of the criterion would let the two
    figures disagree about which CMEs were overlapped.

    Parameters
    ----------
    overlaps_path : str
        Path to DONKI_cme_hi1_overlaps.json.
    overlap_coefficient : float, default=0.1
        Least share of position angles a pair must have in common.
    overlap_minutes : int, default=80
        Least time a pair must share the field.
    elongation_separation : float, default=helper_functions.ELONGATION_SEPARATION_DEG
        Greatest median elongation separation a pair may have, the width of one STRUDL
        annotation and so the point below which both fronts fall in the same mask.

    Returns
    -------
    dict
        `overlapping_ids` plus the three criteria and the pair and CME counts.

    Raises
    ------
    RuntimeError
        If the overlaps file predates the elongation separation criterion.
    """

    overlaps_df = pd.read_json(overlaps_path)
    if 'medianElongationSeparation' not in overlaps_df.columns:
        raise RuntimeError(f"{overlaps_path} predates the elongation separation criterion; "
                           "re-run get_hi1a_overlaps in run_pre_fitting_pipeline.py")

    # coerced in case every pair came back null, which leaves an object dtype that cannot be
    # compared against a float
    separation = pd.to_numeric(overlaps_df['medianElongationSeparation'], errors='coerce')

    # the separation test runs the other way round: less separation is more obscuring. A pair
    # that never shared a position angle has NaN separation and fails it, which is correct since
    # it fails the position angle test too.
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
    Build the detected and missed populations the false-negative analysis compares.

    Both versions of the figure draw from this, so a change of criterion cannot leave them
    disagreeing about which CMEs were missed. `get_stereo_heeq_lon` goes to JPL Horizons, so this
    is computed once and handed to both plotting functions through their `prepared` argument.

    Measured on the raw DONKI parameters rather than the ecliptic-corrected ones: the question is
    what the instrument and the detector saw, not what the CME did in the ecliptic.

    Parameters
    ----------
    donki_df : pandas.DataFrame
        Deduplicated analyses from `load_donki_data`.
    strudl_cmes_path : str
        Path to STRUDL_cmes.json, read for the track count and the false positives.
    overlaps_path, overlap_coefficient, overlap_minutes, elongation_separation
        Passed to `_overlapping_cme_ids`.

    Returns
    -------
    dict
        `visible` (the HI-1A population, with `phi`, `phiSigned`, `absLatitude` and
        `overlapping` added), `populations` with its 'TP' and 'FN' frames,
        `false_positives`, `n_strudl_tracks`, and the overlap criteria and counts.
    """

    visible = donki_df[donki_df['couldBeInHI1AFov'].fillna(False).astype(bool)].copy()

    sta_lon = helper_functions.get_stereo_heeq_lon(list(visible['time21_5']), sc='A')
    # both are kept: ELCon needs the magnitude, the polar panels need the side. Signed is
    # positive when the CME apex sits at a lower HEEQ longitude than the spacecraft.
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
    """Sample-level part of the false-negative figure's summary, with an empty `panels` dict for
    `make_strudl_fn_analysis_plot_v2` to fill."""

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
    """Path effect laying a width-`lw` line over a wider `stroke`, or None for no effect. Serves
    the false-negative and tracking panels."""

    return None if stroke is None else [pe.Stroke(linewidth=lw + 1.5, foreground=stroke), pe.Normal()]


def _plot_kde_panel(ax, populations, colors, strokes, key, xlabel, unit, clip_display, font_params,
                    rng, annotate=True):
    """
    Draw the TP and FN distributions of one quantity and test the difference between them.

    Serves `make_strudl_fn_analysis_plot_v2`. Each panel is normalised to its own taller peak, so
    it shows the two populations against each other and drops the absolute density, which no
    reader takes off a panel like this.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    populations : dict
        'TP' and 'FN' frames from `_fn_analysis_populations`.
    colors, strokes : dict
        Per population, a fill colour and an outline colour or None.
    key : str
        Column to plot.
    xlabel : str
        Axis label, mathmode allowed.
    unit : str
        Suffix for the median labels.
    clip_display : bool
        Cut the upper limit at the 99th percentile, for a quantity with a long tail.
    font_params : dict
        Font sizes.
    rng : numpy.random.Generator
        Passed to `cliffs_delta_with_ci`. Shared across panels, so the panels must be drawn in a
        fixed order for the intervals to be reproducible.
    annotate : bool, default=True
        False drops the median legend and the statistics box.

    Returns
    -------
    dict
        Both medians, the Mann-Whitney `u` and `p`, and Cliff's `delta` with its interval.
    """

    values = {cls: populations[cls][key].to_numpy(dtype=float) for cls in ('TP', 'FN')}

    low = min(v.min() for v in values.values())
    high = max(v.max() for v in values.values())
    pad = 0.1 * (high - low)
    # a non-negative quantity is not padded into the negatives; a signed one is, since clamping
    # it would cut off half its distribution
    lower = max(low - pad, 0.0) if low >= 0.0 else low - pad
    x_grid = np.linspace(lower, high + pad, 400)

    # one divisor for both curves, so their relative height survives the normalisation
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
    Draw how often each population shared the field with an obscuring CME, and test the difference.

    Serves `make_strudl_fn_analysis_plot_v2`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    populations : dict
        'TP' and 'FN' frames from `_fn_analysis_populations`, carrying `overlapping`.
    colors, strokes : dict
        Per population, a fill colour and an outline colour or None.
    font_params : dict
        Font sizes.
    annotate : bool, default=True
        False drops the statistics box.

    Returns
    -------
    dict
        The four cell counts, the conditional odds ratio with its interval, and Fisher's exact `p`.
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
    """Font sizes `by` steps up, for the two multi-panel figures. Returns a copy, so the shared
    `font_params` built in __main__ keeps the sizes the other figures were tuned at."""

    return {name: size + by for name, size in font_params.items()}


def _population_legend(fig, populations, colors, strokes, font_params, y=0.99):
    """
    Add the figure-wide key naming each population and its size.

    Serves `make_strudl_fn_analysis_plot_v2`. One legend for the whole figure leaves each panel's
    own legend free for what differs between panels, which is the median of each distribution.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add the legend to.
    populations : dict
        'TP' and 'FN' frames, read for their sizes.
    colors, strokes : dict
        Per population, a fill colour and an outline colour or None.
    font_params : dict
        Font sizes.
    y : float, default=0.99
        Figure-coordinate top edge of the legend.

    Returns
    -------
    matplotlib.legend.Legend
    """

    handles = [Patch(facecolor=colors[cls], edgecolor=strokes[cls],
                     lw=0.0 if strokes[cls] is None else 1.0,
                     label=f'{cls}, N={len(populations[cls])}')
               for cls in ('TP', 'FN')]

    return fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, y), ncol=2,
                      frameon=False, fontsize=font_params['legend_fontsize'])


def _place_panel_letters(fig, panels, font_params, pad=0.012):
    """
    Label panels in figure coordinates, aligned by row and column.

    Serves `make_strudl_fn_analysis_plot_v2` and `make_forecast_error_plot`. Letters placed in each
    panel's own axes coordinates land wherever the axes happen to sit, and a square polar axes
    inset in a wider cell starts further right than a rectangular one above it; one anchor per
    group of panels puts them on a common line.

    Positions are read straight from the axes, which holds because neither caller uses a layout
    engine that would move them afterwards, and bbox_inches='tight' crops without shifting.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to write into.
    panels : iterable of tuple
        (axes, letter) pairs.
    font_params : dict
        Font sizes.
    pad : float, default=0.012
        Figure-coordinate gap between a panel's top edge and its letter.
    """

    boxes = [(ax.get_position(), letter) for ax, letter in panels]

    def column(box):
        centre = 0.5 * (box.x0 + box.x1)
        return 'left' if centre < 0.45 else ('right' if centre > 0.55 else 'middle')

    # one left edge per column and one top per row, shared by every panel in them
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


# the geometry of the two polar panels, in units of the radius of the wedge ring's outer edge
_R_SUN = 0.30
_R_RING_INNER = 0.42
_R_RING_WIDTH = 0.58
_R_SPACECRAFT = 1.09
_R_LIMIT = 1.15


def detection_odds_colormap(color_dict):
    """
    Build the diverging colormap the detection-odds wedges are painted on.

    Serves `make_strudl_fn_analysis_plot_v2`. The midpoint is neutral grey, so an odds ratio of one
    reads as the absence of a difference rather than as a value of its own.

    Parameters
    ----------
    color_dict : dict
        Palette built in __main__.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """

    return LinearSegmentedColormap.from_list(
        'detection_odds',
        [color_dict['geosphere_midnight_green_70'], '#EEEEEE', color_dict['geosphere_lime']])


def binned_detection_odds(populations, key, edges, min_expected_detections=5.0):
    """
    Odds that STRUDL detected a CME in each bin of `key`, against the rest of the sample.

    Each bin is scored on the table [[tp_in, fn_in], [tp_out, fn_out]], so an odds ratio above one
    means the detected fraction inside the bin exceeds the fraction everywhere else.

    Parameters
    ----------
    populations : dict
        'TP' and 'FN' frames from `_fn_analysis_populations`.
    key : str
        Column to bin.
    edges : array-like
        Bin edges, from `_detection_bin_edges`.
    min_expected_detections : float, default=5.0
        A bin expected to hold fewer detections than this is flagged `sparse` and left uncoloured
        and untested. Such a ratio is limited by its smallest cell, which is the detection count:
        a bin expecting one detection moves by a factor of two on a single event, however many
        CMEs went into it. The cut is on the expected count under the overall detection rate, not
        the observed one, so the rule cannot see the result it decides on; gating on observed
        detections would remove the very bins the figure exists to show. Five is Cochran's rule of
        thumb, binding for the asymptotic `detection_homogeneity_test` and borrowed here so that
        the coloured and the tested bins are the same set.

    Returns
    -------
    list of dict
        One record per bin: `low`, `high`, `nTp`, `nFn`, `expectedTp`, the conditional `oddsRatio`
        with `ciLow`/`ciHigh`, `p`, `sparse`, and `oddsRatioPlot`. The last is what the wedges are
        painted with: the conditional estimate collapses to zero in a bin with no detection, and
        zero has no logarithm and so no colour, so this one adds the Haldane-Anscombe half to every
        cell and stays finite. NaN in an empty bin, which has no odds either way.

    Raises
    ------
    ValueError
        If any value falls outside `edges`.
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

        # NaN rather than the correction alone, which in an empty bin would give the ratio of the
        # two population sizes and say nothing about the bin
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
    Test whether the detected fraction depends on the binned quantity at all.

    One chi-squared test over every non-sparse bin at once, so the quantity gets a single statement
    rather than one borderline p-value per wedge. Sparse bins are excluded because this test, unlike
    the exact per-bin statistics, is asymptotic and misstates its p-value on thin bins.

    Parameters
    ----------
    bins : list of dict
        Records from `binned_detection_odds`.

    Returns
    -------
    tuple
        (chi2, p, dof), all NaN and dof 0 where fewer than two bins survive the cut.
    """

    table = np.array([[b['nTp'], b['nFn']] for b in bins if not b['sparse']], dtype=float)

    # an empty bin is an all-zero row, whose expected frequencies the test cannot divide by. No
    # positive threshold admits one, but the test must not depend on the threshold to stay valid.
    if table.size:
        table = table[table.sum(axis=1) > 0]

    if table.shape[0] < 2 or not table.sum(axis=0).all():
        return float('nan'), float('nan'), 0

    chi2, p_value, dof, _ = chi2_contingency(table)

    return float(chi2), float(p_value), int(dof)


def _check_bin_width(bin_width_deg):
    """
    Reject a bin width that does not divide 360 and 180 deg exactly.

    Serves `make_strudl_fn_analysis_plot_v2` and `detection_bin_width_sensitivity`, so no width can
    be admitted to one and refused by the other. A bin straddling +-180 deg of separation or a pole
    is not a bin of anything: the quantity is continuous across that edge, the histogram is not, so
    the events either side would be counted apart.

    Parameters
    ----------
    bin_width_deg : float
        Candidate width.

    Raises
    ------
    ValueError
        If the width divides either span inexactly.
    """

    for span in (360.0, 180.0):
        if abs(round(span / bin_width_deg) - span / bin_width_deg) > 1e-9:
            raise ValueError(f'bin_width_deg={bin_width_deg} does not divide {span:.0f} deg '
                             'exactly, so a bin would straddle the end of the range')


def _detection_bin_edges(bin_width_deg):
    """Signed-separation and latitude bin edges of one width, as (phiSigned, latitude). Shared by
    `make_strudl_fn_analysis_plot_v2` and `detection_bin_width_sensitivity`, so the scan bins
    exactly as the figure does."""

    return (np.linspace(-180.0, 180.0, int(round(360.0 / bin_width_deg)) + 1),
            np.linspace(-90.0, 90.0, int(round(180.0 / bin_width_deg)) + 1))


def detection_bin_width_sensitivity(populations, widths=(15.0, 20.0, 30.0, 45.0, 60.0),
                                    min_expected_detections=5.0, verbose=True):
    """
    Repeat `detection_homogeneity_test` across bin widths, so the choice of width is visible.

    The width is a decision the test is not indifferent to: a narrow binning resolves more
    structure but pushes more bins under the expected-count threshold, where they are dropped and
    left uncoloured, while a wide one tests nearly everything at coarser resolution. Both the
    p-value and the CMEs behind it move accordingly, so the claim to quote is whether the structure
    holds across the row, not what it reaches at any one width.

    Parameters
    ----------
    populations : dict
        'TP' and 'FN' frames from `_fn_analysis_populations`.
    widths : tuple of float, default=(15.0, 20.0, 30.0, 45.0, 60.0)
        Bin widths to scan; each must pass `_check_bin_width`.
    min_expected_detections : float, default=5.0
        Passed to `binned_detection_odds`.
    verbose : bool, default=True
        Print the scan as a table.

    Returns
    -------
    dict
        One list per quantity, 'phiSigned' and 'latitude', with a record per width: `bin_width_deg`,
        `chi2`, `dof`, `p`, `n_bins`, `n_sparse`, `n_discarded` and `discarded_deg`. `n_discarded`
        is the CMEs the threshold removed from the test, the cost of a narrow binning and the
        quantity to argue a width from, since it can be read without looking at a p-value.
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
    """Draw the Sun at the centre of a polar panel. Serves both polar panel plotters."""

    theta = np.linspace(-np.pi, np.pi, 181)
    # a polygon closed over a full turn seams across the disc, so the fill carries no edge and the
    # rim is a constant-radius arc of its own
    ax.fill_between(theta, 0.0, _R_SUN, color=color_dict['geosphere_yellow'], lw=0.0, zorder=3)
    ax.plot(theta, np.full_like(theta, _R_SUN), color=color_dict['geosphere_orange'], lw=1.2,
            zorder=4)


def _draw_odds_wedges(ax, bins, cmap, norm):
    """
    Draw one wedge per bin as a ring around the Sun, coloured by its odds of detection.

    Serves both polar panel plotters. Sparse bins are drawn grey and hatched rather than coloured.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Polar axes to draw into.
    bins : list of dict
        Records from `binned_detection_odds`.
    cmap : matplotlib.colors.Colormap
        From `detection_odds_colormap`.
    norm : matplotlib.colors.Normalize
        Maps log2 of the odds ratio onto `cmap`.
    """

    centres = np.deg2rad([0.5 * (b['low'] + b['high']) for b in bins])
    widths = np.deg2rad([b['high'] - b['low'] for b in bins])
    facecolors = ['0.85' if b['sparse'] else cmap(norm(np.log2(b['oddsRatioPlot'])))
                  for b in bins]

    wedges = ax.bar(centres, height=_R_RING_WIDTH, width=widths, bottom=_R_RING_INNER,
                    color=facecolors, edgecolor='white', lw=0.6, zorder=2)

    # a hatch takes the edge colour, so a sparse wedge needs a darker edge of its own
    for wedge, record in zip(wedges, bins):
        if record['sparse']:
            wedge.set_hatch('///')
            wedge.set_edgecolor('0.45')


# past this many bin boundaries on one panel, only every second one is labelled
_MAX_POLAR_TICK_LABELS = 12


def _bin_edge_ticks(bins, full_turn):
    """
    Angular ticks of a polar panel, taken from the bin boundaries the wedges are drawn on.

    Serves both polar panel plotters, so every label sits on a wedge boundary whatever width the
    bins were built at.

    Parameters
    ----------
    bins : list of dict
        Records from `binned_detection_odds`.
    full_turn : bool
        True drops the last boundary, which is the first one seen again from the other side.

    Returns
    -------
    tuple
        (ticks, labelled), the second a flag per tick. Past `_MAX_POLAR_TICK_LABELS` boundaries
        only every second one is labelled, counted out from the boundary nearest zero so that the
        panel's reference direction keeps its label.
    """

    ticks = np.array([b['low'] for b in bins] + ([] if full_turn else [bins[-1]['high']]))
    step = 2 if len(ticks) > _MAX_POLAR_TICK_LABELS else 1
    anchor = int(np.argmin(np.abs(ticks)))

    return ticks, [(i - anchor) % step == 0 for i in range(len(ticks))]


def _finish_polar_panel(ax, font_params):
    """Set the radial axis and clear the spines of a polar panel. Serves both polar panel
    plotters."""

    ax.set_ylim(0.0, _R_LIMIT)
    ax.set_yticks([])
    # cleared by colour rather than by visibility, which a polar axes recomputes as it draws
    for spine in ax.spines.values():
        spine.set_color('none')
    ax.grid(False)
    ax.tick_params(axis='x', labelsize=font_params['ticks_fontsize'] - 2, pad=2)


def plot_separation_polar_panel(ax, bins, cmap, norm, color_dict, font_params):
    """
    Draw where around STEREO-A the CMEs sit, seen from solar north.

    Serves `make_strudl_fn_analysis_plot_v2`. The polar angle is the signed separation
    phi = wrap180(sta_lon - cme_lon), zero along the Sun-STEREO-A line and growing clockwise on the
    page, which puts HEEQ longitude growing counter-clockwise: the sense of a view down from solar
    north.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Full-turn polar axes to draw into.
    bins : list of dict
        Records from `binned_detection_odds` over 'phiSigned'.
    cmap, norm : Colormap, Normalize
        Passed to `_draw_odds_wedges`.
    color_dict, font_params : dict
        Palette and font sizes.
    """

    ax.set_theta_zero_location('S')
    ax.set_theta_direction(-1)

    _draw_odds_wedges(ax, bins, cmap, norm)
    _draw_sun_disc(ax, color_dict)

    ax.plot([0.0], [_R_SPACECRAFT], marker='^', ms=14, color='black', clip_on=False, zorder=4)

    # wrapped onto [0, 360): a negative tick widens the axis past a full turn and the ring then
    # wraps over itself
    ticks, labelled = _bin_edge_ticks(bins, full_turn=True)
    labels = [('STEREO-A' if t == 0.0 else
               ('±180°' if t == -180.0 else f'{t:+.0f}°')) if show else ''
              for t, show in zip(ticks, labelled)]
    ax.set_thetagrids(ticks % 360.0, labels)

    _finish_polar_panel(ax, font_params)


def plot_latitude_polar_panel(ax, bins, cmap, norm, color_dict, font_params):
    """
    Draw the latitudes of the CMEs, seen edge-on.

    Serves `make_strudl_fn_analysis_plot_v2`. The polar angle is the DONKI latitude, north up.
    Only one limb is drawn: a band of constant latitude falls on both limbs of an edge-on view as
    mirror images, so the second carries nothing the first does not.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Polar axes to draw into; cut to a [-90, 90] deg sector here.
    bins : list of dict
        Records from `binned_detection_odds` over 'latitude'.
    cmap, norm : Colormap, Normalize
        Passed to `_draw_odds_wedges`.
    color_dict, font_params : dict
        Palette and font sizes.
    """

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_thetamin(-90.0)
    ax.set_thetamax(90.0)

    ax.plot([0.0, 0.0], [0.0, _R_LIMIT], color='0.4', ls='--', lw=1.2, zorder=1)

    _draw_odds_wedges(ax, bins, cmap, norm)
    _draw_sun_disc(ax, color_dict)

    # signed ticks here, unlike the separation panel: inside a sector the wrapped
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
    Draw the five-panel false-negative analysis and return its numbers.

    Panels (a) and (b) compare the detected and the missed CMEs on speed and on half-width, and
    panel (e) on whether the field was shared with an obscuring CME. Panels (c) and (d) draw the
    signed separation and the latitude as polar views, from solar north and edge-on. Each bin is
    painted with its odds of detection against the rest of the sample, on one colour scale across
    both panels; neither carries a statistic of its own, so both tests are returned instead.

    Saves strudl_fn_analysis_v2[_bare].pdf and .png.

    Parameters
    ----------
    donki_df : pandas.DataFrame
        Deduplicated analyses from `load_donki_data`. Ignored when `prepared` is given.
    strudl_cmes_path, overlaps_path, overlap_coefficient, overlap_minutes, elongation_separation
        Passed to `_fn_analysis_populations`. Ignored when `prepared` is given.
    color_dict, font_params : dict
        Palette and font sizes; the font sizes are raised by `_enlarged_fonts`.
    SAVE_PLOT_PAPER_PATH : str
        Directory to write into, trailing separator included.
    bin_width_deg : float, default=45.0
        Width of both sets of polar bins. Must divide 360 and 180 exactly, see `_check_bin_width`.
    min_expected_detections : float, default=5.0
        Passed to `binned_detection_odds`.
    log2_odds_cap : float, default=3.0
        Bounds the colour scale at a factor of 2**cap either way, so one extreme bin cannot
        flatten the rest.
    prepared : dict, optional
        Output of `_fn_analysis_populations`, to avoid repeating its Horizons query across the two
        runs of this figure.
    annotate : bool, default=True
        False drops the key, the panel letters, the median legends and the statistics boxes, and
        writes under the `_bare` name. The colourbar stays, being the scale the wedges are read
        against rather than a key to them. The numbers are identical, the bootstrap being seeded.

    Returns
    -------
    dict
        `_fn_analysis_summary` extended with `bin_width_deg`, `min_expected_detections`,
        `bin_width_sensitivity`, and a `panels` entry per panel.
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

    # one scale over both panels, from the coloured bins only, so a colour means the same odds
    # ratio in either view
    spread = [abs(np.log2(b['oddsRatioPlot'])) for b in separation_bins + latitude_bins
              if not b['sparse']]
    limit = min(float(np.ceil(max(spread))), float(log2_odds_cap)) if spread else 1.0
    cmap = detection_odds_colormap(color_dict)
    norm = plt.Normalize(vmin=-limit, vmax=limit)

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

    for key, panel_bins in (('phiSigned', separation_bins), ('latitude', latitude_bins)):
        chi2, p_value, dof = detection_homogeneity_test(panel_bins)
        summary['panels'][key] = {'bin_width_deg': float(bin_width_deg), 'chi2': chi2, 'dof': dof,
                                  'p': p_value, 'bins': panel_bins}

    # the drawn width joins the standard set, so the scan always contains this figure's own test
    summary['bin_width_sensitivity'] = detection_bin_width_sensitivity(
        populations, widths=tuple(sorted({15.0, 20.0, 30.0, 45.0, 60.0, float(bin_width_deg)})),
        min_expected_detections=min_expected_detections)

    scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_map.set_array([])
    cbar = fig.colorbar(scalar_map, cax=axes[3].inset_axes([1.06, 0.15, 0.035, 0.70]),
                        orientation='vertical')
    # the odds ratio is multiplicative: the scale is linear in log2 but labelled in odds ratios
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

    return summary


def _read_paired_tracks(paired_tracks_path):
    """
    Read the paired tracks that carry an automatic counterpart. Serves `_paired_track_profiles`.

    Narrower than STRUDL having detected the CME: `strudlTrackAvailable` is the flag
    `build_paired_track_records` sets for the tracks surviving the ecliptic-correction and
    Earth-position-angle cuts, 80 of the 166 hand-corrected tracks.

    Parameters
    ----------
    paired_tracks_path : str
        Path to manual_vs_strudl_tracks.json.

    Returns
    -------
    pandas.DataFrame
        The rows with `strudlTrackAvailable` set.
    """

    paired_df = pd.read_json(paired_tracks_path)

    return paired_df[paired_df['strudlTrackAvailable'].astype(bool)]


def _paired_track_overlap(row):
    """
    Pair up one track's two elongation profiles over the time range they share.

    Serves `_paired_track_profiles`. The manual profile is interpolated onto the STRUDL timestamps,
    which STRUDL being the thing judged and the denser sampled of the two. The shared window
    matters because most STRUDL tracks begin before the first manual point, and `np.interp` holds
    the manual elongation flat outside its range rather than refusing, so the whole STRUDL track
    would measure that flat extension instead of the disagreement between two tracks.

    Parameters
    ----------
    row : pandas.Series
        One paired track, carrying `time_manual`, `time_strudl`, `elongation_manual` and
        `elongation_strudl`.

    Returns
    -------
    tuple of numpy.ndarray
        (interpolated manual, STRUDL) elongations on the shared STRUDL timestamps.
    """

    time_manual = pd.to_datetime(pd.Series(row['time_manual'])).to_numpy()
    time_strudl = pd.to_datetime(pd.Series(row['time_strudl'])).to_numpy()
    elongation_manual = np.asarray(row['elongation_manual'], dtype=float)
    elongation_strudl = np.asarray(row['elongation_strudl'], dtype=float)

    shared = ((time_strudl >= max(time_manual.min(), time_strudl.min()))
              & (time_strudl <= min(time_manual.max(), time_strudl.max())))

    # date2num because np.interp cannot interpolate over a datetime64 abscissa
    manual_interp = np.interp(mdates.date2num(time_strudl[shared]),
                              mdates.date2num(time_manual), elongation_manual)

    return manual_interp, elongation_strudl[shared]


def _paired_track_profiles(paired_tracks_path):
    """
    Reduce the paired tracks to pooled point pairs and per-event elongation spans.

    Serves `make_strudl_tracking_performance_plot`. The point pairs are pooled because the
    agreement and residual panels never ask which track a point came from. The spans are taken over
    the whole track rather than the shared window, the retention panel asking how far each track
    got rather than where the two overlap in time.

    Parameters
    ----------
    paired_tracks_path : str
        Path to manual_vs_strudl_tracks.json.

    Returns
    -------
    tuple
        (manual, strudl, spans, n_events). The first two are pooled elongations; `spans` is one
        row per event, (manual min, manual max, strudl min, strudl max).
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



# where the retention curve is evaluated; it stops at 24 deg because HI-1A's field of view does
_RETENTION_GRID = np.arange(5.0, 24.01, 0.25)


def midnight_green_colormap(color_dict):
    """
    Build the sequential light-to-dark colormap for the tracking-profile hexbin.

    Parameters
    ----------
    color_dict : dict
        Palette built in __main__.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
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
    Fraction of the tracks reaching each elongation manually that STRUDL reaches too.

    Serves `_retention_with_band`. A track counts at an elongation when its span covers it, so a
    STRUDL track dropped early leaves the numerator while staying in the denominator.

    Parameters
    ----------
    spans : numpy.ndarray
        Per-event spans from `_paired_track_profiles`.
    grid : numpy.ndarray
        Elongations to evaluate at.

    Returns
    -------
    tuple of numpy.ndarray
        (retention, n_manual). The denominator falls away at both ends of the grid and is what
        decides where the curve is worth showing.
    """

    manual = (spans[:, None, 0] <= grid) & (spans[:, None, 1] >= grid)
    both = manual & (spans[:, None, 2] <= grid) & (spans[:, None, 3] >= grid)

    with np.errstate(invalid='ignore'):
        return both.sum(0) / manual.sum(0), manual.sum(0)


def _retention_with_band(spans, grid, n_bootstrap=2000, min_tracks=10, seed=0):
    """
    Add a bootstrap band to the retention curve. Serves `make_strudl_tracking_performance_plot`.

    Resampled over events rather than track points: the question is how the curve would move with
    a different sample of 80 CMEs, where resampling points would give the far narrower uncertainty
    of having measured these CMEs densely.

    Parameters
    ----------
    spans : numpy.ndarray
        Per-event spans from `_paired_track_profiles`.
    grid : numpy.ndarray
        Elongations to evaluate at.
    n_bootstrap : int, default=2000
        Event resamples behind the band.
    min_tracks : int, default=10
        Fewest manual tracks an elongation needs before it is shown.
    seed : int, default=0
        Fixed, so the band is the same on every run of the figure.

    Returns
    -------
    tuple of numpy.ndarray
        (retention, low, high, n_manual, valid). `low` and `high` are NaN outside `valid`.
    """

    retention, n_manual = _tracking_retention(spans, grid)
    valid = n_manual >= min_tracks

    rng = np.random.default_rng(seed)
    boot = np.array([_tracking_retention(spans[rng.integers(0, len(spans), len(spans))], grid)[0]
                     for _ in range(n_bootstrap)])

    # only the shown part, which also keeps nanpercentile off the all-NaN columns beyond the grid
    low, high = np.full(grid.shape, np.nan), np.full(grid.shape, np.nan)
    low[valid], high[valid] = np.nanpercentile(boot[:, valid], [2.5, 97.5], axis=0)

    return retention, low, high, n_manual, valid


def make_strudl_tracking_performance_plot(paired_tracks_path, color_dict, font_params,
                                          SAVE_PLOT_PAPER_PATH):
    """
    Draw how closely STRUDL's time-elongation profiles follow the hand-corrected ones.

    Three panels on the same paired tracks: (a) whether the two agree where both exist, as the
    density of matched points about the 1:1 line; (b) how far along the manual track STRUDL stays;
    (c) whether the disagreement in (a) is a bias and whether it grows, as the residual against
    elongation.

    Conditioned on an automatic track existing, the cut `_read_paired_tracks` applies, so (b) is
    how much of a track STRUDL keeps once it has one rather than how often it finds one at all;
    the false-negative figure answers the second question. Saves strudl_tracking_profiles.pdf
    and .png.

    Parameters
    ----------
    paired_tracks_path : str
        Path to manual_vs_strudl_tracks.json.
    color_dict, font_params : dict
        Palette and font sizes; the font sizes are raised by `_enlarged_fonts`.
    SAVE_PLOT_PAPER_PATH : str
        Directory to write into, trailing separator included.

    Returns
    -------
    dict
        `n_events`, `n_matched_points`, `retention` read off at 10, 15 and 20 deg with its
        interval and track count, `median_residual`, and `residual_quantiles` per elongation bin.
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
    # an inset rather than space taken out of the axes, which would shrink (a) below the other two
    colorbar = fig.colorbar(hexes, cax=ax.inset_axes([1.04, 0.0, 0.045, 1.0]))
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

    # the tracks behind the curve: a retention of 40 per cent means something different over 60
    # tracks than over 12
    for elongation in np.arange(5, 25, 5):
        ax.text(elongation, -0.22, f'{n_manual[np.argmin(np.abs(grid - elongation))]}',
                transform=ax.get_xaxis_transform(), ha='center',
                fontsize=font_params['ticks_fontsize'] - 4, color='0.35')

    # (c) the residual against elongation
    ax = axes[2]
    residual = strudl - manual
    edges = np.arange(4, 25, 2)
    centres = 0.5 * (edges[1:] + edges[:-1])
    quantiles = []
    for low_edge, high_edge in zip(edges[:-1], edges[1:]):
        in_bin = (manual >= low_edge) & (manual < high_edge)
        # under 20 points a 5th and a 95th percentile are just the bin's two extreme points
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
        ax.set_box_aspect(1)

    fig.tight_layout()
    # tight_layout does not see the inset colourbar, so the gaps are set by hand
    fig.subplots_adjust(bottom=0.2, wspace=0.48)

    plt.savefig(SAVE_PLOT_PAPER_PATH + 'strudl_tracking_profiles.pdf', bbox_inches='tight')
    plt.savefig(SAVE_PLOT_PAPER_PATH + 'strudl_tracking_profiles.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)

    return summary


# ---------------------------------------------------------------------------------------------
# Contingency-table verification, after Verbeke et al. (2019), doi:10.1029/2018SW002046
# ---------------------------------------------------------------------------------------------

# everything but these counts as a predicted non-arrival, except 'no_fit', which is no forecast
PREDICTED_ARRIVAL = 'arrival'
NOT_A_FORECAST = 'no_fit'


def _ratio(numerator, denominator):
    """Safe division for `contingency_scores`; NaN on a zero denominator."""

    return float(numerator) / float(denominator) if denominator else np.nan


def contingency_scores(hits, misses, false_alarms, correct_rejections):
    """
    The eight skill scores of Verbeke et al. (2019), Table 9, from one contingency table.

    Parameters
    ----------
    hits, misses, false_alarms, correct_rejections : int
        The four cells.

    Returns
    -------
    dict
        The cells as 'H', 'M', 'FA', 'CR', their `total`, and POD, SR, FAR, bias, CSI, accuracy,
        POFD and HK. The first five carry no correct-rejection term and so do not move when the
        population boundary does; accuracy, POFD and HK do, and with CR an order of magnitude
        above FA the last two collapse onto POD, so they only mean anything against the stated
        population. NaN where a denominator is zero.
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


# which VENN_SETS a CME must be in for each model to predict an arrival. The masks come from
# get_venn_set_masks, so the scores cannot drift from the Venn figure.
GEOMETRY_MODELS = {
    'Ellipse': ('A',),
    'Ellipse + STRUDL': ('A', 'S'),
    'STRUDL': ('S',),
}

GEOMETRY_POPULATION = 'V'
GEOMETRY_OBSERVED = 'I'

# scored to show where the combined model's precision comes from, not as a pipeline to run
GEOMETRY_REFERENCE = 'STRUDL'


def geometry_contingency(population, predicted, observed):
    """
    Score one binary predictor. Serves `geometry_skill_summary`.

    No hit window: the prediction is a yes or no on whether the CME can reach Earth at all, so an
    event falls in exactly one cell and the four cells sum to the population.

    Parameters
    ----------
    population, predicted, observed : pandas.Series
        Boolean masks on a common index.

    Returns
    -------
    dict
        As `contingency_scores`.
    """

    return contingency_scores(hits=int((population & predicted & observed).sum()),
                              misses=int((population & ~predicted & observed).sum()),
                              false_alarms=int((population & predicted & ~observed).sum()),
                              correct_rejections=int((population & ~predicted & ~observed).sum()))


def geometry_skill_summary(donki_df):
    """
    Score each GEOMETRY_MODELS predictor of an Earth arrival.

    The population is every CME that could have appeared in the HI-1A field of view, the
    observation a DONKI in-situ link at Earth. Requiring a STRUDL detection can only shrink the
    set of predicted arrivals, so it trades recall for precision and the only question is which
    way the trade goes. The masks come from `get_venn_set_masks`, so each model's cells are Venn
    regions by construction.

    Parameters
    ----------
    donki_df : pandas.DataFrame
        Deduplicated analyses from `load_donki_data`.

    Returns
    -------
    dict
        `models`, one `contingency_scores` dict per model; `reference`, the model named by
        GEOMETRY_REFERENCE; and `population` with `n_visible`, `n_arrived` and the set totals.
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
    Reduce a forecast results table to one verification event per forecast.

    Serves `forecast_error_table`. Rows whose DBM fit was rejected are excluded rather than scored
    as predicted non-arrivals, which would credit the model for having said nothing; they are
    counted separately as forecast yield.

    Parameters
    ----------
    results_df : pandas.DataFrame
        From `load_forecast_results`.

    Returns
    -------
    tuple
        (events, not_forecast). Each event carries both CME ids, the `featureCode`, `predicted`
        and `observed` flags, and `dt_h`, the signed arrival time error in hours, NaN where either
        time is missing.
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
    Verbeke's hit-only error metrics with confidence intervals. Serves `forecast_error_summary`.

    Parameters
    ----------
    errors : array-like
        Signed arrival time errors in hours; non-finite values are dropped.
    n_bootstrap : int, default=10000
        Resamples behind the MAE and RMSE intervals.
    seed : int, default=42
        Fixed, so the intervals are the same on every run.

    Returns
    -------
    dict
        `n`, `ME`, `MAE`, `RMSE`, `sd` and an interval for each of the first three. ME takes a
        Student-t interval, being a mean; MAE and RMSE are not means of the quantity whose spread
        is estimated, so they take a percentile bootstrap. All NaN on an empty sample.
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


# colour, marker and outline per pipeline, read only by the arrival-time error figure. The
# 'ELEvo' entries are unused: the only call site supplies the two ELEvoHI arms alone.
VERIFICATION_COLOR_KEYS = {'ELEvo': 'verification_elevo',
                           'STRUDL-ELEvoHI': 'geosphere_lime',
                           'Manual-ELEvoHI': 'geosphere_midnight_green'}

VERIFICATION_MARKERS = {'ELEvo': '^', 'STRUDL-ELEvoHI': '^', 'Manual-ELEvoHI': 'o'}

# a pale fill needs a stroke to hold its edge against the page; a dark one does not and is omitted
VERIFICATION_STROKE_KEYS = {'STRUDL-ELEvoHI': 'geosphere_lime_shade'}


# colour, marker and whether to annotate, per model of the performance diagram
GEOMETRY_STYLES = {
    'Ellipse': ('geosphere_midnight_green', 'o', True),
    'Ellipse + STRUDL': ('geosphere_lime_midnight_mix', 's', True),
    'STRUDL': ('geosphere_lime', '^', True),
}

# marker rim per model; None means the white default
GEOMETRY_EDGE_KEYS = {
    'Ellipse': None,
    'Ellipse + STRUDL': None,
    'STRUDL': 'geosphere_lime_shade',
}

# fallback for a model with no style of its own; every GEOMETRY_MODELS entry has one
GEOMETRY_REFERENCE_COLOR = '0.45'

# annotation offset in points per model, set for the positions the current data put them at
GEOMETRY_LABEL_OFFSETS = {
    'Ellipse': (14, 8),
    'Ellipse + STRUDL': (14, 8),
    'STRUDL': (14, -18),
}


def make_performance_diagram(summary, color_dict, font_params, SAVE_PLOT_PAPER_PATH,
                             stem='performance_diagram'):
    """
    Draw POD against the success ratio, with CSI and frequency-bias contours.

    POD, SR, FAR, bias and CSI can all be read off this one diagram, and none of them carries a
    correct-rejection term, which makes it the comparison that does not move when the population
    boundary does -- the right figure for a population whose correct rejections outnumber
    everything else.

    Each model is a single point, the prediction being untimed: either the geometry admits an
    arrival or it does not. Requiring a STRUDL detection can only remove predicted arrivals, so
    the step between markers is a pure recall-for-precision trade. Saves
    performance_diagram.pdf and .png.

    Parameters
    ----------
    summary : dict
        From `geometry_skill_summary`. The model named by its `reference` key is labelled as such
        rather than as a pipeline to run.
    color_dict, font_params : dict
        Palette and font sizes.
    SAVE_PLOT_PAPER_PATH : str
        Directory to write into, trailing separator included.
    stem : str, default='performance_diagram'
        Filename stem, without extension.
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
    ax.legend(fontsize=font_params['legend_fontsize'] - 2, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    fig.savefig(f'{SAVE_PLOT_PAPER_PATH}{stem}.pdf', bbox_inches='tight')
    fig.savefig(f'{SAVE_PLOT_PAPER_PATH}{stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


# fewest errors a half violin is drawn from; a smaller group draws its members instead, a kernel
# density estimate over four or five points being a shape rather than a distribution
MIN_VIOLIN_N = 10

# what the scatter panels plot the error against, as (column, axis label, plain form)
FORECAST_ERROR_AXES = (('speed', 'Initial CME speed [km s$^{-1}$]',
                        'fitted initial CME speed [km/s]'),
                       ('separation', 'CME-STEREO-A separation $\\phi$ [deg]',
                        'CME-STEREO-A separation, unsigned [deg]'))

FORECAST_ERROR_GROUPS = (('Overlapping', True), ('Not overlapping', False))


def forecast_error_table(results_df, overlapping_ids):
    """
    Build one row per scoreable forecast, with its signed error and the quantities it is plotted
    against.

    The error comes from `forecast_events` rather than being recomputed, so the figure and the
    contingency scores cannot end up with different conventions; it is forecast minus observation
    in hours. Only events with both a predicted and an observed arrival have an error at all,
    which is what cuts the automatic arm to nine.

    Parameters
    ----------
    results_df : pandas.DataFrame
        From `load_forecast_results`.
    overlapping_ids : set
        CME ids from `_overlapping_cme_ids`.

    Returns
    -------
    pandas.DataFrame
        `error`, `speed`, `separation`, `overlapping`, and `error_low`/`error_high` for the
        ensemble range. `separation` is the apex-to-STEREO-A angle `phi`, already unsigned, not
        `delta_target`, which measures the apex against L1 rather than against the imager.
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

    # the observation is one fixed instant per CME, so the ensemble bounds carry over unshifted
    forecast = pd.to_datetime(table['arrival_time [UT]'])
    table['error_low'] = (forecast - pd.to_datetime(table['arrival_time_min [UT]'])
                          ).dt.total_seconds() / 3600.0
    table['error_high'] = (pd.to_datetime(table['arrival_time_max [UT]']) - forecast
                           ).dt.total_seconds() / 3600.0

    return table


def _forecast_error_correlations(table):
    """Spearman's rho of one arm's error against each FORECAST_ERROR_AXES quantity, NaN under three
    events. Serves `forecast_error_summary` and `_forecast_error_scatter_panel`."""

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
    Split one arm's error on the overlap flag and test the difference.

    Serves `forecast_error_summary`. The flag comes from `_overlapping_cme_ids`, the same criterion
    the false-negative figure judges detection on, so a CME obscured there is obscured here.

    Parameters
    ----------
    table : pandas.DataFrame
        One arm, from `forecast_error_table`.

    Returns
    -------
    dict
        Per FORECAST_ERROR_GROUPS name, the `n`, `median`, `ME` and `MAE`, plus the Mann-Whitney
        `p` or NaN where either side is empty.
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
    Compare the two arms on the events both of them forecast. Serves `forecast_error_summary`.

    The only unconfounded comparison: the arms do not cover the same CMEs, the automatic one
    reaching nine of the manual one's forty-three, so their headline errors differ partly because
    the event mixes do. On the shared events that cancels and what is left is the cost of the
    tracking. Paired by CME, hence a signed-rank rather than a rank-sum test.

    Parameters
    ----------
    tables : dict
        One `forecast_error_table` per arm.
    labels : tuple of str
        (manual, automatic). The difference is reported automatic minus manual.

    Returns
    -------
    dict
        {'n': 0} with no shared events; otherwise `n`, `labels`, the per-arm MAE, the mean and
        median difference, and the Wilcoxon `p`.
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

    # a signed-rank test needs a spread to rank; one pair gives it nothing
    paired['p'] = float(wilcoxon(difference).pvalue) if len(difference) > 1 else np.nan

    return paired


def forecast_error_summary(tables, labels):
    """
    Collect the arrival-time error numbers for both arms. Serves `make_forecast_error_figure`.

    Parameters
    ----------
    tables : dict
        One `forecast_error_table` per arm.
    labels : tuple of str
        (manual, automatic).

    Returns
    -------
    dict
        `labels`; `arms`, per arm the event count, `hit_error_statistics`,
        `_forecast_error_correlations` and `_forecast_error_overlap`; and `paired`, the
        `_forecast_error_paired` comparison the figure itself cannot show.
    """

    summary = {'arms': {}, 'labels': list(labels)}

    for label, table in tables.items():
        summary['arms'][label] = {
            'n': int(len(table)),
            'errors': hit_error_statistics(table['error'].to_numpy(float)),
            'correlations': _forecast_error_correlations(table),
            'overlap': _forecast_error_overlap(table)}

    summary['paired'] = _forecast_error_paired(tables, labels)

    return summary


def _forecast_error_scatter_panel(ax, tables, colors, strokes, key, xlabel, font_params,
                                  annotate=True):
    """
    Draw both arms' errors against one quantity the forecast might depend on, with its correlation.

    Serves `make_forecast_error_plot`. Each point carries the range its DBM ensemble spanned.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    tables : dict
        One `forecast_error_table` per arm.
    colors, strokes : dict
        Per arm, a marker colour and an outline colour or None.
    key : str
        Column to plot the error against.
    xlabel : str
        Axis label, mathmode allowed.
    font_params : dict
        Font sizes.
    annotate : bool, default=True
        False drops the correlation boxes, leaving the arms identified by colour and marker alone.
    """

    for row, (label, table) in enumerate(tables.items()):
        ax.errorbar(table[key], table['error'],
                    yerr=[table['error_low'], table['error_high']],
                    fmt='none', ecolor=strokes[label] or colors[label], elinewidth=0.9, alpha=0.45,
                    capsize=2, capthick=0.9, zorder=1)

        ax.scatter(table[key], table['error'], s=42, alpha=0.8, color=colors[label],
                   marker=VERIFICATION_MARKERS.get(label, 'o'),
                   edgecolor=strokes[label] or 'white',
                   linewidth=0.6, label=f'{label} (N={len(table)})', zorder=2)

        rho, p_value = spearmanr(table[key], table['error'])
        # n travels with rho: the same coefficient means something different at n=9 than at n=43
        if annotate:
            ax.text(0.97, 0.97 - 0.07 * row,
                    f'{label}: $\\rho$ = {rho:+.2f}, $p$ = {p_value:.2g} ($n$ = {len(table)})',
                    transform=ax.transAxes, ha='right', va='top',
                    color=strokes[label] or colors[label],
                    fontsize=font_params['legend_fontsize'] - 4)

    ax.axhline(0, color='0.5', lw=0.8, zorder=0)
    ax.set_xlabel(xlabel, fontsize=font_params['label_fontsize'])
    ax.tick_params(labelsize=font_params['ticks_fontsize'])
    ax.spines[['top', 'right']].set_visible(False)


def _half_violin(ax, position, errors, side, color, width=0.4):
    """
    Draw one side of a split violin, filled and outlined. Serves `_forecast_error_overlap_panel`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    position : float
        Abscissa of the group's centre line.
    errors : numpy.ndarray
        The group's errors.
    side : {'left', 'right'}
        Which half of `position` to draw on.
    color : str
        Fill and outline colour.
    width : float, default=0.4
        Half-width at the density's peak, in data units.
    """

    grid = np.linspace(errors.min(), errors.max(), 300)
    density = gaussian_kde(errors)(grid)
    density = density / density.max() * width

    edge = position - density if side == 'left' else position + density

    ax.fill_betweenx(grid, np.minimum(edge, position), np.maximum(edge, position),
                     color=color, alpha=0.25, lw=0, zorder=1)
    ax.plot(edge, grid, color=color, lw=1.2, alpha=0.9, zorder=1)


def _half_points(ax, position, errors, low, high, side, color, rng, stroke=None, width=0.4):
    """
    Draw one group's members, each with the range its DBM ensemble spanned.

    Serves `_forecast_error_overlap_panel`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    position : float
        Abscissa of the group's centre line.
    errors, low, high : numpy.ndarray
        The group's errors and their distance to either end of the ensemble range.
    side : {'left', 'right'}
        Which half of `position` to draw on.
    color : str
        Marker colour.
    rng : numpy.random.Generator
        Jitters the markers off the centre line.
    stroke : str, optional
        Marker and bar outline colour; white when None.
    width : float, default=0.4
        Greatest jitter, in data units.
    """

    offset = rng.uniform(0.08 * width, 0.75 * width, len(errors))
    x = position + (-offset if side == 'left' else offset)

    ax.errorbar(x, errors, yerr=[low, high], fmt='none', ecolor=stroke or color,
                elinewidth=0.9, alpha=0.45, capsize=2, capthick=0.9, zorder=2)
    ax.scatter(x, errors, s=42, color=color, alpha=0.9, edgecolor=stroke or 'white',
               linewidth=0.6, zorder=3)


def _half_strip(ax, position, errors, side, color, stroke=None, width=0.4):
    """
    Draw a group's median as a bar, in place of a density it is too small to support.

    Serves `_forecast_error_overlap_panel`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    position : float
        Abscissa of the group's centre line.
    errors : numpy.ndarray
        The group's errors.
    side : {'left', 'right'}
        Which half of `position` to draw on.
    color : str
        Bar colour.
    stroke : str, optional
        Outline colour; none when None.
    width : float, default=0.4
        Half-width of the bar, in data units.
    """

    ax.hlines(np.median(errors),
              position - 0.75 * width if side == 'left' else position,
              position if side == 'left' else position + 0.75 * width,
              color=color, lw=2.0, zorder=4, path_effects=_outline(stroke, 2.0))


def _forecast_error_overlap_panel(ax, tables, colors, strokes, labels, font_params, seed=4,
                                  annotate=True):
    """
    Draw the error split by whether the CME shared the field with an obscuring one.

    Serves `make_forecast_error_plot`. Every forecast is drawn as its own point. A group of at
    least MIN_VIOLIN_N gets a density behind its points, a smaller one a median bar, so the two
    arms may carry different backdrops in the same panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    tables : dict
        One `forecast_error_table` per arm.
    colors, strokes : dict
        Per arm, a fill colour and an outline colour or None.
    labels : tuple of str
        (manual, automatic); the first is drawn on the left of each position.
    font_params : dict
        Font sizes.
    seed : int, default=4
        Seeds the point jitter, so the panel is the same on every run.
    annotate : bool, default=True
        False drops the group counts.
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
    Draw each arm's error distribution as an empirical CDF, with its mean.

    Serves `make_forecast_error_plot`. Unsmoothed: a kernel density of nine points says more about
    the bandwidth than about the forecast, while an ECDF is exact at any sample size and puts both
    arms on one axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    tables : dict
        One `forecast_error_table` per arm.
    colors, strokes : dict
        Per arm, a line colour and an outline colour or None.
    font_params : dict
        Font sizes.
    annotate : bool, default=True
        False drops the legend, which is what carries the two means.
    """

    for label, table in tables.items():
        errors = np.sort(table['error'].to_numpy(float))
        # closed at both ends, so the curve starts at zero rather than partway up
        ax.step(np.concatenate(([errors[0]], errors, [errors[-1]])),
                np.concatenate(([0.0], np.arange(1, len(errors) + 1) / len(errors), [1.0])),
                where='post', color=colors[label], lw=2.2,
                path_effects=_outline(strokes[label], 2.2),
                label=f'{label} (N={len(errors)}, ME = {errors.mean():+.1f} h)')

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
    Draw the four-panel arrival time error figure for the manual and the automatic arm.

    Where the performance diagram says how often each arm was right, this says by how much it was
    wrong and against what. The automatic arm reaches nine scoreable forecasts, so every panel is
    built to keep nine points legible as nine: the scatter panels carry the n beside each
    correlation, the overlap panel falls back to drawing its members, and the distribution panel is
    an ECDF rather than a density. Saves `stem`.pdf and .png.

    Parameters
    ----------
    tables : dict
        One `forecast_error_table` per arm.
    labels : tuple of str
        (manual, automatic).
    color_dict, font_params : dict
        Palette and font sizes.
    SAVE_PLOT_PAPER_PATH : str
        Directory to write into, trailing separator included.
    stem : str, default='forecast_error'
        Filename stem, without extension.
    annotate : bool, default=True
        Passed to each panel; False also drops the panel letters and the extra headroom the
        correlation boxes need.
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
                                      annotate=annotate)

    _forecast_error_overlap_panel(ax_overlap, tables, colors, strokes, labels, font_params,
                                  annotate=annotate)
    _forecast_error_ecdf_panel(ax_ecdf, tables, colors, strokes, font_params, annotate=annotate)

    error_label = 'Arrival time error [h]\n(predicted $-$ in situ)'
    for ax in (ax_speed, ax_overlap):
        ax.set_ylabel(error_label, fontsize=font_params['label_fontsize'])

    # the error is nothing like symmetric about zero, so the limits come from the data. The
    # annotated figure needs the extra top margin for the correlation boxes.
    errors = np.concatenate([table['error'].to_numpy(float) for table in tables.values()])
    low = np.concatenate([(table['error'] - table['error_low']).to_numpy(float)
                          for table in tables.values()])
    high = np.concatenate([(table['error'] + table['error_high']).to_numpy(float)
                           for table in tables.values()])

    span = errors.max() - errors.min()
    ax_speed.set_ylim(low.min() - 0.12 * span, high.max() + (0.30 if annotate else 0.12) * span)

    # panel (d) carries no bars, so it keeps the bare extent of the errors
    ax_ecdf.set_xlim(errors.min() - 0.12 * span, errors.max() + 0.12 * span)

    # padded on its own drawn extent: the violins already run past the largest error
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
    Assemble both forecast arms, draw the error figure and return its numbers.

    Parameters
    ----------
    manual_df, strudl_df : pandas.DataFrame
        The manually corrected and the automatic arm, from `load_forecast_results`.
    overlaps_path : str
        Passed to `_overlapping_cme_ids`.
    pipeline_label_dict : dict
        `elevohi_corrected_label` and `elevohi_uncorrected_label`, the arms' names on the figure.
    color_dict, font_params : dict
        Palette and font sizes.
    SAVE_PLOT_PAPER_PATH : str
        Directory to write into, trailing separator included.
    stem : str, default='forecast_error'
        Filename stem; the un-annotated run appends `_bare`.
    annotate : bool, default=True
        False drops the correlation boxes, the group counts, the legend and the panel letters, and
        writes under the `_bare` name. The returned summary is the same either way, the bootstraps
        being seeded.

    Returns
    -------
    dict
        As `forecast_error_summary`.
    """

    overlapping_ids = _overlapping_cme_ids(overlaps_path)['overlapping_ids']

    labels = (pipeline_label_dict['elevohi_corrected_label'],
              pipeline_label_dict['elevohi_uncorrected_label'])

    tables = {labels[0]: forecast_error_table(manual_df, overlapping_ids),
              labels[1]: forecast_error_table(strudl_df, overlapping_ids)}

    summary = forecast_error_summary(tables, labels)

    make_forecast_error_plot(tables, labels, color_dict, font_params, SAVE_PLOT_PAPER_PATH,
                             stem=stem if annotate else f'{stem}_bare', annotate=annotate)

    return summary


def load_forecast_results(path):
    """
    Read a forecast results table, renaming the columns the scoring code expects.

    Parameters
    ----------
    path : str
        Path to an ELEvoHI ensemble results JSON file.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If the file predates the `forecast_outcome` field.
    """

    df = pd.read_json(path)

    if 'L1_ist_obs' in df.columns:
        df = df.rename(columns={'L1_ist_obs': 'L1_ist_obs [UT]'})

    if 'forecast_outcome' not in df.columns:
        raise ValueError(f'{path} predates the forecast_outcome field; re-run the forecast')

    return df


if __name__ == '__main__':

    SAVE_PLOTS = True
    SAVE_PLOT_PAPER_PATH = 'paper_plots/'
    # the un-annotated copies, whose numbers the manuscript text carries instead
    SAVE_PLOT_PAPER_BARE_PATH = 'paper_plots/paper_plots_bare/'

    geosphere_purple = '#701F80'
    geosphere_green = '#75CC41'
    geosphere_yellow = '#F9F200'
    geosphere_orange = '#F29707'
    my_pink = '#FF69B4'

    # the corporate design document's two palettes, each base colour with its 70 / 40 / 15 %
    # tints over white
    geosphere_midnight_green = '#052E37'
    geosphere_midnight_green_70 = '#546D73'
    geosphere_midnight_green_40 = '#9DABAE'
    geosphere_midnight_green_15 = '#DAE0E1'

    geosphere_lime = '#BFCE40'
    geosphere_lime_70 = '#D2DD65'
    geosphere_lime_40 = '#E5ECA3'
    geosphere_lime_15 = '#F6F8DC'
    # 40 % of Lime over black, not in the design document, which gives tints and no shades
    geosphere_lime_shade = '#4C521A'

    # only used by the Venn diagram, see VENN_SET_COLOR_KEYS
    venn_blue = '#2A78D6'
    venn_red = '#B22222'

    # Lime and Midnight Green in equal parts, for the performance diagram's combined model
    geosphere_lime_midnight_mix = '#627E3C'

    # the retired verification palette, kept for the unused 'ELEvo' slot of
    # VERIFICATION_COLOR_KEYS against a third pipeline being scored
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
    
    # the bare path is nested inside the other, so this creates both
    os.makedirs(SAVE_PLOT_PAPER_BARE_PATH, exist_ok=True)

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
                                    SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_BARE_PATH,
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

    # scored on the Venn masks, so the diagram and the Venn figure cannot disagree
    contingency = geometry_skill_summary(donki_df_dedupe)

    make_performance_diagram(contingency, color_dict=color_dict, font_params=font_params,
                             SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

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
        font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_BARE_PATH,
        annotate=False)


