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
from matplotlib.colors import to_rgba
from scipy.optimize import minimize
from sunpy.coordinates import frames, get_horizons_coord
import astropy.units as u
import seaborn as sns

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
    fig, ax = plt.subplots(figsize=(15, 6))

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

            fig = plt.figure(figsize=(20, 6), constrained_layout=False)

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

    # only used by the Venn diagram, see VENN_SET_COLOR_KEYS
    venn_blue = '#2A78D6'
    venn_red = '#B22222'

    color_dict = {
        "geosphere_purple": geosphere_purple,
        "geosphere_green": geosphere_green,
        "geosphere_yellow": geosphere_yellow,
        "geosphere_orange": geosphere_orange,
        "my_pink": my_pink,
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

    # make Figure 1
    plot_df = make_figure_donki_stats(donki_df_dedupe, donki_df_filtered, color_dict=color_dict, label_dict=label_dict, font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

    # make Figure 2
    venn_regions, venn_totals = make_donki_venn_plot(donki_df_dedupe, color_dict=color_dict, font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

    STRUDL_SEGMENTATION_PATH = 'STRUDL/segmentation_dictionary_mean_45_2024_05_01_2025_05_01.npy'
    strudl_seg = np.load(STRUDL_SEGMENTATION_PATH, allow_pickle=True).item()

    RUNDIF_PATH = 'HI_Data_Processing/stereo_processed/running_difference/data/A/'
    CHOSEN_RUNDIF_TIMES = [datetime(2024,8,20,15,58,11), datetime(2024,7,23,11,18,11)]

    # make Figure 4
    make_strudl_segmentation_plot(strudl_seg, RUNDIF_PATH, CHOSEN_RUNDIF_TIMES, color_dict=color_dict, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH, rdif_freq=4)

    ELEVO_RESULTS_PATH = 'ELEvo/data/elevo_results_20240501_20250501.json'
    STRUDL_CORRECTED_RESULTS_PATH = 'STRUDL_Tracking_Analysis/data/mean_45_2024_05_01_2025_05_01_strudl_tracks_corrected.json'
    STRUDL_UNCORRECTED_RESULTS_PATH = 'STRUDL_Tracking_Analysis/data/mean_45_2024_05_01_2025_05_01_strudl_tracks.json'
    ELEVOHI_CORRECTED_RESULTS_PATH = 'ELEvoHI/results/elevohi_ensembles_results_cleaned.json'
    ELEVOHI_UNCORRECTED_RESULTS_PATH = 'ELEvoHI/results/elevohi_ensembles_results_raw.json'

    # get results Table 1
    elevo_results_df, elevohi_results_uncorrected_df, elevohi_results_corrected_df, uncorrected_results_df, corrected_results_df =  get_results_table_one(ELEVO_RESULTS_PATH, STRUDL_CORRECTED_RESULTS_PATH, STRUDL_UNCORRECTED_RESULTS_PATH, ELEVOHI_UNCORRECTED_RESULTS_PATH, ELEVOHI_CORRECTED_RESULTS_PATH)

    STRUDL_ELEVOHI_LABEL = "STRUDL-ELEvoHI"
    MANUAL_ELEVOHI_LABEL = "Manual-ELEvoHI"
    ELEVO_LABEL = "ELEvo"

    pipeline_label_dict = {
        "elevohi_uncorrected_label": STRUDL_ELEVOHI_LABEL,
        "elevohi_corrected_label": MANUAL_ELEVOHI_LABEL,
        "elevo_label": ELEVO_LABEL
    }

    # Make Figure 7
    make_pipeline_comparison_plot(elevohi_results_uncorrected_df, elevohi_results_corrected_df, elevo_results_df, pipeline_label_dict, color_dict=color_dict, font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

    CHOSEN_DBM_EVENTS = ['2024-07-28T02:24:00-CME-001_SH', '2024-05-08T22:24:00-CME-001_SH', '2024-12-29T18:24:00-CME-001_LE', '2024-05-08T22:24:00-CME-001_LE']

    # Make Figure 8
    ELEVOHI_CORRECTED_PATH = 'ELEvoHI/results/elevohi_results_cleaned.json'
    make_dbm_comparison_plot(elevohi_path=ELEVOHI_CORRECTED_PATH, elevo_path=ELEVO_RESULTS_PATH, chosen_events=CHOSEN_DBM_EVENTS, color_dict=color_dict, font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)


    # Make Figure 6
    make_plot_elongation_difference(uncorrected_results_df=uncorrected_results_df, color_dict=color_dict, font_params=font_params, SAVE_PLOT_PAPER_PATH=SAVE_PLOT_PAPER_PATH)

    # Make results
    make_results_table(plot_df=plot_df, elevo_results_df=elevo_results_df, uncorrected_results_df=uncorrected_results_df, corrected_results_df=corrected_results_df, elevohi_results_uncorrected_df=elevohi_results_uncorrected_df, elevohi_results_corrected_df=elevohi_results_corrected_df, pipeline_label_dict=pipeline_label_dict, save_path=SAVE_PLOT_PAPER_PATH)


