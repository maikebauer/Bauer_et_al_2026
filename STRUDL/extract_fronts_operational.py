import numpy as np
import datetime
from astropy.wcs import FITSFixedWarning
import warnings
warnings.simplefilter('ignore', category=FITSFixedWarning)
from utils import parse_yml
import models
import torch 
import dataset
import os
from skimage import morphology
from skimage.measure import label
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from astropy.io import fits 
from astropy import wcs
import copy
from plantcv import plantcv as pcv
from collections import defaultdict




def aggregate_fronts(fronts, fronts_pixels, max_diff_elon=3, areas=None):
    """
    Groups CME fronts that likely belong to the same CME.

    Args:
        fronts (list of arrays): List of CME fronts. Each front is an Nx2 array of (elongation, PA) coordinates.
        fronts_pixels (list of arrays): List of pixel coordinates corresponding to the CME fronts.
        max_diff (float): Maximum allowed difference in mean elongation for association.
        areas (list of arrays): List of areas corresponding to the CME fronts (optional).

    Returns:
        list of lists: Grouped fronts, where each group is a list of original front indices.
    """
    n = len(fronts)

    if n <= 1:
        # return [[i] for i in range(n)]  # No aggregation needed
        if areas != None:
            return fronts,fronts_pixels,areas  # No aggregation needed
        else:
            return fronts,fronts_pixels
        
    mean_elongations = [np.mean(front[1]) for front in fronts]
    uf = UnionFind(n)

    for i in range(n):
        diffs = []
        for j in range(n):
            if i == j: # Don't take difference between front and itself
                continue

            diff_elon = abs(mean_elongations[i] - mean_elongations[j])
            if diff_elon >= max_diff_elon:
                continue  # No point checking PA overlap if elongation is too different
            
            diffs.append((j, diff_elon))

            # # Compute PA overlap
            # pa_i = set(fronts[i][0])
            # pa_j = set(fronts[j][0])
            # overlap = len(pa_i & pa_j)
            # pa_overlap_ratio = overlap / min(len(pa_i), len(pa_j))
            # if pa_overlap_ratio <= max_overlap_pa:
            #     diffs.append((j, diff_elon))

        # Find the one with the minimum difference
        if diffs:
            best_match = min(diffs, key=lambda x: x[1])[0]
            uf.union(i, best_match)

    # Group by root parent
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    agg_groups = list(groups.values())
    img_fronts_agg = [[]]*len(agg_groups)
    img_fronts_pixels_agg = [[]]*len(agg_groups)

    if areas != None:
        img_fronts_areas_agg = [[]]*len(agg_groups)

    for group_id,group in enumerate(agg_groups):
        elon_group = []
        pa_group = []
        x_pix_group = []
        y_pix_group = []

        if areas != None:
            areas_group = []

        for index in group:
            elon_group.append(fronts[index][1])
            pa_group.append(fronts[index][0])
            x_pix_group.append(fronts_pixels[index][0])
            y_pix_group.append(fronts_pixels[index][1])

            if areas != None:
                areas_group.append(areas[index])

        elon_group = np.concatenate(elon_group)
        pa_group = np.concatenate(pa_group)
        x_pix_group = np.concatenate(x_pix_group)
        y_pix_group = np.concatenate(y_pix_group)

        # Create a mask that keeps only the largest elongation per PA
        keep_indices = []
        unique_pas = np.unique(pa_group)

        for pa in unique_pas:
            indices = np.where(pa_group == pa)[0]
            if len(indices) == 1:
                keep_indices.append(indices[0])
            else:
                # Pick index of max elongation
                max_idx = indices[np.argmax(elon_group[indices])]
                keep_indices.append(max_idx)

        # Apply the mask to both WCS and pixel data
        keep_indices = np.array(keep_indices)

        pa_group = pa_group[keep_indices]
        elon_group = elon_group[keep_indices]
        x_pix_group = x_pix_group[keep_indices]
        y_pix_group = y_pix_group[keep_indices]

        if areas != None:
            areas_group = np.sum(areas_group,axis=0)

        elon_group = np.array([y for (y,x) in sorted(zip(elon_group,pa_group), key=lambda pair: pair[1])])
        x_pix_group = np.array([y for (y,x) in sorted(zip(x_pix_group,pa_group), key=lambda pair: pair[1])])
        y_pix_group = np.array([y for (y,x) in sorted(zip(y_pix_group,pa_group), key=lambda pair: pair[1])])
        pa_group = np.array(sorted(pa_group))
        
        img_fronts_agg[group_id] = [pa_group, elon_group]
        img_fronts_pixels_agg[group_id] = [x_pix_group, y_pix_group]
        
        if areas != None:
            img_fronts_areas_agg[group_id] = areas_group

    if areas != None:
        return img_fronts_agg, img_fronts_pixels_agg, img_fronts_areas_agg
    else:
        return img_fronts_agg, img_fronts_pixels_agg


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x


def identify_new_cme(associated_fronts, threshold_elon=8):
    """
    Identifies which CME fronts could be the start of a new CME based on mean elongation.

    Args:
        associated_fronts (list): List of fronts (each a [PA, elongation] 2-array).
        threshold_elon (float): Maximum mean elongation for a front to be flagged as a potential new CME.

    Returns:
        list: Binary flags (1 = could be new CME, 0 = not) for each front.
    """
    flags = []
    for front in associated_fronts:
        pa, elon = front
        mean_elon = np.mean(elon)
        if mean_elon < threshold_elon:
            flags.append(1)
        else:
            flags.append(0)
    return flags

def aggregate_fronts_sequential(fronts, fronts_pixels, max_diff_elon=3, areas=None):

    n = len(fronts)

    if n <= 1:
        if areas is not None:
            return fronts, fronts_pixels, areas
        else:
            return fronts, fronts_pixels

    max_elons = [np.max(f[1]) if len(f[1]) > 0 else -999 for f in fronts]

    sorted_indices = np.argsort(max_elons)[::-1]  # Descending order

    used = set()
    discarded = set()

    groups = []

    for i in sorted_indices:
        if i in used or i in discarded:
            continue
        
        if len(fronts[i][1]) == 0:
            # print(f"Front {i} has no elongation data, skipping.")
            discarded.add(i)
            continue
        # Start a new group with front i
        group = [i]
        used.add(i)

        # Aggregate data for matching
        combined_elon = fronts[i][1].copy()
        combined_pa = fronts[i][0].copy()

        changed = True
        while changed:
            changed = False
            for j in range(n):
                if j in used or j in discarded:
                    continue
                
                if len(fronts[j][1]) == 0:
                    # print(f"Front {j} has no elongation data, skipping.")
                    discarded.add(j)
                    continue
                
                # diff = np.abs(np.max(fronts[j][1]) - np.max(combined_elon))
                diff = np.abs(np.mean(fronts[j][1]) - np.mean(combined_elon))
                if diff <= max_diff_elon:
                    # Add to group
                    group.append(j)
                    used.add(j)
                    combined_elon = np.concatenate((combined_elon, fronts[j][1]))
                    combined_pa = np.concatenate((combined_pa, fronts[j][0]))
                    changed = True

        groups.append(group)

    # Now build the aggregated outputs
    img_fronts_agg = []
    img_fronts_pixels_agg = []
    img_fronts_areas_agg = [] if areas is not None else None

    for group in groups:
        elon_group = []
        pa_group = []
        x_pix_group = []
        y_pix_group = []
        areas_group = []

        for idx in group:
            elon_group.append(fronts[idx][1])
            pa_group.append(fronts[idx][0])
            x_pix_group.append(fronts_pixels[idx][0])
            y_pix_group.append(fronts_pixels[idx][1])
            if areas is not None:
                areas_group.append(areas[idx])

        elon_group = np.concatenate(elon_group)
        pa_group = np.concatenate(pa_group)
        x_pix_group = np.concatenate(x_pix_group)
        y_pix_group = np.concatenate(y_pix_group)

        # Keep only largest elongation per PA
        keep_indices = []
        unique_pas = np.unique(pa_group)
        for pa in unique_pas:
            indices = np.where(pa_group == pa)[0]
            if len(indices) == 1:
                keep_indices.append(indices[0])
            else:
                max_idx = indices[np.argmax(elon_group[indices])]
                keep_indices.append(max_idx)

        keep_indices = np.array(keep_indices)
        elon_group = elon_group[keep_indices]
        pa_group = pa_group[keep_indices]
        x_pix_group = x_pix_group[keep_indices]
        y_pix_group = y_pix_group[keep_indices]

        # Sort by PA
        sort_idx = np.argsort(pa_group)
        elon_group = elon_group[sort_idx]
        pa_group = pa_group[sort_idx]
        x_pix_group = x_pix_group[sort_idx]
        y_pix_group = y_pix_group[sort_idx]

        img_fronts_agg.append([pa_group, elon_group])
        img_fronts_pixels_agg.append([x_pix_group, y_pix_group])

        if areas is not None:
            total_area = np.sum(areas_group, axis=0)
            img_fronts_areas_agg.append(total_area)

    if areas is not None:
        return img_fronts_agg, img_fronts_pixels_agg, img_fronts_areas_agg
    else:
        return img_fronts_agg, img_fronts_pixels_agg




def remove_outliers_from_fronts(img_front_wcs, img_fronts_pixels, fits_headers, image_areas=None, window=3, threshold=1.5):
    cleaned_fronts = []
    cleaned_fronts_pixels = []

    if image_areas is not None:
        cleaned_fronts_areas = []

    for img_idx, img_fronts in enumerate(img_front_wcs):
        cleaned_img = []
        cleaned_pixels = []

        if image_areas is not None:
            cleaned_areas = []

        wcoord = wcs.WCS(fits_headers[img_idx])

        for front_idx, front in enumerate(img_fronts):

            pa_array, elon_array = front
            pixel_x_array = np.array(img_fronts_pixels[img_idx][front_idx])[0]
            pixel_y_array = np.array(img_fronts_pixels[img_idx][front_idx])[1]

            pa_array = np.round(pa_array).astype(int)
            elon_array = np.array(elon_array, dtype=np.float32)

            if len(pa_array) < 6:
                # Not enough data to apply outlier detection
                cleaned_img.append(np.array([pa_array, elon_array]))
                cleaned_pixels.append(np.array([pixel_x_array, pixel_y_array]))

                if image_areas is not None:
                    cleaned_areas.append(image_areas[img_idx][front_idx])
                    
                continue

            cleaned_elon = elon_array.copy()
            outlier_mask = np.zeros_like(elon_array, dtype=bool)

            cleaned_pix_arr = np.array([pixel_x_array, pixel_y_array])
            
            for i, pa in enumerate(pa_array):
                # Look for surrounding PAs within ±window
                surrounding_values_left = []
                surrounding_values_right = []

                for offset in range(1, window + 1):
                    neighbor_pa_left = pa - offset
                    neighbor_pa_right = pa + offset

                    if neighbor_pa_left in pa_array:
                        idx_left = np.where(pa_array == neighbor_pa_left)[0][0]
                        surrounding_values_left.append(elon_array[idx_left])      

                    if neighbor_pa_right in pa_array:
                        idx_right = np.where(pa_array == neighbor_pa_right)[0][0]
                        surrounding_values_right.append(elon_array[idx_right])

                if len(surrounding_values_left) >= 2 and len(surrounding_values_right) >= 2:
                    surrounding_mean_left = np.nanmean(surrounding_values_left)
                    surrounding_mean_right = np.nanmean(surrounding_values_right)
                    diff_left = np.abs(elon_array[i] - surrounding_mean_left)
                    diff_right = np.abs(elon_array[i] - surrounding_mean_right)

                    if diff_left > threshold and diff_right > threshold:
                        outlier_mask[i] = True

                elif len(surrounding_values_left) >= 2:
                    surrounding_mean_left = np.nanmean(surrounding_values_left)
                    diff_left = np.abs(elon_array[i] - surrounding_mean_left)

                    if diff_left > threshold:
                        outlier_mask[i] = True
                
                elif len(surrounding_values_right) >= 2:
                    surrounding_mean_right = np.nanmean(surrounding_values_right)
                    diff_right = np.abs(elon_array[i] - surrounding_mean_right)

                    if diff_right > threshold:
                        outlier_mask[i] = True

            # Interpolate over outliers
            if np.any(outlier_mask):
                good_idx = ~outlier_mask

                if np.any(good_idx):
                    cleaned_elon[outlier_mask] = np.interp(
                        pa_array[outlier_mask],
                        pa_array[good_idx],
                        elon_array[good_idx]
                    )
                    cleaned_elon_rad = np.deg2rad(cleaned_elon)
                    pa_array_rad = np.deg2rad(pa_array)

                    tx = np.rad2deg(np.arctan2(-np.sin(cleaned_elon_rad)*np.sin(pa_array_rad), np.cos(cleaned_elon_rad)))
                    ty = np.rad2deg(np.arcsin(np.sin(cleaned_elon_rad)*np.cos(pa_array_rad)))

                    pix_y, pix_x = wcoord.all_world2pix(tx,ty, 0)

                    if np.any(np.round(pix_x/8,0).astype(int) > 128):
                        for remove_ind in reversed(np.where(np.round(pix_x/8,0).astype(int) > 128)[0]):
                            pa_array = np.delete(pa_array, remove_ind)
                            elon_array = np.delete(elon_array, remove_ind)
                            cleaned_elon = np.delete(cleaned_elon, remove_ind)
                            pix_y = np.delete(pix_y, remove_ind)
                            pix_x = np.delete(pix_x, remove_ind)

                    if np.any(np.round(pix_x/8,0).astype(int) < 0):
                        for remove_ind in reversed(np.where(np.round(pix_x/8,0).astype(int) < 0)[0]):
                            pa_array = np.delete(pa_array, remove_ind)
                            elon_array = np.delete(elon_array, remove_ind)
                            cleaned_elon = np.delete(cleaned_elon, remove_ind)
                            pix_y = np.delete(pix_y, remove_ind)
                            pix_x = np.delete(pix_x, remove_ind)

                    if np.any(np.round(pix_y/8,0).astype(int) > 128):
                        for remove_ind in reversed(np.where(np.round(pix_y/8,0).astype(int) > 128)[0]):
                            pa_array = np.delete(pa_array, remove_ind)
                            elon_array = np.delete(elon_array, remove_ind)
                            cleaned_elon = np.delete(cleaned_elon, remove_ind)
                            pix_y = np.delete(pix_y, remove_ind)
                            pix_x = np.delete(pix_x, remove_ind)

                    if np.any(np.round(pix_y/8,0).astype(int) < 0):
                        for remove_ind in reversed(np.where(np.round(pix_y/8,0).astype(int) < 0)[0]):
                            pa_array = np.delete(pa_array, remove_ind)
                            elon_array = np.delete(elon_array, remove_ind)
                            cleaned_elon = np.delete(cleaned_elon, remove_ind)
                            pix_y = np.delete(pix_y, remove_ind)
                            pix_x = np.delete(pix_x, remove_ind)

                    cleaned_pix_arr = np.array([np.round(pix_x/8,0).astype(int), np.round(pix_y/8,0).astype(int)])

                else:
                    # All points in the front are considered outliers, discard front
                    continue

            cleaned_pixels.append(np.array(cleaned_pix_arr))
            cleaned_img.append(np.array([pa_array, cleaned_elon]))

            if image_areas is not None:
                cleaned_areas.append(image_areas[img_idx][front_idx])

        cleaned_fronts.append(cleaned_img)
        cleaned_fronts_pixels.append(cleaned_pixels)

        if image_areas is not None:
            cleaned_fronts_areas.append(cleaned_areas)

    if image_areas is not None:
        return cleaned_fronts, cleaned_fronts_pixels, cleaned_fronts_areas
    
    else:
        return cleaned_fronts,cleaned_fronts_pixels
    

def getwcords_new(header, xv, yv):

    wcoord = wcs.WCS(header)

    thetax, thetay = wcoord.all_pix2world(yv, xv, 0)
    tx = thetax*np.pi/180
    ty = thetay*np.pi/180
    pa_reg = np.arctan2(-np.cos(ty)*np.sin(tx), np.sin(ty))

    elon_reg = np.arctan2(np.sqrt((np.cos(ty)**2)*(np.sin(tx)**2)+(np.sin(ty)**2)), np.cos(ty)*np.cos(tx))


    return elon_reg,pa_reg


def get_fitsfiles(input_days_set, input_dates):
    print('Getting fits files...')
    rundif_paths = ['../HI_Data_Processing/stereo_processed/reduced/data/A/' + day + '/science/hi_1/' for day in input_days_set]

    # Import all fts files in ordered list

    fts_files = []

    for path in rundif_paths:
        temp_files = []
        for day in input_dates:
            temp_files.extend([path + file for file in os.listdir(path) if file.endswith('.fts') and day in file])
        fts_files.append(temp_files)

    
    fits_headers = []

    for files in fts_files:
        for file in files:
            with fits.open(file) as hdul:
                fits_headers.append(hdul[0].header)
    
    return fits_headers

def get_cmes(labeled_clusters_connected, return_area=False):
    print('Getting CMEs...')
    pad_width = 3
    image_outlines = []

    if return_area:
        areas_connected = []

    for k,label_img in enumerate(labeled_clusters_connected):
        
        outl = []

        if return_area:
            outa = []

        if np.max(label_img) != 0:
            for h in range(1,np.max(label_img)+1):
                
                temp_img = np.where(label_img == h, 1, 0).astype(np.float64)
                temp_img = gaussian(temp_img, sigma=3)
                temp_img = np.where(temp_img > 0.5, 1, 0).astype(np.float64)
                temp_img = np.pad(temp_img, pad_width=pad_width, mode='constant', constant_values=0)
                
                edge_img = find_boundaries(temp_img, mode='inner').astype(np.float64)

                #edge_img = feature.canny(temp_img, sigma=3)
                edge_img = edge_img[pad_width:-pad_width, pad_width:-pad_width]
                edge_ind = np.where(edge_img == 1)

                temp_img = temp_img[pad_width:-pad_width, pad_width:-pad_width]
                outl.append(edge_ind)

                if return_area:
                    outa.append(temp_img)

        image_outlines.append(outl)

        if return_area:
            areas_connected.append(outa)
        
    if return_area:
        return image_outlines, areas_connected
    else:
        return image_outlines
    
def get_outline(image_outlines, fits_headers):
    print('Getting outline...')
    image_outlines_wcs = []
    image_outline_pixels = []

    for i, cent in enumerate(image_outlines):
        current_header = fits_headers[i]
        wcoord = wcs.WCS(current_header)

        temp_coords = []
        temp_pixels = []
        for c in cent:
            xv, yv = c
            thetax, thetay = wcoord.all_pix2world(yv*8, xv*8, 0)
            tx = thetax*np.pi/180
            ty = thetay*np.pi/180

            pa_reg = np.arctan2(-np.cos(ty)*np.sin(tx), np.sin(ty))
            elon_reg = np.arctan2(np.sqrt((np.cos(ty)**2)*(np.sin(tx)**2)+(np.sin(ty)**2)), np.cos(ty)*np.cos(tx))

            temp_coords.append([pa_reg*180/np.pi, elon_reg*180/np.pi])
            temp_pixels.append([xv, yv])

        image_outlines_wcs.append(temp_coords)
        image_outline_pixels.append(temp_pixels)

    return image_outlines_wcs, image_outline_pixels


def connect_cmes_new(input_imgs, input_dates, img_front_wcs, img_front_pixels, input_dates_obs, labeled_areas=None, max_time_gap=140, max_start_elon=7.3, min_elon_end=12.9, elon_diff_thresholds=[-3.2,6.7], min_total_duration=None):

    previous_date = input_dates[0]
    previous_fronts = []
    cme_counter = -1
    cme_dictionary = {} 

    previous_cme_keys = []  # tracks which CME string each previous front belongs to

    for idx in range(len(input_imgs)):
        matched_previous_fronts = []

        current_date = input_dates[idx]

        delta_t = (current_date - previous_date).total_seconds()/60

        skip_val = 0

        for frnt in img_front_wcs[idx]:
            if len(frnt[0]) == 0 or len(frnt[1]) == 0:
                skip_val += 1
        
        if skip_val == len(img_front_wcs[idx]):
            print(f"Skipping {current_date} due to no fronts detected")
            continue

        if labeled_areas != None:
            current_fronts, current_fronts_pix, current_areas = aggregate_fronts(img_front_wcs[idx], img_front_pixels[idx], max_diff_elon=3, areas=labeled_areas[idx])
        else:
            current_fronts, current_fronts_pix = aggregate_fronts(img_front_wcs[idx], img_front_pixels[idx], max_diff_elon=3)
        
        new_cme_flag = identify_new_cme(current_fronts, threshold_elon=max_start_elon)
        current_cme_keys = []

        #print('delta_t', delta_t)
        if len(current_fronts) > 0 and len(previous_fronts) > 0 and delta_t < max_time_gap:
            #print(f"CME continues from {previous_date} to {current_date}")
            elongation_differences = [[] for _ in range(len(current_fronts))]

            for i, current in enumerate(current_fronts):
                pa_current, elon_current = current
                pa_current = np.round(pa_current).astype(int)
                elon_current = np.array(elon_current)

                for j, previous in enumerate(previous_fronts):
                    pa_previous, elon_previous = previous
                    pa_previous = np.round(pa_previous).astype(int)
                    elon_previous = np.array(elon_previous)

                    # Build common PA grid
                    all_pas = np.union1d(pa_current, pa_previous)
                    elon_current_grid = np.full_like(all_pas, np.nan, dtype=np.float32)
                    elon_previous_grid = np.full_like(all_pas, np.nan, dtype=np.float32)

                    current_map = dict(zip(pa_current, elon_current))
                    previous_map = dict(zip(pa_previous, elon_previous))

                    for k, pa in enumerate(all_pas):
                        if pa in current_map:
                            elon_current_grid[k] = current_map[pa]
                        if pa in previous_map:
                            elon_previous_grid[k] = previous_map[pa]

                    shared_mask = ~np.isnan(elon_current_grid) & ~np.isnan(elon_previous_grid)
                    if np.any(shared_mask):
                        diff_vector = elon_current_grid[shared_mask] - elon_previous_grid[shared_mask]
                        mean_diff = np.mean(diff_vector)
                        abs_mean_diff = np.mean(np.abs(diff_vector))
                        weight = np.sum(shared_mask) / len(all_pas)
                        weighted_diff = abs_mean_diff * (1 - weight)

                        if elon_diff_thresholds[0] <= mean_diff <= elon_diff_thresholds[1]:
                            elongation_differences[i].append((j, weighted_diff))

            best_matches = []
            for diffs in elongation_differences:
                if diffs:
                    best_match = min(diffs, key=lambda x: x[1])  # minimum absolute mean difference
                    best_matches.append(best_match[0])
                else:
                    best_matches.append(None)


            for i, match in enumerate(best_matches):
                if match is not None:
                    #print(f"Front {i} continues")
                    cme_key = previous_cme_keys[match]
                    cme_dictionary[cme_key]['cme_front_wcs'].append(current_fronts[i])
                    cme_dictionary[cme_key]['cme_front_pixels'].append(current_fronts_pix[i])
                    cme_dictionary[cme_key]['times'].append(input_dates[idx])
                    cme_dictionary[cme_key]['times_obs'].append(input_dates_obs[idx])

                    if labeled_areas != None:
                        cme_dictionary[cme_key]['areas'].append(current_areas[i])

                    current_cme_keys.append(cme_key)
                    matched_previous_fronts.append(current_fronts[i])

                elif new_cme_flag[i] == 1:
                    #print(f"Front {i} is a new CME")
                    cme_counter += 1
                    new_key = f"CME_{cme_counter}"

                    if labeled_areas != None:
                        cme_dictionary[new_key] = {
                            'cme_front_wcs': [current_fronts[i]],
                            'cme_front_pixels': [current_fronts_pix[i]],
                            'times': [input_dates[idx]],
                            'times_obs': [input_dates_obs[idx]],
                            'areas': [current_areas[i]]
                        }

                    else:
                        cme_dictionary[new_key] = {
                            'cme_front_wcs': [current_fronts[i]],
                            'cme_front_pixels': [current_fronts_pix[i]],
                            'times': [input_dates[idx]],
                            'times_obs': [input_dates_obs[idx]]
                        }     

                    current_cme_keys.append(new_key)
                    matched_previous_fronts.append(current_fronts[i])

            previous_date = current_date
            previous_fronts = matched_previous_fronts.copy()
            previous_cme_keys = current_cme_keys

        elif len(previous_fronts) == 0:
            #print(f'New sequence at {current_date}')
            for i in range(len(current_fronts)):
                if new_cme_flag[i] == 1:
                    #print(f'Front {i} is new CME')
                    cme_counter += 1
                    new_key = f"CME_{cme_counter}"
                    
                    if labeled_areas != None: 
                        cme_dictionary[new_key] = {
                            'cme_front_wcs': [current_fronts[i]],
                            'cme_front_pixels': [current_fronts_pix[i]],
                            'times': [input_dates[idx]],
                            'times_obs': [input_dates_obs[idx]],
                            'areas': [current_areas[i]]
                        }
                    
                    else:
                        cme_dictionary[new_key] = {
                            'cme_front_wcs': [current_fronts[i]],
                            'cme_front_pixels': [current_fronts_pix[i]],
                            'times': [input_dates[idx]],
                            'times_obs': [input_dates_obs[idx]]
                        }
                                            
                    current_cme_keys.append(new_key)
                    matched_previous_fronts.append(current_fronts[i])

            previous_date = current_date
            previous_fronts = matched_previous_fronts.copy()
            previous_cme_keys = current_cme_keys
        
        else:
            previous_date = current_date
            previous_fronts = []
            current_fronts = []
            previous_cme_keys = []

    if min_total_duration is not None:
        for entry in list(cme_dictionary):
            time_beginning = sorted(list(set(cme_dictionary[entry]['times'])))[0]
            time_end = sorted(list(set(cme_dictionary[entry]['times'])))[-1]
            time_diff = (time_end - time_beginning).total_seconds()/3600 # total duration of CME in hours

            if time_diff < min_total_duration:
                del cme_dictionary[entry]

    for entry in list(cme_dictionary):
        if np.nanmax(cme_dictionary[entry]['cme_front_wcs'][-1][1]) < min_elon_end: #if maximum elongation reached is below threshold
            del cme_dictionary[entry]

    # Merge duplicate timestep entries
    for cme_key, cme_data in cme_dictionary.items():
        merged_data = defaultdict(lambda: {'wcs': [], 'pixels': [], 'areas': []})
        
        for i, time in enumerate(cme_data['times']):
            merged_data[time]['wcs'].append(cme_data['cme_front_wcs'][i])
            merged_data[time]['pixels'].append(cme_data['cme_front_pixels'][i])
            if 'areas' in cme_data:
                merged_data[time]['areas'].append(cme_data['areas'][i])
        
        # Sort times
        sorted_times = sorted(merged_data.keys())
        
        cme_data['times'] = sorted_times
        cme_data['times_obs'] = [cme_data['times_obs'][cme_data['times'].index(t)] for t in sorted_times]
        cme_data['cme_front_wcs'] = []
        cme_data['cme_front_pixels'] = []

        if 'areas' in cme_data:
            cme_data['areas'] = []

        for t in sorted_times:
            cme_data['cme_front_wcs'].append(np.concatenate(merged_data[t]['wcs'], axis=1))
            cme_data['cme_front_pixels'].append(np.concatenate(merged_data[t]['pixels'], axis=1))
            if 'areas' in cme_data:
                cme_data['areas'].append(np.sum(merged_data[t]['areas'], axis=0))

    # Sort by PA value

    for cme_key, cme_data in cme_dictionary.items():
        for i, time in enumerate(cme_data['times']):

            # Extract current WCS and pixel data
            wcs_data = cme_data['cme_front_wcs'][i]
            pixel_data = cme_data['cme_front_pixels'][i]

            # Shape is (2, N) with [PA, elongation]
            pa_values = wcs_data[0]
            elongations = wcs_data[1]

            x_pix_values = pixel_data[0]
            y_pix_values = pixel_data[1]

            elongations = np.array([e for _, e in sorted(zip(pa_values, elongations), key=lambda pair: pair[0])])
            x_pix_values = np.array([x for _, x in sorted(zip(pa_values, x_pix_values), key=lambda pair: pair[0])])
            y_pix_values = np.array([y for _, y in sorted(zip(pa_values, y_pix_values), key=lambda pair: pair[0])])
            pa_values = np.array(sorted(pa_values))

            wcs_data = [pa_values,elongations]
            pixel_data = [x_pix_values, y_pix_values]

            cme_data['cme_front_wcs'][i] = wcs_data
            cme_data['cme_front_pixels'][i] = pixel_data

    cme_names = np.array(['CME_' + str(i) for i in range(len(cme_dictionary))])
    
    cme_dictionary_clean = {}

    for i, entry in enumerate(list(cme_dictionary)):
        cme_dictionary_clean[cme_names[i]] = cme_dictionary[entry].copy()

    return cme_dictionary_clean

def load_results(path, load_gt=False):

    print('Loading results...')
    results = np.load(path)
    filenames = results['filenames']
    pred = results['masks']
    
    if load_gt:
        gt = results['gt']
        return filenames, pred, gt
    
    else:
        return filenames, pred

def get_front(image_outlines_wcs, image_outline_pixels, fits_headers, image_areas=None):
    print('Getting front...')
    img_front_wcs = []
    img_front_pixels = []
    
    if image_areas is not None:
        img_front_areas = []

    for out_num, outline in enumerate(image_outlines_wcs):

        temp_coords = []
        temp_pixels = []

        if image_areas is not None:
            temp_areas = []

        if(len(image_outline_pixels[out_num])>0):
        
            indices = copy.deepcopy(image_outline_pixels[out_num])
            for i,idx in enumerate(indices):
                img = np.zeros((128,128))
                idx_arr = np.array(copy.deepcopy(idx))
                
                if len(idx_arr[0]) == 0:
                    continue

                img[idx_arr[0,:],idx_arr[1,:]] = 1.0

                points = [(xx,yy)  for xx,yy in zip(idx_arr[0,:],idx_arr[1,:])]

                area = copy.deepcopy(image_areas[out_num][i])
                header = copy.deepcopy(fits_headers[out_num])

                if(len(points)>3):

                    skeleton = morphology.skeletonize(area)
                    pruned_skeleton, _, _ = pcv.morphology.prune(skel_img=skeleton.astype(np.uint8), size=10)
                    pruned_coordinates = np.where(pruned_skeleton == 1)
                    skeleton_x = pruned_coordinates[0]
                    skeleton_y = pruned_coordinates[1]

                    elongs_skeleton, pas_skeleton = getwcords_new(header, skeleton_x*8, skeleton_y*8)

                    center_pa_skeleton_deg = np.rad2deg(pas_skeleton)
                    center_elong_skeleton_deg = np.rad2deg(elongs_skeleton)

                    temp_coords.append([center_pa_skeleton_deg, center_elong_skeleton_deg])
                    temp_pixels.append([np.clip(np.round(skeleton_x,0),0,127).astype(np.uint32), np.clip(np.round(skeleton_y,0),0,127).astype(np.uint32)])

                    if image_areas is not None:
                        temp_areas.append(area)


        img_front_wcs.append(temp_coords)
        img_front_pixels.append(temp_pixels)

        if image_areas is not None:
            img_front_areas.append(temp_areas)   


    if image_areas is not None:
        return img_front_wcs, img_front_pixels, img_front_areas
    else:
        return img_front_wcs, img_front_pixels
    

def post_processing(pred_final, t=0.45, return_labeled=True):

    # print('Post-processing...')
    
    images_proc_thresh = pred_final.copy().astype(float)

    if return_labeled:
        labeled_clusters_connected = np.zeros_like(images_proc_thresh)

    # Label the connected components in the binary image (using skimage.clusters label)
    for i in range(len(images_proc_thresh)):
        # Pre-processing

        images_proc_thresh[i] = np.where(images_proc_thresh[i] > t, 1, 0)
        images_proc_thresh[i] = morphology.remove_small_objects(images_proc_thresh[i].astype(bool), max_size=64, connectivity=2)
        images_proc_thresh[i] = morphology.remove_small_holes(images_proc_thresh[i].astype(bool), max_size=50, connectivity=2)
        
        images_proc_thresh[i] = images_proc_thresh[i].astype(float)
        
        if return_labeled:
            labeled_clusters_connected[i] = label(images_proc_thresh[i])

    if return_labeled:
        return images_proc_thresh, labeled_clusters_connected.astype(np.uint8)
    
    else:
        return images_proc_thresh



def create_dictionaries(load_path, save_path, t=0.90):
    print('Creating dictionaries...')
    filenames, pred = load_results(load_path)

    input_dates = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S').strftime('%Y%m%d_%H%M%S') for name in filenames]
    input_days = [datetime.datetime.strptime(name.split('/')[-1][:15], '%Y%m%d_%H%M%S').strftime('%Y%m%d') for name in filenames]

    input_days_set = sorted(set(input_days))
    pred = pred[:len(input_dates)]

    images_proc, labeled_clusters_connected = post_processing(pred, t=t)
    image_outlines, image_areas = get_cmes(labeled_clusters_connected, return_area=True)

    fits_headers = get_fitsfiles(input_days_set, input_dates)

    input_dates_obs = [datetime.datetime.strptime(fits_headers[i]['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') for i in range(len(fits_headers))]

    image_outlines_wcs, image_outline_pixels = get_outline(image_outlines, fits_headers)
    img_front_wcs, img_front_pixels, img_front_areas = get_front(image_outlines_wcs, image_outline_pixels, fits_headers, image_areas=image_areas)
    img_front_wcs_clean, img_front_pixels_clean, img_front_areas_clean = remove_outliers_from_fronts(img_front_wcs, img_front_pixels, fits_headers, img_front_areas, window=3, threshold=1)

    input_dates_obs = [datetime.datetime.strptime(datetime.datetime.strptime(fits_headers[i]['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S'),'%Y%m%d_%H%M%S') for i in range(len(fits_headers))]
    input_dates_datetime = [datetime.datetime.strptime(name[:15], '%Y%m%d_%H%M%S') for name in input_dates]
    
    cme_dictionary = connect_cmes_new(images_proc, input_dates_datetime, img_front_wcs_clean, img_front_pixels_clean, input_dates_obs, img_front_areas_clean)
    
    np.save(save_path, cme_dictionary)

def segmentation_operational(config,timepairs):
    for pair_idx, pair in enumerate(timepairs):

        batch_size = 4
        num_workers = 2
        device = config['model']['device']
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model = models.CNN3D(1,1)

        checkpoint = torch.load(config['model_name'], weights_only=True,map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)
        
        width_par = config['dataset']['width']
        win_size = config['dataset']['win_size']
        stride = config['dataset']['stride']

        dataset_test = dataset.RundifSequence_Test(data_path=config['dataset']['data_path'][pair_idx],pair=pair,win_size=win_size,stride=stride,width_par=width_par)

        data_loader = torch.utils.data.DataLoader(
                                                    dataset_test,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=False
                                                )

        test_list_names = np.array([l.tolist() for l in dataset_test.files_strided])

        n_val = len(set(test_list_names.flatten()))
        input_names = np.chararray((n_val), itemsize=len(test_list_names[0][0]))

        model.eval()


        num_batch = 0
        dict_output = {}
        dict_output_count = {}

        with torch.no_grad():
            for num, data in enumerate(data_loader):
                print(f'Processing: {num} of {dataset_test.__len__()}')
                input_data = data['image'].float().to(device)

                name_data = np.array(data['names'])
                input_data = torch.permute(input_data, (0, 2, 1, 3, 4))
                name_data = np.transpose(name_data)

                pred_comb = model(input_data)

                for b in range(pred_comb.shape[0]):
                    for k in range(win_size):

                        if str(name_data[b,k]) in dict_output.keys():
                            dict_output[str(name_data[b,k])]['pred'] = torch.nansum(torch.cat([dict_output[str(name_data[b,k])]['pred'][None,...],pred_comb[b,0,k,:,:][None,...]],0),axis=0)
                            dict_output_count[str(name_data[b,k])] = dict_output_count[str(name_data[b,k])] + 1
                        else:
                            dict_output[str(name_data[b,k])] = {'pred':pred_comb[b,0,k,:,:]}
                            dict_output_count[str(name_data[b,k])] = 1

                    num_batch = num_batch + 1

        pred_final = np.zeros((n_val, width_par, width_par))

        input_names = []
        for idx,k in enumerate(dict_output.keys()):
            mean_output = dict_output[k]['pred'] / dict_output_count[k]
            pred_final[idx] = mean_output.cpu().numpy()
            input_names.append(k)


        np.savez_compressed("./segmentation_masks_mean_"+pair['start']+"_"+pair['end']+".npz", masks=pred_final.astype(np.float16), filenames=input_names)

def main(mdl, ml_path, timepairs, best_method, best_threshold):

    for  pair in timepairs:
        start = pair['start']
        end = pair['end']

        mdl_path = ml_path + mdl + '/'

        if not os.path.exists(mdl_path):
            os.makedirs(mdl_path)
        
        load_path = mdl_path + 'segmentation_masks_'+best_method+'_'+start+'_'+end+'.npz'
        save_path = mdl_path + 'segmentation_dictionary_'+best_method+'_'+str(best_threshold).split('.')[-1]+'_'+start+'_'+end+'.npy'

        create_dictionaries(load_path, save_path, t=best_threshold)

if __name__ == "__main__":

    config = parse_yml('config_evaluation.yaml')

    mdls_operational = config['mdls_operational']
    
    ml_path = config['paths']['ml_path']
    

    time_pairs = config['time_pairs']
    timepairs = [{'start': time_pairs['start'][i], 'end': time_pairs['end'][i]} for i in range(len(time_pairs['start']))]


    best_threshold = 0.45
    best_method= 'mean'

    segmentation_operational(config,timepairs)
    main(mdl=mdls_operational, ml_path=ml_path, timepairs=timepairs, best_method=best_method, best_threshold=best_threshold)

