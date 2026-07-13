import os
import glob
import pickle
from collections import defaultdict
from itertools import combinations
from datetime import datetime, timedelta
from scipy import stats
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import numpy as np

from cross_group_functions import tracks_crossing_info, dbscan_time_groups

cam_names = {
    "cam01-bear_outflow": False,
    "cam07-grass_outflow": False,
    "cam0610-big_outflow": True,
    "cam09-trail_outflow": False,
}


def closest_all_fish_distances(
    xlog, ylog, save, all_distances=True, max_time=120, all_max_time=10000, num_bins=100
):
    """
    Plots a figure with the closest distance from one fish to the next as well as all distances from one fish to another

        Args:
            xlog (bool): if True, the xscale of the plots will be "log"
            ylog (bool): if True, the yscale of the plots will be "log"
            save (bool): if True, the figure will be saved in the current file directory
            all_distances (bool): if True, the figure will contain plots showing the distances from one fish to all of its neighbors
            max_time (int): max time to be considered on the plot of closest distances if using linear xscale
            all_max_time (int): max time to be considered on the plot of all distances if using linear xscale
            num_bins (int): number of bins for each histogram to have
        Returns:
            Plots a figure with subplots for each day containing the closest fish distances and potentailly the distances from one fish to all others in a day if all_distances is True.
    """
    xscale = "log" if xlog else "linear"
    yscale = "log" if ylog else "linear"
    fig = plt.figure(figsize=(15, 15))
    for i, (cam, leftright) in enumerate(cam_names.items()):
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        close_dists = []
        all_dists = []
        for tracks_file in tracks_files:
            crossing_data = tracks_crossing_info(tracks_file, 1250, leftright)
            dists_dict = defaultdict(list)
            for (tag1, times1), (tag2, times2) in combinations(
                crossing_data.items(), 2
            ):
                min_dist = 100000  # arbitrarily large
                for time1 in times1:
                    for time2 in times2:
                        if abs(time1 - time2) < min_dist:
                            min_dist = abs(time1 - time2)
                dists_dict[tag1].append(min_dist)
                dists_dict[tag2].append(min_dist)
            for dists in dists_dict.values():
                close_dists.append(min(dists))
                for dist in dists:
                    all_dists.append(dist)
        ax = fig.add_subplot(len(cam_names), 2, 2 * i + 1, xscale=xscale, yscale=yscale)
        close_dists = np.array(close_dists)
        if xlog:
            _, bins = np.histogram(close_dists[close_dists > 0], bins=num_bins)
            bins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            ax.hist(close_dists[close_dists > 0], bins=bins)
        else:
            ax.hist(close_dists[close_dists <= max_time], bins=num_bins)
            ax.set_xlim(0, max_time)
        ax.set_xlabel("Distance (seconds)")
        ax.set_ylabel("Number of fish")
        if not all_distances:
            continue
        ax = fig.add_subplot(len(cam_names), 2, 2 * i + 2, xscale=xscale, yscale=yscale)
        all_dists = np.array(all_dists)
        if xlog:
            _, bins = np.histogram(all_dists[all_dists > 0], bins=num_bins)
            bins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            ax.hist(all_dists[all_dists > 0], bins=bins)
        else:
            ax.hist(all_dists[all_dists <= all_max_time], bins=num_bins)
            ax.set_xlim(0, all_max_time)
        ax.set_title(f"All distances for {cam}")
        ax.set_xlabel("Distance (seconds)")
        ax.set_ylabel("Number of fish")
    plt.tight_layout()
    if save:
        plt.savefig(f"time_dist-{xscale}_{yscale}.jpg", dpi=500)
    plt.show()


def interfish_time_dist_plot(xlog, ylog, altogether, save, line=1250):
    """
    Plots a figure with subplots showing the distance from one fish to the next as it crosses the line

        Args:
            xlog (bool): if True, the scale for the x-axis will be logarithmic
            ylog (bool): if True, the scale for the y-axis will be logarithmic
            save (bool):if True, the figure will be saved to the current directory
            line (int): location of the vertical line to determin crossing
        Returns:
            plots a figure showing the time from one fish to another
    """
    xscale = "log" if xlog else "linear"
    yscale = "log" if ylog else "linear"
    fig = plt.figure(figsize=(8, 8))
    full_distances = []
    for i, (cam, leftright) in enumerate(cam_names.items()):
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        interfish_distances = []
        for tracks_file in tracks_files:
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            all_tags = list(crossing_data)
            for j, tag in enumerate(all_tags):
                if j == len(all_tags) - 1:
                    continue
                distances = []
                for time1 in crossing_data[tag]:
                    for time2 in crossing_data[all_tags[j + 1]]:
                        distances.append(abs(time1 - time2))
                interfish_distances.append(min(distances))
        if not interfish_distances:
            continue
        if altogether:
            full_distances += interfish_distances
            continue
        interfish_distances = np.array(interfish_distances)
        ax = fig.add_subplot(len(cam_names), 1, i + 1, xscale=xscale, yscale=yscale)
        if xlog:
            _, bins = np.histogram(interfish_distances[interfish_distances > 0])
            bins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            ax.hist(interfish_distances[interfish_distances > 0], bins=bins)
        else:
            ax.hist(interfish_distances[interfish_distances <= 200])
        ax.set_title(
            f"{cam}: Distances from one fish to its next chronological neighbor"
        )
        ax.set_xlabel("Distance (seconds)")
        ax.set_ylabel("Number of fish")
    if altogether:
        full_distances = np.array(full_distances)
        ax = fig.add_subplot(111)
        if xlog:
            _, bins = np.histogram(full_distances[full_distances > 0])
            bins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            ax.hist(full_distances[full_distances > 0], bins=bins)
        else:
            ax.hist(full_distances[full_distances <= 200])
        ax.set_title(
            "Distance from one fish to its next chronological neighbor for all three cameras"
        )
        ax.set_xlabel("Distance (seconds)")
        ax.set_ylabel("Number of fish")
    plt.tight_layout()
    if save:
        description = "all_cameras" if altogether else "each_camera"
        plt.savefig(f"{description}-{xscale}_{yscale}_scale.jpg", dpi=500)
    plt.show()


def fish_group_distr_hist(
    proximity, weighted_groups, save, size1=True, line=1250, probability=True
):
    """
    Plots a histogram of the group size distribution. If weighted_groups is True, the histogram gives the probability of a fish (the individual) being in a group of each size. If not, it gives the probability distribution for the size of a random group

        Args:
            proximity (int): specifies how close (in seconds) fish must be to each other to be in a group
            weighted_groups (bool): indicates if groups should be weighted based on number of individuals in the group
            save (bool): if True, the figure will be saved to the current file directory
            size1 (bool): indicates whether groups of size 1 (singleton fish) will be considered
            line (int): x value of the vertical lineused to determine fish crossing
        Returns:
            plots a histogram from the empirical data, based on the given arguments
    """
    all_sizes = []
    for cam, leftright in cam_names.items():
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        cam_sizes = []
        for tracks_file in tracks_files:
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            groups = dbscan_time_groups(crossing_data, proximity)
            if size1:
                group_fish = [fish for group in groups for fish in group]
                for tag in crossing_data:
                    if tag not in group_fish:
                        groups.append([tag])
            if weighted_groups:
                for group in groups:
                    cam_sizes += [len(group)] * len(group)
            else:
                cam_sizes += [len(group) for group in groups]
        all_sizes.append(cam_sizes)
    sizes = [size for cam_sizes in all_sizes for size in cam_sizes]
    bins = np.arange(-0.5, max(sizes) + 1.5, 1)
    width = 0.2
    x = np.arange(0, max(sizes) + 1)
    all_counts = []
    for cam_sizes in all_sizes:
        if not probability:
            counts, _ = np.histogram(cam_sizes, bins=bins)
            all_counts.append(counts)
            continue
        counts, _ = np.histogram(
            cam_sizes, bins=bins, weights=np.ones(len(cam_sizes)) / len(cam_sizes)
        )
        all_counts.append(counts)
    colors = ["tab:blue", "tab:orange", "#009E73", "#CCB000"]
    for i, (counts, cam) in enumerate(zip(all_counts, cam_names)):
        plt.bar(x + (i - 1) * width, counts, width=width, label=cam, color=colors[i])
    start = "Probability distribution" if probability else "Histogram"
    then = "for individual fish" if weighted_groups else "for group sizes"
    plt.title(f"{start} {then} across cameras")
    plt.legend()
    # plt.xticks(x)
    if save:
        first = "probability" if probability else "hist"
        second = "individuals" if weighted_groups else "group-sizes"
        plt.savefig(f"{first}_{second}_all-cams-{proximity}.jpg", dpi=500)
    plt.show()


def make_lollipop_plots(proximity, save, total_plot=False, line=1250):
    dates = ["2025-08-07", "2025-08-10"]
    fig = plt.figure(figsize=(25, 15))
    num_rows = len(cam_names) * 2 if total_plot else len(cam_names)
    cam_repeat = 4 if total_plot else 2
    for i, (cam, leftright) in enumerate(cam_names.items()):
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        for j, tracks_file in enumerate([tracks_files[3], tracks_files[6]]):
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            total = 0
            total_fish = []
            total_times = []
            if total_plot:
                for tag, times in crossing_data.items():
                    total += 1
                    total_fish.append(total)
                    total_times.append(np.mean(times) / 3600)
                ax = fig.add_subplot(num_rows, 2, 3 + j + cam_repeat * i)
                ax.plot(total_times, total_fish)
                ax.set_xlim(15, 24)
                ax.set_xlabel("Time of day (hours)")
                if i == 0:
                    ax.set_ylabel("Number of fish crossed")
                ax.set_title(f"Total fish crossing over {dates[j]} for {cam}")
            groups = dbscan_time_groups(crossing_data, proximity)
            group_fish = [fish for group in groups for fish in group]
            singles = []
            for tag in crossing_data:
                if tag not in group_fish:
                    singles.append(tag)
            group_times = []
            for group in groups:
                group_times.append(
                    [np.mean(crossing_data[group[0]]) / 3600, len(group)]
                )
            single_times = []
            for single in singles:
                single_times.append([np.mean(crossing_data[single]) / 3600, 1])
            group_times = np.array(group_times)
            single_times = np.array(single_times)
            ax = fig.add_subplot(num_rows, 2, 1 + j + cam_repeat * i)
            if group_times.size > 0:
                ax.stem(
                    group_times[:, 0],
                    group_times[:, 1],
                    basefmt=" ",
                )
            if single_times.size > 0:
                ax.stem(
                    single_times[:, 0],
                    single_times[:, 1],
                    basefmt=" ",
                    linefmt="tab:orange",
                    markerfmt="Dr",
                )
            ax.set_xlim(15, 24)
            ax.set_ylim(
                0,
            )
            ax.set_title(f"Group sizes over time on {dates[j]} for {cam}")
            ax.set_xlabel("Time of day (hours)")
            if i == 0:
                ax.set_ylabel("Group size")
    plt.tight_layout()
    if save:
        last = "_with totals" if total_plot else ""
        plt.savefig(f"group_size_lollipop{last}-{proximity}.jpg", dpi=500)
    plt.show()


def gaussian_distr_check(save, line=1250, new_mid=18.5):
    """
    Plots a figure showing how well the gaussian cumulative probability density function aligns with the empirical data

        Args:
            save (bool): if True, the figure will be saved in the working directory
            line (int): x-value of the vertical line to measure crossing
            new_mid (float): the location (in hours) to which the data will be centered
        Returns:
            Plots a figure showing the cumulative probability distributions of the empirical data and the Gaussian distribution
    """
    means = []
    stds = []
    x_vals = np.linspace(15, 24, 901)
    y = []
    for cam, leftright in cam_names.items():
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        for tracks_file in [tracks_files[3], tracks_files[6]]:
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            all_times = np.array(
                [np.mean(times) / 3600 for times in crossing_data.values()]
            )
            all_times = all_times[all_times >= 15]
            all_times = [
                time - (all_times[round(len(all_times) / 2)] - new_mid)
                for time in all_times
            ]
            for j, time in enumerate(all_times):
                if j == 0:
                    continue
                if time == all_times[j - 1]:
                    all_times[j] += 0.00000000001
            total = 0
            total_fish = []
            for time in all_times:
                total += 1
                total_fish.append(total / len(all_times))
            me, st = stats.norm.fit(all_times)
            means.append(me)
            stds.append(st)
            plt.plot(all_times, total_fish, color="gray", alpha=0.3)
            # if cam == "cam0610-big_outflow":
            #     plt.plot(all_times, total_fish, color="#009E73")
            # else:
            #     plt.plot(all_times, total_fish, color="gray", alpha=0.3)
            f = interp1d(all_times, total_fish, bounds_error=False, fill_value=(0, 1))
            y.append(f(x_vals))
    y_vals = np.nanmean(y, axis=0)
    plt.plot(x_vals, y_vals, color="tab:blue", label="mean")

    avg, std_dev = np.mean(means), np.mean(stds)
    print(avg)
    a, b = (15 - avg) / std_dev, (24 - avg) / std_dev

    plt.plot(
        x_vals,
        stats.truncnorm.cdf(x_vals, a=a, b=b, loc=avg, scale=std_dev),
        color="tab:orange",
        label=rf"Gaussian distribution, $\sigma$={round(std_dev, 3)}",
    )
    plt.legend()
    plt.xlabel("Hour of day")
    plt.ylabel("Cumulative proportion of fish arriving")
    if save:
        plt.savefig("cummulative_prop-gaussian.jpg", dpi=500)
    plt.show()


def empirical_totals_plot(time_step, save, line=1250, new_mid=18.5):
    """
    Plots a figure with a subplot for each of the two days for each camera, showing the fraction of fish arriving at each time step

        Args:
            time_step (float): number of seconds per time step
            save (bool): if True, the figure will be saved in the working directory
            line (int): x-value of the vertical line used for crossing
            new_mid (float): time location (in hours) for the new middle of the data once adjusted
        Returns:
            Plots a figure with subplots showing the percentage of fish arriving per time step for 08/07/2026 and 08/10/2026
    """
    t_vals = np.linspace(15, 24, int(9 * 3600 / time_step) + 1)
    bins = np.linspace(15, 24, len(t_vals) - 1)
    dates = ["2025-08-07", "2025-08-10"]
    fig = plt.figure(figsize=(15, 10))
    for i, (cam, leftright) in enumerate(cam_names.items()):
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        for j, tracks_file in enumerate([tracks_files[3], tracks_files[6]]):
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            times = [np.mean(times) / 3600 for times in crossing_data.values()]
            times = np.array(
                [time - (times[round(len(times) / 2)] - new_mid) for time in times]
            )
            times = times[times >= 15]

            hist_data = []
            for time in times:
                for n, t in enumerate(t_vals[1:]):
                    if t_vals[n] < time <= t:
                        hist_data.append(t)
            ax = fig.add_subplot(len(cam_names), 2, 2 * i + j + 1)
            ax.hist(
                hist_data, bins=bins, weights=np.ones(len(hist_data)) / len(hist_data)
            )
            ax.set_title(f"{cam}: {dates[j]}")
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel("Fraction of fish arriving")
    plt.tight_layout()
    if save:
        plt.savefig(f"empirical_totals-days_and_cams-{time_step}.jpg", dpi=500)
    plt.show()
