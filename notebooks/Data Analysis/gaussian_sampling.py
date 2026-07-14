import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from cross_group_functions import dbscan_time_groups, tracks_crossing_info


def gaussian_sample_generator(direction_data, min_val=54000, max_val=86400):
    """
    Uses given direction data to get random samples in the same time range using a Gaussian distribution

        Args:
            direction_data (dict): gives crossing times for each fish
                keys: fish tag number
                values (list): list of crossing times
            min_val: minimum time value (in seconds) to consider for the random samples and distribution
            max_val: maximum time value (in seconds) to consider for the random samples and distribution
        Returns:
            random_sample_dict (dict): mimics the format of direction_data to give a crossing time for a ficticious fish
                keys: ficticious fish tag number
                values (list): list of single time value of crossing time
    """
    direction_times = np.array([np.mean(times) for times in direction_data.values()])
    direction_times = direction_times[direction_times >= min_val]
    mean_val = np.mean(direction_times)
    std_dev = np.std(direction_times)
    a, b = (min_val - mean_val) / std_dev, (max_val - mean_val) / std_dev
    distribution = stats.truncnorm(a=a, b=b, loc=mean_val, scale=std_dev)
    random_samples = distribution.rvs(size=len(direction_times))
    random_sample_dict = {i: [time] for i, time in enumerate(sorted(random_samples))}
    return random_sample_dict


cam_names = {
    "cam01-bear_outflow": False,
    "cam07-grass_outflow": False,
    "cam05-airport_outflow": False,
    "cam09-trail_outflow": False,
    "cam0610-big_outflow": True,
    "cam1006-left_arm_big_confluence": True,
}


def group_size_box_plot_gaussian(proximity, ylog, save, line=1250, cam_names=cam_names):
    """
    Returns a figure with a single plot where the x-axis gives the run size and the y axis gives the group size and there is a box plot at each of the run sizes for the empirical data as well as for data from a gaussian draw

        Args:
            proximity (float): amount of seconds used to determine if a fish is in a group with a neighor or not
            ylog (bool): if True, the y-axis will be on a log scale. If not, it will be on a linear scale
            save (bool): if True, the figure will be saved in the working directory
            line (int): x-value of the vertical line used to measure fish crossing
            cam_names (dict): keys are camera names and the values are bools that indicated whether upstream is left to right (True) or right to left (False)
        Returns:
            Plots a figure  with one supblots and boxplots at eat run size for the true data and for data from a gaussian draw
    """
    fig = plt.figure(figsize=(15, 5))
    emp_points = []
    gauss_points = []
    for i, (cam, leftright) in enumerate(cam_names.items()):
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        for j, tracks_file in enumerate([tracks_files[3], tracks_files[6]]):
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            # remove all fish from before 3 in the counts
            pop_tags = []
            for tag, times in crossing_data.items():
                if np.mean(times) < 54000:
                    pop_tags.append(tag)
            for tag in pop_tags:
                crossing_data.pop(tag)
            groups = dbscan_time_groups(crossing_data, proximity)
            group_fish = [fish for group in groups for fish in group]
            for tag in crossing_data:
                if tag not in group_fish:
                    groups.append([tag])
            group_data = []
            for group in groups:
                group_data += [len(group)] * len(group)
            emp_points.append(group_data)

            if not crossing_data:
                gauss_points.append([])
                continue
            gauss_data = gaussian_sample_generator(crossing_data)
            groups = dbscan_time_groups(gauss_data, proximity)
            group_fish = [fish for group in groups for fish in group]
            for tag in gauss_data:
                if tag not in group_fish:
                    groups.append([tag])
            group_data = []
            for group in groups:
                group_data += [len(group)] * len(group)
            gauss_points.append(group_data)

    lengths = np.array([len(vals) for vals in emp_points])
    values = sorted(zip(lengths, emp_points))
    pos = [vals[0] for vals in values]
    boxes = [vals[1] for vals in values]
    # the widths will have to be changed if the xscale is not log
    widths = []
    for p in pos:
        if p == 0:
            widths.append(0)
            continue
        widths.append(10 ** (np.log10(p) + 0.01) - 10 ** (np.log10(p) - 0.01))
    for i, width in enumerate(widths):
        pos[i] += 0.5 * width
    yscale = "log" if ylog else "linear"
    ax = fig.add_subplot(111, xscale="log", yscale=yscale)
    bp = ax.boxplot(
        boxes, positions=pos, widths=widths, patch_artist=True, label="empirical data"
    )
    for box, median, flier in zip(bp["boxes"], bp["medians"], bp["fliers"]):
        box.set_alpha(0.3)
        median.set_color("black")
        flier.set_markerfacecolor("tab:blue")
        flier.set_alpha(0.3)

    values = sorted(zip([len(vals) for vals in gauss_points], gauss_points))
    pos = [vals[0] for vals in values]
    boxes = [vals[1] for vals in values]
    widths = []
    for p in pos:
        if p == 0:
            widths.append(0)
            continue
        widths.append(10 ** (np.log10(p) + 0.01) - 10 ** (np.log10(p) - 0.01))
    for i, width in enumerate(widths):
        pos[i] -= 0.5 * width
    bp = ax.boxplot(
        boxes,
        positions=pos,
        widths=widths,
        patch_artist=True,
        label="data from Gaussian draw",
    )
    for box, median, flier in zip(bp["boxes"], bp["medians"], bp["fliers"]):
        box.set_facecolor("tab:orange")
        box.set_alpha(0.3)
        median.set_color("black")
        flier.set_markerfacecolor("tab:orange")
        flier.set_alpha(0.3)
    ax.set_xlim(1, 10**3)
    if ylog:
        ax.set_ylim(0.8, 10**2)
    else:
        ax.set_ylim(0, 30)
    ax.minorticks_off()
    xticks = lengths[lengths > 0]
    ax.set_xticks(xticks, labels=xticks, fontsize=6)
    ax.legend()
    ax.set_xlabel("Daily run size")
    ax.set_ylabel("Group size")
    ax.set_title("Distribution of group sizes of individual fish at variuos run sizes")
    if save:
        plt.savefig(f"group_size-individuals_boxplot-{proximity}.jpg", dpi=500)
    plt.show()


cam_names = {
    "cam01-bear_outflow": False,
    "cam07-grass_outflow": False,
    "cam09-trail_outflow": False,
    "cam0610-big_outflow": True,
}


def interfish_emp_gauss(xlog, ylog, save, line=1250, cam_names=cam_names):
    """
    Plots a figure that will compare the interfish distances between each camera and a Gaussian draw of the same size

        Args:
            xlog (bool): if True, the x-axis will be on a log scale. Otherwise, it will be a linear scale
            ylog (bool): if True, the y-axis will be on a log scale. Otherwise, it will be a linear scale
            save (bool): if True, the figure will be saved to the current working directory
            line (int): x value of the vertical line used for fish crossing
            cam_names (dict): keys give camera names and values are bools informing if upstream is left to right (True) or right to left (False)
        Returns:
            plots a figure comparing empirical and Gaussian data and their corresponding interfish distances
    """
    xscale = "log" if xlog else "linear"
    yscale = "log" if ylog else "linear"
    fig = plt.figure(figsize=(20, 10))
    for i, (cam, leftright) in enumerate(cam_names.items()):
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        emp_distances = []
        gauss_distances = []
        for tracks_file in [tracks_files[3], tracks_files[6]]:
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            all_tags = list(crossing_data)
            for j, tag in enumerate(all_tags):
                if j == len(all_tags) - 1:
                    continue
                distances = []
                for time1 in crossing_data[tag]:
                    for time2 in crossing_data[all_tags[j + 1]]:
                        distances.append(abs(time1 - time2))
                emp_distances.append(min(distances))

            gauss_data = gaussian_sample_generator(crossing_data)
            all_tags = list(gauss_data)
            for j, tag in enumerate(all_tags):
                if j == len(all_tags) - 1:
                    continue
                distances = []
                for time1 in gauss_data[tag]:
                    for time2 in gauss_data[all_tags[j + 1]]:
                        distances.append(abs(time1 - time2))
                gauss_distances.append(min(distances))

        ax = fig.add_subplot(2, len(cam_names), i + 1, xscale=xscale, yscale=yscale)
        emp_distances = np.array(emp_distances)
        if xlog:
            nums, bins = np.histogram(emp_distances[emp_distances > 0])
            bins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            ax.hist(emp_distances[emp_distances > 0], bins=bins)
            ax.set_xlim(10 ** (-2), 10**4)
        else:
            nums, _, _ = ax.hist(emp_distances[emp_distances <= 200])
        if ylog:
            ax.set_ylim(10 ** (-2), 9 * 10**2)
        elif xlog:
            ax.set_ylim(0, 0.5 * max(nums))
        else:
            ax.set_ylim(0, 1.1 * max(nums))
        ax.set_title(f"{cam}: Empirical interfish distances")
        ax.set_xlabel("Distance (seconds)")
        ax.set_ylabel("Number of fish")

        ax = fig.add_subplot(2, len(cam_names), 4 + i + 1, xscale=xscale, yscale=yscale)
        gauss_distances = np.array(gauss_distances)
        if xlog:
            _, bins = np.histogram(gauss_distances[gauss_distances > 0])
            bins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            ax.hist(gauss_distances[gauss_distances > 0], bins=bins)
            ax.set_xlim(10 ** (-2), 10**4)
        else:
            ax.hist(gauss_distances[gauss_distances <= 200])
        if ylog:
            ax.set_ylim(10 ** (-2), 9 * 10**2)
        elif xlog:
            ax.set_ylim(0, 0.5 * max(nums))
        else:
            ax.set_ylim(0, 1.1 * max(nums))
        ax.set_title(f"{cam}: Gaussian interfish distances")
        ax.set_xlabel("Distance (seconds)")
        ax.set_label("Number of fish")
    plt.tight_layout()
    if save:
        plt.savefig(f"interfish_dist-each_cam-{xscale}_{yscale}_scale.jpg", dpi=500)
    plt.show()
