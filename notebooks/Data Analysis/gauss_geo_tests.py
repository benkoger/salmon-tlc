import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from statsmodels.graphics.gofplots import qqplot
from scipy import stats

from cross_group_functions import tracks_crossing_info, dbscan_time_groups
from gaussian_sampling import gaussian_time_sample_generator

# the camera names will little to no fish are commented out to only consider the others
cam_names = {
    "cam01-bear_outflow": False,
    "cam07-grass_outflow": False,
    # "cam05-airport_outflow": False,
    "cam09-trail_outflow": False,
    "cam0610-big_outflow": True,
    # "cam1006-left_arm_big_confluence": False,
}
dates = ["08-07", "08-10"]


def gauss_tests(times, alpha):
    """
    Uses Shapiro, D'Agostino and Pearson, and Anderson tests to evaluate how well the normal distribution fits a data set, specifically the distribution of crossing times

        Args:
            times (list): list of the crossing times, either for groups or individuals
            alpha (float): significance level for the tests
        Returns:
            Prints a statement for each test, evaluating if the null was reject or not
    """
    # uses the shaprio test
    _, p = stats.shapiro(times)
    if p > alpha:
        print("Using Shapiro, fail to reject null:", p)
    else:
        print("Using Shapiro, reject null:", p)
    # uses the D'Agostino and Pearson test
    _, p = stats.normaltest(times)
    if p > alpha:
        print("Using D'Agostino and Pearon, fail to reject null:", p)
    else:
        print("Using D'Agostino and Pearon, reject null:", p)
    # uses the anderson test
    result = stats.anderson(times)
    for i, (sl, cv) in enumerate(zip(result.significance_level, result.critical_values)):
        if result.statistic < cv:
            print("Using Anderson, fail to reject null:", sl, cv)
        else:
            print("Using Anderson, reject null:", sl, cv)


def group_gauss_check(proximity, data, full, each_cam, save, line=1250, cam_names=cam_names, alpha=0.05, min_time=54000):
    """
    Using a qqplot and the tests in gauss_tests (function), shows whether the distribution of crossing times of groups might be normally distributed or not, based on a comparison to a truncated normal distribution

        Args:
            proximity (int): gives the number of seconds used for proximity of groups
            data (bool): if True, values for the tests from gauss_tests will be printed along with reject/fail to reject null for each test
            full (bool): if True, all of the cameras will be considered together
            each_cam (bool): if True and full is False, each camera will be considered individually. If False and full is False, each day for each camera will be considered separately.
            save (bool): if True, the figure will be saved in the working directory
            line (int): gives the x-value of the line used to determine crossing
            cam_names (dict): keys give the names of the cameras as stored in the files and values are True if upstream is left to right and False if upstream is right to left
            alpha (float): significance leve for the tests in gauss_tests
            min_times (int): gives the minimum time of day (in seconds) to be considered for the truncated normal distribution (note that the max is automatically set to 86400 (24hrs)
        Returns:
            if data, statements will be printed showing the results of various tests for normalilty. In any case, a figure containing one or more qq plots (depending on if full, each_cam) will be plotted.
    """
    # note that times will be given at different parts of the function depending on if all cameras are seen together or independently
    # first the figsize is determined
    if full: 
        times = []
        fig = plt.figure(figsize=(5,5))
    elif each_cam:
        fig = plt.figure(figsize=(30,7))
    else:
        fig = plt.figure(figsize=(30,15))

    # then each camera  and date is iterated over
    for i, (cam, leftright) in enumerate(cam_names.items()):
        if not full and each_cam:
            times = []
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        # note that only the 3rd and 6th files are used because that is where most fish are
        # looking at the normal distribution of the other days would not make sense because of the extremely small sample sizes
        for j, (tracks_file, date) in enumerate(zip([tracks_files[3], tracks_files[6]], dates)):
            if not full and not each_cam:
                times = []
            #this next block looks at the groups after the minimum time, adding singletons accordingly
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            pop_tags = []
            for tag, ts in crossing_data.items():
                if np.mean(ts) < min_time:
                    pop_tags.append(tag)
            for tag in pop_tags:
                crossing_data.pop(tag)
            groups = dbscan_time_groups(crossing_data, proximity)
            group_fish = [tag for group in groups for tag in group]
            for tag in crossing_data:
                if tag not in group_fish:
                    groups.append([tag])
            for group in groups:
                raw_times = []
                for tag in group:
                    for time in crossing_data[tag]:
                        raw_times.append(time)
                times.append(np.median(raw_times))
                # maybe would be of more use to use the smallest time
            #plots will be made at different points depending on if we are looking at every day, every camera, or all of it altogether.
            if not full and not each_cam:
                times = np.array(times)
                ax = fig.add_subplot(2, len(cam_names), 4 * j + i + 1)
                m = np.median(times)
                st = np.std(times)
                a, b = (min_time - m) / st, (86400 - m) / st
                qqplot(times, line="s", ax=ax, dist=stats.truncnorm, distargs=(a,b))
                ax.set_ylim(min_time, 86400)
                ax.set_title(f"{cam}, {date}: {len(times)} groups")

                if data:
                    print(f"{cam}, {date}: {len(times)} groups")
                    gauss_tests(times, alpha)
        if not full and each_cam:
            times = np.array(times)
            ax = fig.add_subplot(1, len(cam_names), i + 1)
            m = np.mean(times)
            st = np.std(times)
            a, b = (min_time - m) / st, (86400 - m) / st
            qqplot(times, line="s", ax=ax, dist=stats.truncnorm, distargs=(1,b))
            ax.set_ylim(min_time, 86400)
            ax.set_title(f"{cam}: {len(times)} groups")

            if data:
                print(f"{cam}: {len(times)} groups")
                gauss_tests(times, alpha)
    if full:
        times = np.array(times)
        ax = fig.add_subplot(111)
        m = np.mean(times)
        st = np.std(times)
        a, b = (min_time - m) / st, (86400 - m) / st
        qqplot(times, line="s", ax=ax, dist=stats.truncnorm, distargs=(a,b))
        ax.set_ylim(min_time, 86400)
        ax.set_title(f"All cams: {len(times)} groups")

        if data:
            print(f"All cams: {len(times)} groups")
            gauss_tests(times, alpha)
    plt.tight_layout()
    if save:
        if full:
            name = "all_cams"
        elif each_cam:
            name = "each_cam"
        else:
            name = "each_cam-day"
        plt.savefig(f"qqplot-{name}-{proximity}-truncnorm.jpg", dpi=500)
    plt.show()


def emp_geo_group_distr(proximity, each_day, individual, save, probability=True, line=1250, cam_names=cam_names):
    """
    Gives a histogram comparing the empirical distribution of group sizes to group sizes from a geometric distribution

        Args:
            proximity (int): gives the proximity (seconds) for considering groups
            each_day (bool): if True, each day will be considered individually. If False, data from each camera will be considered together
            individual (bool): if True, the group size an individual experiences is given. If False, the group sizes are given
            save (bool): if True, the figure will be saved in the working directory
            probability (bool): if true, the probability distribution will be given, as opposed to the raw number of groups/individuals. Note that if this becomes true, the draw size for the geometric distributions will need to be lowered
            line (int): x value of the line used to determine crossing
            cam_names (dict): keys give the names of the cameras as found in the files. Values are True if upstream is left to right and False if upstream is right to left.
        Returns:
            Plots a figure showing the empirical distribution of group sizes compared to that from a draw from a Geometric distribution
    """
    # first, dates and labels are set, along with the figure
    dates = ["08-07", "08-10"]
    labels = ["Empirical data", "Geometric draw"]
    fig = plt.figure(figsize=(32, 4) if each_day else (32, 8))
    # then each camera and big date are considered
    for i, (cam, leftright) in enumerate(cam_names.items()):
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        all_emp_sizes = []
        for j, tracks_file in enumerate([tracks_files[3], tracks_files[6]]):
            # the groups and sizes are then considered
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            pop_tags = []
            for tag, ts in crossing_data.items():
                if np.mean(ts) < 54000:
                    pop_tags.append(tag)
            for tag in pop_tags:
                crossing_data.pop(tag)
            groups = dbscan_time_groups(crossing_data, proximity)
            group_fish = [tag for group in groups for tag in group]
            for tag in crossing_data:
                if tag not in group_fish:
                    groups.append([tag])
            if individual:
                emp_sizes = [len(group) for group in groups for _ in range(len(group))]
            else:
                emp_sizes = [len(group) for group in groups]

            if not each_day:
                all_emp_sizes += emp_sizes
                continue
            # then the bins for the histograms are made
            height = emp_sizes.count(1) / len(emp_sizes)
            geo_sizes = stats.geom.rvs(p=height, size = 10000)
            sizes = [size for sizes in [emp_sizes, geo_sizes] for size in sizes]
            bins = np.arange(-0.5, max(sizes) + 1.5, 1)
            width = 0.2
            x = np.arange(0, max(sizes) + 1)
            ax = fig.add_subplot(1, len(cam_names) * 2, 2 * i + j + 1)
            for n, sizes in enumerate([emp_sizes, geo_sizes]):
                if probability:
                    counts, _ = np.histogram(sizes, bins=bins, weights=np.ones(len(sizes))/len(sizes))
                else:
                    counts, _ = np.histogram(sizes, bins=bins)
                ax.bar(x + (n - 1) * width, counts, width=width, label=labels[n])
            ax.set_title(f"{cam}, {dates[j]}")
            ax.legend()
            ax.set_xlabel("Group size")
            if probability and individual:
                ax.set_ylabel("Fraction of fish")
            elif individual:
                ax.set_ylabel("Number of fish")
            elif probability:
                ax.set_ylabel("Fraction of groups")
            else:
                ax.set_ylabel("Number of groups")
        if not each_day:
            height = all_emp_sizes.count(1) / len(all_emp_sizes)
            geo_sizes = stats.geom.rvs(p=height, size=10000)
            sizes = [size for sizes in [all_emp_sizes, geo_sizes] for size in sizes]
            bins = np.arange(-0.5, max(sizes) + 1.5, 1)
            width = 0.2
            x = np.arange(0, max(sizes) + 1)
            ax = fig.add_subplot(1, len(cam_names), i + 1)
            for n, sizes in enumerate([all_emp_sizes, geo_sizes]):
                if probability:
                    counts, _ = np.histogram(sizes, bins=bins, weights=np.ones(len(sizes))/len(sizes))
                else:
                    counts, _ = np.histogram(sizes, bins=bins)
                ax.bar(x + (n - 1) * width, counts, width=width, label=labels[n])
            ax.set_title(cam)
            ax.legend()
            ax.set_xlabel("Group size")
            if probability and individual:
                ax.set_ylabel("Fraction of fish")
            elif individual:
                ax.set_ylabel("Number of fish")
            elif probability:
                ax.set_ylabel("Fraction of groups")
            else:
                ax.set_ylabel("Number of groups")
    plt.tight_layout()
    if save:
        one = "individual" if individual else"group"
        two = "fraction" if probability else "total"
        three = "day" if each_day else "cam"
        plt.savefig(f"emp-geo_hist-{one}-{two}-{three}-{proximity}.jpg", dpi=500)
    plt.show()


# this is just an idea for the overall distribution
# groups are distributed by a truncated normal distribution
# their sizes are given by a geometric distribution
# based on the sparrow study comparison, truncated power law may be better
def emp_fake_distr(proximity, iterations, save, line=1250, cam_names=cam_names):
    """
    Using a Gaussian distribution for time distribution of groups and a geometric distribution for the group size to try to model the empirical data

        Args:
            proximity (int): number of seconds for group consideration
            iterations (int): number of times to repeat the Gaussian distribution draw
            save (bool): if True, the figure will be saved in the working directory
            line (int): x-value of the line used to determine crossing
            cam_names (dict): keys are names of cameras as found in the files and values are True if upstream is left to right and False if upstream is right to left
        Returns:
            Plots a figure comparing the empirical data to this "model"
    """
    dates = ["08-07", "08-10"]
    fig = plt.figure(figsize=(32,8))
    for i, (cam, leftright) in enumerate(cam_names.items()):
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        for j, tracks_file in enumerate([tracks_files[3], tracks_files[6]]):
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            pop_tags = []
            for tag, ts in crossing_data.items():
                if np.mean(ts) < 54000:
                    pop_tags.append(tag)
            for tag in pop_tags:
                crossing_data.pop(tag)
            groups = dbscan_time_groups(crossing_data, proximity)
            group_fish = [tag for group in groups for tag in group]
            for tag in crossing_data:
                if tag not in group_fish:
                    groups.append([tag])
            group_sizes = [len(group) for group in groups]
            heights, _ = np.histogram(group_sizes, bins=20, weights=np.ones(len(group_sizes))/len(group_sizes))
            # this gives the sizes based on the geometric distribution, using the probability that an individual is in a 1-group as the probability
            g_sizes = stats.geom.rvs(p=max(heights), size=len(groups))
            # then I find the times for each group to look at the distribution
            emp_times = {}
            times = []
            for group in groups:
                raw_times = []
                for tag in group:
                    for time in crossing_data[tag]:
                        raw_times.append(time)
                # note that using the first time instead of the median could be better??
                times.append(np.median(raw_times))
                emp_times[np.median(raw_times) / 3600] = len(group)
            ax = fig.add_subplot(2, 8, 2 * i + j + 1)
            ax.stem(list(emp_times), list(emp_times.values()), basefmt=" ")
            ax.set_title(f"{cam}, {dates[j]}, empirical data")
            ax.set_xlabel("Time of day (hours)")
            ax.set_ylabel("Group size")
            ax.set_xlim(15, 24)

            # then I iterate over the number of iterations to get lots of Gaussian draws and use the one with the median of the medians.
            # choosing the median of the medians is somewhat arbitrary. I just wanted to take into account more than just one draw.
            gaussian_times = {}
            # meds = []
            for _ in range(iterations):
                g_times = gaussian_time_sample_generator(list(times))
                gaussian_times[np.median(g_times)] = g_times
                # meds.append(np.median(g_times))
            gauss_times = gaussian_times[sorted(gaussian_times)[round(len(gaussian_times) / 2)]]
            # this next line ensures that it is in a similar format to the emp_times dict
            fake_times = {time / 3600: size for time, size in zip(gauss_times, g_sizes)}
            ax = fig.add_subplot(2, 8, 2 * i + j + 9)
            ax.stem(list(fake_times), list(fake_times.values()),basefmt=" ", linefmt = "tab:orange")
            ax.set_title(f"{cam}, {dates[j]}, Gaussian/Geo data")
            ax.set_xlabel("Time of day (hours)")
            ax.set_ylabel("Group size")
            ax.set_xlim(15, 24)

    plt.tight_layout()
    if save:
        plt.savefig(f"emp-gauss_geo-{proximity}.jpg", dpi=500)
    plt.show()
                          