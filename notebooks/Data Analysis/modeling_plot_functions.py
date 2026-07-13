from modeling_functions import (
    ruggedness_func,
    empirical_data_return,
    asocial_data_return,
    asocial_data_return_true_size,
    social_data_return,
    social_data_return_true_size,
    plot5_y_vals_return,
)
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


def make_full_figure_new(
    step_size, alpha, gamma, save, line=1250, new_mid=18.5, iterations=1000
):
    """
    Plots a figure comparing the empirical, asocial, and social models using the functions from modeling_functions.py. The 4th and 5th subplots are in the same form as the plots_d_e function below.

        Args:
            step_size (float): nmber of seconds per time step
            alpha (float): probability of a fish leaving the staging pond
            gamma (float): probability of a fish following a fish that has left
            save (bool): if True, the figure will be saved in the working directory
            line (int): x-value of the vertical line used to measure crossing
            new_mid (float): time location (hours) of the middle of the centered data
            iterations (int): number of times to repeat the models for the fourth and fifth plots
        Returns:
            Plots a figure with 6 subplots in the same form as the plot from the 2017 paper by Andrew Berdahl
    """
    t_vals = np.linspace(15, 24, int(9 * 3600 / step_size) + 1)
    fig = plt.figure(figsize=(15, 10))
    empirical_data = empirical_data_return(t_vals, line, new_mid)
    asocial_data = asocial_data_return_true_size(t_vals)
    social_data = social_data_return_true_size(t_vals, alpha, gamma)

    names = ["empirical data", "asocial model", "social model"]
    colors = ["tab:blue", "#009E73", "tab:orange"]

    # for subplots a - c
    bins = np.linspace(15, 24, len(t_vals) - 1)
    hist_data = [empirical_data[1], asocial_data[1], social_data[1]]
    for i, data in enumerate(hist_data):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.hist(
            data, bins=bins, weights=np.ones(len(data)) / len(data), color=colors[i]
        )
        ax.set_xlim(15, 24)
        ax.set_ylim(0, 0.05)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Fraction of fish arriving")
        ax.set_title(names[i])

    range_index = round(0.05 * iterations / 2)
    ax = fig.add_subplot(2, 3, 4)

    emp_fracs, _ = np.histogram(
        empirical_data[1],
        bins=bins,
        weights=np.ones(len(empirical_data[1])) / len(empirical_data[1]),
    )
    emp_y_vals = plot5_y_vals_return(empirical_data[0], t_vals)

    asocial_fracs = []
    asocial_y_vals = []
    asocial_totals = []
    social_fracs = []
    social_y_vals = []
    social_totals = []
    for i in range(iterations):
        asocial_data = asocial_data_return_true_size(t_vals)
        asoc_fracs, _ = np.histogram(
            asocial_data[1],
            bins=bins,
            weights=np.ones(len(asocial_data[1])) / len(asocial_data[1]),
        )
        asocial_fracs.append(asoc_fracs)
        asocial_y_vals.append(plot5_y_vals_return(asocial_data[0], t_vals))

        social_data = social_data_return_true_size(t_vals, alpha, gamma)
        soc_fracs, _ = np.histogram(
            social_data[1],
            bins=bins,
            weights=np.ones(len(social_data[1])) / len(social_data[1]),
        )
        social_fracs.append(soc_fracs)
        social_y_vals.append(plot5_y_vals_return(social_data[0], t_vals))
        if i < 20:
            for totals in asocial_data[0]:
                asocial_totals += [totals]
            for totals in social_data[0]:
                social_totals += [totals]
    asoc_max = max([max(fracs) for fracs in asocial_fracs])
    soc_max = max([max(fracs) for fracs in social_fracs])
    max_frac = max([max(emp_fracs), asoc_max, soc_max])
    bins = np.linspace(0, max_frac, 15)

    for i, model_fracs in enumerate([asocial_fracs, social_fracs]):
        x_vals, y_vals = [], []
        for fracs in model_fracs:
            heights, edges = np.histogram(fracs, bins=bins, density=True)
            if not list(x_vals):
                x_vals = (edges[:-1] + edges[1:]) / 2
            y_vals.append(heights)
        med_vals = np.median(y_vals, axis=0)
        new_vals = [[] for _ in range(len(y_vals[0]))]
        for vals in y_vals:
            for j, y in enumerate(vals):
                new_vals[j].append(y)
        new_vals = [sorted(vals) for vals in new_vals]
        lower = [vals[range_index] for vals in new_vals]
        upper = [vals[-range_index] for vals in new_vals]
        ax.semilogy(x_vals, med_vals, color=colors[i + 1], label=names[i + 1])
        ax.fill_between(x_vals, lower, upper, color=colors[i + 1], alpha=0.15)
    heights, edges = np.histogram(emp_fracs, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.semilogy(centers, heights, color=colors[0], label=names[0])
    ax.legend()
    ax.set_xlabel("Fraction of daily run arriving per time step")
    ax.set_ylabel("PDF")

    ax = fig.add_subplot(2, 3, 5)
    for i, y_vals in enumerate([asocial_y_vals, social_y_vals]):
        med_vals = np.median(y_vals, axis=0)
        new_vals = [[] for _ in range(len(y_vals[0]))]
        for vals in y_vals:
            for j, y in enumerate(vals):
                new_vals[j].append(y)
        new_vals = [sorted(vals) for vals in new_vals]
        lower = [vals[range_index] for vals in new_vals]
        upper = [vals[-range_index] for vals in new_vals]
        ax.plot(t_vals, med_vals, color=colors[i + 1], label=names[i + 1])
        ax.fill_between(t_vals, lower, upper, color=colors[i + 1], alpha=0.15)
    ax.plot(t_vals, emp_y_vals, color=colors[0], label=names[0])
    ax.legend()
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("SD (cummulative fraction arrived)")

    names = ["empirical data", "social model", "asocial model"]
    colors = ["tab:blue", "tab:orange", "#009E73"]
    ax = fig.add_subplot(2, 3, 6)
    raw_totals = [asocial_totals, social_totals, empirical_data[0]]
    for i, model_totals in enumerate(raw_totals):
        points = []
        for totals in model_totals:
            new_totals = [0]
            for j, total in enumerate(totals[1:]):
                new_totals.append(total - totals[j])
            ruggedness = ruggedness_func(new_totals)
            points.append([totals[-1], ruggedness])
        if i == 2:
            new_points = np.array(sorted(points))
            X = sm.add_constant(new_points[:, 0])
            model = sm.OLS(new_points[:, 1], X).fit()
            pred = model.get_prediction(X)
            ci = pred.summary_frame(alpha=0.05)
            ax.plot(new_points[:, 0], ci["mean"], color=colors[2 - i])
            ax.fill_between(
                new_points[:, 0],
                ci["mean_ci_lower"],
                ci["mean_ci_upper"],
                color=colors[2 - i],
                alpha=0.15,
            )
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], color=colors[2 - i], label=names[2 - i])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_xlabel("Daily run size")
    ax.set_ylabel("Ruggedness")
    fig.suptitle(
        rf"{int( 9 * 3600 / (len(t_vals) - 1))}-second time steps with $\alpha$={alpha} and $\gamma$={gamma}"
    )
    if save:
        plt.savefig(
            f"{step_size}_seconds-{alpha}-{gamma}_empirical_run_size.jpg", dpi=500
        )
    plt.show()


def make_full_figure(
    t_vals, empirical_data, asocial_data, social_data, alpha, gamma, save
):
    """
    Plots a figure comparing the empirical, asocial, and social data as obtained using the functions above

        Args:
            t_vals (np array): np.linspace time mesh of the desired time with desired time step size
            empirical_data (list): list of both totals and hist_data as received from the empirical_data_return or empirica_data_return_true_size function
            asocial_data (list): list of both totals and hist data as received from the asocial_data_return or asocial_data_return_true_size function
            social_data (list): list of both totals and hist_data as received from the social_data_return or social_data_return_true_size function
            alpha (float): the probability of a fish leaving the staging pool
            gamma (float): the probability of a fish following a fish that leaves the staging pond
            save (bool): if True, the figure will be saved to the directory
        Returns:
            plots a figure comparing the data sets
    """
    names = ["empirical_data", "asocial model", "social model"]
    colors = ["tab:blue", "#009E73", "tab:orange"]
    fig = plt.figure(figsize=(15, 10))

    all_fracs = []
    bins = np.linspace(15, 24, len(t_vals) - 1)
    hist_data = [empirical_data[1], asocial_data[1], social_data[1]]
    for i, data in enumerate(hist_data):
        fracs, _ = np.histogram(data, bins=bins, weights=np.ones(len(data)) / len(data))
        all_fracs.append(fracs)
        ax = fig.add_subplot(2, 3, i + 1)
        ax.hist(
            data, bins=bins, weights=np.ones(len(data)) / len(data), color=colors[i]
        )
        ax.set_xlim(15, 24)
        ax.set_ylim(0, 0.05)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Fraction of fish arriving")
        ax.set_title(names[i])

    ax = fig.add_subplot(2, 3, 4)
    full_fracs = [frac for fracs in all_fracs for frac in fracs]
    bins = np.linspace(0, max(full_fracs), 15)
    for i, fracs in enumerate(all_fracs):
        heights, edges = np.histogram(fracs, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.semilogy(centers, heights, color=colors[i], label=names[i])
    ax.legend()
    ax.set_xlabel("Fraction of daily run arriving per time step")
    ax.set_ylabel("PDF")

    all_raw_totals = [empirical_data[0], asocial_data[0], social_data[0]]

    ax = fig.add_subplot(2, 3, 5)
    for i, model_totals in enumerate(all_raw_totals):
        new_model_totals = []
        for totals in model_totals:
            if totals[-1] == 0:
                new_model_totals.append([0] * len(totals))
            else:
                new_model_totals.append([total / totals[-1] for total in totals])
        y_vals = []
        for j, t in enumerate(t_vals):
            val_list = [totals[j] for totals in new_model_totals]
            y_vals.append(np.std(val_list))
        ax.plot(t_vals, y_vals, color=colors[i], label=names[i])
    ax.legend()
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("SD (cummulative fraction arrived)")

    ax = fig.add_subplot(2, 3, 6)
    for i, model_totals in enumerate(all_raw_totals):
        points = []
        for totals in model_totals:
            new_totals = [0]
            for j, total in enumerate(totals[1:]):
                new_totals.append(total - totals[j])
            ruggedness = ruggedness_func(new_totals)
            points.append([totals[-1], ruggedness])
        if i == 0:
            new_points = np.array(sorted(points))
            X = sm.add_constant(new_points[:, 0])
            model = sm.OLS(new_points[:, 1], X).fit()
            pred = model.get_prediction(X)
            ci = pred.summary_frame(alpha=0.05)
            ax.plot(new_points[:, 0], ci["mean"], color=colors[i])
            ax.fill_between(
                new_points[:, 0],
                ci["mean_ci_lower"],
                ci["mean_ci_upper"],
                color=colors[i],
                alpha=0.15,
            )
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], color=colors[i], label=names[i])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_xlabel("Daily run size")
    ax.set_ylabel("Ruggedness")
    fig.suptitle(
        rf"{int( 9 * 3600 / (len(t_vals) - 1))}-second time steps with $\alpha$={alpha} and $\gamma$={gamma}"
    )
    if save:
        plt.savefig(
            f"{int(9 * 3600 / (len(t_vals) - 1))}_seconds-{alpha}-{gamma}_empirical_run_size.jpg",
            dpi=500,
        )
    plt.show()


def get_data_and_fig(step_size, alpha, gamma, save, line=1250, new_mid=18.5):
    """
    Uses make_full_figure2 and the other above function to get a figure with the empirical data and data from the models

        Args:
            step_size (float): number of seconds desired per time step
            alpha (float): probability of a fish leaving the staging pool
            gamma (float): probability of a fish following a fish that leaves the pool
            save (bool): if True, will save the figure created
            line (int): location of the vertical line used to determine crossing
            new_mid (float): number of hours to set the new mid point of the data to
        Returns:
            plots a figure with 6 subplots to compare the empirical data and the data from the models
    """
    t_vals = np.linspace(15, 24, 9 * int(3600 / step_size) + 1)

    empirical_data = empirical_data_return(t_vals, line, new_mid)
    asocial_data = asocial_data_return_true_size(t_vals)
    social_data = social_data_return_true_size(t_vals, alpha, gamma)

    make_full_figure(
        t_vals, empirical_data, asocial_data, social_data, alpha, gamma, save
    )


def plots_d_e(time_step, alpha, gamma, iterations, save, num_bins=15, scatter=False):
    """
    Returns plots d and e from the 2017 study, plotting the median of the simulated data as well as the range for the middle 95% of the simulated data

        Args:
            time_step (float): number of seconds per time step
            alpha (float): probability of a fish leaving the staging pool
            gamma (float): probability of a fish following a fish that has left the pool
            iterations (int): amount of times each model will be run
            save (bool): if True, the figure will be saved in the folder the function is run
            num_bins (int): number of bins for the histogram that is used for the PDF of the data
            scatter (bool): if True, the values from all of the iterations will be plotted as points in a scatter plot
        Returns:
            plots a figure with two supblots, one for plot d (the PDFs) and one for plot e (SD over time)
    """
    t_vals = np.linspace(15, 24, int(9 * 3600 / time_step) + 1)
    colors = ["tab:blue", "#009E73", "tab:orange"]
    names = ["empirical_data", "asocial_model", "social_model"]
    range_index = round(0.05 * iterations / 2)
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)
    bins = np.linspace(15, 24, len(t_vals) - 1)
    empirical_totals, empirical_data = empirical_data_return(t_vals, 1250, 18.5)

    emp_fracs, _ = np.histogram(
        empirical_data,
        bins=bins,
        weights=np.ones(len(empirical_data)) / len(empirical_data),
    )

    emp_y_vals = plot5_y_vals_return(empirical_totals, t_vals)

    asocial_fracs = []
    asocial_y_vals = []
    social_fracs = []
    social_y_vals = []
    for _ in range(iterations):
        asocial_totals, asocial_data = asocial_data_return_true_size(t_vals)
        asoc_fracs, _ = np.histogram(
            asocial_data,
            bins=bins,
            weights=np.ones(len(asocial_data)) / len(asocial_data),
        )
        asocial_fracs.append(asoc_fracs)
        asocial_y_vals.append(plot5_y_vals_return(asocial_totals, t_vals))

        social_totals, social_data = social_data_return_true_size(t_vals, alpha, gamma)
        soc_fracs, _ = np.histogram(
            social_data, bins=bins, weights=np.ones(len(social_data)) / len(social_data)
        )
        social_fracs.append(soc_fracs)
        social_y_vals.append(plot5_y_vals_return(social_totals, t_vals))

    asoc_max = max([max(fracs) for fracs in asocial_fracs])
    soc_max = max([max(fracs) for fracs in social_fracs])
    max_frac = max([max(emp_fracs), asoc_max, soc_max])
    bins = np.linspace(0, max_frac, 15)

    for i, model_fracs in enumerate([asocial_fracs, social_fracs]):
        x_vals, y_vals = [], []
        for fracs in model_fracs:
            heights, edges = np.histogram(fracs, bins=bins, density=True)
            if not list(x_vals):
                x_vals = (edges[:-1] + edges[1:]) / 2
            y_vals.append(heights)
        med_y_vals = np.median(y_vals, axis=0)
        new_vals = [[] for _ in range(len(y_vals[0]))]
        for vals in y_vals:
            for j, y in enumerate(vals):
                new_vals[j].append(y)
        new_vals = [sorted(vals) for vals in new_vals]
        lower = [vals[range_index] for vals in new_vals]
        upper = [vals[-range_index] for vals in new_vals]
        ax.semilogy(x_vals, med_y_vals, color=colors[i + 1], label=names[i + 1])
        ax.fill_between(x_vals, lower, upper, color=colors[i + 1], alpha=0.15)
    heights, edges = np.histogram(emp_fracs, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.semilogy(centers, heights, color=colors[0], label=names[0])
    ax.legend()
    ax.set_xlabel("Fraction of daily run arriving per time step")
    ax.set_ylabel("PDF")

    ax = fig.add_subplot(122)
    for i, y_vals in enumerate([asocial_y_vals, social_y_vals]):
        med_vals = np.median(y_vals, axis=0)
        new_vals = [[] for _ in range(len(y_vals[0]))]
        for vals in y_vals:
            for j, y in enumerate(vals):
                new_vals[j].append(y)
        new_vals = [sorted(vals) for vals in new_vals]
        lower = [vals[range_index] for vals in new_vals]
        upper = [vals[-range_index] for vals in new_vals]
        ax.plot(t_vals, med_vals, color=colors[i + 1], label=names[i + 1])
        ax.fill_between(t_vals, lower, upper, color=colors[i + 1], alpha=0.15)
    ax.plot(t_vals, emp_y_vals, color=colors[0], label=names[0])
    ax.legend()
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("SD (cummulative fraction arrived)")

    fig.suptitle(
        rf"{time_step}-second time steps, with {iterations} iterations, $\alpha$={alpha}, $\gamma$={gamma}"
    )
    if save:
        plt.savefig(f"plots4-5_{time_step}-{alpha}_{gamma}-{iterations}.jpg", dpi=500)
    plt.show()


# NOTE: these next two functions are older version where you can specify the desired size of the run of fish and the iteration number and get the figure with 6 subplots as in the 2017 study


def old_make_full_figure(t_vals, empirical_data, asocial_data, social_data, total_fish):
    """
    Plots an entire figure based on the empirical, asocial, and social data as retreived from the functions above

        Args:
            t_vals (numpy array): a np.linspace array of the range of times with the desired step size
            empirical_data (list): list of both raw times totals and hist data for empirical data
            asocial_data (list): list of both raw time totals and hist_data for the asocial model
            social_data (list): list of both raw time totals and hist_data for the social model
        Returns:
            figure with 6 total subplots in 2 rows and three columns
    """
    names = ["empirical data", "asocial model", "social model"]
    colors = ["tab:blue", "#009E73", "tab:orange"]
    fig = plt.figure(figsize=(15, 10))

    all_fracs = []
    # num_bins = len(t_vals) - 1
    bins = np.linspace(15, 24, len(t_vals) - 1)
    hist_data = [empirical_data[1], asocial_data[1], social_data[1]]
    for i, data in enumerate(hist_data):
        fracs, _ = np.histogram(data, bins=bins, weights=np.ones(len(data)) / len(data))
        all_fracs.append(fracs)
        ax = fig.add_subplot(2, 3, i + 1)
        ax.hist(
            data, weights=np.ones(len(data)) / len(data), bins=bins, color=colors[i]
        )
        ax.set_xlim(15, 24)
        ax.set_ylim(0, 0.05)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Fraction of fish arriving")
        ax.set_title(names[i])

    ax = fig.add_subplot(2, 3, 4)
    full_fracs = [frac for fracs in all_fracs for frac in fracs]
    x_vals = np.linspace(0, max(full_fracs), 1000)
    for i, fracs in enumerate(all_fracs):
        # kde = stats.gaussian_kde(fracs)
        # ax.semilogy(x_vals, kde.logpdf(x_vals), color=colors[i], label=names[i])
        heights, edges = np.histogram(fracs, bins=6, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.semilogy(centers, heights, color=colors[i], label=names[i])
    ax.legend()
    ax.set_xlabel("Fraction of daily run arriving per time step")
    ax.set_ylabel("PDF")
    ax.set_ylim(10 ** (-1), 10**3)

    all_raw_totals = [empirical_data[0], asocial_data[0], social_data[0]]

    ax = fig.add_subplot(2, 3, 5)
    for i, model_totals in enumerate(all_raw_totals):
        new_model_totals = []
        for totals in model_totals:
            if totals[-1] == 0:
                new_model_totals.append([0] * len(totals))
            else:
                new_model_totals.append([total / totals[-1] for total in totals])
        # new_model_totals = [
        #     [total / totals[-1] for total in totals] for totals in model_totals
        # ]
        y_vals = []
        for j, t in enumerate(t_vals):
            val_list = [totals[j] for totals in new_model_totals]
            y_vals.append(np.std(val_list))
        ax.plot(t_vals, y_vals, color=colors[i], label=names[i])
    ax.legend()
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("SD (cummulative fraction arrived)")

    ax = fig.add_subplot(2, 3, 6)
    for i, model_totals in enumerate(all_raw_totals):
        points = []
        for totals in model_totals:
            # print(totals)
            new_totals = [0]
            for j, total in enumerate(totals[1:]):
                new_totals.append(total - totals[j])
            ruggedness = ruggedness_func(new_totals)
            points.append([totals[-1], ruggedness])
        if i == 0:
            new_points = np.array(sorted(points))
            X = sm.add_constant(new_points[:, 0])
            model = sm.OLS(new_points[:, 1], X).fit()
            pred = model.get_prediction(X)
            ci = pred.summary_frame(alpha=0.05)
            ax.plot(new_points[:, 0], ci["mean"], color=colors[i])
            ax.fill_between(
                new_points[:, 0],
                ci["mean_ci_lower"],
                ci["mean_ci_upper"],
                color=colors[i],
                alpha=0.15,
            )
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], color=colors[i], label=names[i])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_xlabel("Daily run size")
    ax.set_ylabel("Ruggedness")
    fig.suptitle(
        rf"{int((9 * 3600) / (len(t_vals) - 1))}-second time steps with {total_fish} desired fish per day"
    )

    plt.show()


def old_get_data_and_fig(
    step_size, total_fish, alpha, gamma, iterations, line=1250, new_mid=18.5
):
    """
    Combines all the functions to get the empirical data and the data from the models to plot it all to compare

        Args:
            step_size (float): number of seconds desired for each time step
            total_fish (int): total number of fish desired for the daily run
            alpha (float): probability of a fish leaving the staging pool for the social model
            gamma (float): probability of a fish following another fish that left the staging pool in the social model
            iterations (int): number of times to repeat the process for the asocial and social models
            line (int): x value of the vertical line used to determine crossing
            new_mid (float): new mid point of the empirical data so that it lines up
        Returns:
            Plots a figure showing and comapring the data
    """
    t_vals = np.linspace(15, 24, (9 * int(3600 / step_size)) + 1)

    empirical_data = empirical_data_return(t_vals, line, new_mid)
    asocial_data = asocial_data_return(t_vals, total_fish, iterations)
    social_data = social_data_return(t_vals, total_fish, alpha, gamma, iterations)

    old_make_full_figure(t_vals, empirical_data, asocial_data, social_data, total_fish)
