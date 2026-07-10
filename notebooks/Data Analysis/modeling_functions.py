import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats

from cross_group_functions import crossing_times


def gauss_prob(x, avg=18.5, sigma=1.216, min_time=15, max_time=24):
    """
    Sets the basis for the gaussian probability distribution used for the Poisson draw in the models

        Args:
            x (numpy array or list): time mesh (in this case) of the desired range of time
            avg (float): mean value for the distribution
            sigma (float): value of the standard deviation, determined to be 1.216 on average for the data
            min_time (int): minimum time (in hours) to be considered in the truncated normal distrubition
            max_time (int): maximum time (in hours) to be considered in the truncated normal distribution
        Returns:
            Returns the cumulative probality distribution function for the x range and with the specified average, standard deviation, and boundary values
    """
    a, b = (min_time - avg) / sigma, (max_time - avg) / sigma
    return stats.truncnorm.cdf(x, a=a, b=b, loc=avg, scale=sigma)


def ruggedness_func(totals):
    """
    Uses the ruggedness idea that Andrew used in the 2017 paper to determine how much the behavior comes in pulses

        Args:
            totals (list): contians total values for the time steps
        Returns:
            ruggedness: value associated with the ruggedness of the total values
    """
    term1 = max(totals) / sum(totals)
    numerator = 0
    for i, total in enumerate(totals[1:]):
        numerator += abs(total - totals[i])
    term2 = numerator / (2 * sum(totals))
    ruggedness = (term1 + term2) / 2
    return ruggedness


# EMPIRICAL DATA FUNCTIONS


def empirical_data_extraction(tracks_file, cam_name, line):
    """
    Uses the track data to get the crossing information for a track file

        Args:
            tracks_file (str): directory location of the pickle file with the tracks
            cam_name (str): name of the camera as it appears in the directory
            line (int): x value of the vertical line used to determine crossing
        Returns:
            rl (dict): gives the fish tag number with time(s) that it crossed the line
                keys: fish tag number
                values (list): list of times the fish crossed the line
    """
    tracks = pickle.load(open(tracks_file, "rb"))
    lr = crossing_times(tracks, line, True)
    rl = crossing_times(tracks, line, False)
    for tag, lr_times in lr.items():
        if tag in rl:
            rl_times = rl[tag]
            if min(lr_times) < min(rl_times) and max(lr_times) > max(rl_times):
                rl.pop(tag)
    return rl


def empirical_tracks_processing(tracks_file, cam_name, t_vals, line, new_mid):
    """
    Processes the tracks to return aggregate totals of fish every time step

        Args:
            tracks_file (str): directory location of the pickle file with the tracks
            cam_name (str): name of camera as it appears in the directory
            t_vals (numpy array): numpy linspace array of the time range with the respective time steps
            line (int): x value of the vertical line used to determine crossing
            new_mid (float): time location (hours) of the new middle of the data so the tracks line up
        Returns:
            raw_time_totals (list): same size as t_vals so that the ith entry corresponds to the total fish that have crossed after i time steps
            hist_data (list): list of the time steps where fish crossing with corresponding multiplicity
    """
    rl = empirical_data_extraction(tracks_file, cam_name, line)
    hist_data = []
    times = np.array([np.mean(times) / 3600 for times in sorted(rl.values())])
    times = times[times >= 15]
    times = [time - (times[round(len(times) / 2)] - new_mid) for time in times]
    for time in times:
        for i, t in enumerate(t_vals[1:]):
            if t_vals[i] < time <= t:
                hist_data.append(t)
    raw_time_totals = []
    total = 0
    for t in t_vals:
        for time in hist_data:
            if t == time:
                total += 1
        raw_time_totals.append(total)
    return raw_time_totals, hist_data


cam_names = [
    "cam01-bear_outflow",
    "cam07-grass_outflow",
    "cam09-trail_outflow",
]


def empirical_data_return(t_vals, line, new_mid, cam_names=cam_names):
    """
    Uses empirical_tracks_processing functionf for all tracks on 08-07 and 08-10 to return the data for the histogram and of the totals

        Args:
            cam_names (list): list of all of the camera names to be considered
            line (int): x value of the vertical line to determine crossing
            new_mid (float): time value (hours) of the new middle of the data so the data all lines up
        Returns:
            raw_empirical_time_totals (list): list of the raw_time_totals lists for the tracks, as returned by empirical_tracks_processing
            empirical_hist_data (list): list of the hist_data as returned by the empirical_tracks_processsing function
    """
    raw_empirical_time_totals = []
    empirical_hist_data = []
    for cam in cam_names:
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        for tracks_file in [tracks_files[3], tracks_files[6]]:
            raw_time_totals, hist_data = empirical_tracks_processing(
                tracks_file, cam, t_vals, line, new_mid
            )
            raw_empirical_time_totals.append(raw_time_totals)
            empirical_hist_data += hist_data
            # empirical_hist_data.append(hist_data)
    return raw_empirical_time_totals, empirical_hist_data


# ASOCIAL MODEL FUNCTIONS


def asocial_poisson_draw(t_vals, total_fish):
    """
    Based on a Gaussian distribution, performs random draws from a Poisson distribution to get the fabricated data

        Args:
            t_vals (numpy array): an array of np.linspace set up with the desired time step size
            total_fish (int): total number of desired fish in the day
        Returns:
            pois_nums (list): list of the same length as t_vals giving how many fish go though in each time interval
    """
    gauss_vals = gauss_prob(t_vals)
    pois_nums = []
    for i, t in enumerate(t_vals):
        if i == 0:
            pois_nums.append(stats.poisson.rvs(gauss_vals[i] * total_fish))
        else:
            pois_nums.append(
                stats.poisson.rvs((gauss_vals[i] - gauss_vals[i - 1]) * total_fish)
            )
    return pois_nums


def asocial_data_return(t_vals, total_fish, iterations):
    """
    Uses the asocial_poisson_draw at each time step to determine how many fish crossed in the asocial model

        Args:
            t_vals (numpy array): numpy linspace array of desired size with desired time steps
            total_fish (int): total number of desired fish for the day
            iterations (int): number of times to repeat the process for verification
        Returns:
            raw_asocial_time_totals (list): list of time totals for each iteration
            asocial_hist_data (list): list of the time steps where fish cross to be used in a histogram later
    """
    raw_asocial_time_totals = []
    asocial_hist_data = []
    for _ in range(iterations):
        pois_nums = asocial_poisson_draw(t_vals, total_fish)
        time_totals = []
        total = 0
        for t, step_total in zip(t_vals, pois_nums):
            asocial_hist_data += [t] * step_total
            total += step_total
            time_totals.append(total)
        raw_asocial_time_totals.append(time_totals)
    return raw_asocial_time_totals, asocial_hist_data


# this function uses the pre-determined run sizes from the empricial data to get the simluated runs
def asocial_data_return_true_size(t_vals):
    """
    Uses the asocial_poisson_draw at each time step to determine how many fish should be crossing in the asocial model

        Args:
            t_vals (numpy array): numpy linspace array of desired size with desired time steps
        Returns:
            raw_asocial_time_totals (list): list of the time totals for each iteration
            asocial_hist_data (list): list of the time stemps wehre fish cross to be used in a histogram below
    """
    raw_asocial_time_totals = []
    asocial_hist_data = []
    for total_fish in [37, 51, 144, 57, 103, 159]:
        pois_nums = asocial_poisson_draw(t_vals, total_fish)
        time_totals = []
        hist_data = []
        total = 0
        for t, step_total in zip(t_vals, pois_nums):
            hist_data += [t] * step_total
            total += step_total
            time_totals.append(total)
        for i, total in enumerate(time_totals):
            if total >= round(max(time_totals) / 2) and time_totals[i - 1] < round(
                max(time_totals) / 2
            ):
                mid = t_vals[i]
                # 105 ONLY works when step size is 120 seconds
                index_diff = i - list(t_vals).index(18.5)
        # hist_data = sorted(hist_data)
        hist_data = [time - (mid - 18.5) for time in hist_data]
        asocial_hist_data += hist_data
        if index_diff < 0:
            time_totals = [0] * -index_diff + time_totals[:index_diff]
        elif index_diff > 0:
            time_totals = time_totals[index_diff:] + [max(time_totals)] * index_diff
        raw_asocial_time_totals.append(time_totals)

    return raw_asocial_time_totals, asocial_hist_data


# SIMPLE SOCIAL MODEL FUNCTIONS


def poisson_pool_and_cascade(t_vals, total_fish, alpha, gamma):
    """
    Uses same distribution as asocial_poisson_draw to get a pool of fish and then simulates how many fish move on

        Args:
            t_vals (numpy array): an array using np.linspace so that time steps are of the desired length
            total_fish (int): total number of desired fish for the day
            alpha (float): probability of a fish leaving the staging pool
            gamma (float): probability of a fish following a fish that leaves the pool

        Returns:
            leaving_fish (list): list of list of fish leaving; same size as t_vals so that the ith entry corresponds to the ith entry of t_vals
    """
    leaving_fish = []
    W = 0
    gauss_vals = gauss_prob(t_vals)
    for i, t in enumerate(t_vals):
        if i == 0:
            E = stats.poisson.rvs(gauss_vals[i] * total_fish)
        else:
            E = stats.poisson.rvs((gauss_vals[i] - gauss_vals[i - 1]) * total_fish)
        W += E
        L = stats.binom.rvs(W, alpha)
        L_C = [L]
        while L_C[-1] > 0 and sum(L_C) < W:
            C = stats.binom.rvs(W - sum(L_C), 1 - (1 - gamma) ** L_C[-1])
            L_C.append(C)
        W -= sum(L_C)
        leaving_fish.append(sum(L_C))
    return leaving_fish


def social_data_return(t_vals, total_fish, alpha, gamma, iterations):
    """
    Uses poisson_pool_and_cascade to simulate results for the social model

        Args:
            t_vals (numpy array): a np linspace array set up for the range of time and time steps desired
            total_fish (int): total number of fish desired for the day
            alpha (float): probability of a fish leaving the staging pool
            gamma (float): probability of a fish following a fish that left the staging pool
            interations (int): number of times to repeat the process
        Returns:
            raw_social_time_totals (list): list of lists of fish that cross in each time step, with the ith entry corresponding to the ith time step
            social_hist_data (list): list of the time steps in which fish cross with the respective multiplicity
    """
    raw_social_time_totals = []
    social_hist_data = []
    for _ in range(iterations):
        leaving_fish = poisson_pool_and_cascade(t_vals, total_fish, alpha, gamma)
        time_totals = []
        total = 0
        for t, step_total in zip(t_vals, leaving_fish):
            social_hist_data += [t] * step_total
            total += step_total
            time_totals.append(total)
        raw_social_time_totals.append(time_totals)
    return raw_social_time_totals, social_hist_data


# this function uses the pre-determined run sizes of the empirical data to get 6 runs
def social_data_return_true_size(t_vals, alpha, gamma):
    """
    Uses poisson_pool_and_cascade to simulate results for the social model

        Args:
            t_vals (numpy array): a np.linspace array set up for the range of time and time steps desired
            alpha (float): probability of a fish leaving the staging pool
            gamma (float): probability of a fish following a fish that left the staging pool
        Returns:
            raw_social_time_totals (list): list of lists of the number of fish that cross in each time step, with the ith entry corresponding to the ith time step
            social_hist_data (list): list of the time steps in which fish cross with the respective multiplicity
    """
    raw_social_time_totals = []
    social_hist_data = []
    for total_fish in [37, 51, 144, 57, 103, 159]:
        leaving_fish = poisson_pool_and_cascade(t_vals, total_fish, alpha, gamma)
        time_totals = []
        hist_data = []
        total = 0
        for t, step_total in zip(t_vals, leaving_fish):
            hist_data += [t] * step_total
            total += step_total
            time_totals.append(total)
        for i, total in enumerate(time_totals):
            if total >= round(max(time_totals) / 2) and time_totals[i - 1] < round(
                max(time_totals) / 2
            ):
                mid = t_vals[i]
                index_diff = i - list(t_vals).index(18.5)
        hist_data = [time - (mid - 18.5) for time in hist_data]
        social_hist_data += hist_data
        if index_diff < 0:
            time_totals = [0] * -index_diff + time_totals[:index_diff]
        elif index_diff > 0:
            time_totals = time_totals[index_diff:] + [max(time_totals)] * index_diff
        raw_social_time_totals.append(time_totals)
    return raw_social_time_totals, social_hist_data


# this next function is specifically just for making the 4th and 5th subplots of the full figure
def plot5_y_vals_return(model_totals, t_vals):
    """
    Gives the y_vals that will be used for the SD over time plot (plot e)

        Args:
            model_totals (list): list of totals for each day/run, as returned from the data return function for each model
            t_vals (numpy array): np.linspace array of the time mesh
        Returns:
            model_y_vals (list): list of standard deviations, with the ith entry corresponding to the ith time step
    """
    new_totals = []
    for totals in model_totals:
        new_totals.append([total / totals[-1] for total in totals])
    model_y_vals = []
    for j in range(len(t_vals)):
        val_list = [totals[j] for totals in new_totals]
        model_y_vals.append(np.std(val_list))
    return model_y_vals
