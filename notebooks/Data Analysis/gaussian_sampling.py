import numpy as np
from scipy import stats


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