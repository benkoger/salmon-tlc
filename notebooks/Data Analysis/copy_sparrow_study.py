import os
import glob
from scipy import stats
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import numpy as np

from cross_group_functions import tracks_crossing_info, dbscan_time_groups

cam_names = {
    "cam01-bear_outflow": False,
    "cam07-grass_outflow": False,
    "cam09-trail_outflow": False,
    "cam0610-big_outflow": True,
}


def sparrow_study_distr(proximity, xlog, save, line=1250, cam_names=cam_names):
    all_sizes = []
    raw_sizes = []
    for cam, leftright in cam_names.items():
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        for tracks_file in [tracks_files[3], tracks_files[6]]:
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            groups = dbscan_time_groups(crossing_data, proximity)
            group_fish = [tag for group in groups for tag in group]
            for tag in crossing_data:
                if tag not in group_fish:
                    groups.append([tag])

            all_sizes += [len(group) for group in groups for _ in range(len(group))]
            raw_sizes += [len(group) for group in groups]

    Np = np.mean(all_sizes)
    n = len(raw_sizes)
    M = sum(raw_sizes)
    S = sum([np.log(N) for N in raw_sizes])
    sizes = np.array(list(set(raw_sizes)))
    x_vals = np.log(sizes) if xlog else sizes
    probs = np.array([raw_sizes.count(size) / len(raw_sizes) for size in sizes])
    plt.scatter(x_vals, np.log(probs), c="black", marker="x", label="empirical")


    # Poisson distribution
    lam_hat = M / n # np.mean(raw_sizes)
    pois = lambda N: lam_hat ** N * np.exp(-lam_hat) / np.prod(range(1, N + 1))
    poisy = [pois(N) for N in sizes]
    poisy = [10 ** (-12) if val <= 0 else val for val in poisy]
    plt.plot(x_vals, np.log(poisy), label="Poisson")

    # geometric distribution
    p_hat = 1 / lam_hat # = n / M
    geo = lambda N: p_hat * (1 - p_hat) ** (N - 1)
    plt.plot(x_vals, np.log(geo(sizes)), label="geometric", linestyle="--", color="black")

    # power law
    a_hat = 1 + n / S
    k_hat = 1 / sum([N ** (-a_hat) for N in sizes])
    power = lambda N: k_hat * N ** (-a_hat)
    plt.plot(x_vals, np.log(power(sizes)), label="power law", linestyle=":", color="black")

    # logarithmic distribution
    Npjs = np.linspace(Np - 2.5, Np + 1.5, 81)
    vals = [-n * np.log(np.log(Npj)) - S + M * np.log(1 - 1 / Npj) for Npj in Npjs]
    Np_hat = Npjs[np.argmax(vals)]
    log_distr = lambda N: 1 / np.log(Np_hat) * 1 / N * (1 - 1 / Np_hat) ** N
    plt.plot(x_vals, np.log(log_distr(sizes)), label="logarithmic", color="black")

    # truncated power law
    ajs = np.linspace(0.45, 2, 156)
    cjs = np.linspace(0.5, 1.5, 101)
    def k_hat_func(a, c):
        return 1 / sum([N ** (-a) * c ** N for N in sizes])
    vals = []
    acs = []
    for aj in ajs:
        for cj in cjs:
            vals.append(n * np.log(k_hat_func(aj, cj)) - aj * S + M * np.log(cj))
            acs.append([aj, cj])
    a_hat, c_hat = acs[np.argmax(vals)]
    trunc_power = lambda N: k_hat_func(a_hat, c_hat) * N ** (-a_hat) * c_hat ** N
    plt.plot(x_vals, np.log(trunc_power(sizes)), label="truncated power", color="red")

    plt.legend()
    if xlog:
        plt.xlabel("log(Group size)")
    else:
        plt.xlabel("Group size")
    plt.ylabel("log(Probability)")
    plt.xlim(0, )
    plt.ylim(-7, 0)
    if save:
        one = "log" if xlog else "linear"
        plt.savefig(f"sparrow-study-comparison_{"log" if xlog else "linear"}-log-{proximity}.jpg", dpi=500)
    plt.show()


def sparrow_study_aic(proximity, line=1250, cam_names=cam_names):
    all_sizes = []
    raw_sizes = []
    for cam, leftright in cam_names.items():
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        for tracks_file in [tracks_files[3], tracks_files[6]]:
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            groups = dbscan_time_groups(crossing_data, proximity)
            group_fish = [tag for group in groups for tag in group]
            for tag in crossing_data:
                if tag not in group_fish:
                    groups.append([tag])
            all_sizes += [len(group) for group in groups for _ in range(len(group))]
            raw_sizes += [len(group) for group in groups]

    Np = np.mean(all_sizes)
    n = len(raw_sizes)
    M = sum(raw_sizes)
    S = sum(np.log(raw_sizes))
    sizes = np.array(list(set(raw_sizes)))
    probs = np.array([raw_sizes.count(size) / len(raw_sizes) for size in sizes])
    aics = []

    # Poisson distribution
    lam_hat = M / n
    def pois_l(lam):
        return - lam * n + M * np.log(lam) - sum([np.log(np.prod(range(1, N + 1))) for N in sizes])
    aics.append(2 * 1 - 2 * pois_l(lam_hat))
    
    # Geometric distribution
    p_hat = 1 / lam_hat
    def geo_l(p):
        return n * np.log(p) + (M - n) * np.log(1 - p)
    aics.append(2 * 1 - 2 * geo_l(p_hat))

    # Power law
    a_hat = 1 + n / S
    def p_l(a):
        return - n * np.log(sum(N ** (-a) for N in sizes)) - a * S
    aics.append(2 * 1 - 2 * p_l(a_hat))

    # Log distribution
    Npjs = np.linspace(Np - 2.5, Np + 1.5)
    vals = [-n * np.log(np.log(Npj)) - S + M * np.log(1 - 1 / Npj) for Npj in Npjs]
    aics.append(2 * 1 - 2 * max(vals))

    # Truncated power law
    ajs = np.linspace(0.45, 2, 156)
    cjs = np.linspace(0.5, 1.5)
    def k_hat_func(a, c):
        return 1 / sum([N ** (-a) * c ** N for N in sizes])
    vals = []
    acs = []
    for aj in ajs:
        for cj in cjs:
            vals.append(n * np.log(k_hat_func(aj, cj)) - aj * S + M * np.log(cj))
            acs.append([aj, cj])
    aics.append(2 * 2 - 2 * max(vals))
    
    min_aic = min(aics[1:])
    names = ["Poisson", "Geometric", "Power law", "logarithmic", "Truncated Power law"]
    aics = [aic - min_aic for aic in aics]
    aics = dict(zip(names, aics))
    print(np.array(aics))

    # print(chi2)


def sparrow_study_chi(proximity, line=1250, cam_names=cam_names):
    all_sizes = []
    raw_sizes = []
    for cam, leftright in cam_names.items():
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        for tracks_file in [tracks_files[3], tracks_files[6]]:
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            groups = dbscan_time_groups(crossing_data, proximity)
            group_fish = [tag for group in groups for tag in group]
            for tag in crossing_data:
                if tag not in group_fish:
                    groups.append([tag])
            all_sizes += [len(group) for group in groups for _ in range(len(group))]
            raw_sizes += [len(group) for group in groups]

    chi2 = []
    Np = np.mean(all_sizes)
    n = len(raw_sizes)
    M = sum(raw_sizes)
    S = sum(np.log(raw_sizes))
    sizes = np.array(list(set(raw_sizes)))
    probs = np.array([raw_sizes.count(size) / len(raw_sizes) for size in sizes])
    new_probs = [sum(probs[3*i: 3*(i+1)]) for i in range(5)]
    new_probs.append(sum(probs[15:]))

    # Poisson distribution
    lam_hat = M / n
    pois = lambda N: lam_hat ** N * np.exp(-lam_hat) / np.prod(range(1, N + 1))
    poisy = [pois(N) for N in sizes]
    # x = range(-10, 10)
    # y = [pois(i) for i in x]
    # plt.plot(x, y)
    # plt.show()
    # return
    new_poisy = [sum(poisy[3*i:3*(i+1)]) for i in range(5)]
    new_poisy.append(sum(poisy[15:]))
    new_poisy = np.array(new_poisy)
    new_poisy = new_poisy / sum(new_poisy)
    chi2.append(stats.chisquare(new_probs, f_exp=new_poisy, ddof=1))

    # Geometric distribution
    p_hat = 1 / lam_hat
    geo = lambda N: p_hat * (1 - p_hat) ** (N - 1)
    geoy = geo(sizes)
    new_geoy = [sum(geoy[3*i:3*(i+1)]) for i in range(5)]
    new_geoy.append(sum(geoy[15:]))
    new_geoy = np.array(new_geoy)
    new_geoy = new_geoy / sum(new_geoy)
    chi2.append(stats.chisquare(new_probs, f_exp=new_geoy, ddof=1))

    # Power law
    a_hat = 1 + n / S
    k_hat = 1 / sum([N ** (-a_hat) for N in sizes])
    power = lambda N: k_hat * N ** (-a_hat)
    powery = power(sizes)
    new_powery = [sum(powery[3*i: 3*(i+1)]) for i in range(5)]
    new_powery.append(sum(powery[15:]))
    new_powery = np.array(new_powery)
    new_powery = new_powery / sum(new_powery)
    chi2.append(stats.chisquare(new_probs, f_exp=new_powery, ddof=2))

    # log distribution
    Npjs = np.linspace(Np - 2.5, Np + 1.5)
    vals = [-n * np.log(np.log(Npj)) - S + M * np.log(1 - 1 / Npj) for Npj in Npjs]
    Np_hat = Npjs[np.argmax(vals)]
    log_distr = lambda N: 1 / np.log(Np_hat) * 1 / N * (1 - 1 / Np_hat) ** N
    logy = log_distr(sizes)
    new_logy = [sum(logy[3*i:3*(i+1)]) for i in range(5)]
    new_logy.append(sum(logy[15:]))
    new_logy = np.array(new_logy)
    new_logy = new_logy / sum(new_logy)
    chi2.append(stats.chisquare(new_probs, f_exp=new_logy, ddof=1))

    # Truncated power law
    ajs = np.linspace(0.45, 2, 156)
    cjs=np.linspace(0.5, 1.5, 101)
    def k_hat_func(a, c):
        return 1 / sum([N ** (-a) * c ** N for N in sizes])
    vals = []
    acs = []
    for aj in ajs:
        for cj in cjs:
            vals.append(n * np.log(k_hat_func(aj, cj)) - aj * S + M * np.log(cj))
            acs.append([aj, cj])
    a_hat, c_hat = acs[np.argmax(vals)]
    trunc_power = lambda N: k_hat_func(a_hat, c_hat) * N ** (-a_hat) * c_hat ** N
    truncy = trunc_power(sizes)
    new_truncy = [sum(truncy[3*i:3*(i+1)]) for i in range(5)]
    new_truncy.append(sum(truncy[15:]))
    chi2.append(stats.chisquare(new_probs, f_exp=new_truncy, ddof=2))

    names = ["Poisson", "Geometric", "Power law", "logarithmic", "Truncated Power law"]

    print(dict(zip(names, chi2)))


def sparrow_study_r(proximity, line=1250, cam_names=cam_names):
    all_sizes = []
    raw_sizes = []
    for cam, leftright in cam_names.items():
        folder = f"/project/uwyo-0003/salmon-tlc/processing/tracks-with-true-times_06-18-2026/{cam}"
        tracks_files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        for tracks_file in [tracks_files[3], tracks_files[6]]:
            crossing_data = tracks_crossing_info(tracks_file, line, leftright)
            groups = dbscan_time_groups(crossing_data, proximity)
            group_fish = [tag for group in groups for tag in group]
            for tag in crossing_data:
                if tag not in group_fish:
                    groups.append([tag])
            all_sizes += [len(group) for group in groups for _ in range(len(group))]
            raw_sizes += [len(group) for group in groups]

    r2 = []
    Np = np.mean(all_sizes)
    n = len(raw_sizes)
    M = sum(raw_sizes)
    S = sum(np.log(raw_sizes))
    sizes = np.array(list(set(raw_sizes)))
    # print(len(set(raw_sizes)))
    # print(set(raw_sizes))
    probs = np.array([raw_sizes.count(size) / len(raw_sizes) for size in sizes])

    # Poisson distribution
    lam_hat = M / n
    pois = lambda N: lam_hat ** N * np.exp(-lam_hat) / np.prod(range(1, N + 1))
    poisy = [pois(N) for N in sizes]
    r2.append(r2_score(probs, poisy))
    
    # Geometric distribution
    p_hat = 1 / lam_hat
    geo = lambda N: p_hat * (1 - p_hat) ** (N - 1)
    r2.append(r2_score(probs, geo(sizes)))

    # Power law
    a_hat = 1 + n / S
    k_hat = 1 / sum([N ** (-a_hat) for N in sizes])
    power = lambda N: k_hat * N ** (-a_hat)
    r2.append(r2_score(probs, power(sizes)))

    # log distribution
    Npjs = np.linspace(Np - 2.5, Np + 1.5)
    vals = [-n * np.log(np.log(Npj)) - S + M * np.log(1 - 1 / Npj) for Npj in Npjs]
    Np_hat = Npjs[np.argmax(vals)]
    log_distr = lambda N: 1 / np.log(Np_hat) * 1 / N * (1 - 1 / Np_hat) ** N
    r2.append(r2_score(probs, log_distr(sizes)))

    # Truncated power law
    ajs = np.linspace(0.45, 2, 156)
    cjs=np.linspace(0.5, 1.5, 101)
    def k_hat_func(a, c):
        return 1 / sum([N ** (-a) * c ** N for N in sizes])
    vals = []
    acs = []
    for aj in ajs:
        for cj in cjs:
            vals.append(n * np.log(k_hat_func(aj, cj)) - aj * S + M * np.log(cj))
            acs.append([aj, cj])
    a_hat, c_hat = acs[np.argmax(vals)]
    trunc_power = lambda N: k_hat_func(a_hat, c_hat) * N ** (-a_hat) * c_hat ** N
    r2.append(r2_score(probs, trunc_power(sizes)))

    names = ["Poisson", "Geometric", "Power law", "logarithmic", "Truncated Power law"]
    r2s = dict(zip(names, r2))
    print(r2s)
