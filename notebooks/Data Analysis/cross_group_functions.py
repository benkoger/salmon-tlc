from dbscan1d.core import DBSCAN1D
import numpy as np
import pickle
from collections import defaultdict


def fish_mid(tracks):
    """
    Returns x and y values for the midpoint of each fish at each time in each file

        Args:
            tracks (dict): for each file gives fish tag with corresponding time stamps and location of bounding box
                keys: file number
                values (dict):gives fish with corresponding time stamps and location
                    keys: tag number for fish
                    values (dict): gives time stamps with corresponding location and size of bounding box
                        keys: time stamps (frame number)
                        values (numpy array): top left coordinates of bounding box (x,y) followed by the width and height of the box
        Returns:
            tracks (dict): same format as the parameter, but values for the box location are a list of the x andy midpoints of the box
                keys: file number
                values (dict):gives fish with corresponding time stamps and location
                    keys: tag number for fish
                    values (dict): gives time stamps with corresponding location and size of bounding box
                        keys: time stamps (frame number)
                        values (list): x and y midpoints of the bounding box
    """
    for file, file_data in tracks.items():
        for tag, times_data in file_data.items():
            for time, box_loc in times_data.items():
                if len(box_loc) == 4:
                    x_min, y_min, width, height = box_loc
                    x_mid, y_mid = x_min + width / 2, y_min + height / 2
                    tracks[file][tag][time] = [x_mid, y_mid]
    return tracks


def fish_mid_no_file(tracks):
    """
    Adaptation of the midpoint function for tracks with no file distinction. Gives tracks with x and y midpoints for each bounding box instead of the location and size

        Args:
            tracks (dict): gives fish tag with corresponding time stamps and locations of bounding boxes
                keys: tag number for fish
                values (dict): gives the time stamps with corresponding location and size of bounding boxes
                    keys: time stamps (total seconds)
                    values (numpy array): top left coordinate of bounding box (x,y) followed by the width and height of the box
        Returns:
            tracks (dict): same format as argument, but the values for the box location are the x and y midpoints of the bounding box.
                keys: tag number for fish
                values (dict): gives the time stamps with corresponding locations of bounding boxes
                    keys: time stamps (total seconds)
                    values (list): x midpoint and y midpoint of the bounding box
    """
    for tag, times_data in tracks.items():
        for time, box_loc in times_data.items():
            if len(box_loc) == 4:
                x_min, y_min, width, height = box_loc
                x_mid, y_mid = x_min + width / 2, y_min + height / 2
                tracks[tag][time] = [x_mid, y_mid]
    return tracks


def crossing_times(file_data, line, leftright):
    """
    Returns the crossing time for a specified fish

        Parameters:
            file_data (dict): gives fish for each file with their respective locations at the corresponding times, as returned by the fish_mid function
                keys: fish tag number
                values (dict): gives the location at each tiem the fish appears
                    keys: time (frame number)
                    values (list): list of x and y midpoint values for the bounding box of the fish
            line (int): x-value of vertical line to check crossing
            leftright (bool): determines if consider left to right or right to left crossing
        Returns:
            times (dict): for each fish gives frame count number for when the fish crossed the line
                keys: fish tag number
                values (list): list of times when the fish crossing the vertical line
    """
    times = defaultdict(list)
    for tag, tag_data in file_data.items():
        time_list = list(tag_data)
        # for time, loc in tag_data.items():
        for i, loc in enumerate(tag_data.values()):
            if i < len(time_list) - 1:
                # if time + 1 in tag_data: #maybe if time != max(tag_data):
                if leftright:
                    if loc[0] < line:
                        if tag_data[time_list[i + 1]][0] >= line:
                            times[tag].append(time_list[i])
                else:
                    if loc[0] >= line:
                        if tag_data[time_list[i + 1]][0] < line:
                            times[tag].append(time_list[i])
    return dict(times)


def file_fish_cross(tracks, line):
    """
    Returns the files in which fish cross a vertical line with the time values in which the fish cross the line

        Parameters:
            tracks (dict): gives file number, fish tag, time, and location for files (format as returned from fish_mid function)
                keys: file number
                values (dict): fish tag numbers with respective time and x midpoint location
                    As given by fish_mid function.
                    keys: fish tag number
                    values (dict): bounding box midpoint for each frame number
                        keys: frame number to give time
                        values(list): x and y midpoints of bounding box
            line (int): x-value of vertical line for measuring crossing
        Returns:
            lr (dict): gives crossing times for each fish going left to right.
                keys: file number
                values (dict): gives fish and crossing time for each file
                    keys: fish tag number
                    values (list): frame numbers for crossing times
            rl (dict):gives crossing times for each fish going right to left.
                keys: file number
                values (dict): gives fish and crossing time for each file
                    keys: fish tag number
                    values (list): frame numbers for crossing times
    """
    lr = {}
    rl = {}
    for file, file_data in tracks.items():
        lr_crossing = crossing_times(file_data, line, True)
        if lr_crossing:
            lr[file] = lr_crossing
        rl_crossing = crossing_times(file_data, line, False)
        if rl_crossing:
            rl[file] = rl_crossing
    return lr, rl


def dbscan_time_groups(direction_data, proximity, min_size=2):
    """
    Uses DBSCAN1D to get fish groups of given direction data with specified proximity

        Args:
            direction_data (dict): fish tags with their respective crossing times for an entire day
                keys: fish tag number
                values (list): list of crossing times in seconds
            proximity (int): specifies how close in time classifies a group
        Returns:
            groups (list): list of all groups with their respective fish
    """
    fishes = []
    all_times = []
    for fish, times in direction_data.items():
        for time in times:
            fishes.append(fish)
            all_times.append(time)
    all_times = np.array(all_times)
    dbs = DBSCAN1D(eps=proximity, min_samples=min_size)
    labels = dbs.fit_predict(all_times)
    groups = []
    for label in set(l for l in labels if l != -1):
        group = []
        for i in range(len(labels)):
            if labels[i] == label:
                if fishes[i] not in group:
                    group.append(fishes[i])
        if len(group) > 1:
            groups.append(group)
    return groups


def norm_groups(direction_data, tracks, proximity):
    """
    Gives fish groups of specified direction data based on specified proximity

        Args:
            direction_data (dict): for each file has each fish that crossed the vertical line with the time(s) the fish crossed
                keys: file number
                values (dict): gives fish with the times they crossed
                    keys: fish tag number
                    values (list): list of crossing times (frame count number)
            tracks (dict): for each file gives each fish as well as what time stamp and when they are on the frame, as returned by fish_mid function
                keys: file number
                values (dict): fish tag numbers with respective time and x midpoint location
                    As given by fish_mid function.
                    keys: fish tag number
                    values (dict): bounding box midpoint for each frame number
                        keys: frame number to give time
                        values(list): x and y midpoints of bounding box
            proximity (int): specifies how close a fish must be to another to be considered in a group
        Returns:
            groups (list): list of groups with the fish that are in it for the direction data
    """
    groups = []
    fish_list = []
    for file, file_data in direction_data.items():
        if len(file_data) == 1:
            continue
        for tag, times in file_data.items():
            for time in times:
                tag_loc = tracks[file][tag][time]
                for fish in file_data:
                    if fish == tag:
                        continue
                    if time not in tracks[file][fish]:
                        continue
                    diff = np.linalg.norm(
                        np.array(tag_loc) - np.array(tracks[file][fish][time])
                    )
                    if diff < proximity:
                        if tag not in fish_list:
                            if fish not in fish_list:
                                groups.append([tag, fish])
                                fish_list.append(tag)
                                fish_list.append(fish)
                            else:
                                for i in range(len(groups)):
                                    if fish in groups[i]:
                                        groups[i].append(tag)
                                        fish_list.append(tag)
    return groups


def tracks_crossing_info(tracks_file, line):
    """
    Uses the crossing_times function to get the crossing info for the specific tracks

        Args:
            tracks_file: directory location of the specific tracks file
            line (int): location of the vertical line to determine crossing
        Returns:
            rl (dict): gives the crossing times for each tag that crosses
                keys (int): tag number of fish
                values (list): list of respective crossing_times
    """
    tracks = pickle.load(open(tracks_file, "rb"))
    lr = crossing_times(tracks, 1250, True)
    rl = crossing_times(tracks, 1250, False)
    for tag, lr_times in lr.items():
        if tag in rl:
            rl_times = rl[tag]
            if min(lr_times) < min(rl_times) and max(lr_times) > max(rl_times):
                rl.pop(tag)
    return rl
