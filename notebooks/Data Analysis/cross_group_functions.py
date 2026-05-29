from dbscan1d.core import DBSCAN1D
import numpy as np


def fish_mid(track):
    ''' 
    Returns midpoint of each fish at each time in each file

        Args:
            track (dict): gives fish with corresponding time stamps and location
                keys: tag number for fish
                values (dict): gives time stamps and location and size of bounding box
                    keys: time stamps (frame number)
                    values (list): top left coordinates of bounding box followed by the width and height of the box
        Returns:
            track (dict): same format as the parameter, only with xmidpoint for each bounding box
                keys: tag number for fish
                values (dict): gives time stamps and x midpoint of bounding box
                    keys: time stamps
                    values: x midpoint of bounding box
    '''
    for tag in track:
        for time in track[tag]:
            if len(track[tag][time]) == 4:
                x_min, y_min, width, height = track[tag][time]
                track[tag][time][0] = x_min + width/2
                # track[tag][time][1] = y_min + height/2
                track[tag][time] = track[tag][time][0].tolist()
    return track


def crossing_times(tag, line, leftright):
    ''' 
    Returns the crossing time for a specified fish

        Parameters:
            tag (dict): gives coordinate for fish at respective time stamps
                keys: time stamp
                values: x midpoint for bounding box (from fish_mid function)
            line (int): x-value of vertical line for crossing
        Returns:
            times (list): list of frame count number to give times that fish crossed line
    '''
    times = []
    for time in tag:
        if time+1 in tag:
            if leftright == True:
                if tag[time] < line:
                    if tag[time+1] >= line:
                        times.append(time)
            else:
                if tag[time] >= line:
                    if tag[time+1] < line:
                        times.append(time)
    if times != []:
        return times


def file_fish_cross(tracks, line):
    ''' 
    Returns the files in which fish cross a vertical line

        Parameters:
            tracks (dict): gives file number, fish tag, time, and location for files
                keys: file number
                values (dict): fish tag numbers with respective time and x midpoint location
                    As given by fish_mid function.
                    keys: fish tag number
                    values (dict): x midpoint for each frame number
                        keys: frame number to give time
                        values: x-midpoint of boudning box
            line (int): x-value of vertical line for measuring crossing
        Returns:
            lr (dict): gives crossing times for each fish going left to right.
                keys: file number
                values (dict): gives fish and crossing time for each file 
                    keys: fish tag number
                    values (list): frame number for crossing times
            rl (dict):gives crossing times for each fish going right to left.
                keys: file number
                values (dict): gives fish and crossing time for each file 
                    keys: fish tag number
                    values (list): frame number for crossing times
    '''
    lr = {}
    rl = {}
    for file in tracks:
        for tag in tracks[file]:
            lr_crossing = crossing_times(
                tracks[file][tag], line=line, leftright=True)
            if lr_crossing:
                if file not in lr:
                    lr[file] = {}
                lr[file][tag] = lr_crossing
            rl_crossing = crossing_times(
                tracks[file][tag], line=line, leftright=False)
            if rl_crossing:
                if file not in rl:
                    rl[file] = {}
                rl[file][tag] = rl_crossing
    return lr, rl


def dbscan_dist(direction_data, proximity):
    '''
    Uses DBSCAN1 to get fish groups of given direction data with specified proximity

        Args:
            direction_data (dict): fish tags with their respective crossing times for an entire day
                keys: fish tag number
                values: list of crossing times in seconds
            proximity (int): specifies how close in time classifies a group
        Returns:
            groups (list): list of all groups with their respective fish
    '''
    fishes = []
    all_times = []
    for fish, times in direction_data.items():
        for time in times:
            fishes.append(fish)
            all_times.append(time)
    all_times = np.array(all_times)
    dbs = DBSCAN1D(eps=proximity, min_samples=2)
    labels = dbs.fit_predict(all_times)
    groups = []
    for label in set(labels):
        if label == -1:
            continue
        group = []
        for i in range(len(labels)):
            if labels[i] == label:
                if fishes[i] not in group:
                    group.append(fishes[i])
        if len(group) > 1:
            groups.append(group)
    return groups
