import easyocr
import cv2
from imutils.video import FileVideoStream
from datetime import datetime, timedelta

reader = easyocr.Reader(['en'])


def time_converter(video_file, x_fac, y_fac):
    '''
    Gives the time of day for each of the frames in a video

        Args:
            video_file: mp4 file of the salmon
            x_fac (int): specifies the scaling factor for the x-axis
            y_fac (int): specifies the scaling factor for the y-axis
        Returns:
            output (list): list of times in seconds, with index + 1 corresponding to the frame number
    '''
    fvs = FileVideoStream(video_file).start()
    output = []
    frame_count = 0
    while fvs.more():
        frame = fvs.read()
        if frame is None:
            break
        frame_count += 1
        frame = frame[20:76, 1443:1520]
        if frame_count == 1:
            # year, month, and day are arbitrary
            t1 = datetime(2025, 8, 1, int(video_file[82:84]), int(
                video_file[84:86]), int(video_file[86:88]))
            output.append(t1)
            continue
        frame = cv2.resize(frame, None, fx=x_fac, fy=y_fac)
        secs = str(t1.second).zfill(2) + \
            str((t1 + timedelta(seconds=1)).second).zfill(2)
        second = reader.readtext(frame, allowlist=secs, detail=0)
        if second:
            if second[0] == secs[2:]:
                if frame_count > 15:
                    if output[-1] == output[-14]:
                        t1 += timedelta(seconds=1)
                else:
                    t1 += timedelta(seconds=1)
        else:
            if frame_count >= 16:
                if output[-1] == output[-15]:
                    t1 += timedelta(seconds=1)
        output.append(t1)
    return output


def time_replace(direction_data_file, output):
    '''
    Replaces frame number with corresponding time value in seconds

        Args:
            direction_data_file (dict): gives list of crossing times for each fish that crossed the line
                As returned for each file from file_fish_cross function
                keys: fish tag number
                values (list): list of frame number for crossing times
            output (list): list of time stamps in seconds for the file
                As returned by time_converter() function
        Returns:
            new_data (dict): same structure as direction_data_file but with time values in seconds
    '''
    new_data = {}
    for tag, times in direction_data_file.items():
        new_data[tag] = []
        for time in times:
            if output[time-1] not in new_data[tag]:
                stamp = output[time-1]
                new_data[tag].append(3600 * stamp.hour +
                                     60 * stamp.minute + stamp.second)
    return new_data


def direction_time_replace(lr, rl, video_files, x_fac, y_fac):
    '''Uses time_replace function for all files in the crossing data

        Args:
            lr (dict): gives files where fish crossed the line left to right
                As returned from file_fish_cross fuction
                keys: file number
                values (dict): fish tag number and respective crossing times
                    keys: fish tag number
                    values: list of crossing times
            rl (dict): gives files where fish crossed the line right to left
                As returned from file_fish_cross function
                keys: file number
                values (dict): fish tag number and respective crossing times
                    keys: fish tag number
                    values: list of crossing times
            video_files (list): video files corresponding to the tracks
            x_fac (int): scale factor for x-axis of image
            y_fac (int): scale factor for y-axis of image
        Returns:
            lr (dict): same structure as parameter but with times in seconds
            rl (dict): same structure as parameter but with times in seconds
    '''
    files = []
    for file in lr:
        files.append(file)
        output = time_converter(video_files[file], x_fac, y_fac)
        lr[file] = time_replace(lr[file], output)
        if file in rl:
            rl[file] = time_replace(rl[file], output)
    for file in rl:
        if file in files:
            continue
        files.append(file)
        output = time_converter(video_files[file], x_fac, y_fac)
        rl[file] = time_replace(rl[file], output)
    return lr, rl


def full_day(lr, rl):
    '''
    Combines direction data from all of the files for a day

        Args:
            lr (dict): gives fish and their crossing times from left to right in seconds for each file with a fish that crosses the line
                keys: file number
                values (dict): gives fish tag and its crossing times
                    keys: fish tag number
                    values (list): list of crossing times in seconds
            rl (dict): gives fish and their crossing times from right to left in seconds for each file containing a fish that crosses the line
                keys: file number
                values: fish tag with its crossing times
                    keys: fish tag number
                    values: list of crossing times in seconds
        Returns:
            full_lr (dict): fish with their respective crossing times in seconds
                keys: fish tag number
                values: list of crossing times in seconds
            full_rl (dict): fish with their respective crossing times in seconds
                keys: fish tag number
                values: list of crossing times in seconds
    '''
    full_lr = {}
    full_rl = {}
    for file in lr:
        for tag, times in lr[file].items():
            full_lr[tag] = times
    for file in rl:
        for tag, times in rl[file].items():
            full_rl[tag] = times
    return full_lr, full_rl
