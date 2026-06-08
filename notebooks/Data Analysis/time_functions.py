import easyocr
import cv2
from imutils.video import FileVideoStream
from datetime import datetime, timedelta

reader = easyocr.Reader(["en"])


def time_converter(video_file, x_fac, y_fac, folder_len):
    """
    Gives the time of day for each of the frames in a video

        Args:
            video_file: mp4 file of the salmon
            x_fac (int): specifies the scaling factor for the x-axis
            y_fac (int): specifies the scaling factor for the y-axis
            folder_len (int): length of direction for folder to accurately extract initial time from file name
        Returns:
            output (list): list of times in seconds, with index + 1 corresponding to the frame number
    """
    crop_dict = {
        "20250804": [1426, 1510],
        "20250806": [1435, 1519],
        "20250808": [1460, 1542],
        "20250811": [1426, 1510],
    }
    output = []
    frame_count = 0
    date_info = video_file[folder_len + 11 : folder_len + 26]
    if date_info[:8] in crop_dict:
        x_s = crop_dict[date_info[:8]]
    else:
        x_s = [1442, 1524]
    one_sec = timedelta(seconds=1)
    fvs = FileVideoStream(video_file).start()
    while fvs.more():
        frame = fvs.read()
        if frame is None:
            break
        frame_count += 1
        if frame_count == 1:
            # year, month, and day are arbitrary
            time = datetime(
                int(date_info[:4]),
                int(date_info[4:6]),
                int(date_info[6:8]),
                int(date_info[9:11]),
                int(date_info[11:13]),
                int(date_info[13:]),
            )
            output.append(time)
            continue
        frame = frame[20:76, x_s[0] : x_s[1]]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, None, fx=x_fac, fy=y_fac)
        secs = str(time.second).zfill(2) + str((time + one_sec).second).zfill(2)
        second = reader.readtext(
            frame, allowlist=secs, detail=0
        )  # , text_threshold=0.6, low_text=0.3)
        if second:
            if second[0] == secs[2:]:
                if frame_count > 15:
                    if output[-1] == output[-14]:
                        time += one_sec
                else:
                    time += one_sec
        else:
            if frame_count >= 16:
                if output[-1] == output[-15]:
                    time += one_sec
        output.append(time)
    return output


def time_replace(direction_data_file, output):
    """
    Replaces frame number with corresponding time value in seconds

        Args:
            direction_data_file (dict): gives list of crossing times for each fish that crossed the line
                As returned for each file from file_fish_cross function
                keys: fish tag number
                values (list): list of frame number for crossing times
            output (list): list of time stamps in seconds for the file
                As returned by time_converter() function where the index equals the frame count - 1
        Returns:
            new_data (dict): same structure as direction_data_file but with time values in seconds
    """
    new_data = {}
    for tag, times in direction_data_file.items():
        new_data[tag] = []
        for time in times:
            if output[time - 1] not in new_data[tag]:
                stamp = output[time - 1]
                new_data[tag].append(
                    3600 * stamp.hour + 60 * stamp.minute + stamp.second
                )
    return new_data


def direction_time_replace(lr, rl, video_files, x_fac, y_fac):
    """
    Uses time_replace function for all files in the crossing data

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
    """
    file_list = []
    for file in lr:
        file_list.append(file)
        output = time_converter(video_files[file], x_fac, y_fac)
        lr[file] = time_replace(lr[file], output)
        if file in rl:
            rl[file] = time_replace(rl[file], output)
    for file in rl:
        if file in file_list:
            continue
        file_list.append(file)
        output = time_converter(video_files[file], x_fac, y_fac)
        rl[file] = time_replace(rl[file], output)
    return lr, rl


def full_day(lr, rl):
    """
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
    """
    full_lr = {}
    full_rl = {}
    for file_data in lr.values():
        full_lr.update(file_data)
    for file_data in rl.values():
        full_rl.update(file_data)
    return full_lr, full_rl
