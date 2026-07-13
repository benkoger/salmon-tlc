import easyocr
import cv2
from imutils.video import FileVideoStream
from datetime import datetime, timedelta
import os
import pickle

import matplotlib.pyplot as plt

reader = easyocr.Reader(["en"])


def time_converter(video_file, x_fac=1.4, y_fac=1.4):
    """
    Gives the time of day for each of the frames in a video

        Args:
            video_file: mp4 file of the salmon
            x_fac (int): specifies the scaling factor for the x-axis
            y_fac (int): specifies the scaling factor for the y-axis
            folder_len (int): length of direction for folder to accurately
            extract initial time from file name
        Returns:
            output (list): list of times in seconds, with index + 1 corresponding to the frame number
    """
    vid_split = video_file.split("/")
    file_dir = f"/project/uwyo-0003/salmon-tlc/processing/time-stamp-extraction_07-08-2026/{vid_split[5]}/{vid_split[7]}"
    file_path = os.path.join(file_dir, f"{vid_split[-1][:-4]}-extracted_times.pkl")
    if os.path.exists(file_path):
        return

    crop_dict = {
        "20250804": [1425, 1511],  # used to be 1426, 1510
        "20250806": [1434, 1520],  # used to be 1435, 1519
        "20250808": [1459, 1543],  # used to be 1460, 1542
        "20250811": [1425, 1511],  # used to be 1426, 1510
    }
    output = []
    frame_count = 0
    vid_split2 = vid_split[-1].split("_")
    if vid_split2[1][3:] in crop_dict:
        x_s = crop_dict[vid_split2[1][3:]]
    else:
        x_s = [1441, 1525]  # used to be 1442, 1524
    one_sec = timedelta(seconds=1)
    start = datetime(
        2025,
        8,
        int(vid_split2[1][9:]),
        int(vid_split2[2][:2]),
        int(vid_split2[2][2:4]),
        int(vid_split2[2][4:]),
    )

    fvs = FileVideoStream(video_file).start()
    while fvs.more():
        frame = fvs.read()
        if frame is None:
            break
        frame_count += 1
        frame = frame[19:77, x_s[0] : x_s[1]]  # used to be 20: 76
        if frame.size == 0:
            if frame_count == 1:
                time = start
            output.append(3600 * time.hour + 60 * time.minute + time.second)
            detect = False
            continue
        frame = cv2.copyMakeBorder(frame, 3, 3, 3, 3, cv2.BORDER_REPLICATE)
        frame = cv2.resize(frame, None, fx=x_fac, fy=y_fac)
        # frame = cv2.medianBlur(frame, 3)
        if frame_count == 1:
            second = reader.readtext(
                frame, allowlist="0123456789", text_threshold=0.4, low_text=0.3
            )
            if second and second[0][2] > 0.95 and 0 <= int(second[0][1]) <= 59:
                if int(second[0][1]) == (int(vid_split2[2][4:]) + 1) % 60:
                    time = start + one_sec
                    output.append(3600 * time.hour + 60 * time.minute + time.second)
                    old_frame = frame
                    detect = True
                    continue
            time = start
            output.append(3600 * time.hour + 60 * time.minute + time.second)
            old_frame = frame
            detect = False
            continue
        # Make sure frames of first five time stamps are "mostly" correct values
        for i in range(3, 8):
            if len(set(output)) == i and output[-2] != output[-1]:
                if output[-2] != output[-14] and len(output) >= 17:
                    for j in range(i - 2):
                        output[-16 - (15 * j) : -2 - (15 * j)] = [
                            output[-2 - (15 * j)]
                        ] * 14

        # Ensure that the reader does not get stuck on one number for too long
        if len(output) >= 22 and output[-1] == output[-22]:
            detect = False
            time += one_sec
            output.append(3600 * time.hour + 60 * time.minute + time.second)
            old_frame = frame
            continue

        # Only use ocr if the current frame and the one before are different enough
        if detect:
            diff = cv2.absdiff(frame, old_frame)
            sim_score = (diff > 10).mean()
            if sim_score < 0.00007:  # used to be 0.00005
                output.append(3600 * time.hour + 60 * time.minute + time.second)
                old_frame = frame
                continue

        secs = str(time.second).zfill(2) + str((time + one_sec).second).zfill(2)
        second = reader.readtext(
            frame, allowlist=secs, text_threshold=0.4, low_text=0.3
        )
        if second and second[0][2] > 0.75:  # changed from 0.9 to 0.75
            detect = True
            if second[0][1] == secs[2:]:
                time += one_sec
        else:
            detect = False
            if frame_count >= 16:

                if output[-1] == output[-15]:
                    time += one_sec
        if time.hour == 0 and time.minute == 0 and time.second == 0:
            output.append(86400)
        else:
            output.append(3600 * time.hour + 60 * time.minute + time.second)
        old_frame = frame

    sorted_out = sorted(set(output))
    new_output = []
    for i, t in enumerate(sorted_out):
        num_count = 0
        for sec in output:
            if sec == t:
                num_count += 1
        total = 0
        if i == 0 and num_count < 13:
            for sec in output:
                if sec == t:
                    new_output.append((15 - num_count + total) / 15 + sec)
                    total += 1
            continue
        elif i == len(sorted_out) - 1 and num_count < 13:
            for sec in output:
                if sec == t:
                    new_output.append(total / 15 + sec)
                    total += 1
            continue
        for sec in output:
            if sec == t:
                new_output.append(total / num_count + sec)
                total += 1

    with open(file_path, "wb") as f:
        pickle.dump(new_output, f)
    return  # new_output, output


def get_tracks_folder(date, cam_name):
    """
    Returns the directory location of the tracks folder for specified date and camera name

        Args:
            date (str): date of the year in the format YYYYMMDD
            cam_name (str): name of camera in the format "cam##-{name}"
        Returns:
            folder (str): directory location of the tracks folder for the date and camera name
    """
    folder = f"/project/uwyo-0003/salmon-tlc/processing/predictions_05-27-2026-14-13-10/{cam_name}/{date[:4]}-{date[4:6]}-{date[6:]}"
    return folder


def get_times_folder(date, cam_name):
    """
    Returns the directory location of the time info folder for specified date and camera

        Args:
            date (str): date of the year in the format YYYYMMDD
            cam_name (str): name of camerca in the format "cam##-{name}"
        Returns:
            folder (str): directory location of the times folder for the date and camera name
    """
    folder = f"/project/uwyo-0003/salmon-tlc/processing/time-stamp-extraction_06-17-2026/{cam_name}/{date[:4]}-{date[4:6]}-{date[6:]}"
    return folder


def get_tracks_folder2(date, cam_name):
    """
    Returns the directory location of the tracks folder for the specified date and camera name with the new run of cameras from 06/28

        Args:
            date (str): date of the year in the format YYYYMMDD
            cam_name (str): name of camera in the format "cam##-{name}"
        Returns:
            folder (str):directoy location of the tracks folder for the date and camera name
    """
    folder = f"/project/uwyo-0003/salmon-tlc/processing/model-06-28-2026-14-56-28/{cam_name}/{date[:4]}-{date[4:6]}-{date[6:]}"
    return folder


def get_times_folder2(date, cam_name):
    """
    Returns the directory location of the time info folder for the specified date and camera based on the run from 07/09/2026

        Args:
            date (str): date of the year in the format YYYYMMDD
            cam_name (str): name of camera in the format "cam##-{name}"
        Returns:
            folder (str): directory location of the times folder for the date and camera name
    """
    folder = f"/project/uwyo-0003/salmon-tlc/processing/time-stamp-extraction_07-08-2026/{cam_name}/{date[:4]}-{date[4:6]}-{date[6:]}"
    return folder


def track_time_conversion(track, times, max_tag):
    """
    Returns track with unique fish tags and time stamps as total seconds

        Args:
            track (dict): gives fish tag number with corresponding times and locations
                keys: fish tag number
                values (dict): frame count with corresponding locations
                    keys: time (frame count number)
                    values (numpy array): location of top left conrer of bounding box (x,y) followed by width and height of box
            times (list): gives time stamps as total seconds, where the i-th term corresponds to the i + 1 time stamp
            max_tag (int): gives the max tag number for previous fish for the date
        Returns:
            track (dict): gives unique fish tag number with corresponding time in seconds and location
                keys: fish tag number (made unique)
                values (dict): time in seconds with corresponding locations
                    keys: time in seconds
                    values (numpy array): location of top left corner of bounding box (x,y) followed by width and height of box
            max_tag: new max tag number of fish
    """
    new_track = {}
    new_tags = []
    for tag, tag_data in track.items():
        new_tag = tag + max_tag
        new_tags.append(new_tag)
        new_vals = {}
        for time, loc in tag_data.items():
            new_vals[times[time - 1]] = loc
        new_track[new_tag] = new_vals
    max_tag = max(new_tags) + 1
    return new_track, max_tag
