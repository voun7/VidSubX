"""
@FileName: video_to_frames.py
@desc: Fast frame extraction from videos using Python and OpenCV
"""
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2 as cv
import numpy as np

from logger_setup import get_logger

logger = get_logger(__name__)


def print_progress(iteration: int, total: float, decimals: float = 3, bar_length: int = 50) -> None:
    """
    Call in a loop to create standard out progress bar

    :param iteration: current iteration
    :param total: total iterations
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return: None
    """

    prefix = "Extracting frames from video: "
    suffix = "Complete"
    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),  # write out the bar
    sys.stdout.flush()  # flush to stdout


def extract_frames(video_path: Path, output: Path, sub_area: tuple, overwrite: bool, start: int, end: int,
                   every: int) -> int:
    """
    Extract frames from a video using OpenCVs VideoCapture

    :param video_path: path of the video
    :param output: the directory to save the frames
    :param sub_area: coordinates of the frame containing subtitle
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    x1, y1, x2, y2 = sub_area

    capture = cv.VideoCapture(str(video_path))  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    while frame < end:  # let's loop through the frames until the end

        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety max out at 500
            break

        # sometimes OpenCV reads Nones during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            frame_position = capture.get(cv.CAP_PROP_POS_MSEC)  # get the name of the frame
            file_name = f"{output}/{frame_position}"  # create the file name save path
            if not Path(file_name).exists() or overwrite:  # if it doesn't exist, or we want to overwrite anyway
                cropped_frame = image[y1:y2, x1:x2]  # crop the subtitle area
                np.save(file_name, cropped_frame)  # save the extracted image as np array
                # cv.imwrite(file_name, cropped_frame)  # save the extracted image as image
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved


def video_to_frames(video_path: Path, output: Path, sub_area: tuple, overwrite: bool, every: int,
                    chunk_size: int) -> None:
    """
    Extracts the frames from a video using multiprocessing

    :param video_path: path to the video
    :param output: directory to save the frames
    :param sub_area: coordinates of the frame containing subtitle
    :param overwrite: overwrite frames if they exist
    :param every: extract every this many frames
    :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
    :return: path to the directory where the frames were saved, or None if fails
    """

    capture = cv.VideoCapture(str(video_path))  # load the video
    total = int(capture.get(cv.CAP_PROP_FRAME_COUNT))  # get its total frame count
    capture.release()  # release the capture straight away

    logger.debug(f"Video has {total} frames and is being asked to be split into {chunk_size}. "
                 f"Will split? {total > chunk_size}")
    chunk_size = chunk_size if total > chunk_size else total - 1  # ignore chunk size if it's greater than frame count

    if total < 1:  # if video has no frames, might be and opencv error
        logger.error("Video has no frames. Check your OpenCV installation")

    # split the frames into chunk lists
    frame_chunks = [[i, i + chunk_size] for i in range(0, total, chunk_size)]
    # make sure last chunk has correct end frame, also handles case chunk_size < total
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total - 1)
    logger.debug(f"Frame chunks = {frame_chunks}")

    logger.debug("Using multiprocessing for frames")
    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

        futures = [executor.submit(extract_frames, video_path, output, sub_area, overwrite, f[0], f[1], every)
                   for f in frame_chunks]  # submit the processes: extract_frames(...)

        for i, f in enumerate(as_completed(futures)):  # as each process completes
            print_progress(i, len(frame_chunks) - 1)  # print it's progress
        print("")  # prevent next line from joining previous progress bar
    logger.info("Done extracting frames from video!")
