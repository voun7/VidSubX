import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2 as cv

import utilities.utils as utils

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, frames_dir: Path, key_area: tuple | None, start: int, end: int, every: int) -> None:
    """
    Extract frames from a video using OpenCVs VideoCapture.
    :param video_path: Path of the video.
    :param frames_dir: The directory to save the frames.
    :param key_area: Coordinates of the frame containing subtitle.
    :param start: Start frame.
    :param end: End frame.
    :param every: Frame spacing.
    """
    capture = cv.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)

    while frame < end:  # let's loop through the frames until the end
        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety max's out at 500
            break

        # sometimes OpenCV reads Nones during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            # crop and save key area
            if key_area:
                x1, y1, x2, y2 = key_area
                image = image[y1:y2, x1:x2]
            frame_position = capture.get(cv.CAP_PROP_POS_MSEC)
            save_name = f"{frames_dir}/{frame_position}.jpg"  # create the save path
            cv.imwrite(save_name, image)  # save the extracted image

        frame += 1  # increment our frame count
    capture.release()  # after the while has finished close the capture


def video_to_frames(video_path: str, frames_dir: Path, key_area: tuple | None, start_frame: int = None,
                    stop_frame: int = None) -> None:
    """
    Extracts the frames from a video using multiprocessing.
    :param video_path: path like string to the video
    :param frames_dir: directory to save the frames
    :param key_area: coordinates of the frame containing subtitle
    :param start_frame: The frame where image extractions from video starts.
    :param stop_frame: The frame where image extractions from video stops.
    """
    # extract every this many frames.
    every = utils.Config.frame_extraction_frequency
    # how many frames to split into chunks (one chunk per cpu core process)
    chunk_size = utils.Config.frame_extraction_chunk_size
    # cancel if process has been cancelled by gui.
    prefix = "Frame Extraction"
    if utils.Process.interrupt_process:
        logger.warning(f"{prefix} process interrupted!")
        return

    capture = cv.VideoCapture(video_path)  # load the video
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))  # get its total frame count
    capture.release()  # release the capture straight away

    # ignore chunk size if it's greater than frame count
    chunk_size = chunk_size if frame_count > chunk_size else frame_count - 1

    if frame_count < 1:  # if video has no frames, might be and opencv error
        logger.error("Video has no frames. Check your OpenCV installation")
        return  # end function call

    start_frame, stop_frame = start_frame or 0, stop_frame or frame_count
    # split the frames into chunk lists
    frame_chunks = [[i, i + chunk_size] for i in range(start_frame, stop_frame, chunk_size)]
    frame_chunks[-1][-1] = stop_frame  # make sure last chunk has correct end frame
    no_chunks = len(frame_chunks)
    # create a process pool to execute across multiple cpu cores to speed up processing
    logger.info(f"Starting Multiprocess {prefix} from video...")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(extract_frames, video_path, frames_dir, key_area, f[0], f[1], every)
                   for f in frame_chunks]  # submit the processes: extract_frames(...)
        for i, f in enumerate(as_completed(futures)):  # as each process completes
            f.result()  # Prevents silent bugs. Exceptions raised will now be displayed.
            utils.print_progress(i, no_chunks - 1, prefix)
    logger.info(f"{prefix} done!")
