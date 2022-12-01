import logging
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2 as cv
import numpy as np

from logger_setup import get_log

logger = logging.getLogger(__name__)


class SubtitleExtractor:
    def __init__(self, video_path: Path, sub_area: tuple = None) -> None:
        self.video_path = video_path
        self.video_name = self.video_path.stem
        self.sub_area = sub_area
        self._video_cap = cv.VideoCapture(str(video_path))
        self.fps = int(self._video_cap.get(cv.CAP_PROP_FPS))
        self.frame_count = int(self._video_cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.frame_height = int(self._video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self._video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self._video_cap.release()
        self.vd_output_dir = Path(f"{Path.cwd()}/output/{self.video_name}")
        # Extracted video frame storage directory
        self.frame_output_dir = self.vd_output_dir / "frames"
        # Extracted subtitle file storage directory
        self.subtitle_output_dir = self.vd_output_dir / "subtitle"
        # If the directory does not exist, create the folder
        if not self.frame_output_dir.exists():
            self.frame_output_dir.mkdir(parents=True)
        if not self.subtitle_output_dir.exists():
            self.subtitle_output_dir.mkdir(parents=True)

    def default_subtitle_area(self) -> tuple:
        """
        Returns a default subtitle area that can be used if no subtitle is given.
        :return: Position of subtitle relative to the resolution of the video. x2 = width and y2 = height
        """
        x1, y1, x2, y2 = 0, int(self.frame_height * 0.75), self.frame_width, self.frame_height
        return x1, y1, x2, y2

    @staticmethod
    def rescale_frame(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
        height = int(frame.shape[0] * scale)
        width = int(frame.shape[1] * scale)
        dimensions = (width, height)
        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    @staticmethod
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

    def extract_frames(self, overwrite: bool, start: int, end: int, every: int) -> int:
        """
        Extract frames from a video using OpenCVs VideoCapture
        :param overwrite: to overwrite frames that already exist
        :param start: start frame
        :param end: end frame
        :param every: frame spacing
        :return: count of images saved
        """
        if self.sub_area:
            x1, y1, x2, y2 = self.sub_area
        else:
            x1, y1, x2, y2 = self.default_subtitle_area()

        capture = cv.VideoCapture(str(self.video_path))  # open the video using OpenCV
        capture.set(1, start)  # set the starting frame of the capture
        frame = start  # keep track of which frame we are up to, starting from start
        while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop
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
                # crop and save subtitle area
                cropped_frame = image[y1:y2, x1:x2]
                frame_position = capture.get(cv.CAP_PROP_POS_MSEC)
                save_path = f"{self.frame_output_dir}/{frame_position}.jpg"  # create the save path
                if not overwrite:  # we want to overwrite anyway
                    cv.imwrite(save_path, cropped_frame)  # save the extracted image
                    saved_count += 1  # increment our counter by one

            frame += 1  # increment our frame count

        capture.release()  # after the while has finished close the capture
        return saved_count  # and return the count of the images we saved

    def video_to_frames(self, overwrite: bool, every: int, chunk_size: int) -> None:
        """
        Extracts the frames from a video using multiprocessing
        :param overwrite: overwrite frames if they exist
        :param every: extract every this many frames
        :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
        """

        if self.frame_count < 1:  # if video has no frames, might be and opencv error
            logger.error("Video has no frames. Check your OpenCV + ffmpeg installation")

        # split the frames into chunk lists
        frame_chunks = [[i, i + chunk_size] for i in range(0, self.frame_count, chunk_size)]
        # make sure last chunk has correct end frame, also handles case chunk_size < total
        frame_chunks[-1][-1] = min(frame_chunks[-1][-1], self.frame_count-1)

        for f in frame_chunks:
            self.extract_frames(overwrite, f[0], f[1], every)

        # execute across multiple cpu cores to speed up processing, get the count automatically
        # with ProcessPoolExecutor() as executor:
        #     futures = [executor.submit(self.extract_frames, overwrite, f[0], f[1], every) for f in frame_chunks]
        #     print(futures)
        #     for i, f in enumerate(as_completed(futures)):  # as each process completes
        #         self.print_progress(i, len(frame_chunks)-1)
        print("")

    def empty_cache(self) -> None:
        """
        Delete all cache files produced during subtitle extraction.
        """
        if self.vd_output_dir.exists():
            shutil.rmtree(self.vd_output_dir.parent)

    def run(self) -> None:
        """
        Run through the steps of extracting video.
        """
        start = cv.getTickCount()
        logger.info(f"File Path: {self.video_path}")
        logger.info(f"Frame Rate: {self.fps}, Frame Count: {self.frame_count}")
        logger.info(f"Resolution: {self.frame_width} X {self.frame_height}")
        logger.info(f"Subtitle Area: {self.sub_area or self.default_subtitle_area()}")

        logger.info("Start to extracting video keyframes...")
        self.video_to_frames(overwrite=False, every=1, chunk_size=1000)

        end = cv.getTickCount()
        total_time = (end - start) / cv.getTickFrequency()
        logger.info(f"Subtitle file generated successfully, Total time: {round(total_time, 3)}s")
        # self.empty_cache()


if __name__ == '__main__':
    get_log()
    logger.debug("Logging Started")

    # video = Path("tests/I Can Copy Talents.mp4")
    # video = Path("tests/40,000 Years of the Stars.mp4")
    # video = Path("tests/anime video-cut.mp4")
    video = Path("tests/anime video.mp4")
    se = SubtitleExtractor(video)
    se.run()

    logger.debug("Logging Ended\n")
