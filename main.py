import logging
import multiprocessing
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2 as cv
import numpy as np
from natsort import natsorted

from logger_setup import get_logger

logger = logging.getLogger(__name__)


class SubtitleExtractor:
    def __init__(self, video_path: Path, sub_area: tuple = None) -> None:
        self.video_path = video_path
        self.video_name = self.video_path.stem
        self.video_details = self.__get_video_details()
        self.sub_area = self.__subtitle_area(sub_area)
        self.vd_output_dir = Path(f"{Path.cwd()}/output/{self.video_name}")
        # Extracted video frame storage directory
        self.frame_output = self.vd_output_dir / "frames"
        # Extracted text file storage directory
        self.text_output = self.vd_output_dir / "extracted texts"
        # If the directory does not exist, create the folder
        if not self.frame_output.exists():
            self.frame_output.mkdir(parents=True)
        if not self.text_output.exists():
            self.text_output.mkdir(parents=True)

    def __get_video_details(self) -> tuple:
        capture = cv.VideoCapture(str(self.video_path))
        fps = capture.get(cv.CAP_PROP_FPS)
        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        capture.release()
        return fps, frame_count, frame_height, frame_width

    def __subtitle_area(self, sub_area: None | tuple) -> tuple:
        """
        Returns a default subtitle area that can be used if no subtitle is given.
        :return: Position of subtitle relative to the resolution of the video. x2 = width and y2 = height
        """
        if sub_area:
            return sub_area
        else:
            _, _, frame_height, frame_width = self.video_details
            x1, y1, x2, y2 = 0, int(frame_height * 0.75), frame_width, frame_height
            return x1, y1, x2, y2

    @staticmethod
    def rescale_frame(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
        height = int(frame.shape[0] * scale)
        width = int(frame.shape[1] * scale)
        dimensions = (width, height)
        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    def view_frames(self) -> None:
        video_cap = cv.VideoCapture(str(self.video_path))
        while True:
            success, frame = video_cap.read()
            if not success:
                logger.warning(f"Video has ended!")  # or failed to read
                break
            x1, y1, x2, y2 = self.sub_area
            # draw rectangle over subtitle area
            top_left_corner = (x1, y1)
            bottom_right_corner = (x2, y2)
            color_red = (0, 0, 255)
            cv.rectangle(frame, top_left_corner, bottom_right_corner, color_red, 2)

            # crop and show subtitle area
            subtitle_area = frame[y1:y2, x1:x2]
            resized_cropped_frame = self.rescale_frame(subtitle_area)
            cv.imshow("Cropped frame", resized_cropped_frame)

            frame_resized = self.rescale_frame(frame)
            cv.imshow("Video Output", frame_resized)

            if cv.waitKey(1) == ord('q'):
                break
        video_cap.release()
        cv.destroyAllWindows()

    def empty_cache(self) -> None:
        """
        Delete all cache files produced during subtitle extraction.
        """
        if self.vd_output_dir.exists():
            shutil.rmtree(self.vd_output_dir.parent)
            logger.debug("Emptying cache")

    @staticmethod
    def print_progress(iteration: int, total: float, decimals: float = 3, bar_length: int = 50) -> None:
        """
        Call in a loop to create standard out progress bar.

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
        Extract frames from a video using OpenCVs VideoCapture.

        :param overwrite: whether to overwrite frames that already exist
        :param start: start frame
        :param end: end frame
        :param every: frame spacing
        :return: count of images saved
        """

        x1, y1, x2, y2 = self.sub_area

        capture = cv.VideoCapture(str(self.video_path))  # open the video using OpenCV

        if start < 0:  # if start isn't specified lets assume 0
            start = 0
        if end < 0:  # if end isn't specified assume the end of the video
            end = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

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
                frame_position = capture.get(cv.CAP_PROP_POS_MSEC)  # get the name of the frame
                file_name = f"{self.frame_output}/{frame_position}"  # create the file name save path
                if not Path(file_name).exists() or overwrite:  # if it doesn't exist, or we want to overwrite anyway
                    subtitle_area = image[y1:y2, x1:x2]  # crop the subtitle area
                    rescaled_sub_area = self.rescale_frame(subtitle_area)
                    cv.imwrite(file_name + ".jpg", rescaled_sub_area)  # save the extracted image as jpg image
                    saved_count += 1  # increment our counter by one

            frame += 1  # increment our frame count

        capture.release()  # after the while has finished close the capture
        return saved_count  # and return the count of the images we saved

    def video_to_frames(self, overwrite: bool, every: int, chunk_size: int) -> None:
        """
        Extracts the frames from a video using multiprocessing.

        :param overwrite: overwrite frames if they exist
        :param every: extract every this many frames
        :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
        :return: path to the directory where the frames were saved, or None if fails
        """

        frame_count = self.video_details[1]

        logger.debug(f"Video has {frame_count} frames and is being asked to be split into {chunk_size}. "
                     f"Will split? {frame_count > chunk_size}")
        # ignore chunk size if it's greater than frame count
        chunk_size = chunk_size if frame_count > chunk_size else frame_count - 1

        if frame_count < 1:  # if video has no frames, might be and opencv error
            logger.error("Video has no frames. Check your OpenCV installation")

        # split the frames into chunk lists
        frame_chunks = [[i, i + chunk_size] for i in range(0, frame_count, chunk_size)]
        # make sure last chunk has correct end frame, also handles case chunk_size < frame count
        frame_chunks[-1][-1] = min(frame_chunks[-1][-1], frame_count - 1)
        logger.debug(f"Frame chunks = {frame_chunks}")

        logger.debug("Using multiprocessing for frames")
        # execute across multiple cpu cores to speed up processing, get the count automatically
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = [executor.submit(self.extract_frames, overwrite, f[0], f[1], every) for f in frame_chunks]
            for i, f in enumerate(as_completed(futures)):  # as each process completes
                self.print_progress(i, len(frame_chunks) - 1)  # print it's progress
            print("")  # prevent next line from joining previous progress bar
        logger.info("Done extracting frames from video!")

    def merge_similar_frames(self):
        for file in natsorted(self.frame_output.iterdir()):
            print(file.name)

    @staticmethod
    def timecode(frame_no: float) -> str:
        seconds = frame_no // 1000
        milliseconds = int(frame_no % 1000)
        minutes = 0
        hours = 0
        if seconds >= 60:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
        if minutes >= 60:
            hours = int(minutes // 60)
            minutes = int(minutes % 60)
        smpte_token = ','
        return "%02d:%02d:%02d%s%03d" % (hours, minutes, seconds, smpte_token, milliseconds)

    def _save_subtitle(self, lines: list) -> None:
        name = self.video_path.with_suffix(".srt")
        logger.info(f"Subtitle file successfully generated. Name: {name}")
        with open(name, 'w', encoding="utf-8") as new_sub:
            new_sub.writelines(lines)

    def generate_subtitle(self):
        subtitles = []
        line_code = 0
        for file in self.text_output.iterdir():
            line_code += 1
            frame_start = ''
            frame_end = ''
            file_content = file.read_text(encoding="utf-8")
            subtitle_line = f"{line_code}\n{frame_start} --> {frame_end}\n{file_content}\n"
            subtitles.append(subtitle_line)
        self._save_subtitle(subtitles)
        logger.info("Subtitle generated!")

    def run(self) -> None:
        """
        Run through the steps of extracting video.
        """
        start = cv.getTickCount()
        fps, frame_count, frame_height, frame_width = self.video_details
        logger.info(f"File Path: {self.video_path}")
        logger.info(f"Frame Rate: {fps}, Frame Count: {frame_count}")
        logger.info(f"Resolution: {frame_width} X {frame_height}")
        logger.info(f"Subtitle Area: {self.sub_area}")

        # self.view_frames()
        logger.info("Starting to extracting video keyframes...")
        # self.video_to_frames(overwrite=False, every=2, chunk_size=250)
        logger.info("Merging similar frames...")
        self.merge_similar_frames()
        logger.info("Starting to extracting text from frames...")
        # extract_and_save_text(self.frame_output, self.text_output)
        logger.info("Generating subtitle...")
        # self.generate_subtitle()

        end = cv.getTickCount()
        total_time = (end - start) / cv.getTickFrequency()
        logger.info(f"Subtitle file generated successfully, Total time: {round(total_time, 3)}s")
        # self.empty_cache()


if __name__ == '__main__':
    get_logger()
    logger.debug("Logging Started")

    video = Path("tests/anime video.mp4")
    se = SubtitleExtractor(video)
    se.run()

    logger.debug("Logging Ended\n")
