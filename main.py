import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import pairwise
from pathlib import Path

import cv2 as cv
import numpy as np
from natsort import natsorted
from paddleocr import PaddleOCR
# from skimage.metrics import structural_similarity


class SubtitleExtractor:
    def __init__(self, video_path: Path, sub_area: tuple = None) -> None:
        self.video_path = video_path
        self.video_name = self.video_path.stem
        self.video_details = self.__get_video_details()
        self.sub_area = self.__subtitle_area(sub_area)
        # Create cache directory
        self.vd_output_dir = Path(f"{Path.cwd()}/output/{self.video_name}")
        # Extracted video frame storage directory
        self.frame_output = self.vd_output_dir / "frames"
        # Extracted text file storage directory
        self.text_output = self.vd_output_dir / "extracted texts"
        # Empty cache at the beginning of program run before recreating it
        # self.empty_cache()
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
                print(f"Video has ended!")  # or failed to read
                break
            x1, y1, x2, y2 = self.sub_area
            # draw rectangle over subtitle area
            top_left_corner = (x1, y1)
            bottom_right_corner = (x2, y2)
            color_red = (0, 0, 255)
            cv.rectangle(frame, top_left_corner, bottom_right_corner, color_red, 2)

            # show preprocessed subtitle area
            preprocessed_sub = self.preprocess_sub_frame(frame)
            cv.imshow("Preprocessed Sub", preprocessed_sub)

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
            print("Emptying cache")

    def preprocess_sub_frame(self, frame: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.sub_area
        subtitle_area = frame[y1:y2, x1:x2]  # crop the subtitle area
        rescaled_sub_area = self.rescale_frame(subtitle_area)
        gray_image = cv.cvtColor(rescaled_sub_area, cv.COLOR_BGR2GRAY)
        return gray_image

    @staticmethod
    def print_progress(iteration: int, total: float, prefix: str, decimals: float = 3, bar_length: int = 50) -> None:
        """
        Call in a loop to create standard out progress bar.

        :param iteration: current iteration
        :param total: total iterations
        :param prefix: prefix string
        :param decimals: positive number of decimals in percent complete
        :param bar_length: character length of bar
        :return: None
        """

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
                file_name = f"{self.frame_output}/{frame_position}.jpg"  # create the file name save path and format
                if not Path(file_name).exists() or overwrite:  # if it doesn't exist, or we want to overwrite anyway
                    preprocessed_frame = self.preprocess_sub_frame(image)
                    cv.imwrite(file_name, preprocessed_frame)  # save the extracted image
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

        # ignore chunk size if it's greater than frame count
        chunk_size = chunk_size if frame_count > chunk_size else frame_count - 1

        if frame_count < 1:  # if video has no frames, might be and opencv error
            print("Video has no frames. Check your OpenCV installation")

        # split the frames into chunk lists
        frame_chunks = [[i, i + chunk_size] for i in range(0, frame_count, chunk_size)]
        # make sure last chunk has correct end frame, also handles case chunk_size < frame count
        frame_chunks[-1][-1] = min(frame_chunks[-1][-1], frame_count - 1)
        # print(f"Frame chunks = {frame_chunks}")

        prefix = "Extracting frames from video:"
        # execute across multiple cpu cores to speed up processing, get the count automatically
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.extract_frames, overwrite, f[0], f[1], every) for f in frame_chunks]
            for i, f in enumerate(as_completed(futures)):  # as each process completes
                self.print_progress(i, len(frame_chunks) - 1, prefix)  # print it's progress
            print("")  # prevent next line from joining previous progress bar
        print("Done extracting frames from video!")

    @staticmethod
    def image_similarity(image1: Path, image2: Path) -> float:
        frame1 = cv.imread(str(image1))
        frame2 = cv.imread(str(image2))
        # Compute SSIM between two images
        score = structural_similarity(frame1, frame2, channel_axis=-1)
        return score

    def frame_merger(self, files: list, threshold: float) -> int:
        saved_count = 0
        starting_file = None
        for file1, file2 in pairwise(files):
            saved_count += 2
            similarity = self.image_similarity(file1, file2)
            if similarity > threshold:
                # print(file1.name, file2.name, similarity)
                if not starting_file:
                    starting_file = file1
            else:
                # print(file1.name, file2.name, similarity)
                if not starting_file:
                    starting_file = file1
                ending_file = file1
                if starting_file == ending_file:
                    ending_file = file2
                new_file_name = f"{starting_file.stem}--{ending_file.stem}.jpg"
                starting_file.rename(f"{starting_file.parent}/{new_file_name}")
                starting_file = None
        return saved_count

    def merge_similar_frames(self, chunk_size: int, threshold: float) -> None:
        files = [file for file in natsorted(self.frame_output.iterdir())]
        file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

        prefix = "Merging similar frames from video:"
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.frame_merger, files, threshold) for files in file_chunks]
            for i, f in enumerate(as_completed(futures)):
                self.print_progress(i, len(file_chunks), prefix)
            print("")
        print("Done merging frames from video!")

        print("Deleting excess frames...")
        for file in self.frame_output.iterdir():
            if "--" not in file.name:
                file.unlink()

    # def frames_to_text(self):
    #     from paddleocr import PaddleOCR
    #     ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    #     for file in tqdm(self.frame_output.iterdir(), desc="Extracting texts: "):
    #         name = Path(f"{self.text_output}/{file.stem}.txt")
    #         result = ocr.ocr(str(file), cls=True)
    #         if result[0]:
    #             text = result[0][0][1][0]
    #             with open(name, 'w', encoding="utf-8") as text_file:
    #                 text_file.write(text)
    #     logger.info("Done extracting texts!")

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
        print(f"Subtitle file successfully generated. Name: {name}")
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
        print("Subtitle generated!")

    def run(self) -> None:
        """
        Run through the steps of extracting video.
        """
        start = cv.getTickCount()
        fps, frame_count, frame_height, frame_width = self.video_details
        print(f"File Path: {self.video_path}")
        print(f"Frame Rate: {fps}, Frame Count: {frame_count}")
        print(f"Resolution: {frame_width} X {frame_height}")
        print(f"Subtitle Area: {self.sub_area}")

        # self.view_frames()
        print("Starting to extracting video keyframes...")
        # self.video_to_frames(overwrite=False, every=2, chunk_size=250)
        print("Starting to merge similar frames...")
        # self.merge_similar_frames(chunk_size=100, threshold=0.75)
        print("Starting to extracting text from frames...")
        self.frames_to_text(chunk_size=250)
        print("Generating subtitle...")
        # self.generate_subtitle()

        end = cv.getTickCount()
        total_time = (end - start) / cv.getTickFrequency()
        print(f"Subtitle file generated successfully, Total time: {round(total_time, 3)}s")
        # self.empty_cache()


if __name__ == '__main__':
    video = Path("tests/anime video.mp4")
    se = SubtitleExtractor(video)
    se.run()
