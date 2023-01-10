import logging
import re
import sys
from threading import Thread
from tkinter import *
from tkinter import filedialog
from tkinter import ttk

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

import utilities.utils as utils
from main import SubtitleExtractor
from utilities.logger_setup import get_logger

logger = logging.getLogger(__name__)

get_logger()


class SubtitleExtractorGUI:
    SubEx = SubtitleExtractor()

    def __init__(self, root: ttk) -> None:
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._create_layout()
        self.video_queue = {}
        self.vse_thread = Thread(target=self.extract_subtitle)
        self.current_video = None
        self.video_capture = None
        self.running = False
        self.console_redirector()

    def _create_layout(self) -> None:
        """
        Use ttk to create frames for gui.
        """
        # Window title
        self.root.title("Video Subtitle Extractor")
        # Do not allow window to be resizable.
        self.root.resizable(FALSE, FALSE)

        # Create window menu bar.
        self._menu_bar()

        # Create main frame that will contain other frames.
        self.main_frame = ttk.Frame(self.root, padding=(5, 5, 5, 15))

        # Frames created in main frame.
        self._video_frame()
        self._work_frame()
        self._output_frame()

        # Main frame's position in root window.
        self.main_frame.grid(column=0, row=0, sticky="N, S, E, W")

    def _menu_bar(self) -> None:
        # Remove dashed lines that come default with tkinter menu bar.
        self.root.option_add('*tearOff', FALSE)

        # Create menu bar in root window.
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        # Create menus for menu bar.
        self.menu_file = Menu(menubar)
        menu_settings = Menu(menubar)

        menubar.add_cascade(menu=self.menu_file, label="File")
        menubar.add_cascade(menu=menu_settings, label="Settings")

        # Add menu items.
        self.menu_file.add_command(label="Open file(s)", command=self.open_files)
        self.menu_file.add_command(label="Close", command=self._on_closing)

        menu_settings.add_command(label="Language", command=self._language_settings)
        menu_settings.add_command(label="Extraction", command=self._extraction_settings)

    def _video_frame(self) -> None:
        """
        Frame that contains the widgets for the video.
        """
        # Create video frame in main frame.
        video_frame = ttk.Frame(self.main_frame)
        video_frame.grid(column=0, row=0)

        # Create canvas widget in video frame.
        self.video_canvas = Canvas(video_frame, bg="black")
        self.video_canvas.grid(column=0, row=0)

        # Create frame slider widget in video frame.
        self.video_scale = ttk.Scale(video_frame, command=self._frame_slider, orient=HORIZONTAL, length=600,
                                     state="disabled")
        self.video_scale.grid(column=0, row=1)

    def _work_frame(self) -> None:
        """
        Frame that contains the widgets for working with the video or videos (batch mode).
        """
        # Create work frame in main frame.
        progress_frame = ttk.Frame(self.main_frame)
        progress_frame.grid(column=0, row=1)

        # Create button widget for starting the text extraction.
        self.run_button = ttk.Button(progress_frame, text="Run", command=self._run)
        self.run_button.grid(column=0, row=0, pady=6, padx=10)

        # Create progress bar widget for showing the text extraction progress.
        self.progress_bar = ttk.Progressbar(progress_frame, orient=HORIZONTAL, length=600, mode='determinate')
        self.progress_bar.grid(column=1, row=0, padx=10)

        # Create button widget for previous video in queue for subtitle area selection.
        self.previous_button = ttk.Button(progress_frame, text="Previous Video", command=self._previous_video)

        # Create label widget to show current video number and number of videos.
        self.video_label = ttk.Label(progress_frame)
        self.video_label.grid(column=3, row=0, padx=10)

        # Create button widget for next video in queue for subtitle area selection.
        self.next_button = ttk.Button(progress_frame, text="Next Video", command=self._next_video)

    def _output_frame(self) -> None:
        """
        Frame that contains the widgets for the extraction text output.
        """
        # Create output frame in main frame
        output_frame = ttk.Frame(self.main_frame)
        output_frame.grid(column=0, row=2, sticky="N, S, E, W")

        # Create text widget for showing the text extraction details in the output. Does not allow input from gui.
        self.text_output_widget = Text(output_frame, height=12, state="disabled")
        self.text_output_widget.grid(column=0, row=0, sticky="N, S, E, W")

        # Create scrollbar widget for text widget.
        output_scroll = ttk.Scrollbar(output_frame, orient=VERTICAL, command=self.text_output_widget.yview)
        output_scroll.grid(column=1, row=0, sticky="N,S")

        # Connect text and scrollbar widgets.
        self.text_output_widget.configure(yscrollcommand=output_scroll.set)

        # Resize output frame if main frame is resized.
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)

    def _reset_batch_layout(self) -> None:
        """
        Deactivate the batch layout from the work frame on the gui.
        """
        logger.debug("Batch layout deactivated")
        self.previous_button.grid_remove()
        self.next_button.grid_remove()

    def _set_batch_layout(self) -> None:
        """
        Activate the batch layout from the work frame on the gui.
        """
        logger.debug("Setting batch layout")
        self.progress_bar.configure(length=500)
        self.video_label.configure(state="normal", text=self._video_indexer()[2])
        self.previous_button.grid(column=2, row=0, padx=10)
        self.next_button.grid(column=4, row=0, padx=10)

    def _language_settings(self):
        pass

    def _extraction_settings(self):
        pass

    @staticmethod
    def rescale(frame: np.ndarray = None, subtitle_area: tuple = None, resolution: tuple = None,
                scale: float = 0.5) -> np.ndarray | tuple:
        """
        Method to rescale any frame, subtitle area and resolution.
        """
        if frame is not None:
            height = int(frame.shape[0] * scale)
            width = int(frame.shape[1] * scale)
            dimensions = (width, height)
            return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

        if subtitle_area:
            x1, y1, x2, y2 = subtitle_area
            x1 = x1 * scale
            y1 = y1 * scale
            x2 = x2 * scale
            y2 = y2 * scale
            return x1, y1, x2, y2

        if resolution:
            frame_width, frame_height = resolution
            frame_width = frame_width * scale
            frame_height = frame_height * scale
            return frame_width, frame_height

    def _set_canvas(self) -> None:
        """
        Set canvas size to the size of captured video.
        """
        logger.debug("Setting canvas size")
        _, _, frame_width, frame_height, = self.SubEx.video_details(self.current_video)
        frame_width, frame_height = self.rescale(resolution=(frame_width, frame_height))
        self.video_canvas.configure(width=frame_width, height=frame_height, bg="white")

    def _set_sub_area(self, subtitle_area: tuple) -> None:
        """
        Set current video subtitle area to new area.
        :param subtitle_area: new subtitle area to be used.
        """
        self.current_sub_area = subtitle_area
        self.video_queue[f"{self.current_video}"] = self.current_sub_area

    def draw_subtitle_area(self, subtitle_area: tuple, border_width: int = 4, border_color: str = "green") -> None:
        """
        Draw subtitle on video frame. x1, y1 = top left corner and x2, y2 = bottom right corner.
        """
        if subtitle_area:
            logger.debug(f"Subtitle coordinates are not None. {subtitle_area}")
            x1, y1, x2, y2 = self.rescale(subtitle_area=subtitle_area)
            self.video_canvas.create_rectangle(x1, y1, x2, y2, width=border_width, outline=border_color)
        else:
            logger.debug("Subtitle coordinates are None.")
            _, _, frame_width, frame_height, = self.SubEx.video_details(self.current_video)
            self._set_sub_area(self.SubEx.default_sub_area(frame_width, frame_height, subtitle_area))

    def _display_video_frame(self, second: float | int = 0) -> None:
        """
        Find captured video frame through corresponding second and display on video canvas.
        :param second: default corresponding second
        """
        self.video_capture.set(cv.CAP_PROP_POS_MSEC, second * 1000)
        _, frame = self.video_capture.read()

        cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        frame_resized = self.rescale(cv2image)

        img = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image=img)
        self.video_canvas.create_image(0, 0, image=photo, anchor=NW)
        self.video_canvas.image = photo

    def _frame_slider(self, scale_value: str) -> None:
        """
        Make changes according to the position of the slider.
        :param scale_value: current position of the slider.
        """
        scale_value = float(scale_value)
        self._display_video_frame(scale_value)
        self.draw_subtitle_area(self.current_sub_area)

    def _set_frame_slider(self) -> None:
        """
        Activate the slider, then set the starting and ending values of the slider.
        """
        logger.debug("Setting frame slider")
        fps, frame_total, _, _ = self.SubEx.video_details(self.current_video)
        duration = frame_total / fps

        self.video_scale.configure(state="normal", from_=0, to=duration, value=0)

    def _video_indexer(self) -> tuple:
        """
        Checks the index of the given video in the video queue dictionary using its key.
        """
        index = list(self.video_queue).index(self.current_video)
        queue_len = len(self.video_queue)
        video_index = f"Video {index + 1} of {queue_len}"
        return index, queue_len, video_index

    def _previous_video(self) -> None:
        """
        Change current video to the previous video in queue.
        """
        logger.debug("Previous video button clicked")
        index = self._video_indexer()[0]
        previous_index = index - 1
        self._set_video(previous_index)

    def _next_video(self) -> None:
        """
        Change current video to the next video in queue.
        """
        logger.debug("Next video button clicked")
        index, queue_len, _ = self._video_indexer()
        next_index = index + 1

        if index < queue_len - 1:
            self._set_video(next_index)
        else:
            self._set_video()

    def _set_video(self, video_index: int = 0) -> None:
        """
        Set the gui for the given current video queue index.
        :param video_index: Index of video that should be set to current. Defaults to first index.
        """
        if self.video_capture is not None:
            logger.debug("Closing open video")
            self.video_capture.release()

            if len(self.video_queue) == 1:
                self.video_label.configure(text='')
                self._reset_batch_layout()

        self.current_video = list(self.video_queue.keys())[video_index]
        self.current_sub_area = list(self.video_queue.values())[video_index]
        self.video_capture = cv.VideoCapture(self.current_video)
        self._set_canvas()
        self._set_frame_slider()
        self._display_video_frame()
        self.draw_subtitle_area(self.current_sub_area)

        if len(self.video_queue) > 1:
            self._set_batch_layout()

    def open_files(self) -> None:
        """
        Open file dialog to select a file or files then call required methods.
        """
        logger.debug("Open button clicked")

        title = "Select Video(s)"
        file_types = (("mp4", "*.mp4"), ("mkv", "*.mkv"), ("All files", "*.*"))
        filenames = filedialog.askopenfilenames(title=title, filetypes=file_types)

        # This condition prevents the below methods from being called
        # when button is clicked but no files are selected.
        if filenames:
            logger.debug("New files have been selected, video queue, and text widget output cleared")
            self.video_queue = {}  # Empty the video queue before adding the new videos.
            self.progress_bar.configure(value=0)
            self.set_output()

            # Add all opened videos to a queue.
            for filename in filenames:
                logger.info(f"Opened file: {filename}")
                self.video_queue[filename] = None

            self._set_video()  # Set one of the opened videos to current video.

    def console_redirector(self) -> None:
        """
        Redirect console statements to text widget
        """
        sys.stdout.write = self.write_to_output
        # sys.stderr.write = self.write_to_output

    def set_output(self, text: str = None) -> None:
        """
        Clear all text or clear progress repetition in text widget.
        """
        if text is None:
            logger.debug("Text output cleared")
            self.text_output_widget.configure(state="normal")
            self.text_output_widget.delete("1.0", "end")
            self.text_output_widget.configure(state="disabled")
            return

        progress_pattern = re.compile(r'.+\s\|[#-]+\|\s[\d.]+%\s')
        if progress_pattern.search(text):
            previous_line = self.text_output_widget.get("end-2l", "end-1l")
            if progress_pattern.search(previous_line):
                logger.debug(f"pattern found ----- {previous_line}")
                self.text_output_widget.configure(state="normal")
                self.text_output_widget.delete("end-2l", "end-1l")
                self.text_output_widget.configure(state="disabled")

    def write_to_output(self, text: str) -> None:
        """
        Write text to the output frame's text widget.
        :param text: text to write.
        """
        self.set_output(text)
        self.text_output_widget.configure(state="normal")
        self.text_output_widget.insert("end", f"{text}")
        self.text_output_widget.see("end")
        self.text_output_widget.configure(state="disabled")

    def extract_subtitle(self) -> None:
        """
        Use the main module extraction class to extract text from subtitle.
        """
        queue_len = len(self.video_queue)
        self.progress_bar.configure(maximum=queue_len)
        self.video_label.configure(text=f"{self.progress_bar['value']} of {queue_len} Video(s) Completed")
        for video, sub_area in self.video_queue.items():
            self.running = True
            if utils.process_state():
                logger.warning("Process interrupted")
                self.running = False
                self._stop_run()
                return
            self.SubEx.run(video, sub_area)
            self.progress_bar['value'] += 1
            self.video_label.configure(text=f"{self.progress_bar['value']} of {queue_len} Video(s) Completed")
        self.running = False
        self._stop_run()

    def _stop_run(self) -> None:
        """
        Stop program from running.
        """
        logger.debug("Stop button clicked")
        utils.interrupt_process(True)
        # self.vse_thread.join()
        if not self.running:
            self.run_button.configure(text="Run", command=self._run)
            self.menu_file.entryconfig("Open file(s)", state="normal")
            self.current_video = None

    def _run(self) -> None:
        """
        Start the text extraction from video frames.
        """
        logger.debug("Run button clicked")
        if self.current_video:
            utils.interrupt_process(False)
            self.run_button.configure(text='Stop', command=self._stop_run)
            self.menu_file.entryconfig("Open file(s)", state="disabled")
            self.video_capture.release()
            self.video_scale.configure(state="disabled")
            self.progress_bar.configure(value=0)
            self._reset_batch_layout()
            self.vse_thread.start()
        else:
            logger.info("No video has been selected!")

    def _on_closing(self) -> None:
        """
        Method called when window is closed.
        """
        self._stop_run()
        self.root.destroy()
        exit()


if __name__ == '__main__':
    logger.debug("\n\nGUI program Started.")
    rt = Tk()
    SubtitleExtractorGUI(rt)
    rt.mainloop()
    logger.debug("GUI program Ended.\n\n")
