import ctypes
import logging
import platform
import re
import sys
import time
from pathlib import Path
from threading import Thread
from tkinter import *
from tkinter import filedialog
from tkinter import ttk

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
from winotify import Notification, audio

import utilities.utils as utils
from main import SubtitleDetector, SubtitleExtractor
from utilities.logger_setup import get_logger

logger = logging.getLogger(__name__)


def set_dpi_scaling() -> None:
    """
    0 = DPI unaware. This app does not scale for DPI changes and is always assumed to have a scale factor of
    100% (96 DPI). It will be automatically scaled by the system on any other DPI setting.

    1 = System DPI aware. This app does not scale for DPI changes. It will query for the DPI once and use that value
    for the lifetime of the app. If the DPI changes, the app will not adjust to the new DPI value. It will be
    automatically scaled up or down by the system when the DPI changes from the system value.

    2 = Per monitor DPI aware. This app checks for the DPI when it is created and adjusts the scale factor whenever the
    DPI changes. These applications are not automatically scaled by the system.
    """
    operating_system = platform.system()
    if operating_system == "Windows":
        # Query DPI Awareness (Windows 10 and 8)
        awareness = ctypes.c_int()
        logger.debug(f"OS = {operating_system}, DPI awareness = {awareness}")

        # Set DPI Awareness  (Windows 10 and 8)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception as dpi_error:
            logger.exception(f"An error occurred while setting the dpi: {dpi_error}")


class SubtitleExtractorGUI:
    def __init__(self, root: ttk) -> None:
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._create_layout()
        self.sub_ex = SubtitleExtractor()
        self.video_queue = {}
        self.current_video = None
        self.video_capture = None
        self.running = False
        self._console_redirector()

    def _create_layout(self) -> None:
        """
        Use ttk to create frames for gui.
        """
        # Window title and icon
        self.window_title = "Video Subtitle Extractor"
        self.icon_file = "VSE.ico"
        self.root.title(self.window_title)
        self.root.iconbitmap(self.icon_file)
        # Do not allow window to be resizable.
        self.root.resizable(FALSE, FALSE)

        # Create window menu bar.
        self._menu_bar()

        # Create main frame that will contain other frames.
        self.main_frame = ttk.Frame(self.root, padding=(5, 5, 5, 15))
        # Main frame's position in root window.
        self.main_frame.grid(column=0, row=0, sticky="N, S, E, W")

        # Frames created in main frame.
        self._video_frame()
        self._work_frame()
        self._output_frame()

    def _menu_bar(self) -> None:
        # Remove dashed lines that come default with tkinter menu bar.
        self.root.option_add('*tearOff', FALSE)

        # Create menu bar in root window.
        self.menubar = Menu(self.root)
        self.root.config(menu=self.menubar)

        # Create menus for menu bar.
        self.menu_file = Menu(self.menubar)

        self.menubar.add_cascade(menu=self.menu_file, label="File")
        self.menubar.add_command(label="Preferences", command=self._preferences)
        self.menubar.add_command(label="Detect Subtitles", command=self.run_sub_detection, state="disabled")

        # Add menu items to file menu.
        self.menu_file.add_command(label="Open file(s)", command=self._open_files)
        self.menu_file.add_command(label="Close", command=self._on_closing)

    def _video_frame(self) -> None:
        """
        Frame that contains the widgets for the video.
        """
        # Create video frame in main frame.
        video_frame = ttk.Frame(self.main_frame)
        video_frame.grid(column=0, row=0)

        # Create canvas widget in video frame.
        self.canvas = Canvas(video_frame, bg="black", cursor="tcross")
        self.canvas.grid(column=0, row=0)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_motion)

        # Create frame slider widget in video frame and label to display value.
        video_work_frame = ttk.Frame(video_frame)
        video_work_frame.grid(column=0, row=1, sticky="W")
        self.video_scale = ttk.Scale(
            video_work_frame, command=self._frame_slider, orient=HORIZONTAL, length=600, state="disabled"
        )
        self.video_scale.grid(column=0, row=1, padx=60)
        self.scale_value = ttk.Label(video_work_frame)
        self.scale_value.grid(column=1, row=1)

    def _work_frame(self) -> None:
        """
        Frame that contains the widgets for working with the video or videos (batch mode).
        """
        # Create work frame in main frame.
        progress_frame = ttk.Frame(self.main_frame)
        progress_frame.grid(column=0, row=1)

        # Create button widget for starting the text extraction.
        self.run_button = ttk.Button(progress_frame, text="Run", command=self._run_sub_extraction)
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
        self.progress_bar.configure(length=600)

    def _set_batch_layout(self) -> None:
        """
        Activate the batch layout from the work frame on the gui.
        """
        logger.debug("Setting batch layout")
        self.progress_bar.configure(length=500)
        self.video_label.configure(text=self._video_indexer()[2])
        self.previous_button.grid(column=2, row=0, padx=10)
        self.next_button.grid(column=4, row=0, padx=10)

    def _preferences(self) -> None:
        self.preference_window = PreferencesUI(self.icon_file)

    def get_scaler(self) -> float:
        """
        Use the frame height to determine which value will be used to scale the video.
        :return: frame scale
        """
        if self.frame_height <= 480:
            return 1.125
        elif self.frame_height <= 720:
            return 0.75
        elif self.frame_height <= 1080:
            return 0.5
        elif self.frame_height <= 1440:
            return 0.375
        elif self.frame_height <= 2160:
            return 0.25
        else:
            logger.debug("frame height above 2160")
            return 0.1

    def rescale(self, frame: np.ndarray = None, subtitle_area: tuple = None, resolution: tuple = None,
                scale: float = None) -> np.ndarray | tuple:
        """
        Method to rescale any frame, subtitle area and resolution.
        """
        if not scale:
            scale = self.get_scaler()

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
            return int(x1), int(y1), int(x2), int(y2)

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
        frame_width, frame_height = self.rescale(resolution=(self.frame_width, self.frame_height))
        self.canvas.configure(width=frame_width, height=frame_height, bg="white")

    def _set_sub_area(self, subtitle_area: tuple) -> None:
        """
        Set current video subtitle area to new area.
        :param subtitle_area: New subtitle area to be used.
        """
        if not self.running:  # prevents new sub areas from being set while program has a process running.
            self.current_sub_area = subtitle_area
            self.video_queue[f"{self.current_video}"] = self.current_sub_area

    def _on_click(self, event: Event) -> None:
        """
        Fires when user clicks on the background ... binds to current rectangle.
        """
        if self.current_video:
            self.mouse_start = event.x, event.y
            self.canvas.bind('<Button-1>', self._on_click_rectangle)
            self.canvas.bind('<B1-Motion>', self._on_motion)

    def _on_click_rectangle(self, event: Event) -> None:
        """
        Fires when the user clicks on a rectangle ... edits the clicked on rectangle.
        """
        if self.current_video:
            x1, y1, x2, y2 = self.canvas.coords(self.current_sub_rect)
            if abs(event.x - x1) < abs(event.x - x2):
                # opposing side was grabbed; swap the anchor and mobile side
                x1, x2 = x2, x1
            if abs(event.y - y1) < abs(event.y - y2):
                y1, y2 = y2, y1
            self.mouse_start = x1, y1

    def _on_motion(self, event: Event) -> None:
        """
        Fires when the user drags the mouse ... resizes currently active rectangle.
        """
        if self.current_video:
            self.canvas.coords(self.current_sub_rect, *self.mouse_start, event.x, event.y)
            rect_coords = tuple(self.canvas.coords(self.current_sub_rect))
            scale = self.frame_height / int(self.canvas['height'])
            self._set_sub_area(self.rescale(subtitle_area=rect_coords, scale=scale))

    def _draw_subtitle_area(self, subtitle_area: tuple, border_width: int = 4, color: str = "green") -> None:
        """
        Draw subtitle on video frame. x1, y1 = top left corner and x2, y2 = bottom right corner.
        """
        if subtitle_area is None:
            logger.debug("Subtitle coordinates are None.")
            def_sub = self.sub_ex.default_sub_area(self.frame_width, self.frame_height, subtitle_area)
            self._set_sub_area(def_sub)
            x1, y1, x2, y2 = self.rescale(subtitle_area=def_sub)
            self.current_sub_rect = self.canvas.create_rectangle(x1, y1, x2, y2, width=border_width, outline=color)
            self.canvas.event_generate("<Button-1>")
        else:
            self.canvas.coords(self.current_sub_rect, self.rescale(subtitle_area=subtitle_area))
            self.canvas.tag_raise(self.current_sub_rect)

    def _display_video_frame(self, millisecond: float = 0.0) -> None:
        """
        Find captured video frame through corresponding second and display on video canvas.
        :param millisecond: Default corresponding millisecond.
        """
        self.video_capture.set(cv.CAP_PROP_POS_MSEC, millisecond)
        _, frame = self.video_capture.read()

        cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        frame_resized = self.rescale(cv2image)

        img = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=photo, anchor=NW)
        self.canvas.image = photo

    def _frame_slider(self, scale_value: str) -> None:
        """
        Make changes according to the position of the slider.
        :param scale_value: current position of the slider.
        """
        scale_value = float(scale_value)
        current_time = self.sub_ex.timecode(scale_value).replace(",", ":")
        total_time = self.sub_ex.timecode(self._video_duration()).replace(",", ":")
        self.scale_value.configure(text=f"{current_time}/{total_time}")
        self._display_video_frame(scale_value)
        self._draw_subtitle_area(self.current_sub_area)

    def _video_duration(self) -> float:
        """
        Returns the total duration of the current_video in milliseconds.
        """
        fps, frame_total, _, _ = self.sub_ex.video_details(self.current_video)
        milliseconds_duration = ((frame_total / fps) * 1000) - 1000
        return milliseconds_duration

    def _set_frame_slider(self) -> None:
        """
        Activate the slider, then set the starting and ending values of the slider.
        """
        logger.debug("Setting frame slider")
        self.video_scale.configure(state="normal", from_=0.0, to=self._video_duration(), value=0)

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
        _, _, self.frame_width, self.frame_height = self.sub_ex.video_details(self.current_video)
        self.video_capture = cv.VideoCapture(self.current_video)
        self._set_canvas()
        self._set_frame_slider()
        self._display_video_frame()
        self._draw_subtitle_area(self.current_sub_area)
        self.root.title(f"{self.window_title} - {Path(self.current_video).name}")
        self.scale_value.configure(text="")

        if len(self.video_queue) > 1:
            self._set_batch_layout()

    def _open_files(self) -> None:
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
            self.menubar.entryconfig(2, state="normal")
            self.clear_output()
            logger.info("Opening video(s)...")
            # Add all opened videos to a queue.
            for filename in filenames:
                logger.info(f"Opened file: {Path(filename).name}")
                self.video_queue[filename] = None
            logger.info("All video(s) opened!\n")
            self._set_video()  # Set one of the opened videos to current video.

    def _console_redirector(self) -> None:
        """
        Redirect console statements to text widget.
        """
        sys.stdout.write = self.write_to_output
        # sys.stderr.write = self.write_to_output

    def clear_output(self, start: str = "1.0", stop: str = "end") -> None:
        """
        Delete text in text widget.
        :param start: Text start position index.
        :param stop: Text stop position index.
        """
        self.text_output_widget.configure(state="normal")
        self.text_output_widget.delete(start, stop)
        self.text_output_widget.configure(state="disabled")

    def _set_progress_output(self, text: str) -> None:
        """
        Overwrite progress bar text in text widget, if detected in previous line.
        """
        progress_pattern = re.compile(r'.+\s\|[#-]+\|\s[\d.]+%\s')
        if progress_pattern.search(text):
            start, stop = 'end - 1 lines', 'end - 1 lines lineend'
            previous_line = self.text_output_widget.get(start, stop)
            if progress_pattern.search(previous_line):
                self.clear_output(start, stop)

    def write_to_output(self, text: str) -> None:
        """
        Write text to the output frame's text widget.
        :param text: Text to write.
        """
        self._set_progress_output(text)
        self.text_output_widget.configure(state="normal")
        self.text_output_widget.insert("end", text)
        self.text_output_widget.see("end")
        self.text_output_widget.configure(state="disabled")

    def send_notification(self, title: str, message: str = "") -> None:
        operating_system = platform.system()
        if operating_system == "Windows":
            toast = Notification(
                app_id=self.window_title,
                title=title,
                msg=message,
                icon=str(Path(self.icon_file).absolute()),
                duration="long"
            )
            toast.set_audio(audio.Default, loop=True)
            toast.show()

    def detect_subtitles(self) -> None:
        """
        Detect sub area of videos in the queue and set as new sub area.
        """
        logger.info("Detecting subtitle area in video(s)...")
        start = time.perf_counter()
        self.running = True
        for video in self.video_queue.keys():
            if utils.Process.interrupt_process:
                logger.warning("Process interrupted\n")
                self.running = False
                self._stop_sub_detection_process()
                return
            logger.info(f"File name: {Path(video).name}")
            sub_dt = SubtitleDetector(video)
            new_sub_area = sub_dt.get_sub_area()
            self.video_queue[video] = new_sub_area
            logger.info(f"New sub area = {new_sub_area}\n")
        self.running = False
        self._stop_sub_detection_process()
        self._set_video(self._video_indexer()[0])
        end = time.perf_counter()
        completion_message = f"Done detecting subtitle(s)! Total time: {round(end - start, 3)}s"
        self.send_notification("Subtitle Detection Completed!", completion_message)
        logger.info(f"{completion_message}\n")

    def _stop_sub_detection_process(self) -> None:
        """
        Stop sub detection from running.
        """
        logger.debug("Stop detection button clicked")
        utils.Process.stop_process()
        if not self.running:
            self._set_run_state("normal", "detection")
            self.menubar.entryconfig(2, label="Detect Subtitles", command=self.run_sub_detection)

    def run_sub_detection(self) -> None:
        """
        Create a thread to run subtitle detection.
        """
        utils.Process.start_process()
        self._set_run_state("disabled", "detection")
        self.menubar.entryconfig(2, label="Stop Sub Detection", command=self._stop_sub_detection_process)
        Thread(target=self.detect_subtitles, daemon=True).start()

    def extract_subtitles(self) -> None:
        """
        Use the main module extraction class to extract text from subtitle.
        """
        queue_len = len(self.video_queue)
        self.progress_bar.configure(maximum=queue_len)
        self.video_label.configure(text=f"{self.progress_bar['value']} of {queue_len} Video(s) Completed")
        logger.info(f"Subtitle Language: {utils.Config.ocr_rec_language}\n")
        self.running = True
        for video, sub_area in self.video_queue.items():
            if utils.Process.interrupt_process:
                logger.warning("Process interrupted\n")
                self.running = False
                self._stop_sub_extraction_process()
                return
            self.sub_ex.run_sub_extraction(video, sub_area)
            self.progress_bar['value'] += 1
            self.video_label.configure(text=f"{self.progress_bar['value']} of {queue_len} Video(s) Completed")
        self.running = False
        self._stop_sub_extraction_process()
        self.send_notification("Subtitle Extraction Completed!")

    def _stop_sub_extraction_process(self) -> None:
        """
        Stop program from running.
        """
        logger.debug("Stop button clicked")
        utils.Process.stop_process()
        if not self.running:
            self.run_button.configure(text="Run", command=self._run_sub_extraction)
            self._set_run_state("normal")

    def _run_sub_extraction(self) -> None:
        """
        Start the text extraction from video frames.
        """
        logger.debug("Run button clicked")
        if self.video_queue and self.current_video:
            self.current_video = None
            self.video_capture.release()
            utils.Process.start_process()
            self.run_button.configure(text='Stop', command=self._stop_sub_extraction_process)
            self._set_run_state("disabled", "extraction")
            self.progress_bar.configure(value=0)
            self._reset_batch_layout()
            Thread(target=self.extract_subtitles, daemon=True).start()
        elif self.video_queue:
            logger.info("Open new video(s)!")
        else:
            logger.info("No video has been opened!")

    def _set_run_state(self, state: str, process_name: str = None) -> None:
        """
        Set state for widgets while process is running.
        """
        logger.debug("Setting run state")
        self.menu_file.entryconfig(0, state=state)
        self.menubar.entryconfig(1, state=state)

        if process_name == "detection":
            self.run_button.configure(state=state)

        if process_name == "extraction":
            self.menubar.entryconfig(2, state=state)
            self.video_scale.configure(state=state)

    def _on_closing(self) -> None:
        """
        Method called when window is closed.
        """
        self._stop_sub_extraction_process()
        self.root.quit()
        exit()


class PreferencesUI(Toplevel):
    def __init__(self, icon_file: str) -> None:
        super().__init__()
        self.icon_file = icon_file
        self.focus()
        self.grab_set()
        self._create_layout()

    def _create_layout(self) -> None:
        """
        Create layout for preferences window.
        """
        self.title("Preferences")
        self.iconbitmap(self.icon_file)
        self.resizable(FALSE, FALSE)

        # Create main frame that will contain notebook.
        main_frame = ttk.Frame(self, padding=(5, 5, 5, 5))
        main_frame.grid(column=0, row=0)

        # Create notebook that will contain tab frames.
        self.notebook_tab = ttk.Notebook(main_frame)
        self.notebook_tab.grid(column=0, row=0)

        # Shared widget values.
        self.entry_size = 15
        self.spinbox_size = 13
        self.wgt_x_padding = 40
        self.wgt_y_padding = 20

        # Add tabs to notebook.
        self._subtitle_detection_tab()
        self._frame_extraction_tab()
        self._text_extraction_tab()
        self._subtitle_generator_tab()

        # Add buttons to window.
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(column=0, row=1, sticky="E")

        self.reset_button = ttk.Button(button_frame, text="Reset", command=self._reset_settings, state="disabled")
        self.reset_button.grid(column=0, row=0, padx=4, pady=4)

        ok_button = ttk.Button(button_frame, text="Ok", command=self._save_settings)
        ok_button.grid(column=1, row=0, padx=4, pady=4)

        cancel_button = ttk.Button(button_frame, text="Cancel", command=self.destroy)
        cancel_button.grid(column=2, row=0, padx=4, pady=4)

        # Set the reset button when layout is created.
        self._set_reset_button()

    def _subtitle_detection_tab(self) -> None:
        """
        Creates widgets in the Subtitle detection preferences tab frame.
        """
        subtitle_detection_frame = ttk.Frame(self.notebook_tab)
        subtitle_detection_frame.grid(column=0, row=0)
        subtitle_detection_frame.grid_columnconfigure(1, weight=1)
        self.notebook_tab.add(subtitle_detection_frame, text=utils.Config.sections[3])

        ttk.Label(subtitle_detection_frame, text="Split Start Part:").grid(
            column=0, row=0, padx=60, pady=self.wgt_y_padding
        )
        self.split_start = DoubleVar(value=utils.Config.split_start)
        self.split_start.trace_add("write", self._set_reset_button)
        ttk.Spinbox(
            subtitle_detection_frame,
            from_=0, to=0.5,
            increment=0.02,
            textvariable=self.split_start,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=0)

        ttk.Label(subtitle_detection_frame, text="Split Stop Part:").grid(column=0, row=1)
        self.split_stop = DoubleVar(value=utils.Config.split_stop)
        self.split_stop.trace_add("write", self._set_reset_button)
        ttk.Spinbox(
            subtitle_detection_frame,
            from_=0.5, to=1.0,
            increment=0.02,
            textvariable=self.split_stop,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=1)

        ttk.Label(subtitle_detection_frame, text="No of Frames:").grid(column=0, row=2, pady=self.wgt_y_padding)
        self.no_of_frames = IntVar(value=utils.Config.no_of_frames)
        self.no_of_frames.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        ttk.Entry(
            subtitle_detection_frame,
            textvariable=self.no_of_frames,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        ).grid(column=1, row=2)

        ttk.Label(subtitle_detection_frame, text="X Axis Padding:").grid(column=0, row=3)
        self.sub_area_x_padding = IntVar(value=utils.Config.sub_area_x_padding)
        self.sub_area_x_padding.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        ttk.Entry(
            subtitle_detection_frame,
            textvariable=self.sub_area_x_padding,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        ).grid(column=1, row=3)

        ttk.Label(subtitle_detection_frame, text="Y Axis Padding:").grid(column=0, row=4, pady=self.wgt_y_padding)
        self.sub_area_y_padding = IntVar(value=utils.Config.sub_area_y_padding)
        self.sub_area_y_padding.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        ttk.Entry(
            subtitle_detection_frame,
            textvariable=self.sub_area_y_padding,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        ).grid(column=1, row=4)

    def _frame_extraction_tab(self) -> None:
        """
        Creates widgets in the Frame extraction preferences tab frame.
        """
        frame_extraction_frame = ttk.Frame(self.notebook_tab)
        frame_extraction_frame.grid(column=0, row=0)
        frame_extraction_frame.grid_columnconfigure(1, weight=1)
        self.notebook_tab.add(frame_extraction_frame, text=utils.Config.sections[0])

        ttk.Label(frame_extraction_frame, text="Frame Extraction Frequency:").grid(
            column=0, row=0, padx=self.wgt_x_padding, pady=self.wgt_y_padding
        )
        self.frame_extraction_frequency = IntVar(value=utils.Config.frame_extraction_frequency)
        self.frame_extraction_frequency.trace_add("write", self._set_reset_button)
        ttk.Spinbox(
            frame_extraction_frame,
            from_=1.0, to=10,
            textvariable=self.frame_extraction_frequency,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=0)

        ttk.Label(frame_extraction_frame, text="Frame Extraction Chunk Size:").grid(column=0, row=1)
        self.frame_extraction_chunk_size = IntVar(value=utils.Config.frame_extraction_chunk_size)
        self.frame_extraction_chunk_size.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        ttk.Entry(
            frame_extraction_frame,
            textvariable=self.frame_extraction_chunk_size,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        ).grid(column=1, row=1)

    def _text_extraction_tab(self) -> None:
        """
        Creates widgets in the Text extraction preferences tab frame.
        """
        text_extraction_frame = ttk.Frame(self.notebook_tab)
        text_extraction_frame.grid(column=0, row=0)
        text_extraction_frame.grid_columnconfigure(1, weight=1)
        self.notebook_tab.add(text_extraction_frame, text=utils.Config.sections[1])

        ttk.Label(text_extraction_frame, text="Text Extraction Chunk Size:").grid(
            column=0, row=0, padx=self.wgt_x_padding, pady=self.wgt_y_padding
        )
        self.text_extraction_chunk_size = IntVar(value=utils.Config.text_extraction_chunk_size)
        self.text_extraction_chunk_size.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        ttk.Entry(
            text_extraction_frame,
            textvariable=self.text_extraction_chunk_size,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        ).grid(column=1, row=0)

        ttk.Label(text_extraction_frame, text="OCR Max Processes:").grid(column=0, row=1)
        self.ocr_max_processes = IntVar(value=utils.Config.ocr_max_processes)
        self.ocr_max_processes.trace_add("write", self._set_reset_button)
        ttk.Spinbox(
            text_extraction_frame,
            from_=1.0, to=24,
            textvariable=self.ocr_max_processes,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=1)

        ttk.Label(text_extraction_frame, text="OCR Recognition Language:").grid(
            column=0, row=2, pady=self.wgt_y_padding
        )
        self.ocr_rec_language = StringVar(value=utils.Config.ocr_rec_language)
        self.ocr_rec_language.trace_add("write", self._set_reset_button)
        languages = ["ch", "en", "ru", "fr", "it", "japan", "korean", "chinese_cht"]
        ttk.Combobox(
            text_extraction_frame,
            textvariable=self.ocr_rec_language,
            values=languages,
            state="readonly",
            width=13
        ).grid(column=1, row=2)

    def _subtitle_generator_tab(self) -> None:
        """
        Creates widgets in the Subtitle generator preferences tab frame.
        """
        subtitle_generator_frame = ttk.Frame(self.notebook_tab)
        subtitle_generator_frame.grid(column=0, row=0)
        subtitle_generator_frame.grid_columnconfigure(1, weight=1)
        self.notebook_tab.add(subtitle_generator_frame, text=utils.Config.sections[2])

        ttk.Label(subtitle_generator_frame, text="Text Similarity Threshold:").grid(
            column=0, row=0, padx=self.wgt_x_padding, pady=self.wgt_y_padding
        )
        self.text_similarity_threshold = DoubleVar(value=utils.Config.text_similarity_threshold)
        self.text_similarity_threshold.trace_add("write", self._set_reset_button)
        ttk.Spinbox(
            subtitle_generator_frame,
            from_=0, to=1.0,
            increment=0.05,
            textvariable=self.text_similarity_threshold,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=0)

    def _set_reset_button(self, *args) -> None:
        """
        Set the reset button based on the value of the text variables.
        :param args: Info of the variable that called the method.
        """
        logger.debug(f"Reset button set by -> {args}")
        default_values = (
            utils.Config.default_frame_extraction_frequency,
            utils.Config.default_frame_extraction_chunk_size,
            utils.Config.default_text_extraction_chunk_size,
            utils.Config.default_ocr_max_processes,
            utils.Config.default_ocr_rec_language,
            utils.Config.default_text_similarity_threshold,
            utils.Config.default_split_start,
            utils.Config.default_split_stop,
            utils.Config.default_no_of_frames,
            utils.Config.default_sub_area_x_padding,
            utils.Config.default_sub_area_y_padding
        )

        try:
            values = (
                self.frame_extraction_frequency.get(),
                self.frame_extraction_chunk_size.get(),
                self.text_extraction_chunk_size.get(),
                self.ocr_max_processes.get(),
                self.ocr_rec_language.get(),
                self.text_similarity_threshold.get(),
                self.split_start.get(),
                self.split_stop.get(),
                self.no_of_frames.get(),
                self.sub_area_x_padding.get(),
                self.sub_area_y_padding.get()
            )
        except TclError:
            values = None

        if default_values == values:
            self.reset_button.configure(state="disabled")
        else:
            self.reset_button.configure(state="normal")

    @staticmethod
    def _check_integer(new_val: str) -> bool:
        """
        Check if the value entered into the entry widget is valid.
        """
        return new_val.isnumeric() or new_val == ""

    def _reset_settings(self) -> None:
        """
        Change the values of the text variables to the default values.
        """
        self.frame_extraction_frequency.set(utils.Config.default_frame_extraction_frequency)
        self.frame_extraction_chunk_size.set(utils.Config.default_frame_extraction_chunk_size)
        self.text_extraction_chunk_size.set(utils.Config.default_text_extraction_chunk_size)
        self.ocr_max_processes.set(utils.Config.default_ocr_max_processes)
        self.ocr_rec_language.set(utils.Config.default_ocr_rec_language)
        self.text_similarity_threshold.set(utils.Config.default_text_similarity_threshold)
        self.split_start.set(utils.Config.default_split_start)
        self.split_stop.set(utils.Config.default_split_stop)
        self.no_of_frames.set(utils.Config.default_no_of_frames)
        self.sub_area_x_padding.set(utils.Config.default_sub_area_x_padding)
        self.sub_area_y_padding.set(utils.Config.default_sub_area_y_padding)

    def _save_settings(self) -> None:
        """
        Save the values of the text variables to the config file.
        """
        try:
            utils.Config.set_config(
                frame_extraction_frequency=self.frame_extraction_frequency.get(),
                frame_extraction_chunk_size=self.frame_extraction_chunk_size.get(),
                text_extraction_chunk_size=self.text_extraction_chunk_size.get(),
                ocr_max_processes=self.ocr_max_processes.get(),
                ocr_rec_language=self.ocr_rec_language.get(),
                text_similarity_threshold=self.text_similarity_threshold.get(),
                split_start=self.split_start.get(),
                split_stop=self.split_stop.get(),
                no_of_frames=self.no_of_frames.get(),
                sub_area_x_padding=self.sub_area_x_padding.get(),
                sub_area_y_padding=self.sub_area_y_padding.get()
            )
        except TclError:
            logger.warning("An error occurred value(s) not saved!")
        self.destroy()


if __name__ == '__main__':
    get_logger()
    logger.debug("\n\nGUI program Started.")
    set_dpi_scaling()
    rt = Tk()
    SubtitleExtractorGUI(rt)
    rt.mainloop()
    logger.debug("GUI program Ended.\n\n")
