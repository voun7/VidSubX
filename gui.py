import logging
import re
import sys
import time
from threading import Thread
from tkinter import *
from tkinter import filedialog
from tkinter import ttk

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

import utilities.utils as utils
from main import video_details, default_sub_area, SubtitleDetector, SubtitleExtractor
from utilities.logger_setup import get_logger

logger = logging.getLogger(__name__)


class SubtitleExtractorGUI:
    def __init__(self, root: ttk) -> None:
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._create_layout()
        self.video_queue = {}
        self.current_video = None
        self.video_capture = None
        self.running = False
        self.console_redirector()

    def _create_layout(self) -> None:
        """
        Use ttk to create frames for gui.
        """
        # Window title and icon
        self.root.title("Video Subtitle Extractor")
        self.root.iconbitmap("VSE.ico")
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
        self.menu_file.add_command(label="Open file(s)", command=self.open_files)
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

    def _preferences(self):
        self.preference_window = PreferencesUI()

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
        _, _, frame_width, frame_height, = video_details(self.current_video)
        frame_width, frame_height = self.rescale(resolution=(frame_width, frame_height))
        self.canvas.configure(width=frame_width, height=frame_height, bg="white")

    def _set_sub_area(self, subtitle_area: tuple) -> None:
        """
        Set current video subtitle area to new area.
        :param subtitle_area: new subtitle area to be used.
        """
        if not self.running:  # prevents new sub areas from being set while program has a process running.
            self.current_sub_area = subtitle_area
            self.video_queue[f"{self.current_video}"] = self.current_sub_area

    def _on_click(self, event):
        """
        Fires when user clicks on the background ... binds to current rectangle
        """
        if self.current_video:
            self.mouse_start = event.x, event.y
            self.canvas.bind('<Button-1>', self._on_click_rectangle)
            self.canvas.bind('<B1-Motion>', self._on_motion)

    def _on_click_rectangle(self, event):
        """
        Fires when the user clicks on a rectangle ... edits the clicked on rectangle
        """
        if self.current_video:
            x1, y1, x2, y2 = self.canvas.coords(self.current_sub_rect)
            if abs(event.x - x1) < abs(event.x - x2):
                # opposing side was grabbed; swap the anchor and mobile side
                x1, x2 = x2, x1
            if abs(event.y - y1) < abs(event.y - y2):
                y1, y2 = y2, y1
            self.mouse_start = x1, y1

    def _on_motion(self, event):
        """
        Fires when the user drags the mouse ... resizes currently active rectangle
        """
        if self.current_video:
            self.canvas.coords(self.current_sub_rect, *self.mouse_start, event.x, event.y)
            rect_coords = tuple(self.canvas.coords(self.current_sub_rect))
            self._set_sub_area(self.rescale(subtitle_area=rect_coords, scale=2))

    def _draw_subtitle_area(self, subtitle_area: tuple, border_width: int = 4, color: str = "green") -> None:
        """
        Draw subtitle on video frame. x1, y1 = top left corner and x2, y2 = bottom right corner.
        """
        if subtitle_area is None:
            logger.debug("Subtitle coordinates are None.")
            _, _, frame_width, frame_height, = video_details(self.current_video)
            def_sub = default_sub_area(frame_width, frame_height, subtitle_area)
            self._set_sub_area(def_sub)
            x1, y1, x2, y2 = self.rescale(subtitle_area=def_sub)
            self.current_sub_rect = self.canvas.create_rectangle(x1, y1, x2, y2, width=border_width, outline=color)
            self.canvas.event_generate("<Button-1>")
        else:
            self.canvas.coords(self.current_sub_rect, self.rescale(subtitle_area=subtitle_area))
            self.canvas.tag_raise(self.current_sub_rect)

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
        self.canvas.create_image(0, 0, image=photo, anchor=NW)
        self.canvas.image = photo

    def _frame_slider(self, scale_value: str) -> None:
        """
        Make changes according to the position of the slider.
        :param scale_value: current position of the slider.
        """
        scale_value = float(scale_value)
        self._display_video_frame(scale_value)
        self._draw_subtitle_area(self.current_sub_area)

    def _set_frame_slider(self) -> None:
        """
        Activate the slider, then set the starting and ending values of the slider.
        """
        logger.debug("Setting frame slider")
        fps, frame_total, _, _ = video_details(self.current_video)
        duration = (frame_total / fps) - 1

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
        self._draw_subtitle_area(self.current_sub_area)

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
            self.menubar.entryconfig(2, state="normal")
            self.set_output()
            logger.info("Opening video(s)...")
            # Add all opened videos to a queue.
            for filename in filenames:
                logger.info(f"Opened file: {filename}")
                self.video_queue[filename] = None
            logger.info("All video(s) opened!\n")
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

    def detect_subtitles(self) -> None:
        """
        Detect sub area of videos in the queue and set as new sub area.
        """
        logger.info("Detecting subtitle area in video(s)...")
        start = time.perf_counter()
        self.run_button.configure(state="disabled")
        self.menubar.entryconfig(2, state="disabled")
        self.running = True
        for video in self.video_queue.keys():
            logger.info(f"File: {video}")
            sub_dt = SubtitleDetector(video)
            new_sub_area = sub_dt.get_sub_area()
            self.video_queue[video] = new_sub_area
            logger.info(f"New sub area = {new_sub_area}\n")
        self.run_button.configure(state="normal")
        self.menubar.entryconfig(2, state="normal")
        self.running = False
        self._set_video(self._video_indexer()[0])
        end = time.perf_counter()
        logger.info(f"Done detecting subtitle(s)! Total time: {round(end - start, 3)}s\n")

    def run_sub_detection(self) -> None:
        """
        Create a thread to run subtitle detection.
        """
        Thread(target=self.detect_subtitles, daemon=True).start()

    def extract_subtitles(self) -> None:
        """
        Use the main module extraction class to extract text from subtitle.
        """
        sub_ex = SubtitleExtractor()
        queue_len = len(self.video_queue)
        self.progress_bar.configure(maximum=queue_len)
        self.video_label.configure(text=f"{self.progress_bar['value']} of {queue_len} Video(s) Completed")
        logger.info(f"Subtitle Language: {utils.Config.ocr_rec_language}\n")
        for video, sub_area in self.video_queue.items():
            self.running = True
            if utils.Process.interrupt_process:
                logger.warning("Process interrupted")
                self.running = False
                self._stop_run()
                return
            sub_ex.run(video, sub_area)
            self.progress_bar['value'] += 1
            self.video_label.configure(text=f"{self.progress_bar['value']} of {queue_len} Video(s) Completed")
        self.running = False
        self._stop_run()

    def _stop_run(self) -> None:
        """
        Stop program from running.
        """
        logger.debug("Stop button clicked")
        utils.Process.stop_process()
        if not self.running:
            self.run_button.configure(text="Run", command=self._run)
            self.menu_file.entryconfig(0, state="normal")
            self.menubar.entryconfig(1, state="normal")

    def _run(self) -> None:
        """
        Start the text extraction from video frames.
        """
        logger.debug("Run button clicked")
        if self.video_queue and self.current_video:
            self.current_video = None
            self.video_capture.release()
            utils.Process.start_process()
            self.run_button.configure(text='Stop', command=self._stop_run)
            self.menu_file.entryconfig(0, state="disabled")
            self.menubar.entryconfig(1, state="disabled")
            self.menubar.entryconfig(2, state="disabled")
            self.video_scale.configure(state="disabled")
            self.progress_bar.configure(value=0)
            self._reset_batch_layout()
            Thread(target=self.extract_subtitles, daemon=True).start()
        elif self.video_queue:
            logger.info("Open new video(s)!")
        else:
            logger.info("No video has been opened!")

    def _on_closing(self) -> None:
        """
        Method called when window is closed.
        """
        self._stop_run()
        self.root.quit()
        exit()


class PreferencesUI(Toplevel):
    def __init__(self) -> None:
        super().__init__()
        self.focus()
        self.grab_set()
        self._create_layout()

    def _create_layout(self) -> None:
        """
        Create layout for preferences window.
        """
        self.title("Preferences")
        self.iconbitmap("VSE.ico")
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

        # Add tabs to notebook.
        self._frame_extraction_tab()
        self._text_extraction_tab()
        self._subtitle_generator_tab()
        self._subtitle_detection_tab()

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

    def _frame_extraction_tab(self) -> None:
        """
        Creates widgets in the Frame extraction preferences tab frame
        """
        frame_extraction_frame = ttk.Frame(self.notebook_tab)
        frame_extraction_frame.grid(column=0, row=0)
        self.notebook_tab.add(frame_extraction_frame, text=utils.Config.sections[0])

        ttk.Label(frame_extraction_frame, text="Frame Extraction Frequency:").grid(column=0, row=0, padx=25, pady=20)
        self.frame_extraction_frequency = IntVar(value=utils.Config.frame_extraction_frequency)
        self.frame_extraction_frequency.trace_add("write", self._set_reset_button)
        extraction_frequency_entry = ttk.Spinbox(
            frame_extraction_frame,
            from_=1.0, to=10,
            textvariable=self.frame_extraction_frequency,
            state="readonly",
            width=self.spinbox_size
        )
        extraction_frequency_entry.grid(column=1, row=0)

        ttk.Label(frame_extraction_frame, text="Frame Extraction Chunk Size:").grid(column=0, row=1)
        self.frame_extraction_chunk_size = IntVar(value=utils.Config.frame_extraction_chunk_size)
        self.frame_extraction_chunk_size.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        frame_extraction_chunk_size_entry = ttk.Entry(
            frame_extraction_frame,
            textvariable=self.frame_extraction_chunk_size,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        )
        frame_extraction_chunk_size_entry.grid(column=1, row=1)

    def _text_extraction_tab(self) -> None:
        """
        Creates widgets in the Text extraction preferences tab frame
        """
        text_extraction_frame = ttk.Frame(self.notebook_tab)
        text_extraction_frame.grid(column=0, row=0)
        self.notebook_tab.add(text_extraction_frame, text=utils.Config.sections[1])

        ttk.Label(text_extraction_frame, text="Text Extraction Chunk Size:").grid(column=0, row=0, padx=25, pady=20)
        self.text_extraction_chunk_size = IntVar(value=utils.Config.text_extraction_chunk_size)
        self.text_extraction_chunk_size.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        text_extraction_chunk_size_entry = ttk.Entry(
            text_extraction_frame,
            textvariable=self.text_extraction_chunk_size,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        )
        text_extraction_chunk_size_entry.grid(column=1, row=0)

        ttk.Label(text_extraction_frame, text="OCR Max Processes:").grid(column=0, row=1, padx=25)
        self.ocr_max_processes = IntVar(value=utils.Config.ocr_max_processes)
        self.ocr_max_processes.trace_add("write", self._set_reset_button)
        ocr_max_processes_box = ttk.Spinbox(
            text_extraction_frame,
            from_=1.0, to=24,
            textvariable=self.ocr_max_processes,
            state="readonly",
            width=self.spinbox_size
        )
        ocr_max_processes_box.grid(column=1, row=1)

        ttk.Label(text_extraction_frame, text="OCR Recognition Language:").grid(column=0, row=2, padx=25, pady=20)
        self.ocr_rec_language = StringVar(value=utils.Config.ocr_rec_language)
        self.ocr_rec_language.trace_add("write", self._set_reset_button)
        languages = ["ch", "en"]
        ocr_rec_language_box = ttk.Combobox(
            text_extraction_frame,
            textvariable=self.ocr_rec_language,
            values=languages,
            state="readonly",
            width=13
        )
        ocr_rec_language_box.grid(column=1, row=2)

    def _subtitle_generator_tab(self) -> None:
        """
        Creates widgets in the Subtitle generator preferences tab frame
        """
        subtitle_generator_frame = ttk.Frame(self.notebook_tab)
        subtitle_generator_frame.grid(column=0, row=0)
        self.notebook_tab.add(subtitle_generator_frame, text=utils.Config.sections[2])

        ttk.Label(subtitle_generator_frame, text="Text Similarity Threshold:").grid(column=0, row=0, padx=32, pady=20)
        self.text_similarity_threshold = DoubleVar(value=utils.Config.text_similarity_threshold)
        self.text_similarity_threshold.trace_add("write", self._set_reset_button)
        text_similarity_threshold_box = ttk.Spinbox(
            subtitle_generator_frame,
            from_=0, to=1.0,
            increment=0.05,
            textvariable=self.text_similarity_threshold,
            state="readonly",
            width=self.spinbox_size
        )
        text_similarity_threshold_box.grid(column=1, row=0)

    def _subtitle_detection_tab(self):
        """
        Creates widgets in the Subtitle detection preferences tab frame
        """
        subtitle_detection_frame = ttk.Frame(self.notebook_tab)
        subtitle_detection_frame.grid(column=0, row=0)
        self.notebook_tab.add(subtitle_detection_frame, text=utils.Config.sections[3])

        ttk.Label(subtitle_detection_frame, text="Split Start:").grid(column=0, row=0, padx=25, pady=20)
        self.split_start = IntVar(value=utils.Config.split_start)
        self.split_start.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        split_start_entry = ttk.Entry(
            subtitle_detection_frame,
            textvariable=self.split_start,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        )
        split_start_entry.grid(column=1, row=0)

        ttk.Label(subtitle_detection_frame, text="Split Stop:").grid(column=0, row=1)
        self.split_stop = IntVar(value=utils.Config.split_stop)
        self.split_stop.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        split_stop_entry = ttk.Entry(
            subtitle_detection_frame,
            textvariable=self.split_stop,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        )
        split_stop_entry.grid(column=1, row=1)

        ttk.Label(subtitle_detection_frame, text="No of Frames:").grid(column=0, row=2, padx=25, pady=20)
        self.no_of_frames = IntVar(value=utils.Config.no_of_frames)
        self.no_of_frames.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        no_of_frames_entry = ttk.Entry(
            subtitle_detection_frame,
            textvariable=self.no_of_frames,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        )
        no_of_frames_entry.grid(column=1, row=2)

        ttk.Label(subtitle_detection_frame, text="X Axis Padding:").grid(column=0, row=3)
        self.sub_area_x_padding = IntVar(value=utils.Config.sub_area_x_padding)
        self.sub_area_x_padding.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        x_padding_entry = ttk.Entry(
            subtitle_detection_frame,
            textvariable=self.sub_area_x_padding,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        )
        x_padding_entry.grid(column=1, row=3)

        ttk.Label(subtitle_detection_frame, text="Y Axis Padding:").grid(column=0, row=4, padx=25, pady=20)
        self.sub_area_y_padding = IntVar(value=utils.Config.sub_area_y_padding)
        self.sub_area_y_padding.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        y_padding_entry = ttk.Entry(
            subtitle_detection_frame,
            textvariable=self.sub_area_y_padding,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        )
        y_padding_entry.grid(column=1, row=4)

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
            frame_extraction_chunk_size = self.frame_extraction_chunk_size.get()
        except TclError:
            frame_extraction_chunk_size = 0
        try:
            text_extraction_chunk_size = self.text_extraction_chunk_size.get()
        except TclError:
            text_extraction_chunk_size = 0
        try:
            split_start = self.split_start.get()
        except TclError:
            split_start = 0
        try:
            split_stop = self.split_stop.get()
        except TclError:
            split_stop = 0
        try:
            no_of_frames = self.no_of_frames.get()
        except TclError:
            no_of_frames = 0
        try:
            sub_area_x_padding = self.sub_area_x_padding.get()
        except TclError:
            sub_area_x_padding = 0
        try:
            sub_area_y_padding = self.sub_area_y_padding.get()
        except TclError:
            sub_area_y_padding = 0

        values = (
            self.frame_extraction_frequency.get(),
            frame_extraction_chunk_size,
            text_extraction_chunk_size,
            self.ocr_max_processes.get(),
            self.ocr_rec_language.get(),
            self.text_similarity_threshold.get(),
            split_start,
            split_stop,
            no_of_frames,
            sub_area_x_padding,
            sub_area_y_padding
        )

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
                text_similarity_threshold=self.text_similarity_threshold.get()
            )
        except TclError:
            logger.warning("Empty value not saved!")
        self.destroy()


if __name__ == '__main__':
    get_logger()
    logger.debug("\n\nGUI program Started.")
    rt = Tk()
    SubtitleExtractorGUI(rt)
    rt.mainloop()
    logger.debug("GUI program Ended.\n\n")
