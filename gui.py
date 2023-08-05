import ctypes
import logging
import platform
import re
import sys
import time
import tkinter as tk
from pathlib import Path
from threading import Thread
from tkinter import ttk, filedialog, messagebox

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

import utilities.utils as utils
from main import SubtitleDetector, SubtitleExtractor
from utilities.logger_setup import setup_logging
from utilities.win_notify import Notification, Sound

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
        # Query DPI Awareness (Windows 10 and 8).
        awareness = ctypes.c_int()
        logger.debug(f"OS = {operating_system}, DPI awareness = {awareness}")

        # Set DPI Awareness  (Windows 10 and 8).
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception as dpi_error:
            logger.exception(f"An error occurred while setting the dpi: {dpi_error}")


class CustomMessageBox(tk.Toplevel):
    """
    CustomMessageBox class represents a custom messagebox that appends messages on a single window.
    The class inherits from tk.Toplevel and ensures that only one instance of the messagebox is created.
    """
    instance = None

    def __init__(self, icon_file: str, win_title: str) -> None:
        """
        Initialize the CustomMessageBox instance.
        If an instance of CustomMessageBox already exists, it is reused. Otherwise, a new instance is created.
        """
        if CustomMessageBox.instance is not None and CustomMessageBox.instance.winfo_exists():
            # Reuse the existing instance.
            self.__dict__ = CustomMessageBox.instance.__dict__
            return

        super().__init__()
        CustomMessageBox.instance = self

        self.iconbitmap(icon_file)
        self.title(win_title)

        self.text_box = tk.Text(self, state="disabled", borderwidth=10.0, relief="flat")
        self.text_box.grid(sticky="N, S, E, W")

        # Resize text message box frame if main frame is resized.
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def append_message(self, message: str) -> None:
        """
        Append a message to the CustomMessageBox.
        :param message: The message to be appended to the CustomMessageBox.
        """
        self.text_box.configure(state="normal")
        self.text_box.insert("end", message)
        self.text_box.see("end")  # Auto-scroll to the end.
        self.update_size()
        self.text_box.configure(state="disabled")

    def update_size(self) -> None:
        """
        Dynamically modify the dimensions of the text box widget as it increase.
        """
        widget_width = 0
        widget_height = float(self.text_box.index("end"))
        for line in self.text_box.get("1.0", "end").split("\n"):
            if len(line) > widget_width:
                widget_width = len(line)
        self.text_box.config(width=widget_width, height=widget_height)


class SubtitleExtractorGUI:
    def __init__(self, root: ttk) -> None:
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._create_layout()
        self.sub_ex = SubtitleExtractor()
        self.video_queue = {}
        self.current_video = self.video_capture = self.subtitle_rect = self.non_subarea_rect = None
        self.thread_running = False
        self._console_redirector()

    def _create_layout(self) -> None:
        """
        Use ttk to create frames for gui.
        """
        # Window title and icon.
        self.window_title = "Video Subtitle Extractor"
        self.icon_file = "VSE.ico"
        self.root.title(self.window_title)
        self.root.iconbitmap(self.icon_file)
        # Do not allow window to be resizable.
        self.root.resizable(tk.FALSE, tk.FALSE)

        # Create window menu bar.
        self._menu_bar()

        # Create main frame that will contain other frames.
        self.main_frame = ttk.Frame(self.root, padding=(5, 5, 5, 0))
        # Main frame's position in root window.
        self.main_frame.grid(column=0, row=0, sticky="N, S, E, W")

        # Frames created in main frame.
        self._video_frame()
        self._work_frame()
        self._output_frame()

        self.status_label = tk.Label(self.main_frame)
        self.status_label.grid(column=0, row=3, padx=18, sticky="E")

    def _menu_bar(self) -> None:
        # Remove dashed lines that come default with tkinter menu bar.
        self.root.option_add('*tearOff', tk.FALSE)

        # Create menu bar in root window.
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        # Create menus for menu bar.
        self.menu_file = tk.Menu(self.menubar)

        self.menubar.add_cascade(menu=self.menu_file, label="File")
        self.menubar.add_command(label="Preferences", command=self._preferences)
        self.menubar.add_command(label="Detect Subtitles", command=self._run_sub_detection, state="disabled")
        self.menubar.add_command(label="Hide Non-SubArea", command=self._hide_non_subarea, state="disabled")
        self.menubar.add_command(label="||", state="disabled")
        self.menubar.add_command(label="Set Start Frame", command=self._set_current_start_frame, state="disabled")
        self.menubar.add_command(label="Set Stop Frame", command=self._set_current_stop_frame, state="disabled")

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
        # Border width and highlight thickness set to 0 to prevent hidden rectangle parts.
        self.canvas = tk.Canvas(video_frame, cursor="tcross", borderwidth=0, highlightthickness=0)
        self.canvas.grid(column=0, row=0)
        self.canvas.bind("<Button-1>", self._on_click)  # Bind mouse click to canvas.
        self.canvas.bind("<B1-Motion>", self._on_motion)

        # Create frame slider widget in video frame and label to display value.
        video_work_frame = ttk.Frame(video_frame)
        video_work_frame.grid(column=0, row=1, sticky="W")
        self.video_scale = ttk.Scale(video_work_frame, command=self._frame_slider, orient=tk.HORIZONTAL, length=600,
                                     state="disabled")
        self.video_scale.grid(column=0, row=1, padx=60)
        # Show timecode of the video scale.
        self.current_scale_value = ttk.Label(video_work_frame)
        self.current_scale_value.grid(column=1, row=1)
        self.total_scale_value = ttk.Label(video_work_frame)
        self.total_scale_value.grid(column=2, row=1)

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
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=600, mode='determinate')
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
        # Create output frame in main frame.
        output_frame = ttk.Frame(self.main_frame)
        output_frame.grid(column=0, row=2, sticky="N, S, E, W")

        # Create text widget for showing the text extraction details in the output. Does not allow input from gui.
        self.text_output_widget = tk.Text(output_frame, height=12, state="disabled")
        self.text_output_widget.grid(column=0, row=0, sticky="N, S, E, W")

        # Create scrollbar widget for text widget.
        output_scroll = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.text_output_widget.yview)
        output_scroll.grid(column=1, row=0, sticky="N,S")

        # Connect text and scrollbar widgets.
        self.text_output_widget.configure(yscrollcommand=output_scroll.set)

        # Resize output frame if main frame is resized.
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)

    def bind_keys_to_scale(self) -> None:
        """
        Bind keyboard arrows to scale through root.
        """
        logger.debug("Binding keyboard keys")
        self.root.bind("<Left>", lambda e: self.video_scale.set(self.video_scale.get() - self.current_fps))
        self.root.bind("<Right>", lambda e: self.video_scale.set(self.video_scale.get() + self.current_fps))
        self.root.bind("<Up>", lambda e: self.video_scale.set(self.video_scale.get() + self.current_fps * 15))
        self.root.bind("<Down>", lambda e: self.video_scale.set(self.video_scale.get() - self.current_fps * 15))

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
        """
        Open preferences window and set the icon and window location.
        The windows opening location will always be on top of the main window.
        """
        root_x, root_y = self.root.winfo_rootx(), self.root.winfo_rooty()
        win_x, win_y = root_x + 100, root_y + 50
        self.preference_window = PreferencesUI(self.icon_file, win_x, win_y)

    def _get_scale_value(self, target_height: float = 540.0) -> float:
        """
        Use the frame height to determine which value will be used to scale the current video.
        :return: Rescale factor.
        """
        rescale_factor = target_height / self.current_frame_height
        return rescale_factor

    def rescale(self, frame: np.ndarray = None, subtitle_area: tuple = None, resolution: tuple = None,
                scale: float = None) -> np.ndarray | tuple:
        """
        Method to rescale any frame, subtitle area and resolution.
        """
        scale = scale or self._get_scale_value()

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
        # The current frame size will be rescaled (down scale) to set the canvas size.
        frame_width, frame_height = self.rescale(resolution=(self.current_frame_width, self.current_frame_height))
        self.canvas.configure(width=frame_width, height=frame_height)

    def _set_current_sub_area(self, new_subtitle_area: tuple) -> None:
        """
        Set current video subtitle area to new area.
        :param new_subtitle_area: New subtitle area to be used.
        """
        # Get the relative scale (up scale) for the subtitle area.
        scale = self.current_frame_height / int(self.canvas['height'])
        self.current_sub_area = self.rescale(subtitle_area=new_subtitle_area, scale=scale)
        self.video_queue[f"{self.current_video}"][0] = self.current_sub_area  # Set new sub area.

    def _on_click(self, event: tk.Event) -> None:
        """
        Fires when user clicks on the background ... binds to current rectangle.
        """
        # Only allow clicks on canvas when there is currently a video frame being displayed and no thread is running.
        if self.current_video and not self.thread_running:
            self.mouse_start = event.x, event.y
            self.canvas.bind('<Button-1>', self._on_click_rectangle)

    def _on_click_rectangle(self, event: tk.Event) -> None:
        """
        Fires when the user clicks on a rectangle ... edits the clicked on rectangle.
        """
        x1, y1, x2, y2 = self.canvas.coords(self.subtitle_rect)
        if abs(event.x - x1) < abs(event.x - x2):
            # opposing side was grabbed; swap the anchor and mobile side
            x1, x2 = x2, x1
        if abs(event.y - y1) < abs(event.y - y2):
            y1, y2 = y2, y1
        self.mouse_start = x1, y1

    def _on_motion(self, event: tk.Event) -> None:
        """
        Fires when the user drags the mouse ... resizes currently active rectangle.
        """
        # Only allow clicks on canvas when there is currently a video frame being displayed and no thread is running.
        if self.current_video and not self.thread_running:
            # Redraw the rectangle at the given coordinates.
            self.canvas.coords(self.subtitle_rect, *self.mouse_start, event.x, event.y)
            rect_coords = tuple(self.canvas.coords(self.subtitle_rect))  # Get the coordinates of the rectangle.
            self._set_current_sub_area(rect_coords)  # Set new sub area with coordinates of the rectangle.

    def current_non_subarea(self) -> tuple:
        """
        The area of the current video that usually doesn't have subtitles.
        """
        bottom_right_height = int(self.current_frame_height * utils.Config.subarea_height_scaler)
        x1, y1, x2, y2 = 0, 0, self.current_frame_width, bottom_right_height
        return x1, y1, x2, y2

    def _hide_non_subarea(self) -> None:
        """
        Create a rectangle that hides the non subtitle area.
        """
        if self.non_subarea_rect is None:
            logger.debug("Rectangle for non subtitle area created.")
            x1, y1, x2, y2 = self.rescale(subtitle_area=self.current_non_subarea())
            self.non_subarea_rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill="black")
        self.menubar.entryconfig(3, label="Show Non-SubArea", command=self._show_non_subarea)  # Change button config.

    def _show_non_subarea(self) -> None:
        """
        Delete the rectangle covering the non subtitle area.
        """
        logger.debug("Rectangle for non subtitle area deleted.")
        self.canvas.delete(self.non_subarea_rect)
        self.non_subarea_rect = None
        self.menubar.entryconfig(3, label="Hide Non-SubArea", command=self._hide_non_subarea)

    def _set_current_non_subarea(self) -> None:
        """
        Resize the non subtitle area to match the current video.
        """
        if self.non_subarea_rect:
            logger.debug("non_subarea_rect resized.")
            # Rescale (down scale) and redraw the rectangle at the coordinates of current non subtitle area.
            self.canvas.coords(self.non_subarea_rect, self.rescale(subtitle_area=self.current_non_subarea()))

    def _elevate_non_subarea(self) -> None:
        """
        Raise the non subtitle rectangle to the top of the canvas. Prevents rectangle from being hidden.
        """
        if self.non_subarea_rect:
            self.canvas.tag_raise(self.non_subarea_rect)

    def _draw_current_subtitle_area(self) -> None:
        """
        Draw subtitle on video frame. x1, y1 = top left corner and x2, y2 = bottom right corner.
        """
        if self.subtitle_rect is None:
            x1, y1, x2, y2 = self.rescale(subtitle_area=self.current_sub_area)  # Values for creating rectangle.
            self.subtitle_rect = self.canvas.create_rectangle(x1, y1, x2, y2, width=4, outline="green")
            self.canvas.event_generate("<Button-1>")  # Prevents mouse sudden jumps on first canvas mouse click.
        else:
            # Rescale (down scale) and redraw the rectangle at the coordinates of current subtitle_area.
            self.canvas.coords(self.subtitle_rect, self.rescale(subtitle_area=self.current_sub_area))
            self.canvas.tag_raise(self.subtitle_rect)

    def _display_video_frame(self, frame_no: float = 0) -> None:
        """
        Find captured video frame through corresponding frame number and display on video canvas.
        :param frame_no: default corresponding frame_no.
        """
        self.video_capture.set(cv.CAP_PROP_POS_FRAMES, frame_no)  # CAP_PROP_POS_MSEC would be used for milliseconds.
        _, frame = self.video_capture.read()
        if frame is not None:
            cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
            frame_resized = self.rescale(cv2image)  # Make image fit canvas (usually a down scale).

            img = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo

    def _frame_slider(self, scale_value: str) -> None:
        """
        Make changes according to the position of the slider.
        :param scale_value: current position of the slider.
        """
        scale_value = float(scale_value)
        # Update timecode label as slider is moved.
        current_duration = self.sub_ex.frame_no_to_duration(scale_value, self.current_fps)
        self.current_scale_value.configure(text=current_duration)
        self._display_video_frame(scale_value)
        self._elevate_non_subarea()
        self._draw_current_subtitle_area()

    def _set_frame_slider(self) -> None:
        """
        Activate the slider, then set the starting and ending values of the slider.
        """
        logger.debug("Setting frame slider")
        # Set the max size of the frame slider.
        self.video_scale.configure(state="normal", from_=0.0, to=self.current_frame_total, value=0.0)
        # Set the durations labels.
        total_time_duration = self.sub_ex.frame_no_to_duration(self.current_frame_total, self.current_fps)
        self.current_scale_value.configure(text="00:00:00:000")
        self.total_scale_value.configure(text=f"/ {total_time_duration}")
        self.bind_keys_to_scale()

    def _set_current_start_frame(self) -> None:
        """
        Sets the point where frame extraction will start for subtitles for current video.
        """
        logger.debug("Setting start frame.")
        current_frame = self.video_scale.get()
        stop_frame = self.video_queue[f"{self.current_video}"][2]
        if stop_frame and current_frame >= stop_frame:
            self.status_label.configure(text="Start Frame must be before Stop Frame!")
            return
        self.video_queue[f"{self.current_video}"][1] = current_frame  # Start frame changed in dict.
        self._set_status_label()

    def _set_current_stop_frame(self) -> None:
        """
        Sets the point where frame extraction will stop for subtitles for current video.
        """
        logger.debug("Setting stop frame.")
        current_frame = self.video_scale.get()
        start_frame = self.video_queue[f"{self.current_video}"][1]
        if start_frame and current_frame <= start_frame:
            self.status_label.configure(text="Stop Frame must be after Start Frame!")
            return
        self.video_queue[f"{self.current_video}"][2] = current_frame  # Stop frame changed in dict.
        self._set_status_label()

    def _set_status_label(self) -> None:
        """
        Set the status label according to the values of the start and stop frame in video queue.
        """
        start_frame = self.video_queue[f"{self.current_video}"][1]
        stop_frame = self.video_queue[f"{self.current_video}"][2]
        if start_frame or stop_frame:
            start_dur = self.sub_ex.frame_no_to_duration(start_frame, self.current_fps) if start_frame else start_frame
            stop_dur = self.sub_ex.frame_no_to_duration(stop_frame, self.current_fps) if stop_frame else stop_frame
            self.status_label.configure(text=f"Start Frame: {start_dur}, Stop Frame: {stop_dur}")
        else:
            self.status_label.configure(text='')

    def _video_indexer(self) -> tuple:
        """
        Checks the index of the current video in the video queue dictionary using its key.
        """
        index, queue_len = list(self.video_queue).index(self.current_video), len(self.video_queue)
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

    def _remove_video_from_queue(self, video: str) -> None:
        """
        Remove given video from video queue and sets new video if no thread is running.
        :param video: Video to be removed.
        """
        if not self.thread_running:  # To prevent dictionary from changing size during iteration.
            logger.warning(f"Removing {Path(video).name} from queue.\n")
            del self.video_queue[video]
            self._set_video()

    def error_msg(self, error_msg: str) -> None:
        """
        Use tkinter built in error message box to show error message.
        The message is also appended as an error and logged.
        """
        logger.debug(f"ERROR: {error_msg}")
        messagebox.showerror(f"{self.window_title} Error!", error_msg)

    def current_video_exists(self) -> bool:
        """
        Check if a video exists, an error will be sent if the video doesn't exist.
        """
        if Path(self.current_video).exists():
            return True
        else:
            self.error_msg(f"Video: {self.current_video} not found!")
            self.video_scale.configure(state="disabled")
            self._remove_video_from_queue(self.current_video)
            return False

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
        if not self.current_video_exists():  # Prevents errors that happen if the video goes missing.
            return
        self.current_sub_area = list(self.video_queue.values())[video_index][0]
        self.current_fps, self.current_frame_total, self.current_frame_width, self.current_frame_height \
            = self.sub_ex.video_details(self.current_video)
        self.video_capture = cv.VideoCapture(self.current_video)
        self._set_canvas()
        self._set_frame_slider()
        self._set_status_label()
        self._display_video_frame()
        self._set_current_non_subarea()
        self._draw_current_subtitle_area()
        self.root.geometry("")  # Make sure the window is always properly resized after Thread is done opening videos.
        self.root.title(f"{self.window_title} - {Path(self.current_video).name}")

        if len(self.video_queue) > 1:
            self._set_batch_layout()

    def _set_opened_videos(self, filenames: tuple) -> None:
        """
        Add all opened videos to a queue along with default values.
        """
        logger.info("Opening video(s)...")
        self.thread_running = True
        for filename in filenames:
            if utils.Process.interrupt_process:
                logger.debug("Video opening process interrupted\n")
                self.thread_running = False
                self._on_closing()
                return
            logger.info(f"Opened file: {Path(filename).name}")
            _, _, frame_width, frame_height = self.sub_ex.video_details(filename)
            default_subarea = self.sub_ex.default_sub_area(frame_width, frame_height)
            self.video_queue[filename] = [default_subarea, None, None]
        self.thread_running = False
        logger.info("All video(s) opened!\n")
        self._set_gui_state("normal", "opening")
        self._set_video()  # Set one of the opened videos to current video.

    def _open_files(self) -> None:
        """
        Open file dialog to select a file or files then call required methods.
        """
        logger.debug("Open button clicked")
        title = "Select Video(s)"
        file_types = (("All files", "*.*"), ("mp4", "*.mp4"), ("mkv", "*.mkv"))
        filenames = filedialog.askopenfilenames(title=title, filetypes=file_types)

        # This condition prevents the below methods from being called
        # when button is clicked but no files are selected.
        if filenames:
            logger.debug("New files have been selected, video queue, and text widget output cleared")
            self.video_queue = {}  # Empty the video queue before adding the new videos.
            self.clear_output()
            self.progress_bar.configure(value=0)
            utils.Process.start_process()
            self._set_gui_state("disabled", "opening")
            Thread(target=self._set_opened_videos, args=(filenames,), daemon=True).start()

    def error_message_handler(self, text: str) -> None:
        """
        Show the CustomMessageBox and append the error message or messages.
        """
        custom_messagebox = CustomMessageBox(self.icon_file, f"{self.window_title} - Error Message")
        custom_messagebox.append_message(text)

    def _console_redirector(self) -> None:
        """
        Redirect console statements to text widget.
        """
        sys.stdout.write = self.write_to_output
        sys.stderr.write = self.error_message_handler

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
        progress_pattern = re.compile(r'.+\s\|[ #-]+\|\s[\d.]+%\s')
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
            sound = Sound.get_sound_value(utils.Config.win_notify_sound)
            toast.set_audio(sound, loop=utils.Config.win_notify_loop_sound)
            toast.show()

    def _detect_subtitles(self) -> None:
        """
        Detect sub area of videos in the queue and set as new sub area.
        """
        logger.info("Detecting subtitle area in video(s)...")
        start = time.perf_counter()
        use_search_area = utils.Config.use_search_area
        self.thread_running = True
        for video in self.video_queue.keys():
            if utils.Process.interrupt_process:
                logger.warning("Process interrupted\n")
                self.thread_running = False
                self._stop_sub_detection_process()
                return
            sub_dt = SubtitleDetector(video, use_search_area)
            new_sub_area = sub_dt.get_sub_area()
            self.video_queue[video][0] = new_sub_area
        self.thread_running = False
        self._stop_sub_detection_process()
        self.current_sub_area = list(self.video_queue.values())[self._video_indexer()[0]][0]
        self._draw_current_subtitle_area()
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
        if not self.thread_running:
            self._set_gui_state("normal", "detection")
            self.menubar.entryconfig(2, label="Detect Subtitles", command=self._run_sub_detection)

    def _run_sub_detection(self) -> None:
        """
        Create a thread to run subtitle detection.
        """
        utils.Process.start_process()
        self._set_gui_state("disabled", "detection")
        self.menubar.entryconfig(2, label="Stop Sub Detection", command=self._stop_sub_detection_process)
        Thread(target=self._detect_subtitles, daemon=True).start()

    def extract_subtitles(self) -> None:
        """
        Use the main module extraction class to extract text from subtitle.
        """
        queue_len = len(self.video_queue)
        self.progress_bar.configure(maximum=queue_len)
        self.video_label.configure(text=f"{self.progress_bar['value']} of {queue_len} Video(s) Completed")
        logger.info(f"Subtitle Language: {utils.Config.ocr_rec_language}\n")
        self.thread_running = True
        for video, sub_info in self.video_queue.items():
            sub_area, start_frame, stop_frame = sub_info[0], sub_info[1], sub_info[2]
            start_frame = int(start_frame) if start_frame else start_frame
            stop_frame = int(stop_frame) if stop_frame else stop_frame
            if utils.Process.interrupt_process:
                logger.warning("Process interrupted\n")
                self.thread_running = False
                self._stop_sub_extraction_process()
                return
            self.sub_ex.run_extraction(video, sub_area, start_frame, stop_frame)
            self.progress_bar['value'] += 1
            self.video_label.configure(text=f"{self.progress_bar['value']} of {queue_len} Video(s) Completed")
        self.thread_running = False
        self._stop_sub_extraction_process()
        self.send_notification("Subtitle Extraction Completed!")

    def _stop_sub_extraction_process(self) -> None:
        """
        Stop program from running.
        """
        logger.debug("Stop button clicked")
        utils.Process.stop_process()
        if not self.thread_running:
            self.run_button.configure(text="Run", command=self._run_sub_extraction)
            self._set_gui_state("normal")

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
            self._set_gui_state("disabled", "extraction")
            self.progress_bar.configure(value=0)
            self._reset_batch_layout()
            Thread(target=self.extract_subtitles, daemon=True).start()
        elif self.video_queue:
            logger.info("Open new video(s)!")
        else:
            logger.info("No video has been opened!")

    def _set_gui_state(self, state: str, process_name: str = None) -> None:
        """
        Set state for widgets while process is running.
        """
        logger.debug("Setting gui state")
        self.menu_file.entryconfig(0, state=state)  # Open file button.
        self.menubar.entryconfig(1, state=state)  # Preferences button.

        if process_name == "opening":
            self.previous_button.configure(state=state)
            self.next_button.configure(state=state)

        if process_name in ("detection", "opening"):
            self.run_button.configure(state=state)

        if process_name in ("extraction", "opening"):
            self.menubar.entryconfig(2, state=state)  # Detect button.
            self.menubar.entryconfig(3, state=state)  # Hide Non-SubArea button.
            self.menubar.entryconfig(5, state=state)  # Set Start Frame button.
            self.menubar.entryconfig(6, state=state)  # Set Stop Frame button.
            self.video_scale.configure(state=state)

    def _on_closing(self) -> None:
        """
        Method called when window is closed.
        """
        utils.Process.stop_process()
        if not self.thread_running:
            self.root.quit()


class PreferencesUI(tk.Toplevel):
    def __init__(self, icon_file: str, win_x: int, win_y: int) -> None:
        super().__init__()
        self.icon_file = icon_file
        self.geometry(f"+{win_x}+{win_y}")  # Set window position.
        self.focus()
        self.grab_set()
        self._create_layout()

    def _create_layout(self) -> None:
        """
        Create layout for preferences window.
        """
        self.title("Preferences")
        self.iconbitmap(self.icon_file)
        self.resizable(tk.FALSE, tk.FALSE)

        # Create main frame that will contain notebook.
        main_frame = ttk.Frame(self, padding=(5, 5, 5, 5))
        main_frame.grid(column=0, row=0)

        # Create notebook that will contain tab frames.
        self.notebook_tab = ttk.Notebook(main_frame)
        self.notebook_tab.grid(column=0, row=0)

        # Shared widget values.
        self.entry_size = 15
        self.spinbox_size = 13
        self.combobox_size = 15
        self.wgt_x_padding = 70
        self.wgt_y_padding = 20

        # Add tabs to notebook.
        self._subtitle_detection_tab()
        self._frame_extraction_tab()
        self._text_extraction_tab()
        self._subtitle_generator_tab()
        self._notifications_tab()

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

        ttk.Label(subtitle_detection_frame, text="Split Start (Relative position):").grid(
            column=0, row=0, padx=self.wgt_x_padding, pady=self.wgt_y_padding
        )
        self.split_start = tk.DoubleVar(value=utils.Config.split_start)
        self.split_start.trace_add("write", self._set_reset_button)
        ttk.Spinbox(
            subtitle_detection_frame,
            from_=0, to=0.5,
            increment=0.02,
            textvariable=self.split_start,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=0)

        ttk.Label(subtitle_detection_frame, text="Split Stop (Relative position):").grid(column=0, row=1)
        self.split_stop = tk.DoubleVar(value=utils.Config.split_stop)
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
        self.no_of_frames = tk.IntVar(value=utils.Config.no_of_frames)
        self.no_of_frames.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        ttk.Entry(
            subtitle_detection_frame,
            textvariable=self.no_of_frames,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        ).grid(column=1, row=2)

        ttk.Label(subtitle_detection_frame, text="X Axis Padding (Relative):").grid(column=0, row=3)
        self.sub_area_x_rel_padding = tk.DoubleVar(value=utils.Config.sub_area_x_rel_padding)
        self.sub_area_x_rel_padding.trace_add("write", self._set_reset_button)
        ttk.Spinbox(
            subtitle_detection_frame,
            from_=0.5, to=1.0,
            increment=0.01,
            textvariable=self.sub_area_x_rel_padding,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=3)

        ttk.Label(subtitle_detection_frame, text="Y Axis Padding (Absolute):").grid(
            column=0, row=4, pady=self.wgt_y_padding
        )
        self.sub_area_y_abs_padding = tk.IntVar(value=utils.Config.sub_area_y_abs_padding)
        self.sub_area_y_abs_padding.trace_add("write", self._set_reset_button)
        check_int = (self.register(self._check_integer), '%P')
        ttk.Entry(
            subtitle_detection_frame,
            textvariable=self.sub_area_y_abs_padding,
            validate='key',
            validatecommand=check_int,
            width=self.entry_size
        ).grid(column=1, row=4)

        self.use_search_area = tk.BooleanVar(value=utils.Config.use_search_area)
        self.use_search_area.trace_add("write", self._set_reset_button)
        ttk.Checkbutton(
            subtitle_detection_frame,
            text='Use Default Search Area',
            variable=self.use_search_area
        ).grid(column=0, row=5)

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
        self.frame_extraction_frequency = tk.IntVar(value=utils.Config.frame_extraction_frequency)
        self.frame_extraction_frequency.trace_add("write", self._set_reset_button)
        ttk.Spinbox(
            frame_extraction_frame,
            from_=1.0, to=10,
            textvariable=self.frame_extraction_frequency,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=0)

        ttk.Label(frame_extraction_frame, text="Frame Extraction Chunk Size:").grid(column=0, row=1)
        self.frame_extraction_chunk_size = tk.IntVar(value=utils.Config.frame_extraction_chunk_size)
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
        self.text_extraction_chunk_size = tk.IntVar(value=utils.Config.text_extraction_chunk_size)
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
        self.ocr_max_processes = tk.IntVar(value=utils.Config.ocr_max_processes)
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
        self.ocr_rec_language = tk.StringVar(value=utils.Config.ocr_rec_language)
        self.ocr_rec_language.trace_add("write", self._set_reset_button)
        import custom_paddleocr.paddleocr as pd
        languages = list(pd.MODEL_URLS['OCR'][pd.DEFAULT_OCR_MODEL_VERSION]['rec'].keys())
        ttk.Combobox(
            text_extraction_frame,
            textvariable=self.ocr_rec_language,
            values=languages,
            state="readonly",
            width=self.combobox_size
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
        self.text_similarity_threshold = tk.DoubleVar(value=utils.Config.text_similarity_threshold)
        self.text_similarity_threshold.trace_add("write", self._set_reset_button)
        ttk.Spinbox(
            subtitle_generator_frame,
            from_=0, to=1.0,
            increment=0.05,
            textvariable=self.text_similarity_threshold,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=0)

        ttk.Label(subtitle_generator_frame, text="Minimum Consecutive Sub Duration (ms):").grid(column=0, row=1)
        self.min_consecutive_sub_dur_ms = tk.DoubleVar(value=utils.Config.min_consecutive_sub_dur_ms)
        self.min_consecutive_sub_dur_ms.trace_add("write", self._set_reset_button)
        check_float = (self.register(self._check_float), '%P')
        ttk.Entry(
            subtitle_generator_frame,
            textvariable=self.min_consecutive_sub_dur_ms,
            validate='key',
            validatecommand=check_float,
            width=self.entry_size
        ).grid(column=1, row=1)

        ttk.Label(subtitle_generator_frame, text="Max Consecutive Short Durations:").grid(
            column=0, row=2, pady=self.wgt_y_padding
        )
        self.max_consecutive_short_durs = tk.IntVar(value=utils.Config.max_consecutive_short_durs)
        self.max_consecutive_short_durs.trace_add("write", self._set_reset_button)
        ttk.Spinbox(
            subtitle_generator_frame,
            from_=2, to=10,
            increment=1,
            textvariable=self.max_consecutive_short_durs,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=2)

        ttk.Label(subtitle_generator_frame, text="Minimum Sub Duration (ms):").grid(column=0, row=3)
        self.min_sub_duration_ms = tk.DoubleVar(value=utils.Config.min_sub_duration_ms)
        self.min_sub_duration_ms.trace_add("write", self._set_reset_button)
        check_float = (self.register(self._check_float), '%P')
        ttk.Entry(
            subtitle_generator_frame,
            textvariable=self.min_sub_duration_ms,
            validate='key',
            validatecommand=check_float,
            width=self.entry_size
        ).grid(column=1, row=3)

    def _notifications_tab(self) -> None:
        """
        Choose notification tab depending on platform os.
        """
        operating_system = platform.system()
        if operating_system == "Windows":
            self._win_notifications_tab()
        else:
            self.win_notify_sound = tk.StringVar(value=utils.Config.win_notify_sound)
            self.win_notify_loop_sound = tk.BooleanVar(value=utils.Config.win_notify_loop_sound)

    def _win_notifications_tab(self) -> None:
        """
        Creates widgets in the Notifications preferences tab frame.
        """
        notification_frame = ttk.Frame(self.notebook_tab)
        notification_frame.grid(column=0, row=0)
        notification_frame.grid_columnconfigure(1, weight=1)
        self.notebook_tab.add(notification_frame, text=utils.Config.sections[4])

        ttk.Label(notification_frame, text="Notification Sound:").grid(
            column=0, row=0, padx=self.wgt_x_padding, pady=self.wgt_y_padding
        )
        self.win_notify_sound = tk.StringVar(value=utils.Config.win_notify_sound)
        self.win_notify_sound.trace_add("write", self._set_reset_button)
        ttk.Combobox(
            notification_frame,
            textvariable=self.win_notify_sound,
            values=Sound.all_sounds(),
            state="readonly",
            width=self.combobox_size
        ).grid(column=1, row=0)

        self.win_notify_loop_sound = tk.BooleanVar(value=utils.Config.win_notify_loop_sound)
        self.win_notify_loop_sound.trace_add("write", self._set_reset_button)
        ttk.Checkbutton(
            notification_frame,
            text='Loop Notification Sound',
            variable=self.win_notify_loop_sound
        ).grid(column=1, row=1)

    def _set_reset_button(self, *args) -> None:
        """
        Set the reset button based on whether the value of the text variables are the same as the default values.
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
            utils.Config.default_min_consecutive_sub_dur_ms,
            utils.Config.default_max_consecutive_short_durs,
            utils.Config.default_min_sub_duration_ms,
            utils.Config.default_split_start,
            utils.Config.default_split_stop,
            utils.Config.default_no_of_frames,
            utils.Config.default_sub_area_x_rel_padding,
            utils.Config.default_sub_area_y_abs_padding,
            utils.Config.default_use_search_area,
            utils.Config.default_win_notify_sound,
            utils.Config.default_win_notify_loop_sound
        )

        try:
            values = (
                self.frame_extraction_frequency.get(),
                self.frame_extraction_chunk_size.get(),
                self.text_extraction_chunk_size.get(),
                self.ocr_max_processes.get(),
                self.ocr_rec_language.get(),
                self.text_similarity_threshold.get(),
                self.min_consecutive_sub_dur_ms.get(),
                self.max_consecutive_short_durs.get(),
                self.min_sub_duration_ms.get(),
                self.split_start.get(),
                self.split_stop.get(),
                self.no_of_frames.get(),
                self.sub_area_x_rel_padding.get(),
                self.sub_area_y_abs_padding.get(),
                self.use_search_area.get(),
                self.win_notify_sound.get(),
                self.win_notify_loop_sound.get()
            )
        except tk.TclError:
            values = None

        if default_values == values:
            self.reset_button.configure(state="disabled")
        else:
            self.reset_button.configure(state="normal")

    @staticmethod
    def _check_integer(entry_value: str) -> bool:
        """
        Check if the value entered into the entry widget is valid.
        """
        if entry_value == "":
            return True
        try:
            int(entry_value)
            return True
        except ValueError:
            return False

    @staticmethod
    def _check_float(entry_value: str) -> bool:
        """
        Check if the value entered into the entry widget is valid.
        """
        if entry_value == "":
            return True
        try:
            float(entry_value)
            return True
        except ValueError:
            return False

    def _reset_settings(self) -> None:
        """
        Change the values of the text variables to the default values.
        """
        # Frame extraction settings.
        self.frame_extraction_frequency.set(utils.Config.default_frame_extraction_frequency)
        self.frame_extraction_chunk_size.set(utils.Config.default_frame_extraction_chunk_size)
        # Text extraction settings.
        self.text_extraction_chunk_size.set(utils.Config.default_text_extraction_chunk_size)
        self.ocr_max_processes.set(utils.Config.default_ocr_max_processes)
        self.ocr_rec_language.set(utils.Config.default_ocr_rec_language)
        # Subtitle generator settings.
        self.text_similarity_threshold.set(utils.Config.default_text_similarity_threshold)
        self.min_consecutive_sub_dur_ms.set(utils.Config.default_min_consecutive_sub_dur_ms)
        self.max_consecutive_short_durs.set(utils.Config.default_max_consecutive_short_durs)
        self.min_sub_duration_ms.set(utils.Config.default_min_sub_duration_ms)
        # Subtitle detection settings.
        self.split_start.set(utils.Config.default_split_start)
        self.split_stop.set(utils.Config.default_split_stop)
        self.no_of_frames.set(utils.Config.default_no_of_frames)
        self.sub_area_x_rel_padding.set(utils.Config.default_sub_area_x_rel_padding)
        self.sub_area_y_abs_padding.set(utils.Config.default_sub_area_y_abs_padding)
        self.use_search_area.set(utils.Config.default_use_search_area)
        # Notification settings.
        self.win_notify_sound.set(utils.Config.default_win_notify_sound)
        self.win_notify_loop_sound.set(utils.Config.default_win_notify_loop_sound)

    def _save_settings(self) -> None:
        """
        Save the values of the text variables to the config file.
        """
        try:
            utils.Config.set_config(
                **{
                    # Frame extraction settings.
                    utils.Config.keys[0]: self.frame_extraction_frequency.get(),
                    utils.Config.keys[1]: self.frame_extraction_chunk_size.get(),
                    # Text extraction settings.
                    utils.Config.keys[2]: self.text_extraction_chunk_size.get(),
                    utils.Config.keys[3]: self.ocr_max_processes.get(),
                    utils.Config.keys[4]: self.ocr_rec_language.get(),
                    # Subtitle generator settings.
                    utils.Config.keys[5]: self.text_similarity_threshold.get(),
                    utils.Config.keys[6]: self.min_consecutive_sub_dur_ms.get(),
                    utils.Config.keys[7]: self.max_consecutive_short_durs.get(),
                    utils.Config.keys[8]: self.min_sub_duration_ms.get(),
                    # Subtitle detection settings.
                    utils.Config.keys[9]: self.split_start.get(),
                    utils.Config.keys[10]: self.split_stop.get(),
                    utils.Config.keys[11]: self.no_of_frames.get(),
                    utils.Config.keys[12]: self.sub_area_x_rel_padding.get(),
                    utils.Config.keys[13]: self.sub_area_y_abs_padding.get(),
                    utils.Config.keys[14]: self.use_search_area.get(),
                    # Notification settings.
                    utils.Config.keys[15]: self.win_notify_sound.get(),
                    utils.Config.keys[16]: self.win_notify_loop_sound.get()
                }
            )
        except tk.TclError:
            logger.warning("An error occurred value(s) not saved!")
        self.destroy()


if __name__ == '__main__':
    setup_logging()
    logger.debug("\n\nGUI program Started.")
    set_dpi_scaling()
    rt = tk.Tk()
    SubtitleExtractorGUI(rt)
    rt.mainloop()
    logger.debug("GUI program Ended.\n\n")
