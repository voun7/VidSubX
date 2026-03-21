import ctypes
import logging
import platform
import sys
import tkinter as tk
from datetime import timedelta
from multiprocessing import freeze_support
from os import cpu_count
from pathlib import Path
from threading import Thread
from time import perf_counter
from tkinter import ttk, filedialog, messagebox

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
from wakepy import keep

from infra.app_paths import AppPaths
from infra.linux_notify import send_linux_notification
from infra.logger_setup import setup_logging
from infra.win_notify import Notification, Sound
from main import SubtitleDetector, SubtitleExtractor, setup_ocr
from shared.config import CONFIG
from shared.process import Process
from shared.utils import video_details, default_sub_area, frame_no_to_duration, check_for_updates

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
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore
        except Exception as dpi_error:
            logger.exception(f"An error occurred while setting the dpi: {dpi_error}")


def set_title_bar_colour(window_id: int, use_dark: bool) -> None:
    """
    For windows use the Windows DWM API to set dark mode (attribute 20).
    This works for Windows 10 (build 17763+) and Windows 11
    """
    if platform.system() == "Windows":
        try:
            set_window_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute  # type: ignore
            get_parent = ctypes.windll.user32.GetParent  # type: ignore
            hwnd = get_parent(window_id)
            rendering_policy = ctypes.c_int(1 if use_dark else 0)
            # 20 is the ID for the Immersive Dark Mode attribute
            set_window_attribute(hwnd, 20, ctypes.byref(rendering_policy), ctypes.sizeof(rendering_policy))
        except Exception as error:
            logger.exception(f"Dark mode title bar not supported: {error}")


class CustomMessageBox(tk.Toplevel):
    """
    CustomMessageBox class represents a custom messagebox that appends messages on a single window.
    The class inherits from tk.Toplevel and ensures that only one instance of the messagebox is created.
    """
    instance = None

    def __init__(self, icon_file: Path, win_title: str, win_x: int, win_y: int) -> None:
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
        if platform.system() == "Windows":
            self.iconbitmap(icon_file)
        self.title(win_title)
        self.geometry(f"+{win_x}+{win_y}")  # Set window position.
        self.focus()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.text_box = tk.Text(self, state="disabled", borderwidth=10.0, relief="flat")
        self.text_box.grid(sticky="N, S, E, W")
        # Create scrollbar widget for text widget.
        output_scroll = ttk.Scrollbar(self, orient="vertical", command=self.text_box.yview)
        output_scroll.grid(column=1, row=0, sticky="N,S")

        # Resize text message box frame if main frame is resized.
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        if CONFIG.use_dark_mode:
            set_title_bar_colour(self.winfo_id(), True)
            self.text_box.configure(bg="#1e1e1e", fg="white")
        else:
            set_title_bar_colour(self.winfo_id(), False)
            self.text_box.configure(bg="white", fg="black")

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

    def log_errors(self) -> None:
        """
        Send all the error messages from the widget to the log.
        """
        error_msgs = self.text_box.get("1.0", "end")
        logger.debug(f"ERROR: \n{error_msgs}")

    def _on_closing(self) -> None:
        """
        Destroy custom message box.
        """
        self.log_errors()
        self.destroy()


class SubtitleExtractorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._create_layout()
        self.video_queue = {}
        self.current_video = self.video_capture = self.subtitle_rect = self.non_subarea_rect = None
        self.video_target_height = 500
        self.thread_running = False
        self._console_redirector()
        self._update_checker()

    def _create_layout(self) -> None:
        """
        Use ttk to create frames for gui.
        """
        # Window title and icon.
        self.window_title, self.icon_file = AppPaths.program_name, AppPaths.icon_file
        self.root.withdraw()  # Hide root window while layout is being drawn
        self.root.title(self.window_title)
        if platform.system() == "Windows":
            self.root.iconbitmap(self.icon_file)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)  # The main frame
        self.style = ttk.Style()
        self.default_theme = self.style.theme_use()
        # Create window menu bar.
        self._menu_bar()
        # Create main frame that will contain other frames.
        self.main_frame = ttk.Frame(self.root, padding=(5, 5, 5, 0))
        # Main frame's position in root window.
        self.main_frame.grid(column=0, row=1, sticky="N, S, E, W")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)  # Video Frame
        self.main_frame.grid_rowconfigure(1, weight=1)  # Work Frame
        self.main_frame.grid_rowconfigure(2, weight=1)  # Output Frame
        # Frames created in main frame.
        self._video_frame()
        self._work_frame()
        self._output_frame()

        self.status_label = ttk.Label(self.main_frame, text=f"v{AppPaths.version_file.read_text()}")
        self.status_label.grid(column=0, row=3, padx=18, sticky="E")
        # Display window after layout is ready
        self._toggle_theme()
        self.root.update_idletasks()
        self.root.after(10, self.root.deiconify)  # type: ignore
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())  # Custom menubar will always be visible

    def _apply_menu_hover(self, force_update: bool = False) -> None:
        widgets = (self.menu_pref_btn, self.menu_detect_btn, self.menu_hide_btn,
                   self.menu_start_btn, self.menu_stop_btn)
        for widget in widgets:
            # If disabled, or new colors are needed, the binding is removed
            if widget["state"] == "disabled" or force_update:
                widget.unbind("<Enter>")  # Safe to do even if not bound
                widget.unbind("<Leave>")
                widget._hover_bound = False

            # If normal and not currently bound (or just reset above)
            if widget["state"] == "normal" and not getattr(widget, "_hover_bound", False):
                normal_bg = widget.cget("bg")
                active_bg = widget.cget("activebackground") or normal_bg
                widget.bind("<Enter>", lambda _, w=widget, bg=active_bg: w.configure(bg=bg))
                widget.bind("<Leave>", lambda _, w=widget, bg=normal_bg: w.configure(bg=bg))
                widget._hover_bound = True

    def _menu_bar(self) -> None:
        # Remove dashed lines that come default with tkinter menu bar.
        self.root.option_add('*tearOff', tk.FALSE)

        # Create menu bar frame in root window.
        self.menubar_frame = tk.Frame(self.root)
        self.menubar_frame.grid(column=0, row=0, sticky="E, W")
        # Common styles
        menu_style = {"padx": 8, "bd": 0, "highlightthickness": 0}

        # File Menu
        self.file_mb = tk.Menubutton(self.menubar_frame, text="File", **menu_style)
        self.file_menu = tk.Menu(self.file_mb)
        self.file_mb["menu"] = self.file_menu
        self.file_mb.grid(column=0, row=0)

        self.file_menu.add_command(label="Open file(s)", command=self._open_files)
        self.file_menu.add_command(label="Close", command=self._on_closing)
        self.file_menu.add_separator()
        self.check_for_updates = tk.BooleanVar(value=CONFIG.check_for_updates)
        self.file_menu.add_checkbutton(label="Check for Updates", command=self._set_update_checker,
                                       variable=self.check_for_updates)

        # View Menu
        self.view_mb = tk.Menubutton(self.menubar_frame, text="View", **menu_style)
        self.view_menu = tk.Menu(self.view_mb)
        self.view_mb["menu"] = self.view_menu
        self.view_mb.grid(column=1, row=0)

        # View Items
        self.use_dark_mode = tk.BooleanVar(value=CONFIG.use_dark_mode)
        self.view_menu.add_checkbutton(label="Dark Mode", command=self._toggle_theme, variable=self.use_dark_mode)
        self.view_menu.add_separator()
        self.view_menu.add_command(label="Video Zoom In   (Ctrl+Plus)", command=lambda: self.resize_video("equal"))
        self.view_menu.add_command(label="Video Zoom Out  (Ctrl+Minus)", command=lambda: self.resize_video("minus"))
        self.root.bind("<Control-equal>", self.resize_video)  # equal instead of plus. It prevents need for shift key.
        self.root.bind("<Control-minus>", self.resize_video)

        # Direct Action Buttons
        self.menu_pref_btn = tk.Button(self.menubar_frame, text="Preferences", command=self._preferences, **menu_style)
        self.menu_pref_btn.grid(column=2, row=0)

        self.menu_detect_btn = tk.Button(self.menubar_frame, text="Detect Subtitles", command=self._run_sub_detection,
                                         state="disabled", **menu_style)
        self.menu_detect_btn.grid(column=3, row=0)

        self.menu_hide_btn = tk.Button(self.menubar_frame, text="Hide Non-SubArea", command=self._hide_non_subarea,
                                       state="disabled", **menu_style)
        self.menu_hide_btn.grid(column=4, row=0)

        self.sep_label = tk.Label(self.menubar_frame, text="||")
        self.sep_label.grid(column=5, row=0)

        self.menu_start_btn = tk.Button(self.menubar_frame, text="Set Start Frame",
                                        command=self._set_current_start_frame, state="disabled", **menu_style)
        self.menu_start_btn.grid(column=6, row=0)

        self.menu_stop_btn = tk.Button(self.menubar_frame, text="Set Stop Frame", command=self._set_current_stop_frame,
                                       state="disabled", **menu_style)
        self.menu_stop_btn.grid(column=7, row=0)

    def _video_frame(self) -> None:
        """
        Frame that contains the widgets for the video.
        """
        # Create video frame in main frame.
        video_frame = ttk.Frame(self.main_frame)
        video_frame.grid(column=0, row=0)
        video_frame.grid_columnconfigure(0, weight=1)
        video_frame.grid_rowconfigure(0, weight=1)

        # Create canvas widget in video frame.
        # Border width and highlight thickness set to 0 to prevent hidden rectangle parts.
        self.canvas = tk.Canvas(video_frame, cursor="tcross", borderwidth=0, highlightthickness=0)
        self.canvas.grid(column=0, row=0)
        self.canvas.bind("<Button-1>", self._on_click)  # Bind mouse click to canvas.
        self.canvas.bind("<B1-Motion>", self._on_motion)

        # Create frame slider widget in video frame and label to display value.
        video_work_frame = ttk.Frame(video_frame)
        video_work_frame.grid(column=0, row=1)

        self.video_scale = ttk.Scale(video_work_frame, command=self._frame_slider, orient="horizontal", length=600,
                                     state="disabled")
        self.video_scale.grid(column=0, row=1, padx=(0, 20))  # Only the right side is padded.
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
        progress_frame.grid_columnconfigure(0, weight=1)
        progress_frame.grid_rowconfigure(0, weight=1)

        # Create button widget for starting the text extraction.
        self.run_button = ttk.Button(progress_frame, text="Run", command=self._run_sub_extraction)
        self.run_button.grid(column=0, row=0, pady=6, padx=10)

        # Create progress bar widget for showing the text extraction progress.
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=500, mode='determinate')
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
        # Resize output frame if main frame is resized.
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)

        # Create text widget for showing the subtitle extraction details in the output. Does not allow input from gui.
        self.text_output_widget = tk.Text(output_frame, height=12, state="disabled")
        self.text_output_widget.grid(column=0, row=0, sticky="N, S, E, W")

        # Create scrollbar widget for text widget.
        output_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.text_output_widget.yview)
        output_scroll.grid(column=1, row=0, sticky="N,S")

        # Connect text and scrollbar widgets.
        self.text_output_widget.configure(yscrollcommand=output_scroll.set)

    def _set_theme_color(self, config: dict) -> None:
        self.menubar_frame.configure(bg=config["bg"])
        self.file_mb.configure(**config)
        self.view_mb.configure(**config)
        self.menu_pref_btn.configure(**config)
        self.menu_detect_btn.configure(**config)
        self.menu_hide_btn.configure(**config)
        self.sep_label.configure(**config)
        self.menu_start_btn.configure(**config)
        self.menu_stop_btn.configure(**config)
        self.file_menu.configure(**config)
        self.view_menu.configure(**config)
        self.canvas.configure(bg=config["bg"])
        self.text_output_widget.configure(bg=config["bg"], fg=config["fg"])

    def _toggle_theme(self) -> None:
        """
        Set the theme of the gui.
        """
        if self.use_dark_mode.get():
            logger.debug("Dark mode turned on")
            config = {"bg": "#1e1e1e", "fg": "white", "activebackground": "#4a4a4a"}
            set_title_bar_colour(self.root.winfo_id(), True)
            self._set_theme_color(config)
            self.style.theme_use("clam")
            self.style.configure("TFrame", background=config["bg"])
            self.style.configure("TLabel", background=config["bg"], foreground="white")
            self.style.configure("Horizontal.TScale", troughcolor="#2b2b2b")
            self.style.configure("TEntry", fieldbackground=config["bg"], foreground="white", insertcolor="white")
            self.style.configure("TSpinbox", fieldbackground=config["bg"], foreground="white", insertcolor="white")
        else:
            logger.debug("Dark mode turned off")
            config = {"bg": "SystemMenu", "fg": "black", "activebackground": "SystemHighlight"}
            if platform.system() != "Windows":
                config["bg"], config["activebackground"] = "#d9d9d9", "#4a6984"
            set_title_bar_colour(self.root.winfo_id(), False)
            self.style.theme_use(self.default_theme)
            self._set_theme_color(config)
        self._apply_menu_hover(True)
        CONFIG.set_config(use_dark_mode=self.use_dark_mode.get())

    def resize_video(self, args: tk.Event | str) -> None:
        """
        Increase or decrease the video display and canvas size.
        """
        if self.current_video and not self.thread_running:
            args, zoom_value = str(args), 50
            if "minus" in args and self.video_target_height > zoom_value:
                self.video_target_height -= zoom_value
            elif "equal" in args:
                self.video_target_height += zoom_value
            self._set_video(self._video_indexer()[0], self.video_scale.get())
            self._elevate_non_subarea()

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
        self.progress_bar.configure(length=500)

    def _set_batch_layout(self) -> None:
        """
        Activate the batch layout from the work frame on the gui.
        """
        logger.debug("Setting batch layout")
        self.progress_bar.configure(length=400)
        self.video_label.configure(text=self._video_indexer()[2])
        self.previous_button.grid(column=2, row=0, padx=10)
        self.next_button.grid(column=4, row=0, padx=10)

    def _preferences(self) -> None:
        """
        Open preferences window and set the icon and window location.
        The windows opening location will always be on top of the main window.
        """
        root_x, root_y = self.root.winfo_rootx(), self.root.winfo_rooty()
        win_x, win_y = root_x + 100, root_y + 20
        self.preference_window = PreferencesUI(self.icon_file, win_x, win_y)

    def _get_rescale_factor(self) -> float:
        """
        Use the frame height to determine which value will be used to scale the current video.
        :return: Rescale factor.
        """
        logger.debug("Calculating the rescale factor")
        rescale_factor = self.video_target_height / self.current_frame_height
        return rescale_factor

    def rescale(self, frame: np.ndarray = None, subtitle_area: tuple = None, resolution: tuple = None,
                scale: float = None) -> np.ndarray | tuple | None:
        """
        Method to rescale any frame, subtitle area and resolution.
        """
        scale = scale or self.current_rescale_factor

        if frame is not None:
            return cv.resize(frame, None, fx=scale, fy=scale)

        if subtitle_area:
            return tuple(map(lambda c: int(c * scale), subtitle_area))

        if resolution:
            frame_width, frame_height = resolution
            frame_width = frame_width * scale
            frame_height = frame_height * scale
            return frame_width, frame_height
        return None

    def _set_canvas(self) -> None:
        """
        Set canvas size to the size of captured video.
        """
        logger.debug("Setting canvas size")
        self.current_rescale_factor = self._get_rescale_factor()
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
        bottom_right_height = int(self.current_frame_height * CONFIG.subarea_height_scaler)
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
        self.menu_hide_btn.configure(text="Show Non-SubArea", command=self._show_non_subarea)  # Change button config.

    def _show_non_subarea(self) -> None:
        """
        Delete the rectangle covering the non subtitle area.
        """
        logger.debug("Rectangle for non subtitle area deleted.")
        self.canvas.delete(self.non_subarea_rect)
        self.non_subarea_rect = None
        self.menu_hide_btn.configure(text="Hide Non-SubArea", command=self._hide_non_subarea)

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

    def _display_video_frame(self, frame_no: float) -> None:
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
            tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
            self.canvas.image = tk_img

    def _frame_slider(self, scale_value: str) -> None:
        """
        Make changes according to the position of the slider.
        :param scale_value: current position of the slider.
        """
        scale_value = float(scale_value)
        # Update timecode label as slider is moved.
        current_duration = frame_no_to_duration(scale_value, self.current_fps)
        self.current_scale_value.configure(text=current_duration)
        self._display_video_frame(scale_value)
        self._elevate_non_subarea()
        self._draw_current_subtitle_area()

    def _set_frame_slider(self, frame_no: float) -> None:
        """
        Activate the slider, then set the starting and ending values of the slider.
        :param frame_no: Corresponding frame_no.
        """
        logger.debug("Setting frame slider")
        # Set the max size of the frame slider.
        self.video_scale.configure(state="normal", from_=0.0, to=self.current_frame_total, value=frame_no)
        # Set the durations labels.
        total_time_duration = frame_no_to_duration(self.current_frame_total, self.current_fps)
        scale_value = "00:00:00:000" if not frame_no else frame_no_to_duration(frame_no, self.current_fps)
        self.current_scale_value.configure(text=scale_value)
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
        start_dur = frame_no_to_duration(start_frame, self.current_fps) if start_frame else start_frame
        stop_dur = frame_no_to_duration(stop_frame, self.current_fps) if stop_frame else stop_frame
        if start_dur and stop_dur:
            self.status_label.configure(text=f"Start Frame: {start_dur}, Stop Frame: {stop_dur}")
        elif start_dur:
            self.status_label.configure(text=f"Start Frame: {start_dur}")
        elif stop_dur:
            self.status_label.configure(text=f"Stop Frame: {stop_dur}")
        else:
            self.status_label.configure(text="")

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
            new_index = self._video_indexer()[0] - 1  # Get previous video index before removing missing video.
            del self.video_queue[video]
            self._set_video(new_index)

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

    def _set_video(self, video_index: int = 0, frame_no: float = 0) -> None:
        """
        Set the gui for the given current video queue index.
        :param video_index: Index of video that should be set to current. Defaults to first index.
        :param frame_no: Corresponding frame_no.
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
            = video_details(self.current_video)
        self.video_capture = cv.VideoCapture(self.current_video)
        self._set_canvas()
        self._set_status_label()
        self._set_frame_slider(frame_no)
        self._display_video_frame(frame_no)
        self._set_current_non_subarea()
        self._draw_current_subtitle_area()
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
            if Process.interrupt_process:
                logger.debug("Video opening process interrupted\n")
                self.thread_running = False
                self._on_closing()
                return
            logger.info(f"Opened file: {Path(filename).name}")
            _, _, frame_width, frame_height = video_details(filename)
            default_subarea = default_sub_area(frame_width, frame_height)
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
            Process.start_process()
            self._set_gui_state("disabled", "opening")
            Thread(target=self._set_opened_videos, args=(filenames,), daemon=True).start()

    def error_message_handler(self, text: str) -> None:
        """
        Show the CustomMessageBox and append the error message or messages.
        """
        root_x, root_y = self.root.winfo_rootx(), self.root.winfo_rooty()
        win_x, win_y = root_x + 10, root_y + 50
        custom_messagebox = CustomMessageBox(self.icon_file, f"{self.window_title} - Error Message", win_x, win_y)
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
        Overwrite progress bar text in text widget, if detected in present and previous line.
        """
        if " |#" in text or "-| " in text:
            start, stop = 'end - 1 lines', 'end - 1 lines lineend'
            previous_line = self.text_output_widget.get(start, stop)
            if " |#" in previous_line or "-| " in previous_line:
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
            sound = Sound.get_sound_value(CONFIG.win_notify_sound)
            toast.clear()
            toast.set_audio(sound, loop=CONFIG.win_notify_loop_sound)
            toast.show()
        elif operating_system == "Linux":
            send_linux_notification(
                app_id=self.window_title,
                title=title,
                msg=message,
                icon=str(Path(self.icon_file).absolute()),
            )

    @keep.running
    def _detect_subtitles(self) -> None:
        """
        Detect sub area of videos in the queue and set as new sub area.
        """
        logger.info("Detecting subtitle area in video(s)...")
        use_search_area = CONFIG.use_search_area
        self.thread_running = True
        try:
            setup_ocr()
            start_time = perf_counter()
            for video in self.video_queue.keys():
                if Process.interrupt_process:
                    logger.warning("Process interrupted\n")
                    self.thread_running = False
                    self._stop_sub_detection_process()
                    return
                sub_dt = SubtitleDetector(video, use_search_area)
                new_sub_area = sub_dt.get_sub_area()
                if new_sub_area:
                    self.video_queue[video][0] = new_sub_area
        except Exception as error:
            logger.exception(f"\nAn error occurred while detecting subtitles! \nError: {error}")
            start_time = perf_counter()
        self.thread_running = False
        self._stop_sub_detection_process()
        self.current_sub_area = list(self.video_queue.values())[self._video_indexer()[0]][0]
        self._draw_current_subtitle_area()
        done_notif = f"Done detecting subtitle(s)! Duration: {timedelta(seconds=round(perf_counter() - start_time))}"
        self.send_notification("Subtitle Detection Completed!", done_notif)
        logger.info(f"{done_notif}\n")

    def _stop_sub_detection_process(self) -> None:
        """
        Stop sub detection from running.
        """
        logger.debug("Stop detection button clicked")
        Process.stop_process()
        if not self.thread_running:
            self._set_gui_state("normal", "detection")
            self.menu_detect_btn.configure(text="Detect Subtitles", command=self._run_sub_detection)

    def _run_sub_detection(self) -> None:
        """
        Create a thread to run subtitle detection.
        """
        Process.start_process()
        self._set_gui_state("disabled", "detection")
        self.menu_detect_btn.configure(text="Stop Sub Detection", command=self._stop_sub_detection_process)
        Thread(target=self._detect_subtitles, daemon=True).start()

    @keep.running
    def extract_subtitles(self) -> None:
        """
        Use the main module extraction class to extract text from subtitle.
        """
        queue_len = len(self.video_queue)
        self.progress_bar.configure(maximum=queue_len)
        self.video_label.configure(text=f"{self.progress_bar['value']} of {queue_len} Video(s) Completed")
        logger.info(f"Subtitle Language: {CONFIG.ocr_rec_language}\n")
        self.thread_running, sub_ex = True, SubtitleExtractor()
        try:
            setup_ocr()
            start_time = perf_counter()
            for video, sub_info in self.video_queue.items():
                sub_area, start_frame, stop_frame = sub_info[0], sub_info[1], sub_info[2]
                start_frame = int(start_frame) if start_frame else start_frame
                stop_frame = int(stop_frame) if stop_frame else stop_frame
                if Process.interrupt_process:
                    logger.warning("Process interrupted\n")
                    self.thread_running = False
                    self._stop_sub_extraction_process()
                    return
                sub_ex.run_extraction(video, sub_area, start_frame, stop_frame)
                self.progress_bar['value'] += 1
                self.video_label.configure(text=f"{self.progress_bar['value']} of {queue_len} Video(s) Completed")
        except Exception as error:
            logger.exception(f"\nAn error occurred while extracting subtitles! \nError: {error}")
            start_time = perf_counter()
        self.thread_running = False
        self._stop_sub_extraction_process()
        done = f"Subtitle Extraction Completed! Total Duration: {timedelta(seconds=round(perf_counter() - start_time))}"
        self.send_notification(done)
        logger.info(f"{done}\n")

    def _stop_sub_extraction_process(self) -> None:
        """
        Stop program from running.
        """
        logger.debug("Stop button clicked")
        Process.stop_process()
        if not self.thread_running:
            self.run_button.configure(text="Run", command=self._run_sub_extraction)
            self._set_gui_state("normal")

    def _run_sub_extraction(self) -> None:
        """
        Start the text extraction from video frames.
        """
        logger.debug("Run button clicked")
        if self.video_queue and self.current_video:
            confirmation = messagebox.askyesno(title='Confirmation', message='Start Subtitle Extraction?')
            if confirmation:
                self.current_video = None
                self.video_capture.release()
                Process.start_process()
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
        self.file_menu.entryconfig(0, state=state)  # Open File button.
        self.view_mb.configure(state=state)  # type: ignore
        self.menu_pref_btn.configure(state=state)  # type: ignore

        if process_name == "opening":
            self.previous_button.configure(state=state)
            self.next_button.configure(state=state)

        if process_name in ("detection", "opening"):
            self.run_button.configure(state=state)

        if process_name in ("extraction", "opening"):
            self.menu_detect_btn.configure(state=state)  # type: ignore
            self.menu_hide_btn.configure(state=state)  # type: ignore
            self.menu_start_btn.configure(state=state)  # type: ignore
            self.menu_stop_btn.configure(state=state)  # type: ignore
            self.video_scale.configure(state=state)

        self._apply_menu_hover()  # Add or remove hover as the state changes

    def _set_update_checker(self) -> None:
        CONFIG.set_config(check_for_updates=self.check_for_updates.get())

    def _update_checker(self) -> None:
        if getattr(sys, "frozen", False) and self.check_for_updates.get():
            Thread(target=check_for_updates, daemon=True).start()

    def clear_notifications(self) -> None:
        """
        Remove all the previous created notification that are still in the notification window.
        """
        operating_system = platform.system()
        if operating_system == "Windows":
            toast = Notification(self.window_title, "")
            toast.clear()

    def _on_closing(self) -> None:
        """
        Method called when window is closed.
        """
        Process.stop_process()
        if not self.thread_running:
            self.clear_notifications()
            self.root.quit()


class PreferencesUI(tk.Toplevel):
    def __init__(self, icon_file: Path, win_x: int, win_y: int) -> None:
        super().__init__()
        self.icon_file = icon_file
        self.geometry(f"+{win_x}+{win_y}")  # Set window position.
        self.sections_to_skip = {"Misc"}  # Sections from Config
        self.check_int = (self.register(self._check_integer), '%P')
        self.check_float = (self.register(self._check_float), '%P')
        self._create_layout()
        self.focus()
        self.grab_set()

    def _create_layout(self) -> None:
        """
        Create layout for preferences window.
        """
        self.withdraw()
        self.title("Preferences")
        if platform.system() == "Windows":
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
        self.combobox_size = 12
        self.wgt_y_padding = 20

        # Add tabs to notebook.
        self._subtitle_detection_tab()
        self._frame_extraction_tab()
        self._text_extraction_nb()
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

        self._set_reset_button()  # Set the reset button when layout is created.
        self._set_ocr_perf_state()

        set_title_bar_colour(self.winfo_id(), CONFIG.use_dark_mode)
        self.update_idletasks()
        self.deiconify()

    def make_pref_var(self, var):
        """
        Make a variable for the preference gui and add tracing for the reset button.
        """
        type_map = {str: tk.StringVar, int: tk.IntVar, float: tk.DoubleVar, bool: tk.BooleanVar}
        try:
            tk_var_type = type_map[type(var)]
        except KeyError:
            raise TypeError(f"Unknown tk type for variable: {var!r}, type: {type(var)}")
        var = tk_var_type(value=var)
        var.trace_add("write", self._set_reset_button)
        return var

    def _subtitle_detection_tab(self) -> None:
        """
        Creates widgets in the Subtitle detection preferences tab frame.
        """
        subtitle_detection_frame = ttk.Frame(self.notebook_tab)
        subtitle_detection_frame.grid(column=0, row=0)
        subtitle_detection_frame.grid_columnconfigure(0, weight=1)
        subtitle_detection_frame.grid_columnconfigure(1, weight=1)
        self.notebook_tab.add(subtitle_detection_frame, text="Subtitle Detection")

        ttk.Label(subtitle_detection_frame, text="Split Start (Relative position):").grid(
            column=0, row=0, pady=self.wgt_y_padding
        )
        self.split_start = self.make_pref_var(CONFIG.split_start)
        ttk.Spinbox(
            subtitle_detection_frame,
            from_=0, to=0.5,
            increment=0.02,
            textvariable=self.split_start,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=0)

        ttk.Label(subtitle_detection_frame, text="Split Stop (Relative position):").grid(column=0, row=1)
        self.split_stop = self.make_pref_var(CONFIG.split_stop)
        ttk.Spinbox(
            subtitle_detection_frame,
            from_=0.5, to=1.0,
            increment=0.02,
            textvariable=self.split_stop,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=1)

        ttk.Label(subtitle_detection_frame, text="No of Frames:").grid(column=0, row=2, pady=self.wgt_y_padding)
        self.no_of_frames = self.make_pref_var(CONFIG.no_of_frames)
        ttk.Entry(
            subtitle_detection_frame,
            textvariable=self.no_of_frames,
            validate='key',
            validatecommand=self.check_int,
            width=self.entry_size
        ).grid(column=1, row=2)

        ttk.Label(subtitle_detection_frame, text="X Axis Padding (Relative):").grid(column=0, row=3)
        self.sub_area_x_rel_padding = self.make_pref_var(CONFIG.sub_area_x_rel_padding)
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
        self.sub_area_y_abs_padding = self.make_pref_var(CONFIG.sub_area_y_abs_padding)
        ttk.Entry(
            subtitle_detection_frame,
            textvariable=self.sub_area_y_abs_padding,
            validate='key',
            validatecommand=self.check_int,
            width=self.entry_size
        ).grid(column=1, row=4)

        ttk.Label(subtitle_detection_frame, text="BBox Drop Score:").grid(column=0, row=5)
        self.bbox_drop_score = self.make_pref_var(CONFIG.bbox_drop_score)
        ttk.Spinbox(
            subtitle_detection_frame,
            from_=0, to=1.0,
            increment=0.01,
            textvariable=self.bbox_drop_score,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=5)

        self.use_search_area = self.make_pref_var(CONFIG.use_search_area)
        ttk.Checkbutton(
            subtitle_detection_frame,
            text='Use Default Search Area',
            variable=self.use_search_area
        ).grid(column=0, row=6, pady=self.wgt_y_padding)

    def _frame_extraction_tab(self) -> None:
        """
        Creates widgets in the Frame extraction preferences tab frame.
        """
        frame_extraction_frame = ttk.Frame(self.notebook_tab)
        frame_extraction_frame.grid(column=0, row=0)
        frame_extraction_frame.grid_columnconfigure(0, weight=1)
        frame_extraction_frame.grid_columnconfigure(1, weight=1)
        self.notebook_tab.add(frame_extraction_frame, text="Frame Extraction")

        ttk.Label(frame_extraction_frame, text="Frame Extraction Frequency:").grid(
            column=0, row=0, pady=self.wgt_y_padding
        )
        self.frame_extraction_frequency = self.make_pref_var(CONFIG.frame_extraction_frequency)
        ttk.Spinbox(
            frame_extraction_frame,
            from_=1.0, to=10,
            textvariable=self.frame_extraction_frequency,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=0)

        ttk.Label(frame_extraction_frame, text="Frame Extraction Batch Size:").grid(column=0, row=1)
        self.frame_extraction_batch_size = self.make_pref_var(CONFIG.frame_extraction_batch_size)
        ttk.Entry(
            frame_extraction_frame,
            textvariable=self.frame_extraction_batch_size,
            validate='key',
            validatecommand=self.check_int,
            width=self.entry_size
        ).grid(column=1, row=1)

    def _text_extraction_nb(self) -> None:
        """
        A notebook tab for all the options related to text extraction.
        """
        text_extraction_notebook = ttk.Notebook(self.notebook_tab)
        self.notebook_tab.add(text_extraction_notebook, text="Text Extraction")
        self._text_extraction_tab(text_extraction_notebook)
        self._ocr_engine_tab(text_extraction_notebook)
        self._ocr_performance_tab(text_extraction_notebook)

    def _text_extraction_tab(self, notebook_tab) -> None:
        """
        Creates widgets in the Text extraction preferences tab frame.
        """
        text_extraction_frame = ttk.Frame(self.notebook_tab)
        text_extraction_frame.grid(column=0, row=0)
        text_extraction_frame.grid_columnconfigure(0, weight=1)
        text_extraction_frame.grid_columnconfigure(1, weight=1)
        notebook_tab.add(text_extraction_frame, text="Extraction")

        ttk.Label(text_extraction_frame, text="Text Extraction Batch Size:").grid(
            column=0, row=0, pady=self.wgt_y_padding
        )
        self.text_extraction_batch_size = self.make_pref_var(CONFIG.text_extraction_batch_size)
        ttk.Entry(
            text_extraction_frame,
            textvariable=self.text_extraction_batch_size,
            validate='key',
            validatecommand=self.check_int,
            width=self.entry_size
        ).grid(column=1, row=0)

        ttk.Label(text_extraction_frame, text="Text Drop Score:").grid(column=0, row=2)
        self.text_drop_score = self.make_pref_var(CONFIG.text_drop_score)
        ttk.Spinbox(
            text_extraction_frame,
            from_=0, to=1.0,
            increment=0.01,
            textvariable=self.text_drop_score,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=2)

        self.line_break = self.make_pref_var(CONFIG.line_break)
        ttk.Checkbutton(
            text_extraction_frame,
            text='Use Line Break',
            variable=self.line_break
        ).grid(column=0, row=3, pady=self.wgt_y_padding)

    def _ocr_engine_tab(self, notebook_tab) -> None:
        models_frame = ttk.Frame(self.notebook_tab)
        models_frame.grid(column=0, row=0)
        models_frame.grid_columnconfigure(0, weight=1)
        models_frame.grid_columnconfigure(1, weight=1)
        notebook_tab.add(models_frame, text="OCR Engine")

        ttk.Label(models_frame, text="OCR Recognition Language:").grid(column=0, row=0, pady=self.wgt_y_padding)
        self.ocr_rec_language = self.make_pref_var(CONFIG.ocr_rec_language)
        ttk.Combobox(
            models_frame,
            textvariable=self.ocr_rec_language,
            values=['ab', 'ady', 'af', 'ang', 'ar', 'av', 'az', 'ba', 'bal', 'be', 'bg', 'bgc', 'bh', 'bho', 'bs',
                    'bua', 'ca', 'ce', 'ch', 'chinese_cht', 'cs', 'cv', 'cy', 'da', 'dar', 'de', 'el', 'en', 'es', 'et',
                    'eu', 'fa', 'fi', 'fr', 'ga', 'gl', 'gom', 'hi', 'hr', 'hu', 'id', 'inh', 'is', 'it', 'japan',
                    'kaa', 'kbd', 'kk', 'korean', 'ku', 'ku', 'kv', 'ky', 'la', 'lb', 'lez', 'lki', 'lt', 'lv', 'mah',
                    'mai', 'mhr', 'mi', 'mk', 'mn', 'mo', 'mr', 'ms', 'mt', 'ne', 'new', 'nl', 'no', 'oc', 'os', 'pi',
                    'pl', 'ps', 'pt', 'qu', 'rm', 'ro', 'rs_latin', 'ru', 'sa', 'sah', 'sck', 'sd', 'sk', 'sl', 'sq',
                    'sr', 'sv', 'sw', 'ta', 'tab', 'te', 'tg', 'th', 'tl', 'tr', 'tt', 'tyv', 'udm', 'ug', 'uk', 'ur',
                    'uz', 'vi', 'xal'],
            state="readonly",
            width=self.combobox_size
        ).grid(column=1, row=0)

        self.use_text_ori = self.make_pref_var(CONFIG.use_text_ori)
        ttk.Checkbutton(
            models_frame,
            text='Use Text Line Orientation Model',
            variable=self.use_text_ori
        ).grid(column=0, row=1)

        self.use_mobile_model = self.make_pref_var(CONFIG.use_mobile_model)
        ttk.Checkbutton(
            models_frame,
            text='Use Mobile Model',
            variable=self.use_mobile_model
        ).grid(column=1, row=1)

    def _ocr_performance_tab(self, notebook_tab) -> None:
        model_performance_frame = ttk.Frame(self.notebook_tab)
        model_performance_frame.grid(column=0, row=0)
        model_performance_frame.grid_columnconfigure(0, weight=1)
        model_performance_frame.grid_columnconfigure(1, weight=1)
        notebook_tab.add(model_performance_frame, text="OCR Performance")

        ttk.Label(model_performance_frame, text="CPU OCR Processes:").grid(column=0, row=0, pady=self.wgt_y_padding)
        self.cpu_ocr_processes = self.make_pref_var(CONFIG.cpu_ocr_processes)
        self.cpu_ocr_processes_sb = ttk.Spinbox(
            model_performance_frame,
            from_=1, to=cpu_count(),
            textvariable=self.cpu_ocr_processes,
            state="readonly",
            width=self.spinbox_size
        )
        self.cpu_ocr_processes_sb.grid(column=1, row=0)

        ttk.Label(model_performance_frame, text="CPU ONNX Intra Threads:").grid(column=0, row=1)
        self.cpu_onnx_intra_threads = self.make_pref_var(CONFIG.cpu_onnx_intra_threads)
        self.cpu_onnx_intra_threads_sb = ttk.Spinbox(
            model_performance_frame,
            from_=0, to=cpu_count(),
            textvariable=self.cpu_onnx_intra_threads,
            state="readonly",
            width=self.spinbox_size
        )
        self.cpu_onnx_intra_threads_sb.grid(column=1, row=1)

        ttk.Label(model_performance_frame, text="GPU OCR Processes:").grid(column=0, row=2, pady=self.wgt_y_padding)
        self.gpu_ocr_processes = self.make_pref_var(CONFIG.gpu_ocr_processes)
        self.gpu_ocr_processes_sb = ttk.Spinbox(
            model_performance_frame,
            from_=1, to=cpu_count(),
            textvariable=self.gpu_ocr_processes,
            state="readonly",
            width=self.spinbox_size
        )
        self.gpu_ocr_processes_sb.grid(column=1, row=2)

        self.use_gpu = self.make_pref_var(CONFIG.use_gpu)
        ttk.Checkbutton(
            model_performance_frame,
            text='Use GPU If Available',
            variable=self.use_gpu
        ).grid(column=0, row=3)

        self.auto_optimize_perf = self.make_pref_var(CONFIG.auto_optimize_perf)
        self.auto_optimize_perf.trace_add("write", self._set_ocr_perf_state)
        ttk.Checkbutton(
            model_performance_frame,
            text='Auto Optimize Performance',
            variable=self.auto_optimize_perf
        ).grid(column=1, row=3)

    def _set_ocr_perf_state(self, *args) -> None:
        """
        Set the state of other performance variables widgets.
        """
        logger.debug(f"_set_ocr_perf_state: {args}")
        state = "disabled" if self.auto_optimize_perf.get() else "normal"
        self.cpu_ocr_processes_sb.configure(state=state)
        self.cpu_onnx_intra_threads_sb.configure(state=state)
        # self.gpu_ocr_processes_sb.configure(state=state)

    def _subtitle_generator_tab(self) -> None:
        """
        Creates widgets in the Subtitle generator preferences tab frame.
        """
        subtitle_generator_frame = ttk.Frame(self.notebook_tab)
        subtitle_generator_frame.grid(column=0, row=0)
        subtitle_generator_frame.grid_columnconfigure(0, weight=1)
        subtitle_generator_frame.grid_columnconfigure(1, weight=1)
        self.notebook_tab.add(subtitle_generator_frame, text="Subtitle Generator")

        ttk.Label(subtitle_generator_frame, text="Text Similarity Threshold:").grid(
            column=0, row=0, pady=self.wgt_y_padding
        )
        self.text_similarity_threshold = self.make_pref_var(CONFIG.text_similarity_threshold)
        ttk.Spinbox(
            subtitle_generator_frame,
            from_=0, to=1.0,
            increment=0.05,
            textvariable=self.text_similarity_threshold,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=0)

        ttk.Label(subtitle_generator_frame, text="Max Timecode Difference (ms):").grid(column=0, row=1)
        self.max_timecode_diff_ms = self.make_pref_var(CONFIG.max_timecode_diff_ms)
        ttk.Entry(
            subtitle_generator_frame,
            textvariable=self.max_timecode_diff_ms,
            validate='key',
            validatecommand=self.check_int,
            width=self.entry_size
        ).grid(column=1, row=1)

        ttk.Label(subtitle_generator_frame, text="Minimum Consecutive Sub Duration (ms):").grid(
            column=0, row=2, pady=self.wgt_y_padding)
        self.min_consecutive_sub_dur_ms = self.make_pref_var(CONFIG.min_consecutive_sub_dur_ms)
        ttk.Entry(
            subtitle_generator_frame,
            textvariable=self.min_consecutive_sub_dur_ms,
            validate='key',
            validatecommand=self.check_int,
            width=self.entry_size
        ).grid(column=1, row=2)

        ttk.Label(subtitle_generator_frame, text="Max Consecutive Short Durations:").grid(column=0, row=3)
        self.max_consecutive_short_durs = self.make_pref_var(CONFIG.max_consecutive_short_durs)
        ttk.Spinbox(
            subtitle_generator_frame,
            from_=2, to=10,
            increment=1,
            textvariable=self.max_consecutive_short_durs,
            state="readonly",
            width=self.spinbox_size
        ).grid(column=1, row=3)

        ttk.Label(subtitle_generator_frame, text="Minimum Sub Duration (ms):").grid(
            column=0, row=4, pady=self.wgt_y_padding
        )
        self.min_sub_duration_ms = self.make_pref_var(CONFIG.min_sub_duration_ms)
        ttk.Entry(
            subtitle_generator_frame,
            textvariable=self.min_sub_duration_ms,
            validate='key',
            validatecommand=self.check_int,
            width=self.entry_size
        ).grid(column=1, row=4)

    def _notifications_tab(self) -> None:
        """
        Choose notification tab depending on platform os.
        """
        operating_system = platform.system()
        if operating_system == "Windows":
            self._win_notifications_tab()
        else:
            self.win_notify_sound = self.make_pref_var(CONFIG.win_notify_sound)
            self.win_notify_loop_sound = self.make_pref_var(CONFIG.win_notify_loop_sound)

    def _win_notifications_tab(self) -> None:
        """
        Creates widgets in the Notifications preferences tab frame.
        """
        notification_frame = ttk.Frame(self.notebook_tab)
        notification_frame.grid(column=0, row=0)
        notification_frame.grid_columnconfigure(0, weight=1)
        notification_frame.grid_columnconfigure(1, weight=1)
        self.notebook_tab.add(notification_frame, text="Notification")

        ttk.Label(notification_frame, text="Notification Sound:").grid(column=0, row=0, pady=self.wgt_y_padding)
        self.win_notify_sound = self.make_pref_var(CONFIG.win_notify_sound)
        ttk.Combobox(
            notification_frame,
            textvariable=self.win_notify_sound,
            values=Sound.all_sounds(),
            state="readonly",
            width=self.combobox_size + 3
        ).grid(column=1, row=0)

        self.win_notify_loop_sound = self.make_pref_var(CONFIG.win_notify_loop_sound)
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
        default_values = [
            default
            for section_name, section in CONFIG.config_schema.items()
            if section_name not in self.sections_to_skip
            for _, default in section.values()
        ]
        try:
            values = [
                getattr(self, key).get()
                for section_name, section in CONFIG.config_schema.items()
                if section_name not in self.sections_to_skip
                for key in section.keys()
            ]
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
        for section_name, section in CONFIG.config_schema.items():
            if section_name not in self.sections_to_skip:
                for key, (_, default_value) in section.items():
                    getattr(self, key).set(default_value)

    def _save_settings(self) -> None:
        """
        Save the values of the text variables to the config file.
        """
        try:
            settings = {
                key: getattr(self, key).get()
                for section_name, section in CONFIG.config_schema.items()
                if section_name not in self.sections_to_skip
                for key in section.keys()
            }
            CONFIG.set_config(**settings)
        except tk.TclError:
            logger.warning("An error occurred value(s) not saved!")
        self.destroy()


if __name__ == '__main__':
    freeze_support()
    setup_logging()
    logger.debug("\n\nGUI program Started.")
    set_dpi_scaling()
    rt = tk.Tk()
    SubtitleExtractorGUI(rt)
    rt.mainloop()
    logger.debug("GUI program Ended.\n\n")
