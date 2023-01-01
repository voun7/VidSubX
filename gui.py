import time
from threading import Thread
from tkinter import *
from tkinter import filedialog
from tkinter import ttk

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk


class SubtitleExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._create_layout()
        self.video_path = None
        self.video_capture = None

    def _create_layout(self):
        self.root.title("Video Subtitle Extractor")
        self.root.resizable(FALSE, FALSE)

        self._menu_bar()

        self.main_frame = ttk.Frame(self.root, padding=(5, 5, 5, 15))

        self._video_frame()
        self._work_frame()
        self._output_frame()

        self.main_frame.grid(sticky="N, S, E, W")

    def _menu_bar(self):
        self.root.option_add('*tearOff', FALSE)

        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        menu_file = Menu(menubar)
        menu_settings = Menu(menubar)

        menubar.add_cascade(menu=menu_file, label="File")
        menubar.add_cascade(menu=menu_settings, label="Settings")

        menu_file.add_command(label="Open", command=self.open_file)
        menu_file.add_command(label="Close", command=self._on_closing)

        menu_settings.add_command(label="Language", command=self._language_settings)
        menu_settings.add_command(label="Extraction", command=self._extraction_settings)

    def _video_frame(self):
        video_frame = ttk.Frame(self.main_frame)
        video_frame.grid()

        self.video_canvas = Canvas(video_frame, bg="black")
        self.video_canvas.grid()

    def _work_frame(self):
        progress_frame = ttk.Frame(self.main_frame)
        progress_frame.grid(row=1)

        self.run_button = ttk.Button(progress_frame, text="Run", command=self._run)
        self.run_button.grid(pady=6, padx=10)

        self.progress_bar = ttk.Progressbar(progress_frame, orient=HORIZONTAL, length=700, mode='determinate')
        self.progress_bar.grid(column=1, row=0, padx=10)

    def _output_frame(self):
        output_frame = ttk.Frame(self.main_frame)
        output_frame.grid(row=2, sticky="N, S, E, W")

        self.text_output_widget = Text(output_frame, height=12, state="disabled")
        self.text_output_widget.grid(sticky="N, S, E, W")

        output_scroll = ttk.Scrollbar(output_frame, orient=VERTICAL, command=self.text_output_widget.yview)
        output_scroll.grid(column=1, row=0, sticky="N,S")

        self.text_output_widget.configure(yscrollcommand=output_scroll.set)

        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)

    def _language_settings(self):
        pass

    def _extraction_settings(self):
        pass

    @staticmethod
    def rescale_to_frame(frame: np.ndarray = None, subtitle_area: tuple = None, resolution: tuple = None,
                         scale: float = 0.5) -> np.ndarray | tuple:
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

    def video_details(self) -> tuple:
        fps = self.video_capture.get(cv.CAP_PROP_FPS)
        frame_total = int(self.video_capture.get(cv.CAP_PROP_FRAME_COUNT))
        frame_width = int(self.video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        return fps, frame_total, frame_width, frame_height

    def _set_canvas_size(self):
        _, _, frame_width, frame_height, = self.video_details()
        frame_width, frame_height = self.rescale_to_frame(resolution=(frame_width, frame_height))
        self.video_canvas.configure(width=frame_width, height=frame_height)

    def default_subtitle_area(self):
        _, _, frame_width, frame_height, = self.video_details()
        frame_width, frame_height = self.rescale_to_frame(resolution=(frame_width, frame_height))
        x1, y1, x2, y2 = 0, int(frame_height * 0.75), frame_width, frame_height
        return x1, y1, x2, y2

    def draw_subtitle_area(self, x1: int = None, y1: int = None, x2: int = None, y2: int = None) -> None:
        if all(value is not None for value in [x1, y1, x2, y2]):
            print('Subtitle coordinates are not None')
            self.video_canvas.create_rectangle(x1, y1, x2, y2)
        else:
            print('Some Subtitle coordinates are None')
            x1, y1, x2, y2 = self.default_subtitle_area()
            self.video_canvas.create_rectangle(x1, y1, x2, y2)

    def _display_video_frame(self, second=0):
        self.video_capture.set(cv.CAP_PROP_POS_MSEC, second * 1000)
        _, frame = self.video_capture.read()

        cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        frame_resized = self.rescale_to_frame(cv2image)

        img = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image=img)
        self.video_canvas.create_image(0, 0, image=photo, anchor=NW)
        self.video_canvas.image = photo

    def open_file(self):
        print("Open button clicked")
        if self.video_capture is not None:
            print("Closing open video")
            self.video_capture.release()

        title = "Open"
        file_types = (("mp4", "*.mp4"), ("mkv", "*.mkv"), ("All files", "*.*"))
        filename = filedialog.askopenfilename(title=title, filetypes=file_types)
        if filename:
            self.write_to_output(f"Opened file: {filename}")
            self.video_path = filename
            self.video_capture = cv.VideoCapture(str(self.video_path))
            self._set_canvas_size()
            self._display_video_frame()
            self.draw_subtitle_area()

    def _on_closing(self):
        self._stop_run()
        self.root.quit()

    def _stop_run(self):
        print("Stop button clicked")
        self.interrupt = True
        self.run_button.configure(text="Run", command=self._run)

    def write_to_output(self, text):
        self.text_output_widget.configure(state="normal")
        self.text_output_widget.insert("end", f"{text}\n")
        self.text_output_widget.see("end")
        self.text_output_widget.configure(state="disabled")

    def long_running_method(self):
        num = 1000
        self.progress_bar.configure(maximum=num)
        for i in range(0, num):
            if self.interrupt:
                break
            self.write_to_output(f"Line {i} of {num}")
            self.progress_bar['value'] += 1
            time.sleep(0.00001)
        self._stop_run()

    def _run(self):
        print("Run button clicked")
        if self.video_path:
            self.interrupt = False
            self.run_button.configure(text='Stop', command=self._stop_run)
            self.progress_bar['value'] = 0

            Thread(target=self.long_running_method, daemon=True).start()
        else:
            self.write_to_output("No video has been selected!")


if __name__ == '__main__':
    rt = Tk()
    SubtitleExtractorGUI(rt)
    rt.mainloop()
