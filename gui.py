import threading
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

import cv2 as cv


class SubtitleExtractorGUI:
    def __init__(self, root):
        self.root = root
        self._create_layout()
        self.video_paths = None

    def _create_layout(self):
        self.root.title("Video Subtitle Extractor")

        self._menu_bar()

        self.main_frame = ttk.Frame(self.root, padding=(5, 5, 5, 15))

        self._video_frame()
        self._work_frame()
        self._output_frame()

        self.main_frame.grid(sticky="N, S, E, W")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

    def _menu_bar(self):
        self.root.option_add('*tearOff', FALSE)

        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        menu_file = Menu(menubar)
        menu_settings = Menu(menubar)

        menubar.add_cascade(menu=menu_file, label="File")
        menubar.add_cascade(menu=menu_settings, label="Settings")

        menu_file.add_command(label="Open", command=self._open_file)
        menu_file.add_command(label="Open (batch mode)", command=self._open_files_batch)
        menu_file.add_command(label="Close", command=self._close_main_window)

        menu_settings.add_command(label="Language", command=self._language_settings)
        menu_settings.add_command(label="Extraction", command=self._extraction_settings)

    def _video_frame(self):
        video_frame = ttk.Frame(self.main_frame)
        video_frame.grid(sticky="N, S, E, W")

        self.video_canvas = Canvas(video_frame, width=1000, height=600, bg="black")
        self.video_canvas.grid(sticky="N, S, E, W")

        video_frame.grid_columnconfigure(0, weight=1)
        video_frame.grid_rowconfigure(0, weight=1)

    def _work_frame(self):
        progress_frame = ttk.Frame(self.main_frame)
        progress_frame.grid(row=1, sticky="N, S, E, W")

        self.run_button = ttk.Button(progress_frame, text="Run", command=self._run)
        self.run_button.grid(pady=10, padx=30)

        self.progress_bar = ttk.Progressbar(progress_frame, orient=HORIZONTAL, length=800, mode='determinate')
        self.progress_bar.grid(column=2, row=0)

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

    def _open_file(self):
        title = "Select Video file"
        file_types = (("mp4", "*.mp4"), ("mkv", "*.mkv"), ("all files", "*.*"))
        filename = filedialog.askopenfilename(title=title, filetypes=file_types)
        if filename:
            self.write_to_output(f"Opened file: {filename}")
            self.video_paths = filename

    def _open_files_batch(self):
        title = "Select Video files"
        file_types = (("mp4", "*.mp4"), ("mkv", "*.mkv"), ("all files", "*.*"))
        filenames = filedialog.askopenfilenames(title=title, filetypes=file_types)
        if filenames:
            for filename in filenames:
                self.write_to_output(f"Opened file: {filename}")
            self.video_paths = filenames

    def _add_batch_mode_layout(self):
        pass

    def _close_main_window(self):
        self._stop_run()
        self.root.quit()

    def _stop_run(self):
        self.interrupt = True
        self.run_button.configure(text="Run", command=self._run)

    def write_to_output(self, text):
        self.text_output_widget.configure(state="normal")
        self.text_output_widget.insert("end", f"{text}\n")
        self.text_output_widget.see("end")
        self.text_output_widget.configure(state="disabled")

    def long_running_method(self, count=0):
        num = 1000
        self.progress_bar.configure(maximum=num)
        if self.interrupt:
            return
        self.write_to_output(f"Line {count} of {num}")
        self.progress_bar['value'] += 1
        if count == num:
            self._stop_run()
            return
        self.root.after(1, lambda: self.long_running_method(count + 1))

    def _run(self):
        self.interrupt = False
        self.run_button.configure(text='Stop', command=self._stop_run)
        self.progress_bar['value'] = 0

        if self.video_paths:
            pass
            # self.text_to_output(self.video_paths)
        else:
            self.write_to_output("No video has been selected!")

        self.long_running_method()


rt = Tk()
SubtitleExtractorGUI(rt)
rt.mainloop()
