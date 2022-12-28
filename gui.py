from tkinter import *
from tkinter import ttk


class SubtitleExtractorGUI:
    def __init__(self, root):
        self.root = root
        self._create_layout()
        self.interrupt = False

    def _create_layout(self):
        self.root.title("Video Subtitle Extractor")

        self._menu_bar()

        self.main_frame = ttk.Frame(self.root, padding=(5, 5, 5, 15))

        self._video_frame()
        self._progress_frame()
        self._output_frame()

        self.main_frame.grid()

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
        menu_file.add_command(label="Close", command=self.root.quit)

        menu_settings.add_command(label="Language", command=self._language_settings)
        menu_settings.add_command(label="Extraction", command=self._extraction_settings)

    def _video_frame(self):
        video_frame = ttk.Frame(self.main_frame, borderwidth=2, relief="ridge", width=1000, height=600)
        video_frame.grid()

    def _progress_frame(self):
        progress_frame = ttk.Frame(self.main_frame)
        progress_frame.grid(row=1, sticky="W")

        self.run_button = ttk.Button(progress_frame, text="Run", command=self._run)
        self.run_button.grid(pady=10, padx=30)

        self.progress_bar = ttk.Progressbar(progress_frame, orient=HORIZONTAL, length=800, mode='determinate')
        self.progress_bar.grid(column=2, row=0)

    def _output_frame(self):
        output_frame = ttk.Frame(self.main_frame)
        output_frame.grid(row=2)

        self.output_text = Text(output_frame, width=97, height=12, state="disabled")
        self.output_text.grid()

        output_scroll = ttk.Scrollbar(output_frame, orient=VERTICAL, command=self.output_text.yview)
        output_scroll.grid(column=1, row=0, sticky="N,S")

        self.output_text.configure(yscrollcommand=output_scroll.set)

        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)

    def _language_settings(self):
        pass

    def _extraction_settings(self):
        pass

    def _open_file(self):
        pass

    def _stop(self):
        self.interrupt = True
        self.progress_bar.stop()
        self.run_button.configure(text="Run", command=self._run)

    def step(self):
        for i in range(1, 101):
            self.progress_bar['value'] += 1
            self.output_text.configure(state="normal")
            self.output_text.insert("end", f"Line {i} of 100\n")
            self.output_text.see("end")
            self.output_text.configure(state="disabled")
        self._stop()

    def _run(self):
        self.run_button.configure(text='Stop', command=self._stop)
        self.step()


rt = Tk()
SubtitleExtractorGUI(rt)
rt.mainloop()
