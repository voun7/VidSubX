from tkinter import *
from tkinter import ttk


class SubtitleExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.main_frame = None
        self._create_layout()

    def _create_layout(self):
        self.root.title("Video Subtitle Extractor")

        self._menu_bar()

        self.main_frame = ttk.Frame(self.root, padding=(4, 4, 4, 4))

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

        run = ttk.Button(progress_frame, text="Run")
        run.grid(pady=10, padx=30)

        progress_bar = ttk.Progressbar(progress_frame, orient=HORIZONTAL, length=800, mode='determinate')
        progress_bar.grid(column=2, row=0)

    def _output_frame(self):
        output_frame = ttk.Frame(self.main_frame)
        output_frame.grid(row=2)

        output_list = Listbox(output_frame, width=122)
        output_list.grid()

        output_scroll = ttk.Scrollbar(output_frame, orient=VERTICAL, command=output_list.yview)
        output_scroll.grid(column=1, row=0, sticky="N,S")

        output_list['yscrollcommand'] = output_scroll.set
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)
        for i in range(1, 101):
            output_list.insert('end', 'Line %d of 100' % i)

    def _language_settings(self):
        pass

    def _extraction_settings(self):
        pass

    def _open_file(self):
        pass


rt = Tk()
SubtitleExtractorGUI(rt)
rt.mainloop()
