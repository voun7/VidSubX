import os
import platform
from pathlib import Path
from unittest import TestCase

from infra.app_paths import AppPaths

os.chdir(AppPaths.working_dir)

config_file = Path("config.ini")
config_file.unlink(missing_ok=True)  # The tests only work on the default config

from main import SubtitleDetector, setup_ocr

ch_vid = "test data/chinese_vid.mp4"
ch_vid_srt = Path("test data/chinese_vid.srt")
setup_ocr()


class TestSubtitleDetector(TestCase):
    # Changes to padding in config module affects tests. Default config should be used.

    @classmethod
    def setUpClass(cls):
        print("Running setUpClass method...")
        cls.sd = SubtitleDetector(ch_vid, True)

    def test_dir(self):
        print("\nRunning test for output_dir existence...")
        self.assertTrue(self.sd.frame_output.exists())

    def test__get_key_frames(self):
        print("\nRunning test for _get_key_frames method...")
        self.sd.empty_cache()
        self.sd.frame_output.mkdir(parents=True)
        self.sd._get_key_frames()
        no_of_frames = len(list(self.sd.frame_output.iterdir()))
        self.assertEqual(no_of_frames, 20)

    def test__pad_sub_area(self):
        print("\nRunning test for _pad_sub_area method...")
        self.assertEqual(self.sd._pad_sub_area((698, 158), (1218, 224)), ((192, 133), (1728, 249)))

    def test__reposition_sub_area(self):
        print("\nRunning test for _reposition_sub_area method...")
        self.assertEqual(self.sd._reposition_sub_area((288, 148), (1632, 234)), ((288, 958), (1632, 1044)))

    def test_empty_cache(self):
        print("\nRunning test for empty_cache method...")
        self.sd.empty_cache()
        self.sd.frame_output.mkdir(parents=True)
        self.assertTrue(self.sd.frame_output.exists())
        self.sd.empty_cache()
        self.assertFalse(self.sd.frame_output.exists())

    def test_get_sub_area_search_area(self):
        print("\nRunning test for get_sub_area method with search area...")
        sub_area = (192, 941, 1728, 1065)
        result = SubtitleDetector(ch_vid, True).get_sub_area()
        self.assertEqual(sub_area, result)

    def test_get_sub_area_full_area(self):
        print("\nRunning test for get_sub_area method without search area...")
        sub_area = (192, 944, 1728, 1063) if platform.system() == "Windows" else (192, 946, 1728, 1059)
        result = SubtitleDetector(ch_vid, False).get_sub_area()
        self.assertEqual(sub_area, result)
