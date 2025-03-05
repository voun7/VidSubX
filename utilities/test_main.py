import os
from pathlib import Path
from unittest import TestCase

os.chdir(Path(__file__).parent.parent)

from main import SubtitleDetector, SubtitleExtractor

ch_vid = "test files/chinese_vid.mp4"
ch_vid_srt = Path("test files/chinese_vid.srt")


class TestSubtitleDetector(TestCase):
    # Changes to padding in config module affects tests. Default config should be used.

    @classmethod
    def setUpClass(cls):
        print("Running setUpClass method...")
        cls.sd = SubtitleDetector(ch_vid, True)

    def test_dir(self):
        print("\nRunning test for vd_output_dir existence...")
        self.assertTrue(self.sd.sub_ex.vd_output_dir.exists())

    def test__get_key_frames(self):
        print("\nRunning test for _get_key_frames method...")
        self.sd.sub_ex.empty_cache()
        self.sd.frame_output.mkdir(parents=True)
        self.sd._get_key_frames()
        no_of_frames = len(list(self.sd.frame_output.iterdir()))
        self.assertEqual(no_of_frames, 20)

    def test__pad_sub_area(self):
        print("\nRunning test for _pad_sub_area method...")
        self.assertEqual(self.sd._pad_sub_area((698, 158), (1218, 224)), ((288, 143), (1632, 239)))

    def test__reposition_sub_area(self):
        print("\nRunning test for _reposition_sub_area method...")
        self.assertEqual(self.sd._reposition_sub_area((288, 148), (1632, 234)), ((288, 958), (1632, 1044)))

    def test_empty_cache(self):
        print("\nRunning test for _empty_cache method...")
        self.sd.sub_ex.empty_cache()
        self.sd.frame_output.mkdir(parents=True)
        self.assertTrue(self.sd.sub_ex.vd_output_dir.exists())
        self.sd.sub_ex.empty_cache()
        self.assertFalse(self.sd.sub_ex.vd_output_dir.exists())

    def test_get_sub_area_search_area(self):
        print("\nRunning test for get_sub_area method with search area...")
        sub_area = (288, 942, 1632, 1062)
        result = SubtitleDetector(ch_vid, True).get_sub_area()
        self.assertEqual(sub_area, result)

    def test_get_sub_area_full_area(self):
        print("\nRunning test for get_sub_area method without search area...")
        sub_area = (288, 947, 1632, 1060)
        result = SubtitleDetector(ch_vid, False).get_sub_area()
        self.assertEqual(sub_area, result)


class TestSubtitleExtractor(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("\nRunning setUpClass method...")
        cls.se = SubtitleExtractor()
        cls.fps, cls.frame_total, cls.frame_width, cls.frame_height = 30.0, 1830, 1920, 1080
        cls.default_sub_area = 0, 810, 1920, 1080

    def test_video_details(self):
        print("\nRunning tests for video_details method...")
        fps, frame_total, frame_width, frame_height = self.se.video_details(ch_vid)
        self.assertEqual(fps, self.fps)
        self.assertEqual(frame_total, self.frame_total)
        self.assertEqual(frame_width, self.frame_width)
        self.assertEqual(frame_height, self.frame_height)

    def test_default_sub_area(self):
        print("\nRunning tests for default_sub_area method...")
        x1, y1, x2, y2 = self.se.default_sub_area(self.frame_width, self.frame_height)
        self.assertEqual(x1, 0)
        self.assertEqual(y1, 810)
        self.assertEqual(x2, self.frame_width)
        self.assertEqual(y2, self.frame_height)

    def test_frame_no_to_duration(self):
        print("\nRunning tests for frame_no_to_duration method...")
        self.assertEqual(self.se.frame_no_to_duration(457, 30.0), "00:00:15:233")
        self.assertEqual(self.se.frame_no_to_duration(915, 30.0), "00:00:30:500")
        self.assertEqual(self.se.frame_no_to_duration(369, 24.0), "00:00:15:375")
        self.assertEqual(self.se.frame_no_to_duration(739, 24.0), "00:00:30:791")

    def test_similarity(self):
        print("\nRunning tests for similarity method...")
        self.assertEqual(self.se.similarity("这漫天的星辰之中", "竟然还蕴含着星辰之力"), 0.3333333333333333)
        self.assertEqual(self.se.similarity("竟然还蕴含着星辰之力", "竟竞然还蕴含着星辰之力"), 0.9523809523809523)
        self.assertEqual(self.se.similarity("此机会多吸取一点", "大胆人类"), 0.0)
        self.assertEqual(self.se.similarity("颗果实就想打发我们", "颗果实就想打发我们"), 1.0)

    def test_similar_text_name_gen(self):
        print("\nRunning tests for similar_text_name_gen method...")
        start_name, end_name = "3466.666666666667--4733.333333333333", "3466.666666666667--4733.333333333333"
        self.assertEqual(self.se.similar_text_name_gen(start_name, end_name), "3466.666666666667--4733.333333333333")
        start_name, end_name = "5066.666666666666--6200.0", "6333.333333333333--6666.666666666667"
        self.assertEqual(self.se.similar_text_name_gen(start_name, end_name), "5066.666666666666--6666.666666666667")
        start_name, end_name = "9866.666666666668--10466.666666666666", "11200.0--11733.333333333332"
        self.assertEqual(self.se.similar_text_name_gen(start_name, end_name), "9866.666666666668--11733.333333333332")
        start_name, end_name = "59733.333333333336--59800.0", "60933.33333333333--60933.33333333333"
        self.assertEqual(self.se.similar_text_name_gen(start_name, end_name), "59733.333333333336--60933.33333333333")

    def test_name_to_duration(self):
        print("\nRunning tests for name_to_duration method...")
        self.assertEqual(self.se.name_to_duration("5066.666666666666--6666.666666666667"), 1600.000000000001)
        self.assertEqual(self.se.name_to_duration("17800.0--18200.0"), 400.0)
        self.assertEqual(self.se.name_to_duration("20133.333333333332--21133.333333333332"), 1000.0)
        self.assertEqual(self.se.name_to_duration("43533.33333333333--44200.0"), 666.6666666666715)

    def test_timecode(self):
        print("\nRunning tests for timecode method...")
        self.assertEqual(self.se.timecode(4577987976), "1271:39:47,976")
        self.assertEqual(self.se.timecode(97879869), "27:11:19,869")
        self.assertEqual(self.se.timecode(309485036), "85:58:05,036")
        self.assertEqual(self.se.timecode(378786979), "105:13:06,979")
        self.assertEqual(self.se.timecode(25234.7962452), "00:00:25,234")
        self.assertEqual(self.se.timecode(6365.242454), "00:00:06,365")

    def test_run_extraction(self):
        print("\nRunning test for run_extraction method...")
        sub_area = (288, 958, 1632, 1044)
        test_sub_path = self.se.run_extraction(ch_vid, sub_area)
        test_sub_txt = test_sub_path.read_text(encoding="utf-8")
        test_sub_path.unlink()
        self.assertEqual(test_sub_txt, ch_vid_srt.read_text(encoding="utf-8"))
