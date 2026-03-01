import logging
from configparser import ConfigParser, ParsingError
from functools import cache

import psutil

from infra.app_paths import AppPaths

logger = logging.getLogger(__name__)


@cache
def physical_cores() -> int:
    return psutil.cpu_count(logical=False)


class Config:
    subarea_height_scaler = 0.75
    config_schema = {
        "Frame Extraction": {
            "frame_extraction_frequency": (int, 2),
            "frame_extraction_batch_size": (int, 250),
        },
        "Text Extraction": {
            "text_extraction_batch_size": (int, 100),
            "text_drop_score": (float, 0.6),
            "line_break": (bool, True),
        },
        "Subtitle Generator": {
            "text_similarity_threshold": (float, 0.85),
            "min_consecutive_sub_dur_ms": (float, 500.0),
            "max_consecutive_short_durs": (int, 4),
            "min_sub_duration_ms": (float, 120.0),
        },
        "Subtitle Detection": {
            "split_start": (float, 0.25),
            "split_stop": (float, 0.50),
            "no_of_frames": (int, 200),
            "sub_area_x_rel_padding": (float, 0.9),
            "sub_area_y_abs_padding": (int, 25),
            "bbox_drop_score": (float, 0.9),
            "use_search_area": (bool, True),
        },
        "Notification": {
            "win_notify_sound": (str, "Default"),
            "win_notify_loop_sound": (bool, True),
        },
        "OCR Engine": {
            "ocr_rec_language": (str, "ch"),
            "use_mobile_model": (bool, True),
            "use_text_ori": (bool, False),
        },
        "OCR Performance": {
            "cpu_ocr_processes": (int, physical_cores() // 2),
            "cpu_onnx_intra_threads": (int, max(4, int(physical_cores() / 2.5))),
            "gpu_ocr_processes": (int, max(2, physical_cores() // 3)),
            "use_gpu": (bool, True),
            "auto_optimize_perf": (bool, True),
        },
        "Misc": {
            "use_dark_mode": (bool, False),
            "check_for_updates": (bool, True),
        }
    }

    def __init__(self) -> None:
        self.ocr_opts = {"model_save_dir": str(AppPaths.models())}
        self.config_file = AppPaths.config() / "config.ini"
        self.config = ConfigParser()
        self.config_loader()

    def config_loader(self) -> None:
        try:
            if not self.config_file.exists():
                self.create_default_config_file()
            self.config.read(self.config_file)
            self.load_config()
        except (KeyError, ValueError, ParsingError):
            logger.error("Unable to load config file. Resetting config...")
            self.config_file.unlink()
            self.config_loader()

    def create_default_config_file(self) -> None:
        for section, data in self.config_schema.items():
            self.config[section] = {k: str(default) for k, (_, default) in data.items()}

        with open(self.config_file, "w") as f:
            self.config.write(f)

    @staticmethod
    def _convert(value: str, typ):
        if typ is bool:
            return value.lower() in ("1", "true", "yes", "on")
        return typ(value)

    def load_config(self) -> None:
        for section, data in self.config_schema.items():
            for key, (typ, default) in data.items():
                raw = self.config[section].get(key, str(default))
                setattr(self, key, self._convert(raw, typ))

    def set_config(self, **kwargs) -> None:
        for key, val in kwargs.items():
            for section, data in self.config_schema.items():
                if key in data:
                    setattr(self, key, val)
                    self.config[section][key] = str(val)
                    break

        with open(self.config_file, "w") as f:
            self.config.write(f)


CONFIG = Config()
