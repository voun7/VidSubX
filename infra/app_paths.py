import os
import platform
import sys
import tempfile
from pathlib import Path


class AppPaths:
    program_name, exe_name = "VidSubX", "VSX"
    working_dir = Path(__file__).parent.parent
    icon_file = working_dir / "installer/vsx.ico"
    version_file = working_dir / "installer/version.txt"
    inno_installer_file = working_dir / "installer/inno script.iss"

    @classmethod
    def output(cls) -> Path:
        return Path(tempfile.gettempdir()) / cls.program_name

    @classmethod
    def config(cls) -> Path:
        config_dir = cls.working_dir
        if getattr(sys, "frozen", False):  # config dir will be in working dir if not compiled
            operating_system = platform.system()
            if operating_system == "Windows":
                config_dir = Path(os.getenv("APPDATA"), cls.program_name)
        config_dir.mkdir(exist_ok=True)
        return config_dir

    @classmethod
    def logs(cls) -> Path:
        log_dir = cls.working_dir
        if getattr(sys, "frozen", False):  # log dir will be in working dir if not compiled
            operating_system = platform.system()
            if operating_system == "Windows":
                log_dir = Path(os.getenv("LOCALAPPDATA"), cls.program_name)
        log_dir.mkdir(exist_ok=True)
        return log_dir / "logs"

    @classmethod
    def models(cls) -> Path:
        return cls.working_dir / "models"
