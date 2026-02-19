import ctypes
import logging
import platform

logger = logging.getLogger(__name__)


class SleepInhibitor:
    _os = platform.system()
    _es_continuous = 0x80000000  # Persist the state

    @classmethod
    def enable(cls) -> None:
        logger.debug("Sleep Inhibitor Enabled")
        if cls._os == "Windows":
            es_system_required = 0x00000001  # Prevents the system from sleeping
            result = ctypes.windll.kernel32.SetThreadExecutionState(cls._es_continuous | es_system_required)
            if result == 0:
                logger.error("Failed to enable Sleep Inhibitor")
        elif cls._os == "Linux":
            ...

    @classmethod
    def disable(cls) -> None:
        logger.debug("Sleep Inhibitor Disabled")
        if cls._os == "Windows":
            result = ctypes.windll.kernel32.SetThreadExecutionState(cls._es_continuous)
            if result == 0:
                logger.error("Failed to disable Sleep Inhibitor")
        elif cls._os == "Linux":
            ...
