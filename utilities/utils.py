INTERRUPT_PROCESS = False


def interrupt_process(condition: bool) -> None:
    global INTERRUPT_PROCESS
    INTERRUPT_PROCESS = condition


def process_state() -> bool:
    return INTERRUPT_PROCESS
