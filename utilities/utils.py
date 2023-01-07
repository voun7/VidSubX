import sys

INTERRUPT_PROCESS = False


def interrupt_process(condition: bool) -> None:
    global INTERRUPT_PROCESS
    INTERRUPT_PROCESS = condition


def process_state() -> bool:
    return INTERRUPT_PROCESS


def print_progress(iteration, total, prefix='', suffix='', decimals=3, bar_length=50):
    """
    Call in a loop to create standard out progress bar
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return: None
    """

    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))  # write out the bar
    sys.stdout.flush()  # flush to stdout
