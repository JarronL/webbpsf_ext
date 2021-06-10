import sys

import webbpsf, poppy
from . import conf

import logging
_log = logging.getLogger('webbpsf_ext')

_DISABLE_FILE_LOGGING_VALUE = 'none'

import warnings
warnings.filterwarnings('ignore')

### Helper routines for logging: ###

class FilterLevelRange(object):
    def __init__(self, min_level, max_level):
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record):
        if record.levelno >= self.min_level and record.levelno <= self.max_level:
            return 1
        else:
            return 0

def restart_logging(verbose=True):
    """Restart Logging
    
    Restart logging using the same settings as those currently
    stored in conf.logging_level. This function was shamelessly
    stolen from WebbPSF utils.py.

    Parameters
    ----------
    verbose : boolean
        Should this function print the new logging targets to
        standard output?
    """

    level = str(conf.logging_level).upper()
    lognames = ['webbpsf_ext', 'webbpsf', 'poppy']

    root_logger = logging.getLogger()
    root_logger.handlers = []

    if level in ['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']:
        level_id = getattr(logging, level)  # obtain one of the DEBUG, INFO, WARN,
                                            # or ERROR constants
        if verbose:
            print(f"webbpsf_ext log messages of level {level} and above will be shown.")
    elif level == 'NONE':
        root_logger.handlers = []  # n.b. this will clear any handlers other libs/users configured
        return
    else:
        raise ValueError("Invalid logging level: {}".format(level))

    for name in lognames:
        logger = logging.getLogger(name)
        logger.setLevel(level_id)


    # set up screen logging
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.addFilter(FilterLevelRange(
        min_level=logging.DEBUG,
        max_level=logging.INFO
    ))

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.addFilter(FilterLevelRange(
        min_level=logging.WARNING,
        max_level=logging.CRITICAL
    ))
    formatter = logging.Formatter(conf.logging_format_screen)
    stderr_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)

    if verbose:
        print("webbpsf_ext log outputs will be directed to the screen.")

    # set up file logging
    filename = conf.logging_filename
    if filename is None or filename.strip().lower() != _DISABLE_FILE_LOGGING_VALUE:
        hdlr = logging.FileHandler(filename)

        formatter = logging.Formatter(conf.logging_format_file)
        hdlr.setFormatter(formatter)

        root_logger.addHandler(hdlr)

        if verbose:
            print("webbpsf_ext log outputs will also be saved to file {}".format(filename))

def setup_logging(level='INFO', filename=None, verbose=True):
    """Setup Logging
    
    Allows selection of logging detail and output locations
    (screen and/or file). Shamelessly stolen from WebbPSF utils.py.

    This is a convenience wrapper to Python's built-in logging package.
    By default, this sets up log messages to be written to the screen, 
    but the user can also request logging to a file.

    For more advanced log handling, see the Python logging module's
    own documentation.

    Parameters
    -------------
    level : str
        Name of log output to show. Defaults to 'INFO', set to 'DEBUG'
        for more extensive messages, or to 'WARN' or 'ERROR' for fewer.
    filename : str, optional
        Filename to write the log output to. If not set, output will
        just be displayed on screen. (Default: None)

    Examples
    -----------

    >>> webbpsf_ext.setup_logging(filename='webbpsf_ext_log.txt')

    This will save all log messages to 'webbpsf_ext_log.txt' in the current
    directory. 

    >>> webbpsf_ext.setup_logging(level='WARN')

    This will show only WARNING or ERROR messages on screen, and not
    save any logs to files at all (since the filename argument is None)
    """

    # implementation note: All this function actually does is apply the
    # defaults into the configuration system, then calls restart_logging to
    # do the actual work.
    level = str(level).upper()

    # The astropy config system will enforce the limited set of values for the logging_level
    # parameter by raising a TypeError on this next line if we feed in an invalid string.
    conf.logging_level = level

    if filename is None:
        # Use the string 'none' as a sentinel value for astropy.config
        filename = _DISABLE_FILE_LOGGING_VALUE

    conf.logging_filename = filename
    restart_logging(verbose=verbose)
