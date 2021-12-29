"""
webbpsf_ext - A Toolset for extending WebbPSF functionality
----------------------------------------------------------------------------

webbpsf_ext uses WebbPSF (https://webbpsf.readthedocs.io) to generate a series of 
monochromatic PSF simulations, then produces polynomial fits to each pixel. 
Storing the coefficients rather than a library of PSFS allows for quick generation 
(via matrix multiplication) of PSF images for an arbitrary number of wavelengths 
(subject to hardware memory limitations, of course). 

The applications range from quickly creating PSFs for many different stellar
types over wide bandpasses to generating a large number of monochromatic PSFs 
for spectral dispersion.

In addition, each science instrument PSF is dependent on the detector position due to 
field-dependent wavefront errors. Such changes are tracked in WebbPSF, but it becomes
burdensome to generate new PSFs from scratch at location, especially for large 
starfields. Instead, these changes can be tracked by the fitting the residuals of 
the PSF coefficients across an instrument's field of view.

Similarly, JWST's thermal evolution (e.g., changing angle of the sunshield after
slewing to a new target) causes small but significant distortions to the telescope
backplane. WebbPSF has tools to modify OPDs, but high-fidelity simulations take
time to calculate . Since the change to the PSF coefficients varies smoothly
with respect to WFE drift components, it's simple to parameterize the coefficient
residuals.

Developed by Jarron Leisenring at University of Arizona (2015-2021).
"""

import os
import sys
# from warnings import warn
import astropy
from astropy import config as _config

import tempfile

try:
    from .version import __version__
except ImportError:
    __version__ = ''


class Conf(_config.ConfigNamespace):

    # Path to data files for webbpsf_ext. 
    # The environment variable $WEBBPSF_EXT_PATH takes priority.
    
    on_rtd = os.environ.get('READTHEDOCS') == 'True'
    
    if on_rtd:
        path = tempfile.gettempdir()
    else:
        path = os.getenv('WEBBPSF_EXT_PATH')
        if path is None:
            print("WARNING: Environment variable $WEBBPSF_EXT_PATH is not set!")
            import webbpsf
            path = webbpsf.utils.get_webbpsf_data_path()
            print("  Setting WEBBPSF_EXT_PATH to WEBBPSF directory:")
            print("  {}".format(path))
        if not os.path.isdir(path):
            raise IOError("WEBBPSF_EXT_PATH ({}) is not a valid directory path!".format(path))
            
    if '/' not in path[-1]: 
        # Make sure there is a '/' at the end of the path name
        path = path + '/'

    WEBBPSF_EXT_PATH = _config.ConfigItem(path, 'Directory path to data files \
                                    required for webbpsf_ext calculations.')

    autoconfigure_logging = _config.ConfigItem(
        False,
        'Should webbpsf_ext configure logging for itself and others?'
    )
    logging_level = _config.ConfigItem(
        ['INFO', 'DEBUG', 'WARN', 'ERROR', 'CRITICAL', 'NONE'],
        'Desired logging level for webbpsf_ext.'
    )
    default_logging_level = _config.ConfigItem('INFO', 
        'Logging verbosity: one of {DEBUG, INFO, WARN, ERROR, or CRITICAL}')
    logging_filename = _config.ConfigItem("none", "Desired filename to save log messages to.")
    logging_format_screen = _config.ConfigItem(
        '[%(name)10s:%(levelname)s] %(message)s', 'Format for lines logged to the screen.'
    )
    logging_format_file = _config.ConfigItem(
        '%(asctime)s [%(name)s:%(levelname)s] %(filename)s:%(lineno)d: %(message)s',
        'Format for lines logged to a file.'
    )

conf = Conf()

from .logging_utils import setup_logging#, restart_logging
setup_logging(conf.default_logging_level, verbose=False)

# Load a bunch of shortcuts to various functions of interest
from .bandpasses import miri_filter, nircam_filter, bp_2mass, bp_wise, bp_gaia
from .webbpsf_ext_core import MIRI_ext, NIRCam_ext
from .spectra import stellar_spectrum, companion_spec, source_spectrum
from .coords import jwst_point

def _reload(name="webbpsf_ext"):
    """
    Simple reload function to test code changes without restarting python.
    There may be some weird consequences and bugs that show up, such as
    functions and attributes deleted from the code still stick around after
    the reload. Although, this is even true with ``importlib.reload(webbpsf_ext)``.

    Other possible ways to reload on-the-fly: 
       
    from importlib import reload
    reload(webbpsf_ext)

    # Delete classes/modules to reload
    import sys
    del sys.modules['webbpsf_ext.obs_nircam'] 
    """
    import imp
    imp.load_module(name, *imp.find_module(name))

    print("{} reloaded".format(name)) 
