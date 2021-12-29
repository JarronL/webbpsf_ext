import numpy as np
import logging
_log = logging.getLogger('webbpsf_ext')

__epsilon = np.finfo(float).eps

from .utils import pysiaf

# Define these here rather than calling multiple times
# since it takes some time to generate these.
siaf_nrc = pysiaf.Siaf('NIRCam')
siaf_nis = pysiaf.Siaf('NIRISS')
siaf_mir = pysiaf.Siaf('MIRI')
siaf_nrs = pysiaf.Siaf('NIRSpec')
siaf_fgs = pysiaf.Siaf('FGS')
si_match = {'NRC': siaf_nrc, 'NIS': siaf_nis, 'MIR': siaf_mir, 'NRS': siaf_nrs, 'FGS': siaf_fgs}


def dist_image(image, pixscale=None, center=None, return_theta=False):
    """Pixel distances
    
    Returns radial distance in units of pixels, unless pixscale is specified.
    Use the center keyword to specify the position (in pixels) to measure from.
    If not set, then the center of the image is used.

    return_theta will also return the angular position of each pixel relative 
    to the specified center
    
    Parameters
    ----------
    image : ndarray
        Input image to find pixel distances (and theta).
    pixscale : int, None
        Pixel scale (such as arcsec/pixel or AU/pixel) that
        dictates the units of the output distances. If None,
        then values are in units of pixels.
    center : tuple
        Pixel location (x,y) in the array calculate distance. If set 
        to None, then the default is the array center pixel.
    return_theta : bool
        Also return the angular positions as a 2nd element.
    """
    y, x = np.indices(image.shape)
    if center is None:
        center = tuple((a - 1) / 2.0 for a in image.shape[::-1])
    x = x - center[0]
    y = y - center[1]

    rho = np.sqrt(x**2 + y**2)
    if pixscale is not None: 
        rho *= pixscale

    if return_theta:
        return rho, np.arctan2(-x,y)*180/np.pi
    else:
        return rho

def xy_to_rtheta(x, y):
    """Convert (x,y) to (r,theta)
    
    Input (x,y) coordinates and return polar cooridnates that use
    the WebbPSF convention (theta is CCW of +Y)
    
    Input can either be a single value or numpy array.

    Parameters
    ----------
    x : float or array
        X location values
    y : float or array
        Y location values
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(-x,y)*180/np.pi

    if np.size(r)==1:
        if np.abs(x) < __epsilon: x = 0
        if np.abs(y) < __epsilon: y = 0
    else:
        r[np.abs(r) < __epsilon] = 0
        theta[np.abs(theta) < __epsilon] = 0

    return r, theta

def rtheta_to_xy(r, theta):
    """Convert (r,theta) to (x,y)
    
    Input polar cooridnates (WebbPSF convention) and return Carteesian coords
    in the imaging coordinate system (as opposed to RA/DEC)

    Input can either be a single value or numpy array.

    Parameters
    ----------
    r : float or array
        Radial offset from the center in pixels
    theta : float or array
        Position angle for offset in degrees CCW (+Y).
    """
    x = -r * np.sin(theta*np.pi/180.)
    y =  r * np.cos(theta*np.pi/180.)

    if np.size(x)==1:
        if np.abs(x) < __epsilon: x = 0
        if np.abs(y) < __epsilon: y = 0
    else:
        x[np.abs(x) < __epsilon] = 0
        y[np.abs(y) < __epsilon] = 0

    return x, y
    
def xy_rot(x, y, ang):

    """Rotate (x,y) positions to new coords
    
    Rotate (x,y) values by some angle. 
    Positive ang values rotate counter-clockwise.
    
    Parameters
    -----------
    x : float or array
        X location values
    y : float or array
        Y location values
    ang : float or array
        Rotation angle in degrees CCW
    """

    r, theta = xy_to_rtheta(x, y)    
    return rtheta_to_xy(r, theta+ang)


###########################################################################
#    NIRCam SIAF helper functions
###########################################################################


# NIRCam aperture limits 
def get_NRC_v2v3_limits(pupil=None, border=10, return_corners=False, **kwargs):
    """
    V2/V3 Limits for a given module stored within an dictionary

    border : float
        Extend a border by some number of arcsec.
    return_corners : bool
        Return the actual aperture corners.
        Otherwise, values are chosen to be a square in V2/V3.
    """
    
    siaf = pysiaf.Siaf('NIRCam')
    siaf.generate_toc()

    names_dict = {
        'SW' : 'NRCALL_FULL',
        'LW' : 'NRCALL_FULL',
        'SWA': 'NRCAS_FULL', 
        'SWB': 'NRCBS_FULL',
        'LWA': 'NRCA5_FULL',
        'LWB': 'NRCB5_FULL',
    }

    v2v3_limits = {}
    for name in names_dict.keys():
       
        apname = names_dict[name]

        # Do all four apertures for each SWA & SWB
        ap = siaf[apname]
        if ('S_' in apname) or ('ALL_' in apname):
            v2_ref, v3_ref = ap.corners('tel', False)
        else:
            xsci, ysci = ap.corners('sci', False)
            v2_ref, v3_ref = ap.sci_to_tel(xsci, ysci)

        # Offset by 50" if coronagraphy
        if (pupil is not None) and ('LYOT' in pupil):
            v2_ref -= 2.1
            v3_ref += 47.7

        # Add border margin
        v2_avg = np.mean(v2_ref)
        v2_ref[v2_ref<v2_avg] -= border
        v2_ref[v2_ref>v2_avg] += border
        v3_avg = np.mean(v3_ref)
        v3_ref[v3_ref<v3_avg] -= border
        v3_ref[v3_ref>v3_avg] += border

        if return_corners:

            v2v3_limits[name] = {'V2': v2_ref / 60.,
                                 'V3': v3_ref / 60.}
        else:
            v2_minmax = np.array([v2_ref.min(), v2_ref.max()])
            v3_minmax = np.array([v3_ref.min(), v3_ref.max()])
            v2v3_limits[name] = {'V2': v2_minmax / 60.,
                                 'V3': v3_minmax / 60.}
        
    return v2v3_limits

def NIRCam_V2V3_limits(module, channel='LW', pupil=None, rederive=False, return_corners=False, **kwargs):
    """
    NIRCam V2/V3 bounds +10" border encompassing detector.
    """

    # Grab coordinate from pySIAF
    if rederive:
        v2v3_limits = get_NRC_v2v3_limits(pupil=pupil, return_corners=return_corners, **kwargs)

        name = channel + module
        if return_corners:
            return v2v3_limits[name]['V2'], v2v3_limits[name]['V3']
        else:
            v2_min, v2_max = v2v3_limits[name]['V2']
            v3_min, v3_max = v2v3_limits[name]['V3']
    else: # Or use preset coordinates
        if module=='A':
            v2_min, v2_max, v3_min, v3_max = (0.2, 2.7, -9.5, -7.0)
        else:
            v2_min, v2_max, v3_min, v3_max = (-2.7, -0.2, -9.5, -7.0)

        if return_corners:
            return np.array([v2_min, v2_min, v2_max, v2_max]), np.array([v3_min, v3_max, v3_min, v3_max])

    return v2_min, v2_max, v3_min, v3_max 




def ap_radec(ap_obs, ap_ref, coord_ref, pa, base_off=(0,0), dith_off=(0,0),
             get_cenpos=True, get_vert=False):
    """Aperture reference point(s) RA/Dec
    
    Given the (RA, Dec) and position angle of a given reference aperture,
    return the (RA, Dec) associated with the reference point (usually center) 
    of a different aperture. Can also return the corner vertices of the
    aperture. 
    
    Typically, the reference aperture (ap_ref) is used for the telescope 
    pointing information (e.g., NRCALL), but you may want to determine
    locations of the individual detector apertures (NRCA1_FULL, NRCB3_FULL, etc).
    
    Parameters
    ----------
    ap_obs : str
        Name of observed aperture (e.g., NRCA5_FULL)
    ap_ref : str
        Name of reference aperture (e.g., NRCALL_FULL)
    coord_ref : tuple or list
        Center position of reference aperture (RA/Dec deg)
    pa : float
        Position angle in degrees measured from North to V3 axis in North to East direction.
        
    Keyword Args
    ------------
    base_off : list or tuple
        X/Y offset of overall aperture offset (see APT pointing file)
    dither_off : list or tuple
        Additional offset from dithering (see APT pointing file)
    get_cenpos : bool
        Return aperture reference location coordinates?
    get_vert: bool
        Return closed polygon vertices (useful for plotting)?
    """

    if (get_cenpos==False) and (get_vert==False):
        _log.warning("Neither get_cenpos nor get_vert were set to True. Nothing to return.")
        return

    siaf_obs = si_match.get(ap_obs[0:3])
    siaf_ref = si_match.get(ap_ref[0:3])
    ap_siaf_ref = siaf_ref[ap_ref]
    ap_siaf_obs = siaf_obs[ap_obs]

    # RA and Dec of ap ref location and the objects in the field
    ra_ref, dec_ref = coord_ref

    # Field offset as specified in APT Special Requirements
    # These appear to be defined in 'idl' coords
    x_off, y_off  = (base_off[0] + dith_off[0], base_off[1] + dith_off[1])

    # V2/V3 reference location aligned with RA/Dec reference
    # and offset by (x_off, y_off) in 'idl' coords
    v2_ref, v3_ref = np.array(ap_siaf_ref.convert(x_off, y_off, 'idl', 'tel'))

    # Attitude correction matrix relative to reference aperture
    att = pysiaf.utils.rotations.attitude(v2_ref, v3_ref, ra_ref, dec_ref, pa)

    # Get V2/V3 position of observed SIAF aperture and convert to RA/Dec
    if get_cenpos==True:
        v2_obs, v3_obs  = ap_siaf_obs.reference_point('tel')
        ra_obs, dec_obs = pysiaf.utils.rotations.pointing(att, v2_obs, v3_obs)
        cen_obs = (ra_obs, dec_obs)
    
    # Get V2/V3 vertices of observed SIAF aperture and convert to RA/Dec
    if get_vert==True:
        v2_vert, v3_vert  = ap_siaf_obs.closed_polygon_points('tel', rederive=False)
        ra_vert, dec_vert = pysiaf.utils.rotations.pointing(att, v2_vert, v3_vert)
        vert_obs = (ra_vert, dec_vert)

    if (get_cenpos==True) and (get_vert==True):
        return cen_obs, vert_obs
    elif get_cenpos==True:
        return cen_obs
    elif get_vert==True:
        return vert_obs
    else:
        _log.warning("Neither get_cenpos nor get_vert were set to True. Nothing to return.")
        return


def radec_to_coord(coord_objs, siaf_ref_name, coord_ref, pa_ref, 
                   frame_out='tel', base_off=(0,0), dith_off=(0,0)):
    """RA/Dec to 'tel' (arcsec), 'sci' (pixels), or 'det' (pixels)
    
    Convert a series of RA/Dec positions to telescope V2/V3 coordinates (in arcsec).
    
    Parameters
    ----------
    coord_objs : tuple 
        (RA, Dec) positions (deg), where RA and Dec are numpy arrays.    
    siaf_ref_name : str
        Reference SIAF aperture name (e.g., 'NRCALL_FULL') 
    coord_ref : list or tuple
        RA and Dec towards which reference SIAF points
    pa : float
        Position angle in degrees measured from North to V3 axis in North to East direction.
        
    Keyword Args
    ------------
    frame_out : str
        One of 'tel' (arcsec), 'sci' (pixels), or 'det' (pixels).
    base_off : list or tuple
        X/Y offset of overall aperture offset (see APT pointing file)
    dither_off : list or tuple
        Additional offset from dithering (see APT pointing file)
    """

    return radec_to_v2v3(coord_objs, siaf_ref_name, coord_ref, pa_ref, frame_out=frame_out, 
                         base_off=base_off, dith_off=dith_off)


def radec_to_v2v3(coord_objs, siaf_ref_name, coord_ref, pa_ref, frame_out='tel',
                  base_off=(0,0), dith_off=(0,0)):
    """RA/Dec to V2/V3
    
    Convert a series of RA/Dec positions to telescope V2/V3 coordinates (in arcsec).
    
    Parameters
    ----------
    coord_objs : tuple 
        (RA, Dec) positions (deg), where RA and Dec are numpy arrays.    
    siaf_ref_name : str
        Reference SIAF aperture name (e.g., 'NRCALL_FULL') 
    coord_ref : list or tuple
        RA and Dec towards which reference SIAF points
    pa : float
        Position angle in degrees measured from North to V3 axis in North to East direction.
        
    Keyword Args
    ------------
    frame_out : str
        One of 'tel' (arcsec), 'sci' (pixels), or 'det' (pixels).
    base_off : list or tuple
        X/Y offset of overall aperture offset (see APT pointing file)
    dither_off : list or tuple
        Additional offset from dithering (see APT pointing file)
    """
    
    # SIAF object setup
    siaf_ref = si_match.get(siaf_ref_name[0:3])
    siaf_ap = siaf_ref[siaf_ref_name]
    
    # RA and Dec of ap ref location and the objects in the field
    ra_ref, dec_ref = coord_ref
    ra_obj, dec_obj = coord_objs

    # Field offset as specified in APT Special Requirements
    # These appear to be defined in 'idl' coords
    x_off, y_off  = (base_off[0] + dith_off[0], base_off[1] + dith_off[1])

    # V2/V3 reference location aligned with RA/Dec reference
    # and offset by (x_off, y_off) in 'idl' coords
    v2_ref, v3_ref = np.array(siaf_ap.convert(x_off, y_off, 'idl', 'tel'))

    # Attitude correction matrix relative to reference aperture
    att = pysiaf.utils.rotations.attitude(v2_ref, v3_ref, ra_ref, dec_ref, pa_ref)

    # Convert all RA/Dec coordinates into V2/V3 positions for objects
    v2_obj, v3_obj = pysiaf.utils.rotations.getv2v3(att, ra_obj, dec_obj)

    if frame_out=='tel':
        return (v2_obj, v3_obj)
    else:
        return siaf_ap.convert(v2_obj, v3_obj, 'tel', frame_out)


def v2v3_to_pixel(ap_obs, v2_obj, v3_obj, frame='sci'):
    """V2/V3 to pixel coordinates
    
    Convert object V2/V3 coordinates into pixel positions.

    Parameters
    ==========
    ap_obs : str
        Name of observed aperture (e.g., NRCA5_FULL)
    v2_obj : ndarray
        V2 locations of stellar sources.
    v3_obj : ndarray
        V3 locations of stellar sources.

    Keyword Args
    ============
    frame : str
        'det' or 'sci' coordinate frame. 'det' is always full frame reference.
        'sci' is relative to subarray size if not a full frame aperture.
    """
    
    # SIAF object setup
    siaf = si_match.get(ap_obs[0:3])
    ap_siaf = siaf[ap_obs]

    if frame=='det':
        xpix, ypix = ap_siaf.tel_to_det(v2_obj, v3_obj)
    elif frame=='sci':
        xpix, ypix = ap_siaf.tel_to_sci(v2_obj, v3_obj)
    else:
        raise ValueError("Do not recognize frame keyword value: {}".format(frame))
        
    return (xpix, ypix)


def gen_sgd_offsets(sgd_type, slew_std=5, fsm_std=2.5, rand_seed=None):
    """
    Create a series of x and y position offsets for a SGD pattern.
    This includes the central position as the first in the series.
    By default, will also add random movement errors using the
    `slew_std` and `fsm_std` keywords. Returned values in arcsec.
    
    Parameters
    ==========
    sgd_type : str
        Small grid dither pattern. Valid types are
        '9circle', '5box', '5diamond', '3bar', '5bar', '5miri', and '9miri'
        where the first four refer to NIRCam coronagraphic dither
        positions and the last two are for MIRI coronagraphy.
    fsm_std : float
        One-sigma accuracy per axis of fine steering mirror positions.
        This provides randomness to each position relative to the nominal 
        central position. Ignored for central position.
        Values should be in units of mas. 
    slew_std : float
        One-sigma accuracy per axis of the initial slew. This is applied
        to all positions and gives a baseline offset relative to the
        desired mask center. Values should be in units of mas. 
    rand_seed : int
        Input a random seed in order to make reproduceable pseudo-random
        numbers.
    """
    
    if sgd_type=='9circle':
        xoff_msec = np.array([0.0,  0,-15,-20,-15,  0,+15,+20,+15])
        yoff_msec = np.array([0.0,+20,+15,  0,-15,-20,-15,  0,+15])
    elif sgd_type=='5box':
        xoff_msec = np.array([0.0,+15,-15,-15,+15])
        yoff_msec = np.array([0.0,+15,+15,-15,-15])
    elif sgd_type=='5diamond':
        xoff_msec = np.array([0.0,  0,  0,+20,-20])
        yoff_msec = np.array([0.0,+20,-20,  0,  0])
    elif sgd_type=='5bar':
        xoff_msec = np.array([0.0,  0,  0,  0,  0])
        yoff_msec = np.array([0.0,+20,+10,-10,-20])
    elif sgd_type=='3bar':
        xoff_msec = np.array([0.0,  0,  0])
        yoff_msec = np.array([0.0,+15,-15])
    elif sgd_type=='5miri':
        xoff_msec = np.array([0.0,-10,+10,+10,-10])
        yoff_msec = np.array([0.0,+10,+10,-10,-10])
    elif sgd_type=='9miri':
        xoff_msec = np.array([0.0,-10,-10,  0,+10,+10,+10,  0,-10])
        yoff_msec = np.array([0.0,  0,+10,+10,+10,  0,-10,-10,-10])
    else:
        raise ValueError(f"{sgd_type} not a valid SGD type")

    # Create local random number generator to avoid global seed setting
    rng = np.random.default_rng(seed=rand_seed)

    # Add randomized telescope offsets
    if slew_std>0:
        x_point, y_point = rng.normal(scale=slew_std, size=2)
        xoff_msec += x_point
        yoff_msec += y_point

    # Add randomized FSM offsets
    if fsm_std>0:
        x_fsm = rng.normal(scale=fsm_std, size=xoff_msec.shape)
        y_fsm = rng.normal(scale=fsm_std, size=yoff_msec.shape)
        xoff_msec[1:] += x_fsm[1:]
        yoff_msec[1:] += y_fsm[1:]
    
    return xoff_msec / 1000, yoff_msec / 1000


def get_idl_offset(base_offset=(0,0), dith_offset=(0,0), base_std=0, use_ta=True, 
                   dith_std=0, use_sgd=True, rand_seed=None, **kwargs):
    """
    Calculate pointing offsets in 'idl' coordinates with errors. Inputs come from the
    APT's .pointing file. For a sequence of dithers, make sure to only calculate the 
    base offset once, and all dithers independently. For instance:
    
        >>> base_offset = get_idl_offset(base_std=None)
        >>> dith0 = get_idl_offset(base_offset, dith_offset=(0,0), dith_std=None)
        >>> dith1 = get_idl_offset(base_offset, dith_offset=(-0.01,+0.01), dith_std=None)
        >>> dith2 = get_idl_offset(base_offset, dith_offset=(+0.01,+0.01), dith_std=None)
        >>> dith3 = get_idl_offset(base_offset, dith_offset=(+0.01,-0.01), dith_std=None)
        >>> dith4 = get_idl_offset(base_offset, dith_offset=(-0.01,-0.01), dith_std=None)
    
    Parameters
    ==========
    base_offset : array-like
        Corresponds to (BaseX, BaseY) columns in .pointing file (arcsec).
    dith_offset : array-like
        Corresponds to (DithX, DithY) columns in .pointing file (arcsec). 
    base_std : float or array-like or None
        The 1-sigma pointing uncertainty per axis for telescope slew. 
        If None, then standard deviation is chosen to be either 5 mas 
        or 100 mas, depending on `use_ta` setting.
    use_ta : bool
        If observation uses a target acquisition, then assume only 5 mas
        of pointing uncertainty, other 100 mas for "blind" pointing.
    dith_std : float or array-like or None
        The 1-sigma pointing uncertainty per axis for dithers. If None,
        then standard deviation is chosen to be either 2.5 or 5 mas, 
        depending on `use_sgd` setting.
    use_sgd : bool
        If True, then we're employing small-grid dithers with the fine
        steering mirror, which has a ~2.5 mas uncertainty. Otherwise,
        assume standard small angle maneuver, which has ~5 mas uncertainty.
    """
    
    # Create local random number generator to avoid global seed setting
    rng = np.random.default_rng(seed=rand_seed)

    # Convert to arrays (values of mas)
    base_xy = np.asarray(base_offset, dtype='float') * 1000
    dith_xy = np.asarray(dith_offset, dtype='float') * 1000

    # Set telescope slew uncertainty
    if base_std is None:
        base_std = 5.0 if use_ta else 100
    
    # No dither offset
    if (dith_std is None):
        dith_std = 2.5 if use_sgd else 5.0
    
    base_rand = rng.normal(loc=base_xy, scale=base_std)
    dith_rand = rng.normal(loc=dith_xy, scale=dith_std)
    # Add and convert to arcsec
    offset = (base_rand + dith_rand) / 1000
    return offset

def radec_offset(ra, dec, dist, pos_ang):
    """
    Return (RA, Dec) of a position offset relative to some input (RA, Dec).
    
    Parameters
    ----------
    RA : float
        Input RA in deg.
    Dec : float
        Input Dec in deg.
    dist : float
        Angular distance in arcsec.
        Can also be an array of distances.
    pos_ang : float
        Position angle (positive angles East of North) in degrees.
        Can also be an array; must match size of `dist`.
        
    Returns
    -------
    Two elements, RA and Dec of calculated offsets in dec degrees.
    If multiple offsets specified, then this will be two arrays
    of RA and Dec, where eac array has the same size as inputs.
    """
    
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    # Get base coordinates in astropy SkyCoord class
    c1 = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')

    # Get offset position
    position_angle = pos_ang * u.deg
    separation = dist * u.arcsec
    res = c1.directional_offset_by(position_angle, separation)

    # Return degrees
    return res.ra.deg, res.dec.deg
    

class jwst_point(object):

    """ JWST Telescope Pointing information

    Holds pointing coordinates and dither information for a given telescope
    visit.

    Parameters
    ==========
    ap_obs_name : str
        Name of the observed instrument aperture.
    ap_ref_name : str
        Name of the reference instrument aperture. Can be the same as observed,
        but not always.
    ra_ref : float
        RA position (in deg) of the reference aperture.
    dec_ref : float
        Dec position (in deg) of the reference aperture.
    
    Keyword Args
    ============
    pos_ang : float
        Position angle (positive angles East of North) in degrees.
    exp_nums : ndarray or None
        Option to specify exposure numbers associated with each dither
        position. Useful fro Visit book-keeping. If not set, then will
        simply be a `np.arange(self.ndith) + 1`
    base_offset : array-like
        Corresponds to (BaseX, BaseY) columns in .pointing file (arcsec).
    dith_offset : array-like
        Corresponds to (DithX, DithY) columns in .pointing file (arcsec). 
    base_std : float or array-like or None
        The 1-sigma pointing uncertainty per axis for telescope slew. 
        If None, then standard deviation is chosen to be either 5 mas 
        or 100 mas, depending on `use_ta` attribute (default: True).
    dith_std : float or array-like or None
        The 1-sigma pointing uncertainty per axis for dithers. If None,
        then standard deviation is chosen to be either 2.5 or 5 mas, 
        depending on `use_sgd` attribute (default: True).
    rand_seed : None or int
        Random seed to use for generating repeatable random offsets.
    rand_seed_base : None or int
        Use a separate random seed for telescope slew offset.
        Then, `rand_seed` corresponds only to relative dithers.
        Useful for multiple exposures with same initial slew, but
        independent dither pattern realizations. 
    """
    
    def __init__(self, ap_obs_name, ap_ref_name, ra_ref, dec_ref, pos_ang=0, 
        base_offset=(0,0), dith_offsets=[(0,0)], exp_nums=None,
        base_std=None, dith_std=None, rand_seed=None, rand_seed_base=None):

        # SIAF objects configuration
        # Reference instrument and aperture
        siaf_ref_inst = si_match.get(ap_ref_name[0:3])
        self.siaf_ap_ref = siaf_ref_inst[ap_ref_name]

        # Observation instrument and aperture
        self.siaf_inst = si_match.get(ap_obs_name[0:3])
        self.siaf_ap_obs = self.siaf_inst[ap_obs_name]
        
        # Store RA/Dec nominal pointing
        self.ra_ref  = ra_ref
        self.dec_ref = dec_ref
        
        # V2/V3 rotation relative to N/E. (e.g, +45 rotates V3 ccw towards E)
        self.pos_ang = pos_ang
        
        # Pointing shifts 
        # Update these with values from .pointing files
        # Baseline offset
        self.base_offset = base_offset
        # Dither offsets
        self.dith_offsets = dith_offsets

        self._exp_nums = exp_nums
        
        # Include randomized pointing offsets?
        self._base_std = base_std # Initial telescope pointing uncertainty (mas)
        self._dith_std = dith_std # Dither uncertainty value
        self.use_ta  = True   # Do we employ target acquisition to reduce pointing uncertainty?
        self.use_sgd = True   # True for small grid dithers, otherwise small angle maneuver

        self._sgd_sig = 2.5
        self._std_sig = 5.0
        
        # Get nominal RA/Dec of observed aperture's reference point
        # Don't include any offsets here, as they will be added later 
        ra_obs, dec_obs = self.ap_radec()
        self.ra_obs  = ra_obs
        self.dec_obs = dec_obs
        
        # Generate self.position_offsets_act, which houses all position offsets
        # for calculating positions of objects
        self.gen_random_offsets(rand_seed=rand_seed, rand_seed_base=rand_seed_base)

    @property
    def ndith(self):
        """Number of dither positions"""
        return len(self.dith_offsets)

    @property
    def exp_nums(self):
        """Exposure Numbers associated with each dither position"""
        if self._exp_nums is None:
            return np.arange(self.ndith) + 1
        else:
            return self._exp_nums
    @exp_nums.setter
    def exp_nums(self, value):
        """Set Exposure Numbers. Should be numpy array"""
        self._exp_nums = value
    
    @property
    def dith_std(self):
        """Dither pointing uncertainty (mas)"""
        if self._dith_std is None:
            dith_std = self._sgd_sig if self.use_sgd else self._std_sig
        else:
            dith_std = self._dith_std
        return dith_std
    @dith_std.setter
    def dith_std(self, value):
        """Dither pointing uncertainty (mas)"""
        self._dith_std = value
    
    @property
    def base_std(self):
        """Target pointing uncertainty (mas)"""
        if self._base_std is None:
            base_std = 5.0 if self.use_ta else 100
        else:
            base_std = self._base_std
        return base_std
    @base_std.setter
    def base_std(self, value):
        """Target pointing uncertainty (mas)"""
        self._base_std = value
        
    def attitude_matrix(self, idl_off=(0,0), ap_siaf_ref=None, coord_ref=None, **kwargs):
        """Return an attitutde correction matrix
        
        Parameters
        ==========
        idl_off : list or tuple
            X/Y offset of overall aperture offset. Usually a combination
            of base_off + dith_off (both in idl coordinates).
        ap_siaf_ref : pysiaf aperture
            Aperture reference being offset (uses it's V2/V3 ref/center coords).
            By default, uses self.siaf_ap_ref.
        coord_ref : tuple
            RA/Dec (in deg) reference coordinate nominally placed at 
            aperture's reference location prior to dither offset.
            Default is (self.ra_ref, self.dec_ref).
        """

        
        ap_siaf_ref = self.siaf_ap_ref if ap_siaf_ref is None else ap_siaf_ref

        # RA and Dec of ap ref location and the objects in the field
        ra_ref, dec_ref = (self.ra_ref, self.dec_ref) if coord_ref is None else coord_ref
        pa = self.pos_ang

        # Any field offset defined in 'idl' coords
        x_off, y_off = idl_off

        # V2/V3 reference location aligned with RA/Dec reference
        # and offset by (x_off, y_off) in 'idl' coords
        v2_ref, v3_ref = np.array(ap_siaf_ref.convert(x_off, y_off, 'idl', 'tel'))

        # Attitude correction matrix relative to reference aperture
        att = pysiaf.utils.rotations.attitude(v2_ref, v3_ref, ra_ref, dec_ref, pa)
        return att

    def ap_radec(self, idl_off=(0,0), get_cenpos=True, get_vert=False, 
                 ap_siaf_obs=None, **kwargs):
        """Aperture reference point(s) RA/Dec

        Given the (RA, Dec) and position angle some reference aperture,
        return the (RA, Dec) associated with the reference point (usually center) 
        of a different aperture. Can also return the corner vertices of the
        aperture. 

        Typically, the reference aperture (self.siaf_ap_ref) is used for the telescope 
        pointing information (e.g., NRCALL), but you may want to determine
        locations of the individual detector apertures (NRCA1_FULL, NRCB3_FULL, etc).

        Parameters
        ----------
        idl_off : list or tuple
            X/Y offset of overall aperture offset. Usually a combination
            of base_off + dith_off (both in idl coordinates)
        get_cenpos : bool
            Return aperture reference location coordinates?
        get_vert : bool
            Return closed polygon vertices (useful for plotting)?
        ap_siaf_obs : pysiaf aperture
            Specify the aperture for which to obtain RA/Dec reference point.
            The default is self.siaf_ap_obs.
        """

        if (get_cenpos==False) and (get_vert==False):
            _log.warning("Neither get_cenpos nor get_vert were set to True. Nothing to return.")
            return

        ap_siaf_obs = self.siaf_ap_obs if ap_siaf_obs is None else ap_siaf_obs

        # Attitude correction matrix relative to reference aperture
        # att = pysiaf.utils.rotations.attitude(v2_ref, v3_ref, ra_ref, dec_ref, pa)
        att = self.attitude_matrix(idl_off=idl_off, **kwargs)

        try: # Use internal pysiaf conversion wrapper (v0.12+)
            ap_siaf_obs.set_attitude_matrix(att)
            cen_obs  = ap_siaf_obs.reference_point('sky')
            vert_obs = ap_siaf_obs.closed_polygon_points('sky', rederive=False)
            ap_siaf_obs._attitude_matrix = None
        except AttributeError:
            # Get V2/V3 position of observed SIAF aperture and convert to RA/Dec
            if get_cenpos==True:
                v2_obs, v3_obs  = ap_siaf_obs.reference_point('tel')
                ra_obs, dec_obs = pysiaf.utils.rotations.pointing(att, v2_obs, v3_obs)
                cen_obs = (ra_obs, dec_obs)
            # Get V2/V3 vertices of observed SIAF aperture and convert to RA/Dec
            if get_vert==True:
                v2_vert, v3_vert  = ap_siaf_obs.closed_polygon_points('tel', rederive=False)
                ra_vert, dec_vert = pysiaf.utils.rotations.pointing(att, v2_vert, v3_vert)
                vert_obs = (ra_vert, dec_vert)

        if (get_cenpos==True) and (get_vert==True):
            return cen_obs, vert_obs
        elif get_cenpos==True:
            return cen_obs
        elif get_vert==True:
            return vert_obs
        else:
            _log.warning("Neither get_cenpos nor get_vert were set to True. Nothing to return.")
            return

    def radec_to_frame(self, coord_objs, frame_out='tel', idl_offsets=None):
        """RA/Dec to aperture coordinate frame

        Convert a series of RA/Dec positions to desired telescope SIAF coordinate frame 
        within the observed aperture. Will return a list of SIAF coordinates for all objects
        at each position.

        Parameters
        ----------
        coord_objs : tuple 
            (RA, Dec) positions (deg), where RA and Dec are numpy arrays.
        frame_out : str
            One of 'tel' (arcsec), 'sci' (pixels), or 'det' (pixels).
        idl_offsets : None or list of 2-element array
            Option to specify custom offset locations. Normally this is set to None, and
            we return RA/Dec for all telescope point positions defined in 
            `self.position_offsets_act`. However, we can specify offsets here (in 'idl')
            coordinates if you're only interested in a single position or want a custom
            location.
        """

        siaf_ap = self.siaf_ap_obs

        # RA and Dec of ap ref location and the objects in the field
        ra_ref, dec_ref = (self.ra_obs, self.dec_obs)
        # pa_ref = self.pos_ang
        ra_obj, dec_obj = coord_objs

        # Field offset as specified in APT Special Requirements
        # These appear to be defined in 'idl' coords
        if idl_offsets is None:
            idl_offsets = self.position_offsets_act

        out_all = []
        # For each dither position
        for idl_off in idl_offsets:
            # Attitude correction matrix relative to observed aperture
            att = self.attitude_matrix(idl_off=idl_off, ap_siaf_ref=siaf_ap, coord_ref=(ra_ref, dec_ref))

            try:
                # Use internal pysiaf conversion wrapper (v0.12+)
                siaf_ap.set_attitude_matrix(att)
                c1, c2 = siaf_ap.convert(ra_obj, dec_obj, 'sky', frame_out)
                siaf_ap._attitude_matrix = None
            except AttributeError:
                # Convert all RA/Dec coordinates into V2/V3 positions for objects
                v2_obj, v3_obj = pysiaf.utils.rotations.getv2v3(att, ra_obj, dec_obj)

                # Convert from tel to something else?
                if frame_out=='tel':
                    c1, c2 = (v2_obj, v3_obj)
                else:
                    c1, c2 = siaf_ap.convert(v2_obj, v3_obj, 'tel', frame_out)

            out_all.append((c1,c2))

        if len(out_all)==1:
            return out_all[0]
        else:
            return out_all
        
    def gen_random_offsets(self, rand_seed=None, rand_seed_base=None, first_dith_zero=True):
        """Generate randomized pointing offsets for each dither position"""
        
        _log.info('Generating random pointing offsets...')

        # Create a randomly assigned random seed...
        if rand_seed is None:
            rng = np.random.default_rng()
            rand_seed = rng.integers(0, 2**32-1)
        
        # Get initial point offset
        rand_seed_base = rand_seed if rand_seed_base is None else rand_seed_base
        _log.info(f'Pointing uncertainty: {self.base_std:.1f} mas')
        self.base_offset_act = get_idl_offset(base_offset=self.base_offset, base_std=self.base_std,
                                              dith_offset=(0,0), dith_std=0, rand_seed=rand_seed_base)
        
        # Get dither positions, including initial location 
        offsets_actual = []
        rand_seed_i = rand_seed + 1
        for i in range(self.ndith):
            dith_xy = self.dith_offsets[i]

            # Random seed should only increment if a dither offset exists.
            # If there's no offset, passing the same random seed will
            # produce the same random position, which we want.
            if (i>0) and (not np.allclose([dith_xy], self.dith_offsets[i-1])):
                rand_seed_i += 1
            # First position is slew, so no additional dither uncertainty
            dith_std = 0 if (first_dith_zero and i==0) else self.dith_std

            _log.info(f'  Pos {i} dither uncertainty: {dith_std:.1f} mas')
            offset = get_idl_offset(base_offset=self.base_offset_act, base_std=0,
                                    dith_offset=dith_xy, dith_std=dith_std, rand_seed=rand_seed_i)
            
            offsets_actual.append(offset)
            
        self.position_offsets_act = offsets_actual

    def plot_main_apertures(self, fill=False, **kwargs):
        """ Plot main SIAF telescope apertures.
        
        Other matplotlib standard parameters may be passed in via ``**kwargs``
        to adjust the style of the displayed lines.

        Parameters
        -----------
        darkbg : bool
            Plotting onto a dark background? Will make white outlines instead of black.
        detector_channels : bool
            Overplot the detector amplifier channels for all apertures.
        label : bool
            Add text labels stating aperture names
        units : str
            one of 'arcsec', 'arcmin', 'deg'.
        show_frame_origin : str or list
            Plot frame origin (goes to plot_frame_origin()): None, 'all', 'det',
            'sci', 'raw', 'idl', or a list of these.
        mark_ref : bool
            Add markers for the reference (V2Ref, V3Ref) point in each apertyre
        ax : matplotlib.Axes
            Desired destination axes to plot into (If None, current
            axes are inferred from pyplot.)
        fill : bool
            Whether to color fill the aperture
        fill_color : str
            Fill color
        fill_alpha : float
            alpha parameter for filled aperture
        color : matplotlib-compatible color
            Color specification for this aperture's outline,
            passed through to `matplotlib.Axes.plot`
        """
        
        pysiaf.siaf.plot_main_apertures(fill=fill, **kwargs)
        
    def plot_inst_apertures(self, subarrays=False, fill=False, **kwargs):
        """ Plot all apertures in this instrument's SIAF.
        
        Other matplotlib standard parameters may be passed in via ``**kwargs``
        to adjust the style of the displayed lines.

        Parameters
        -----------
        names : list of strings
            A subset of aperture names, if you wish to plot only a subset
        subarrays : bool
            Plot all the minor subarrays if True, else just plot the "main" apertures
        label : bool
            Add text labels stating aperture names
        units : str
            one of 'arcsec', 'arcmin', 'deg'. Only set for 'idl' and 'tel' frames.
        clear : bool
            Clear plot before plotting (set to false to overplot)
        show_frame_origin : str or list
            Plot frame origin (goes to plot_frame_origin()): None, 'all', 'det',
            'sci', 'raw', 'idl', or a list of these.
        mark_ref : bool
            Add markers for the reference (V2Ref, V3Ref) point in each apertyre
        frame : str
            Which coordinate system to plot in: 'tel', 'idl', 'sci', 'det'
        ax : matplotlib.Axes
            Desired destination axes to plot into (If None, current
            axes are inferred from pyplot.)
        fill : bool
            Whether to color fill the aperture
        fill_color : str
            Fill color
        fill_alpha : float
            alpha parameter for filled aperture
        color : matplotlib-compatible color
            Color specification for this aperture's outline,
            passed through to `matplotlib.Axes.plot`
        """
        
        # Ensure units and frame make sense
        # Only allow units to be set if frame is one of [None, 'idl', 'tel']
        units = kwargs.get('units')
        if units is not None:
            frame = kwargs.get('frame')
            if frame not in [None, 'idl', 'tel']:
                kwargs['units'] = None
                
        # Don't clear plot if specifying an axes
        if kwargs.get('ax') is not None:
            if kwargs.get('clear') is None:
                kwargs['clear'] = False

        return self.siaf_inst.plot(subarrays=subarrays, fill=fill, **kwargs)
    
    def plot_ref_aperture(self, fill=False, **kwargs):
        """ Plot reference aperture


        Parameters
        -----------
        names : list of strings
            A subset of aperture names, if you wish to plot only a subset
        subarrays : bool
            Plot all the minor subarrays if True, else just plot the "main" apertures
        label : bool
            Add text labels stating aperture names
        units : str
            one of 'arcsec', 'arcmin', 'deg'. Only set for 'idl' and 'tel' frames.
        clear : bool
            Clear plot before plotting (set to false to overplot)
        show_frame_origin : str or list
            Plot frame origin (goes to plot_frame_origin()): None, 'all', 'det',
            'sci', 'raw', 'idl', or a list of these.
        mark_ref : bool
            Add markers for the reference (V2Ref, V3Ref) point in each apertyre
        frame : str
            Which coordinate system to plot in: 'tel', 'idl', 'sci', 'det', 'sky'.
        ax : matplotlib.Axes
            Desired destination axes to plot into (If None, current
            axes are inferred from pyplot.)
        fill : bool
            Whether to color fill the aperture
        fill_color : str
            Fill color
        fill_alpha : float
            alpha parameter for filled aperture
        color : matplotlib-compatible color
            Color specification for this aperture's outline,
            passed through to `matplotlib.Axes.plot`
        """          
        siaf_ap = self.siaf_ap_ref

        # Set attitude matrix for sky transformations
        frame = kwargs.get('frame')
        if (frame is not None) and (frame=='sky'):
            att = self.attitude_matrix(**kwargs)
            try:
                siaf_ap.set_attitude_matrix(att)
            except AttributeError:
                _log.error("Running outdated version of pysiaf. Need >v12.0 for sky transformations.")

        siaf_ap.plot(fill=fill, **kwargs)
        siaf_ap._attitude_matrix = None
        
    def plot_obs_aperture(self, fill=False, **kwargs):
        """ Plot observed aperture

        Parameters
        -----------
        names : list of strings
            A subset of aperture names, if you wish to plot only a subset
        subarrays : bool
            Plot all the minor subarrays if True, else just plot the "main" apertures
        label : bool
            Add text labels stating aperture names
        units : str
            one of 'arcsec', 'arcmin', 'deg'. Only set for 'idl' and 'tel' frames.
        clear : bool
            Clear plot before plotting (set to false to overplot)
        show_frame_origin : str or list
            Plot frame origin (goes to plot_frame_origin()): None, 'all', 'det',
            'sci', 'raw', 'idl', or a list of these.
        mark_ref : bool
            Add markers for the reference (V2Ref, V3Ref) point in each apertyre
        frame : str
            Which coordinate system to plot in: 'tel', 'idl', 'sci', 'det', 'sky'
        ax : matplotlib.Axes
            Desired destination axes to plot into (If None, current
            axes are inferred from pyplot.)
        fill : bool
            Whether to color fill the aperture
        fill_color : str
            Fill color
        fill_alpha : float
            alpha parameter for filled aperture
        color : matplotlib-compatible color
            Color specification for this aperture's outline,
            passed through to `matplotlib.Axes.plot`
        """ 
        siaf_ap = self.siaf_ap_obs

        # Set attitude matrix for sky transformations
        frame = kwargs.get('frame')
        if (frame is not None) and (frame=='sky'):
            att = self.attitude_matrix(**kwargs)
            try:
                siaf_ap.set_attitude_matrix(att)
            except AttributeError:
                _log.error("Running outdated version of pysiaf. Need >v12.0 for sky transformations.")


        siaf_ap.plot(fill=fill, **kwargs)
        siaf_ap._attitude_matrix = None


def plotAxes(ax, position=(0.9,0.1), label1='V2', label2='V3', dir1=[-1,0], dir2=[0,1],
             angle=0, alength=0.12, width=1.5, headwidth=6, color='w', alpha=1,
             fontsize=11):
    """Compass arrows
    
    Show V2/V3 coordinate axis on a plot. By default, this function will plot
    the compass arrows in the lower right position in sky-right coordinates
    (ie., North/V3 up, and East/V2 to the left). 
    
    Parameters
    ==========
    ax : axis
        matplotlib axis to plot coordiante arrows.
    position : tuple
        XY-location of joined arrows as a fraction (0.0-1.0).
    label1 : str
        Label string for horizontal axis (ie., 'E' or 'V2').
    label2 : str
        Label string for vertical axis (ie, 'N' or 'V3').
    dir1 : array like
        XY-direction values to point "horizontal" arrow.
    dir2 : array like 
        XY-direction values to point "vertical" arrow.
    angle : float
        Rotate coordinate axis by some angle. 
        Positive values rotate counter-clockwise.
    alength : float
        Length of arrow vectors as fraction of plot axis.
    width : float
        Width of the arrow in points.
    headwidth : float
        Width of the base of the arrow head in points.
    color : color
        Self-explanatory.
    alpha : float
        Transparency.
    """
    arrowprops={'color':color, 'width':width, 'headwidth':headwidth, 'alpha':alpha}
    
    dir1 = xy_rot(dir1[0], dir1[1], angle)
    dir2 = xy_rot(dir2[0], dir2[1], angle)
    
    for (label, direction) in zip([label1,label2], np.array([dir1,dir2])):
        ax.annotate("", xytext=position, xy=position + alength * direction,
                    xycoords='axes fraction', arrowprops=arrowprops)
        textPos = position + alength * direction*1.3
        ax.text(textPos[0], textPos[1], label, transform=ax.transAxes,
                horizontalalignment='center', verticalalignment='center',
                color=color, fontsize=fontsize, alpha=alpha)

