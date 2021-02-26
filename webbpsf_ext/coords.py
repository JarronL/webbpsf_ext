import numpy as np
import logging
_log = logging.getLogger('pynrc')

__epsilon = np.finfo(float).eps

from .utils import webbpsf, poppy, pysiaf

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
        Location (x,y) in the array calculate distance. If set 
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
    
    import pysiaf
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

