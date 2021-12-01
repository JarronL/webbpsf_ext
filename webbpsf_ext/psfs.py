# Import libraries
import numpy as np
import multiprocessing as mp

from . import conf
from .utils import poppy, S
from .maths import jl_poly
from .image_manip import krebin, fshift
from .bandpasses import nircam_grism_res, niriss_grism_res

import logging
_log = logging.getLogger('webbpsf_ext')

from scipy.interpolate import griddata, RegularGridInterpolator

__epsilon = np.finfo(float).eps

def nproc_use(fov_pix, oversample, nwavelengths, coron=False):
    """Estimate Number of Processors

    Attempt to estimate a reasonable number of processors to use
    for a multi-wavelength calculation. One really does not want
    to end up swapping to disk with huge arrays.

    NOTE: Requires ``psutil`` package. Otherwise defaults to ``mp.cpu_count() / 2``

    Parameters
    -----------
    fov_pix : int
        Square size in detector-sampled pixels of final PSF image.
    oversample : int
        The optical system that we will be calculating for.
    nwavelengths : int
        Number of wavelengths.
    coron : bool
        Is the nproc recommendation for coronagraphic imaging?
        If so, the total RAM usage is different than for direct imaging.
    """

    try:
        import psutil
    except ImportError:
        nproc = int(mp.cpu_count() // 2)
        if nproc < 1: nproc = 1

        _log.info("No psutil package available, cannot estimate optimal nprocesses.")
        _log.info("Returning nproc=ncpu/2={}.".format(nproc))
        return nproc

    mem = psutil.virtual_memory()
    avail_GB = mem.available / 1024**3
    # Leave 10% for other things
    avail_GB *= 0.9

    fov_pix_over = fov_pix * oversample

    # For multiprocessing, memory accumulates into the main process
    # so we have to subtract the total from the available amount
    reserve_GB = nwavelengths * fov_pix_over**2 * 8 / 1024**3
    # If not enough available memory, then just return nproc=1
    if avail_GB < reserve_GB:
        _log.warn('Not enough available memory ({} GB) to \
                   to hold resulting PSF info ({} GB)!'.\
                   format(avail_GB,reserve_GB))
        return 1

    avail_GB -= reserve_GB

    # Memory formulas are based on fits to memory usage stats for:
    #   fov_arr = np.array([16,32,128,160,256,320,512,640,1024,2048])
    #   os_arr = np.array([1,2,4,8])
    if coron:  # Coronagraphic Imaging (in MB)
        mem_total = (oversample*1024*2.4)**2 * 16 / (1024**2) + 500
        if fov_pix > 1024: mem_total *= 1.6
    else:  # Direct Imaging (also spectral imaging)
        mem_total = 5*(fov_pix_over)**2 * 8 / (1024**2) + 300.

    # Convert to GB
    mem_total /= 1024

    # How many processors to split into?
    nproc = int(avail_GB / mem_total)
    nproc = np.min([nproc, mp.cpu_count(), poppy.conf.n_processes])

    # Each PSF calculation will constantly use multiple processors
    # when not oversampled, so let's divide by 2 for some time
    # and memory savings on those large calculations
    if oversample==1:
        nproc = np.ceil(nproc / 2)

    _log.debug('avail mem {}; mem tot: {}; nproc_init: {:.0f}'.\
        format(avail_GB, mem_total, nproc))

    nproc = np.min([nproc, nwavelengths])
    # Resource optimization:
    # Split iterations evenly over processors to free up minimally used processors.
    # For example, if there are 5 processes only doing 1 iteration, but a single
    #	processor doing 2 iterations, those 5 processors (and their memory) will not
    # 	get freed until the final processor is finished. So, to minimize the number
    #	of idle resources, take the total iterations and divide by two (round up),
    #	and that should be the final number of processors to use.
    np_max = np.ceil(nwavelengths / nproc)
    nproc = int(np.ceil(nwavelengths / np_max))

    if nproc < 1: nproc = 1

    # Multiprocessing can only swap up to 2GB of data from the child
    # process to the master process. Return nproc=1 if too much data.
    im_size = (fov_pix_over)**2 * 8 / (1024**3)
    nproc = 1 if (im_size * np_max) >=2 else nproc

    _log.debug('avail mem {}; mem tot: {}; nproc_fin: {:.0f}'.\
        format(avail_GB, mem_total, nproc))

    return int(nproc)

def gen_image_from_coeff(inst, coeff, coeff_hdr, sp_norm=None, nwaves=None, 
                         use_sp_waveset=False, return_oversample=False):
    
    """Generate PSF

    Create an image (direct, coronagraphic, grism, or DHS) based on a set of
    instrument parameters and PSF coefficients. The image is noiseless and
    doesn't take into account any non-linearity or saturation effects, but is
    convolved with the instrument throughput. Pixel values are in counts/sec.
    The result is effectively an idealized slope image.

    If no spectral dispersers, then this returns a single image or list of 
    images if sp_norm is a list of spectra.

    Parameters
    ----------
    coeff : ndarray
        A cube of polynomial coefficients for generating PSFs. This is
        generally oversampled with a shape (fov_pix*oversamp, fov_pix*oversamp, deg).
    coeff_hdr : FITS header
        Header information saved while generating coefficients.
    sp_norm : :mod:`pysynphot.spectrum`
        A normalized Pysynphot spectrum to generate image. If not specified,
        the default is flat in phot lam (equal number of photons per spectral bin).
        The default is normalized to produce 1 count/sec within that bandpass,
        assuming the telescope collecting area. Coronagraphic PSFs will further
        decrease this flux.
    nwaves : int
        Option to specify the number of evenly spaced wavelength bins to
        generate and sum over to make final PSF. Useful for wide band filters
        with large PSFs over continuum source.
    use_sp_waveset : bool
        Set this option to use `sp_norm` waveset instead of bandpass waveset.
        Useful if user inputs a high-resolution spectrum with line emissions,
        so may wants to keep a grism PSF (for instance) at native resolution
        rather than blurred with the bandpass waveset. TODO: Test.  
    return_oversample: bool
        If True, then instead returns the oversampled version of the PSF.

    Keyword Args
    ------------
    grism_order : int
        Grism spectral order (default=1).
    ND_acq : bool
        ND acquisition square in coronagraphic mask.
    """

    # Sort out any spectroscopic modes
    if (inst.name=='NIRCam') or (inst.name=='NIRISS'):
        is_grism = inst.is_grism
    else:
        is_grism = False
    is_dhs = False

    if (inst.name=='MIRI') or (inst.name=='NIRSpec'):
        is_slitspec = inst.is_slitspec
    else:
        is_slitspec = False

    # Get Bandpass
    bp = inst.bandpass

    # Get wavelength range
    npix = coeff.shape[-1]
    # waveset = create_waveset(bp, npix, nwaves=nwaves, is_grism=is_grism)

    # List of sp observation converted to count rate
    obs_list = create_obslist(bp, npix, nwaves=nwaves, is_grism=is_grism,
                              sp_norm=sp_norm, use_sp_waveset=use_sp_waveset)
    nspec = len(obs_list)

    # Get wavelength range
    waveset = obs_list[0].binwave
    wgood = waveset / 1e4
    w1 = wgood.min()
    w2 = wgood.max()
    wrange = w2 - w1

    # Create a PSF for each wgood wavelength
    use_legendre = True if coeff_hdr['LEGNDR'] else False
    lxmap = [coeff_hdr['WAVE1'], coeff_hdr['WAVE2']]
    psf_fit = jl_poly(wgood, coeff, use_legendre=use_legendre, lxmap=lxmap)

    # Multiply each monochromatic PSFs by the binned e/sec at each wavelength
    # Array broadcasting: [nx,ny,nwave] x [1,1,nwave]
    # Do this for each spectrum/observation
    if nspec==1:
        psf_fit *= obs_list[0].binflux.reshape([-1,1,1])
        psf_list = [psf_fit]
    else:
        psf_list = [psf_fit*obs.binflux.reshape([-1,1,1]) for obs in obs_list]
        del psf_fit

    # The number of pixels to span spatially
    fov_pix = int(coeff_hdr['FOVPIX'])
    oversample = int(coeff_hdr['OSAMP'])
    fov_pix_over = int(fov_pix * oversample)

    # Grism spectroscopy
    if is_grism:
        pupil_mask = inst.pupil_mask
        if 'GRISM0' in pupil_mask:
            pupil_mask = 'GRISMR'
        elif 'GRISM90' in pupil_mask:
            pupil_mask = 'GRISMC'

        # spectral resolution in um/pixel
        # res is in pixels per um and dw is inverse
        grism_order = inst._grism_order
        if inst.name=='NIRCam':
            res, dw = nircam_grism_res(pupil_mask, inst.module, grism_order)
        elif inst.name=='NIRISS':
            res, dw = niriss_grism_res(grism_order)

        # Number of real pixels that spectra will span
        npix_spec = int(wrange // dw + 1 + fov_pix)
        npix_spec_over = int(npix_spec * oversample)

        spec_list = []
        spec_list_over = []
        for psf_fit in psf_list:
            # If GRISMC (along columns) rotate image by 90 deg CW to disperse left-to-right
            if 'GRISMC' in pupil_mask:
                psf_fit = np.rot90(psf_fit, k=1, axes=(1,2)) 

            # Create oversampled spectral image
            spec_over = np.zeros([fov_pix_over, npix_spec_over])
            # Place each PSF at its dispersed location (left-to-right)
            for i, w in enumerate(wgood):
                # Separate shift into an integer and fractional shift
                delx = oversample * (w-w1) / dw # Number of oversampled pixels to shift
                intx = int(delx)
                fracx = delx - intx
                if fracx < 0:
                    fracx = fracx + 1
                    intx = intx - 1

                # TODO: Benchmark and compare these two different methods
                # spec_over[:,intx:intx+fov_pix_over] += fshift(psf_fit[i], delx=fracx, interp='cubic')
                im = psf_fit[i]
                im_part1 = im*(1.-fracx)
                im_part2 = np.roll(im,1,axis=1)*fracx
                im_part2[:,0] = 0 # Right side of PSF rolls over to left side; set to 0 instead
                spec_over[:,intx:intx+fov_pix_over] += (im_part1 + im_part2)

            spec_over[spec_over<__epsilon] = 0 

            # Rotate spectrum to its V2/V3 coordinates
            spec_bin = krebin(spec_over, (fov_pix,npix_spec))
            if 'GRISMC' in pupil_mask: # Rotate image 90 deg CCW to disperse bottom-to-top
                spec_over = np.rot90(spec_over, k=-1)
                spec_bin = np.rot90(spec_bin, k=-1)
            elif (inst.name=='NIRCam') and (inst.module=='B'): 
                # Flip for sci coords to disperse right-to-left
                spec_over = spec_over[:,::-1]
                spec_bin = spec_bin[:,::-1]

            # Rebin ovesampled spectral image to real pixels
            spec_list.append(spec_bin)
            spec_list_over.append(spec_over)

        # Wavelength solutions
        dw_over = dw/oversample
        w1_spec = w1 - dw_over*fov_pix_over/2
        wspec_over = np.arange(npix_spec_over)*dw_over + w1_spec
        wspec = wspec_over.reshape((npix_spec,-1)).mean(axis=1)
        if (inst.name=='NIRCam') and ('GRISMR' in pupil_mask) and (inst.module=='B'): 
            # Flip wavelength for sci coords
            wspec = wspec[::-1]

        if nspec == 1: 
            spec_list = spec_list[0]
            spec_list_over = spec_list_over[0]

        # _log.debug('jl_poly: {:.2f} sec; binflux: {:.2f} sec; disperse: {:.2f} sec'.format(t5-t4, t6-t5, t7-t6))
        # Return list of wavelengths for each horizontal pixel as well as spectral image
        if return_oversample:
            return (wspec_over, spec_list_over)
        else:
            return (wspec, spec_list)

    # DHS spectroscopy
    elif is_dhs:
        raise NotImplementedError('DHS has yet to be fully included')

    # Imaging
    else:
        # Create source image slopes (no noise)
        data_list = []
        data_list_over = []
        eps = np.finfo(float).eps
        for psf_fit in psf_list:
            data_over = psf_fit.sum(axis=0)
            data_over[data_over<=eps] = data_over[data_over>eps].min() / 10
            data_list_over.append(data_over)
            data_list.append(krebin(data_over, (fov_pix,fov_pix)))

        if nspec == 1: 
            data_list = data_list[0]
            data_list_over = data_list_over[0]

        #_log.debug('jl_poly: {:.2f} sec; binflux: {:.2f} sec; PSF sum: {:.2f} sec'.format(t5-t4, t6-t5, t7-t6))
        if return_oversample:
            return data_list_over
        else:
            return data_list


def create_waveset(bp, npix, nwaves=None, is_grism=False):

    waveset = np.copy(bp.wave)
    if nwaves is not None:
        # Evenly spaced wavelengths
        waveset = np.linspace(waveset.min(), waveset.max(), nwaves)
    elif is_grism:
        waveset = waveset
    else:
        # For generating the PSF, let's save some time and memory by not using
        # ever single wavelength in the bandpass.
        # Do NOT do this for dispersed modes.
        binsize = 1
        if npix>2000:
            binsize = 7
        elif npix>1000:
            binsize = 5
        elif npix>700:
            binsize = 3

        if binsize>1:
            excess = waveset.size % binsize
            waveset = waveset[:waveset.size-excess]
            waveset = waveset.reshape(-1,binsize) # Reshape
            waveset = waveset[:,binsize//2] # Use the middle values
            waveset = np.concatenate(([bp.wave[0]],waveset,[bp.wave[-1]]))
    
    return waveset

def create_obslist(bp, npix, nwaves=None, is_grism=False,
                   sp_norm=None, use_sp_waveset=False):

    waveset = create_waveset(bp, npix, nwaves=nwaves, is_grism=is_grism)
    wgood = waveset / 1e4
    w1 = wgood.min()
    w2 = wgood.max()

    # Flat spectrum with equal photon flux in each spectal bin
    if sp_norm is None:
        sp_flat = S.ArraySpectrum(waveset, 0*waveset + 10.)
        sp_flat.name = 'Flat spectrum in flam'

        # Bandpass unit response is the flux (in flam) of a star that
        # produces a response of one count per second in that bandpass
        sp_norm = sp_flat.renorm(bp.unit_response(), 'flam', bp)

    # Make sp_norm a list of spectral objects if it already isn't
    if not isinstance(sp_norm, list): 
        sp_norm = [sp_norm]
    nspec = len(sp_norm)

    # Set up an observation of the spectrum using the specified bandpass
    if use_sp_waveset:
        if nspec>1:
            raise AttributeError("Only 1 spectrum allowed when use_sp_waveset=True.")
        # Modify waveset if use_sp_waveset=True
        obs_list = []
        for sp in sp_norm:
            # Select only wavelengths within bandpass
            waveset = sp.wave
            waveset = waveset[(waveset>=w1*1e4) and (waveset<=w2*1e4)]
            obs_list.append(S.Observation(sp, bp, binset=waveset))
    else:
        # Use the bandpass wavelength set to bin the fluxes
        obs_list = [S.Observation(sp, bp, binset=waveset) for sp in sp_norm]

    # Convert to count rate
    for obs in obs_list: 
        obs.convert('counts')

    return obs_list


def make_coeff_resid_grid(xin, yin, cf_resid, xgrid, ygrid):

    # Create 2D grid arrays of coordinates
    xnew, ynew = np.meshgrid(xgrid,ygrid)
    nx, ny = len(xgrid), len(ygrid)

    _log.warn("Interpolating coefficient residuals onto regular grid...")

    sh = cf_resid.shape
    cf_resid_grid = np.zeros([ny,nx,sh[1],sh[2],sh[3]])

    # Cycle through each coefficient to interpolate onto V2/V3 grid
    for i in range(sh[1]):
        cf_resid_grid[:,:,i,:,:] = griddata((xin, yin), cf_resid[:,i,:,:], (xnew, ynew), method='cubic')

    return cf_resid_grid


def field_coeff_func(v2grid, v3grid, cf_fields, v2_new, v3_new, method='linear'):
    """Interpolation function for PSF coefficient residuals

    Uses `RegularGridInterpolator` to quickly determine new coefficient
    residulas at specified points.

    Parameters
    ----------
    v2grid : ndarray
        V2 values corresponding to `cf_fields`.
    v3grid : ndarray
        V3 values corresponding to `cf_fields`.
    cf_fields : ndarray
        Coefficient residuals at different field points
        Shape is (nV3, nV2, ncoeff, ypix, xpix)
    v2_new : ndarray
        New V2 point(s) to interpolate on. Same units as v2grid.
    v3_new : ndarray
        New V3 point(s) to interpolate on. Same units as v3grid.
    """

    func = RegularGridInterpolator((v3grid, v2grid), cf_fields, method=method, 
                                   bounds_error=False, fill_value=None)

    pts = np.array([v3_new,v2_new]).transpose()
    
    if np.size(v2_new)>1:
        res = np.asarray([func(pt).squeeze() for pt in pts])
    else:
        res = func(pts)

    # If only 1 point, remove first axes
    res = res.squeeze() if res.shape[0]==1 else res
    return res

