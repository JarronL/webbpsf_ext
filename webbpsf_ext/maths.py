# Import libraries
import numpy as np
from numpy.polynomial import legendre
from scipy.special import eval_legendre


from .coords import dist_image
from .image_manip import frebin

###########################################################################
#    Polynomial fitting
###########################################################################

def jl_poly(xvals, coeff, dim_reorder=False, use_legendre=False, lxmap=None, **kwargs):
    """Evaluate polynomial
    
    Replacement for `np.polynomial.polynomial.polyval(wgood, coeff)`
    to evaluate y-values given a set of xvals and coefficients.
    Uses matrix multiplication, which is much faster. Beware, the
    default output array shapes organization may differ from the 
    polyval routine for 2D and 3D results.

    Parameters
    ----------
    xvals : ndarray
        1D array (time, for instance)
    coeff : ndarray
        1D, 2D, or 3D array of coefficients from a polynomial fit.
        The first dimension should have a number of elements equal
        to the polynomial degree + 1. Order such that lower degrees
        are first, and higher degrees are last.

    Keyword Args
    ------------
    dim_reorder : bool
        If true, then result to be ordered (nx,ny,nz), otherwise we
        use the Python preferred ordering (nz,ny,nx)
    use_legendre : bool
        Fit with Legendre polynomial, an orthonormal basis set.
    lxmap : ndarray or None
        Legendre polynomials are normaly mapped to xvals of [-1,+1].
        `lxmap` gives the option to supply the values for xval that
        should get mapped to [-1,+1]. If set to None, then assumes 
        [xvals.min(),xvals.max()].
                       
    Returns
    -------
    float array
        An array of values where each xval has been evaluated at each
        set of supplied coefficients. The output shape has the first 
        dimension equal to the number of xvals, and the final dimensions
        correspond to coeff's latter dimensions. The result is flattened 
        if there is either only one xval or one set of coeff (or both).
    """

    # How many xvals?
    n = np.size(xvals)
    try:
        xdim = len(xvals.shape)
    except AttributeError:
        # Handle list
        xvals = np.array(xvals)
        xdim = len(xvals.shape)
        # Handle single value
        if xdim == 0:
            xvals = np.array([xvals])
            xdim = len(xvals.shape)

    if xdim>1:
        raise ValueError('xvals can only have 1 dimension. Found {} dimensions.'.format(xdim))

    # Check number of dimensions in coefficients
    dim = coeff.shape
    ndim = len(dim)
    if ndim>3:
        raise ValueError('coefficient can only have 1, 2, or 3 dimensions. \
                          Found {} dimensions.'.format(ndim))

    if use_legendre:
        # Values to map to [-1,+1]
        if lxmap is None:
            lxmap = [np.min(xvals), np.max(xvals)]

        # Remap xvals -> lxvals
        dx = lxmap[1] - lxmap[0]
        lxvals = 2 * (xvals - (lxmap[0] + dx/2)) / dx

        # Use Identity matrix to evaluate each polynomial component
        # xfan = legendre.legval(lxvals, np.identity(dim[0]))
        # Below method is faster for large lxvals
        xfan = np.asarray([eval_legendre(n, lxvals) for n in range(dim[0])])
    else:
        # Create an array of exponent values
        parr = np.arange(dim[0], dtype='float')
        # If 3D, this reshapes xfan to 2D
        xfan = xvals**parr.reshape((-1,1)) # Array broadcasting

    # Reshape coeffs to 2D array
    cf = coeff.reshape(dim[0],-1)
    if dim_reorder:
        # Coefficients are assumed (deg+1,nx,ny)
        # xvals have length nz
        # Result to be ordered (nx,ny,nz)
        # Use np.matmul instead of np.dot for speed improvement
        yfit = np.matmul(cf.T, xfan) # cf.T @ xfan

        if ndim==1 or n==1: 
            yfit = yfit.ravel()
        if ndim==3: 
            yfit = yfit.reshape((dim[1],dim[2],n))
    else:
        # This is the Python preferred ordering
        # Coefficients are assumed (deg+1,ny,nx)
        # xvals have length nz
        # Result to be ordered (nz,ny,nx)
        # Use np.matmul instead of np.dot for speed improvement
        yfit = np.matmul(xfan.T, cf) # xfan.T @ cf

        if ndim==1 or n==1: 
            yfit = yfit.ravel()
        if ndim==3: 
            yfit = yfit.reshape((n,dim[1],dim[2]))

    return yfit


def jl_poly_fit(x, yvals, deg=1, QR=True, robust_fit=False, niter=25, use_legendre=False, lxmap=None, **kwargs):
    """Fast polynomial fitting
    
    Fit a polynomial to a function using linear least-squares.
    This function is particularly useful if you have a data cube
    and want to simultaneously fit a slope to all pixels in order
    to produce a slope image.
    
    Gives the option of performing QR decomposition, which provides
    a considerable speed-up compared to simply using `np.linalg.lstsq()`.
    In addition to being fast, it has better numerical stability than
    linear regressions that involve matrix inversions (ie., `dot(x.T,x)`).
    
    Returns the coefficients of the fit for each pixel.
    
    Parameters
    ----------
    x : ndarray
        X-values of the data array (1D).
    yvals : ndarray 
        Y-values (1D, 2D, or 3D) where the first dimension
        must have equal length of x. For instance, if x is
        a time series of a data cube with size NZ, then the 
        data cube must follow the Python convention (NZ,NY,NZ).

    Keyword Args
    ------------
    deg : int
        Degree of polynomial to fit to the data.
    QR : bool
        Perform QR decomposition? Default=True.
    robust_fit : bool
        Perform robust fitting, iteratively kicking out 
        outliers until convergence.
    niter : int
        Maximum number of iterations for robust fitting.
        If convergence is attained first, iterations will stop.
    use_legendre : bool
        Fit with Legendre polynomials, an orthonormal basis set.
    lxmap : ndarray or None
        Legendre polynomials are normally mapped to xvals of [-1,+1].
        `lxmap` gives the option to supply the values for xval that
        should get mapped to [-1,+1]. If set to None, then assumes 
        [xvals.min(),xvals.max()].
    
    Example
    -------
    Fit all pixels in a data cube to get slope image in terms of ADU/sec
    
    >>> nz, ny, nx = cube.shape
    >>> tvals = (np.arange(nz) + 1) * 10.737
    >>> coeff = jl_poly_fit(tvals, cube, deg=1)
    >>> bias = coeff[0]  # Bias image (y-intercept)
    >>> slope = coeff[1] # Slope image (DN/sec)
    """
    
    from .robust import medabsdev
    
    orig_shape = yvals.shape
    ndim = len(orig_shape)
    
    cf_shape = list(yvals.shape)
    cf_shape[0] = deg+1
    
    if ndim==1:
        assert len(x)==len(yvals), 'X and Y must have the same length'
    else:
        assert len(x)==orig_shape[0], 'X and Y.shape[0] must have the same length'

    # Get different components to fit
    if use_legendre:
        # Values to map to [-1,+1]
        if lxmap is None:
            lxmap = [np.min(x), np.max(x)]

        # Remap xvals -> lxvals
        dx = lxmap[1] - lxmap[0]
        lx = 2 * (x - (lxmap[0] + dx/2)) / dx

        # Use Identity matrix to evaluate each polynomial component
        # a = legendre.legval(lx, np.identity(deg+1))
        # Below method is faster for large lxvals
        a = np.asarray([eval_legendre(n, lx) for n in range(deg+1)])
    else:
        # Normalize x values to closer to 1 for numerical stability with large inputs
        xnorm = np.mean(x)
        x = x / xnorm
        a = np.asarray([x**num for num in range(deg+1)], dtype='float')
    b = yvals.reshape([orig_shape[0],-1])

    # Fast method, but numerically unstable for overdetermined systems
    #cov = np.linalg.pinv(np.matmul(a,a.T))
    #coeff_all = np.matmul(cov,np.matmul(a,b))
    
    if QR:
        # Perform QR decomposition of the A matrix
        q, r = np.linalg.qr(a.T, 'reduced')
        # computing Q^T*b (project b onto the range of A)
        # Use np.matmul instead of np.dot for speed improvement
        qTb = np.matmul(q.T,b) # q.T @ b 
        # solving R*x = Q^T*b
        coeff_all = np.linalg.lstsq(r, qTb, rcond=None)[0]
    else:
        coeff_all = np.linalg.lstsq(a.T, b, rcond=None)[0]
        
    if robust_fit:
        # Normally, we would weight both the x and y (ie., a and b) values
        # then plug those into the lstsq() routine. However, this means we
        # can no longer use the same x values for a series of y values. Each
        # fit would have differently weight x-values, requiring us to fit
        # each element individually, which would be very slow. 
        # Instead, we will compromise by "fixing" outliers in order to 
        # preserve the quickness of this routine. The fixed outliers become 
        # the new data that we refit. 

        close_factor = 0.03
        close_enough = np.max([close_factor * np.sqrt(0.5/(x.size-1)), 1e-20])
        err = 0
        for i in range(niter):
            # compute absolute value of residuals (fit minus data)
            yvals_mod = jl_poly(x, coeff_all, use_legendre=use_legendre)
            abs_resid = np.abs(yvals_mod - b)

            # compute the scaling factor for the standardization of residuals
            # using the median absolute deviation of the residuals
            # 6.9460 is a tuning constant (4.685/0.6745)
            abs_res_scale = 6.9460 * np.median(abs_resid, axis=0)

            # standardize residuals
            w = abs_resid / abs_res_scale.reshape([1,-1])

            # exclude outliers
            outliers = w>1
            
            # Create a version with outliers fixed
            # Se
            yvals_fix = b.copy()
            yvals_fix[outliers] = yvals_mod[outliers]
            
            # Ignore fits with no outliers
            ind_fit = outliers.sum(axis=0) > 0
            if ind_fit[ind_fit].size == 0: break
            if QR:
                # Use np.matmul instead of np.dot for speed improvement
                qTb = np.matmul(q.T,yvals_fix[:,ind_fit]) # q.T @ yvals_fix[:,ind_fit] 
                coeff_all[:,ind_fit] = np.linalg.lstsq(r, qTb, rcond=None)[0]
            else:
                coeff_all[:,ind_fit] = np.linalg.lstsq(a.T, yvals_fix[:,ind_fit], rcond=None)[0]

            prev_err = medabsdev(abs_resid, axis=0) if i==0 else err
            err = medabsdev(abs_resid, axis=0)
            
            diff = np.abs((prev_err - err)/err)
            #print(coeff_all.mean(axis=1), coeff_all.std(axis=1), np.nanmax(diff), ind_fit[ind_fit].size)
            if 0 < np.nanmax(diff) < close_enough: break
    
    if not use_legendre:
        parr = np.arange(deg+1, dtype='float')
        coeff_all = coeff_all / (xnorm**parr.reshape([-1,1]))

    return coeff_all.reshape(cf_shape)


###########################################################################
#    Binning and stats
###########################################################################

def hist_indices(values, bins=10, return_more=False):
    """Histogram indices
    
    This function bins an input of values and returns the indices for
    each bin. This is similar to the reverse indices functionality
    of the IDL histogram routine. It's also much faster than doing
    a for loop and creating masks/indices at each iteration, because
    we utilize a sparse matrix constructor. 
    
    Returns a list of indices grouped together according to the bin.
    Only works for evenly spaced bins.
    
    Parameters
    ----------
    values : ndarray
        Input numpy array. Should be a single dimension.
    bins : int or ndarray
        If bins is an int, it defines the number of equal-width bins 
        in the given range (10, by default). If bins is a sequence, 
        it defines the bin edges, including the rightmost edge.
        In the latter case, the bins must encompass all values.
    return_more : bool
        Option to also return the values organized by bin and 
        the value of the centers (igroups, vgroups, center_vals).
    
    Example
    -------
    Find the standard deviation at each radius of an image
    
        >>> rho = dist_image(image)
        >>> binsize = 1
        >>> bins = np.arange(rho.min(), rho.max() + binsize, binsize)
        >>> igroups, vgroups, center_vals = hist_indices(rho, bins, True)
        >>> # Get the standard deviation of each bin in image
        >>> std = binned_statistic(igroups, image, func=np.std)

    """
    
    from scipy.sparse import csr_matrix
    
    values_flat = values.ravel()

    vmin = values_flat.min()
    vmax = values_flat.max()
    N  = len(values_flat)   
    
    try: # assume it's already an array
        binsize = bins[1] - bins[0]
    except TypeError: # if bins is an integer
        binsize = (vmax - vmin) / bins
        bins = np.arange(vmin, vmax + binsize, binsize)
        bins[0] = vmin
        bins[-1] = vmax
    
    # Central value of each bin
    center_vals = bins[:-1] + binsize / 2.
    nbins = center_vals.size

    # TODO: If input bins is an array that doesn't span the full set of input values,
    # then we need to set a warning.
    if (vmin<bins[0]) or (vmax>bins[-1]):
        raise ValueError("Bins must encompass entire set of input values.")
    digitized = ((nbins-1.0) / (vmax-vmin) * (values_flat-vmin)).astype(np.int)
    csr = csr_matrix((values_flat, [digitized, np.arange(N)]), shape=(nbins, N))

    # Split indices into their bin groups    
    igroups = np.split(csr.indices, csr.indptr[1:-1])
    
    if return_more:
        vgroups = np.split(csr.data, csr.indptr[1:-1])
        return (igroups, vgroups, center_vals)
    else:
        return igroups
    

def binned_statistic(x, values, func=np.mean, bins=10):
    """Binned statistic
    
    Compute a binned statistic for a set of data. Drop-in replacement
    for scipy.stats.binned_statistic.

    Parameters
    ----------
    x : ndarray
        A sequence of values to be binned. Or a list of binned 
        indices from hist_indices().
    values : ndarray
        The values on which the statistic will be computed.
    func : func
        The function to use for calculating the statistic. 
    bins : int or ndarray
        If bins is an int, it defines the number of equal-width bins 
        in the given range (10, by default). If bins is a sequence, 
        it defines the bin edges, including the rightmost edge.
        This doesn't do anything if x is a list of indices.
             
    Example
    -------
    Find the standard deviation at each radius of an image
    
        >>> rho = dist_image(image)
        >>> binsize = 1
        >>> radial_bins = np.arange(rho.min(), rho.max() + binsize, binsize)
        >>> radial_stds = binned_statistic(rho, image, func=np.std, bins=radial_bins)
    
    """

    values_flat = values.ravel()
    
    try: # This will be successful if x is not already a list of indices
    
        # Check if bins is a single value
        if (len(np.array(bins))==1) and (bins is not None):
            igroups = hist_indices(x, bins=bins, return_more=False)
            res = np.array([func(values_flat[ind]) for ind in igroups])
        # Otherwise we assume bins is a list or array defining edge locations
        else:
            bins = np.array(bins)
            # Check if binsize is the same for all bins
            bsize = bins[1:] - bins[:-1]
            # Make sure bins encompass full set of input values
            ind_bin = (x>=bins.min()) & (x<=bins.max())
            x = x[ind_bin]
            values_flat = values_flat[ind_bin]
            if np.isclose(bsize.min(), bsize.max()):
                igroups = hist_indices(x, bins=bins, return_more=False)
                res = np.array([func(values_flat[ind]) for ind in igroups])
            else:
                # If non-uniform bins, just use scipy.stats.binned_statistic
                from scipy import stats 
                res, _, _ = stats.binned_statistic(x, values, func, bins)
    except:
        # Assume that input is a list of indices
        igroups = x
        res = np.array([func(values_flat[ind]) for ind in igroups])
    
    return res


def radial_std(im_diff, pixscale=None, oversample=None, supersample=False, func=np.std):
    """Generate contrast curve of PSF difference

    Find the standard deviation within fixed radial bins of a differenced image.
    Returns two arrays representing the 1-sigma contrast curve at given distances.

    Parameters
    ==========
    im_diff : ndarray
        Differenced image of two PSFs, for instance.

    Keywords
    ========
    pixscale : float  
        Pixel scale of the input image
    oversample : int
        Is the input image oversampled compared to detector? If set, then
        the binsize will be pixscale*oversample (if supersample=False).
    supersample : bool
        If set, then oversampled data will have a binsize of pixscale,
        otherwise the binsize is pixscale*oversample.
    func_std : func
        The function to use for calculating the radial standard deviation.

    """

    from astropy.convolution import convolve, Gaussian1DKernel

    # Set oversample to 1 if supersample keyword is set
    oversample = 1 if supersample or (oversample is None) else oversample

    # Rebin data
    data_rebin = frebin(im_diff, scale=1/oversample)

    # Determine pixel scale of rebinned data
    pixscale = 1 if pixscale is None else oversample*pixscale

    # Pixel distances
    rho = dist_image(data_rebin, pixscale=pixscale)

    # Get radial profiles
    binsize = pixscale
    bins = np.arange(rho.min(), rho.max() + binsize, binsize)
    nan_mask = np.isnan(data_rebin)
    igroups, _, rr = hist_indices(rho[~nan_mask], bins, True)
    stds = binned_statistic(igroups, data_rebin[~nan_mask], func=func)
    stds = convolve(stds, Gaussian1DKernel(1))

    # Ignore corner regions
    arr_size = np.min(data_rebin.shape) * pixscale
    mask = rr < (arr_size/2)

    return rr[mask], stds[mask]


def find_closest(A, B):
    """ Find closest indices
    
    Given two arrays, A and B, find the indices in B whose values
    are closest to those in A. Returns an array with size equal to A.
    
    This is much much faster than something like, especially for large arrays:
        `np.argmin(np.abs(A - B[:, np.newaxis]), axis=0)`

    """
    
    # Make sure these are array
    a = np.asarray(A)
    if np.size(B)==1:
        b = np.asarray([B])
    else:
        b = np.asarray(B)
    
    # Flatten a array
    a_shape = a.shape
    if len(a_shape)>1:
        a = a.flatten()
    
    # b needs to be sorted
    isort = np.argsort(B)
    b = b[isort]

    # Find indices of 
    arghigh = np.searchsorted(b,a)
    arglow = np.maximum(arghigh-1,0)
    arghigh = np.minimum(arghigh,len(b)-1)
    
    # Look at deltas and choose closest
    delta_high = np.abs(b[arghigh]-a)
    delta_low = np.abs(b[arglow]-a)
    closest_arg = np.where(delta_high>delta_low,arglow,arghigh)
        
    return isort[closest_arg].reshape(a_shape)


def fit_bootstrap(pinit, datax, datay, function, yerr_systematic=0.0, nrand=1000, return_more=False):
    """Bootstrap fitting routine
    
    Bootstrap fitting algorithm to determine the uncertainties on the fit parameters.
    
    Parameters
    ----------
    pinit : ndarray
        Initial guess for parameters to fit
    datax, datay : ndarray
        X and Y values of data to be fit
    function : func
        Model function 
    yerr_systematic : float or array_like of floats
        Systematic uncertainites contributing to additional error in data. 
        This is treated as independent Normal error on each data point.
        Can have unique values for each data point. If 0, then we just use
        the standard deviation of the residuals to randomize the data.
    nrand : int
        Number of random data sets to generate and fit.
    return_more : bool
        If true, then also return the full set of fit parameters for the randomized
        data to perform a more thorough analysis of the distribution. Otherewise, 
        just reaturn the mean and standard deviations.
    """

    from scipy import optimize
    
    def errfunc(p, x, y):
        return function(x, p) - y

    # Fit first time
    pfit, perr = optimize.leastsq(errfunc, pinit, args=(datax, datay), full_output=0)


    # Get the stdev of the residuals
    residuals = errfunc(pfit, datax, datay)
    sigma_res = np.std(residuals)

    sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

    # Some random data sets are generated and fitted
    randomdataY = datay + np.random.normal(scale=sigma_err_total, size=(nrand, len(datay)))
    ps = []
    for i in range(nrand):

        # randomDelta = np.random.normal(0., sigma_err_total, len(datay))
        # datay_rand = datay + randomDelta
    
        datay_rand = randomdataY[i]
        randomfit, randomcov = optimize.leastsq(errfunc, pinit, args=(datax, datay_rand), full_output=0)

        ps.append(randomfit) 

    ps = np.array(ps)
    mean_pfit = np.mean(ps,axis=0)
    err_pfit = np.std(ps,axis=0)
    
    if return_more:
        return mean_pfit, err_pfit, ps
    else:
        return mean_pfit, err_pfit
    