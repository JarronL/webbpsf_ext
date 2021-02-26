"""
Small collection of robust statistical estimators based on functions from
Henry Freudenriech (Hughes STX) statistics library (called ROBLIB) that have
been incorporated into the AstroIDL User's Library.  Function included are:

  * medabsdev - median absolute deviation
  * biweightMean - biweighted mean estimator
  * mean - robust estimator of the mean of a data set
  * mode - robust estimate of the mode of a data set using the half-sample
    method
  * std - robust estimator of the standard deviation of a data set
  * checkfit - return the standard deviation and biweights for a fit in order 
    to determine its quality
  * linefit - outlier resistant fit of a line to data
  * polyfit - outlier resistant fit of a polynomial to data

For the fitting routines, the coefficients are returned in the same order as
np.polyfit, i.e., with the coefficient of the highest power listed first.

For additional information about the original IDL routines, see:
  http://idlastro.gsfc.nasa.gov/contents.html#C17
"""

from __future__ import division, print_function#, unicode_literals

import numpy as np
from numpy import median

import logging
_log = logging.getLogger('pynrc')

__version__ = '0.4'
__revision__ = '$Rev$'
__all__ = ['medabsdev','biweightMean', 'mean', 'mode', 'std', \
           'checkfit', 'linefit', 'polyfit', \
           '__version__', '__revision__', '__all__']


# Numerical precision
__epsilon = np.finfo(float).eps
__delta = 5.0e-7

def medabsdev(data, axis=None, keepdims=False, nan=True):
    """Median Absolute Deviation
    
    A "robust" version of standard deviation. Runtime is the 
    same as `astropy.stats.funcs.mad_std`.
    
    Parameters
    ----------
    data : ndarray
        The input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the deviation is computed. The
        default is to compute the deviation of the flattened array.
        
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
        This is the equivalent of reshaping the input data and then taking
        the standard devation.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.
    nan : bool, optional
        Ignore NaNs? Default is True.
    """
    medfunc = np.nanmedian if nan else np.median
    meanfunc = np.nanmean if nan else np.mean

    if (axis is None) and (keepdims==False):
        data = data.ravel()
    
    # Scale factor to return result equivalent to standard deviation.
    sig_scale = 0.6744897501960817
    
    med = medfunc(data, axis=axis, keepdims=True)
    absdiff = np.abs(data - med)
    sigma = medfunc(absdiff, axis=axis, keepdims=True)  / sig_scale
    
    # Check if anything is near 0.0 (below machine precision)
    mask = sigma < __epsilon
    if np.any(mask):
        sigma[mask] = (meanfunc(absdiff, axis=axis, keepdims=True))[mask] / 0.8
    mask = sigma < __epsilon
    if np.any(mask):
        sigma[mask] = 0.0
        
        
    if len(sigma)==1:
        return sigma[0]
    elif not keepdims:
        return np.squeeze(sigma)
    else:
        return sigma



def biweightMean(inputData, axis=None, dtype=None, iterMax=25):
    """Biweight Mean
    
    Calculate the mean of a data set using bisquare weighting.  

    Based on the biweight_mean routine from the AstroIDL User's 
    Library.

    .. versionchanged:: 1.0.3
        Added the 'axis' and 'dtype' keywords to make this function more
        compatible with np.mean()
    """

    if axis is not None:
        fnc = lambda x: biweightMean(x, dtype=dtype)
        y0 = np.apply_along_axis(fnc, axis, inputData)
    else:
        y = inputData.ravel()
        if type(y).__name__ == "MaskedArray":
            y = y.compressed()
        if dtype is not None:
            y = y.astype(dtype)
        
        n = len(y)
        closeEnough = 0.03*np.sqrt(0.5/(n-1))
    
        diff = 1.0e30
        nIter = 0
    
        y0 = np.median(y)
        deviation = y - y0
        sigma = std(deviation)
    
        if sigma < __epsilon:
            diff = 0
        while diff > closeEnough:
            nIter = nIter + 1
            if nIter > iterMax:
                break
            uu = ((y-y0)/(6.0*sigma))**2.0
            uu = np.where(uu > 1.0, 1.0, uu)
            weights = (1.0-uu)**2.0
            weights /= weights.sum()
            y0 = (weights*y).sum()
            deviation = y - y0
            prevSigma = sigma
            sigma = std(deviation, Zero=True)
            if sigma > __epsilon:
                diff = np.abs(prevSigma - sigma) / prevSigma
            else:
                diff = 0.0
            
    return y0


def mean(inputData, Cut=3.0, axis=None, dtype=None, keepdims=False, 
    return_std=False, return_mask=False):
    """Robust Mean
    
    Robust estimator of the mean of a data set. Based on the `resistant_mean` 
    function from the AstroIDL User's Library. NaN values are excluded.

    This function trims away outliers using the median and the median 
    absolute deviation. An approximation formula is used to correct for
    the truncation caused by trimming away outliers.

    Parameters
    ==========
    inputData : ndarray
        The input data.

    Keyword Args
    ============
    Cut : float
        Sigma for rejection; default is 3.0.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the deviation is computed. The
        default is to compute the deviation of the flattened array.
        
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
        This is the equivalent of reshaping the input data and then taking
        the standard devation.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.
    return_std : bool
        Also return the std dev calculated using only the "good" data?
    return_mask : bool
        If set to True, then return only boolean array of good (1) and 
        rejected (0) values.

    """

    inputData = np.array(inputData)
    
    if np.isnan(inputData).sum() > 0:
        medfunc = np.nanmedian
        meanfunc = np.nanmean
    else:
        medfunc = np.median
        meanfunc = np.mean

    if axis is None:
        data = inputData.ravel()
    else:
        data = inputData
        
    if type(data).__name__ == "MaskedArray":
        data = data.compressed()
    if dtype is not None:
        data = data.astype(dtype)

    # Scale factor to return result equivalent to standard deviation.
    sig_scale = 0.6744897501960817
        
    # Calculate the median absolute deviation
    data0 = medfunc(data, axis=axis, keepdims=True)
    absdiff = np.abs(data-data0)
    medAbsDev = medfunc(absdiff, axis=axis, keepdims=True) / sig_scale

    mask = medAbsDev < __epsilon
    if np.any(mask):
        medAbsDev[mask] = (meanfunc(absdiff, axis=axis, keepdims=True))[mask] / 0.8

    # First cut using the median absolute deviation
    cutOff = Cut*medAbsDev
    good = absdiff <= cutOff
    data_naned = data.copy()
    data_naned[~good] = np.nan
    dataMean = np.nanmean(data_naned, axis=axis, keepdims=True)
    dataSigma = np.nanstd(data_naned, axis=axis, keepdims=True)
    #dataSigma = np.sqrt( np.nansum((data_naned-dataMean)**2.0) / len(good) )

    # Calculate sigma
    if Cut > 1.0:
        sigmaCut = Cut
    else:
        sigmaCut = 1.0
    if sigmaCut <= 4.5:
        poly_sigcut = -0.15405 + 0.90723*sigmaCut - 0.23584*sigmaCut**2.0 + 0.020142*sigmaCut**3.0
        dataSigma = dataSigma / poly_sigcut

    cutOff = Cut*dataSigma
    good = absdiff <= cutOff

    if return_mask:
        return np.reshape(~np.isnan(data_naned), inputData.shape)

    data_naned = data.copy()
    data_naned[~good] = np.nan
    dataMean = np.nanmean(data_naned, axis=axis, keepdims=True)
    if return_std:
        dataSigma = np.nanstd(data_naned, axis=axis, keepdims=True)
    
    if len(dataMean)==1:
        if return_std:
            return dataMean[0], dataSigma[0]
        else:
            return dataMean[0]
    if not keepdims:
        if return_std:
            return np.squeeze(dataMean), np.squeeze(dataSigma)
        else:
            return np.squeeze(dataMean)
    else:
        if return_std:
            return dataMean, dataSigma
        else:
            return dataMean



def _mean_old(inputData, Cut=3.0, axis=None, dtype=None):
    """Robust mean
    
    Robust estimator of the mean of a data set.  Based on the 
    resistant_mean function from the AstroIDL User's Library.

    .. versionchanged:: 1.0.3
        Added the 'axis' and 'dtype' keywords to make this function more
        compatible with np.mean()
    """
           
    inputData = np.array(inputData)
    if axis is not None:
        fnc = lambda x: _mean_old(x, dtype=dtype)
        dataMean = np.apply_along_axis(fnc, axis, inputData)
    else:
        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)
        
        data0 = np.median(data)
        absdiff = np.abs(data-data0)
        
        medAbsDev = np.median(absdiff) / 0.6745
        if medAbsDev < __epsilon:
            medAbsDev = absdiff.mean() / 0.8000
        
        cutOff = Cut*medAbsDev
        good = np.where( absdiff <= cutOff )
        good = good[0]
        dataMean = data[good].mean()
        dataSigma = np.sqrt( ((data[good]-dataMean)**2.0).sum() / len(good) )

        if Cut > 1.0:
            sigmaCut = Cut
        else:
            sigmaCut = 1.0
        if sigmaCut <= 4.5:
            dataSigma = dataSigma / (-0.15405 + 0.90723*sigmaCut - 0.23584*sigmaCut**2.0 + 0.020142*sigmaCut**3.0)
        
        cutOff = Cut*dataSigma
        good = np.where( absdiff <= cutOff )
        good = good[0]
        dataMean = data[good].mean()
        if len(good) > 3:
            dataSigma = np.sqrt( ((data[good]-dataMean)**2.0).sum() / len(good) )
        
        if Cut > 1.0:
            sigmaCut = Cut
        else:
            sigmaCut = 1.0
        if sigmaCut <= 4.5:
            dataSigma = dataSigma / (-0.15405 + 0.90723*sigmaCut - 0.23584*sigmaCut**2.0 + 0.020142*sigmaCut**3.0)
        
        dataSigma = dataSigma / np.sqrt(len(good)-1)
    
    return dataMean


def mode(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """

    if axis is not None:
        fnc = lambda x: mode(x, dtype=dtype)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:
                wMin = data[-1] - data[0]
                N = int(data.size/2 + data.size%2)
                for i in range(0, N):
                    w = data[i+N-1] - data[i] 
                    if w < wMin:
                        wMin = w
                        j = i
                return _hsm(data[j:j+N])
            
        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)
        
        # The data need to be sorted for this to work
        data = np.sort(data)
    
        # Find the mode
        dataMode = _hsm(data)
    
    return dataMode

def std(inputData, Zero=False, axis=None, dtype=None, keepdims=False, return_mask=False):
    """Robust Sigma
    
    Based on the robust_sigma function from the AstroIDL User's Library.

    Calculate a resistant estimate of the dispersion of a distribution.
    
    Use the median absolute deviation as the initial estimate, then weight 
    points using Tukey's Biweight. See, for example, "Understanding Robust
    and Exploratory Data Analysis," by Hoaglin, Mosteller and Tukey, John
    Wiley & Sons, 1983, or equation 9 in Beers et al. (1990, AJ, 100, 32).

    Parameters
    ==========
    inputData : ndarray
        The input data.

    Keyword Args
    ============
    axis : None or int or tuple of ints, optional
        Axis or axes along which the deviation is computed. The
        default is to compute the deviation of the flattened array.
        
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
        This is the equivalent of reshaping the input data and then taking
        the standard devation.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.
    return_mask : bool
        If set to True, then only return boolean array of good (1) and 
        rejected (0) values.

	"""

    inputData = np.array(inputData)
    
    if np.isnan(inputData).sum() > 0:
        medfunc = np.nanmedian
        meanfunc = np.nanmean
    else:
        medfunc = np.median
        meanfunc = np.mean

    if axis is None:
        data = inputData.ravel()
    else:
        data = inputData

    if type(data).__name__ == "MaskedArray":
        data = data.compressed()
    if dtype is not None:
        data = data.astype(dtype)

    # Scale factor to return result equivalent to standard deviation.
    sig_scale = 0.6744897501960817

    # Calculate the median absolute deviation
    if Zero:
        data0 = 0.0
    else:
        data0 = medfunc(data, axis=axis, keepdims=True)
    absdiff = np.abs(data-data0)
    medAbsDev = medfunc(absdiff, axis=axis, keepdims=True) / sig_scale
    mask = medAbsDev < __epsilon
    if np.any(mask):
        medAbsDev[mask] = (meanfunc(absdiff, axis=axis, keepdims=True))[mask] / 0.8
        
    # These will be set to 0 later
    mask0 = medAbsDev < __epsilon
        
    u = (data-data0) / (6.0 * medAbsDev)
    u2 = u**2.0
    good = u2 <= 1.0

    if return_mask:
        return good & ~np.isnan(data)
    
    # These values will be set to NaN later
    # if fewer than 3 good points to calculate stdev
    ngood = good.sum(axis=axis, keepdims=True)
    mask_nan = ngood < 2
    if mask_nan.sum() > 0:
        print("WARNING: NaN's will be present due to weird distributions")
    
    # Set bad points to NaNs
    u2[~good] = np.nan
    
    numerator = np.nansum( (data - data0)**2 * (1.0 - u2)**4.0, axis=axis, keepdims=True)
    nElements = len(data) if axis is None else data.shape[axis]
    denominator = np.nansum( (1.0 - u2) * (1.0 - 5.0*u2), axis=axis, keepdims=True)
    sigma = np.sqrt( nElements*numerator / (denominator*(denominator-1.0)) )
    
    sigma[mask0] = 0
    sigma[mask_nan] = np.nan

    if len(sigma)==1:
        return sigma[0]
    elif not keepdims:
        return np.squeeze(sigma)
    else:
        return sigma


def _std_old(inputData, Zero=False, axis=None, dtype=None):
    """
    Robust estimator of the standard deviation of a data set.  

    Based on the robust_sigma function from the AstroIDL User's Library.

    .. versionchanged:: 1.0.3
        Added the 'axis' and 'dtype' keywords to make this function more
        compatible with np.std()
    """

    if axis is not None:
        fnc = lambda x: _std_old(x, dtype=dtype)
        sigma = np.apply_along_axis(fnc, axis, inputData)
    else:
        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)
        
        if Zero:
            data0 = 0.0
        else:
            data0 = np.median(data)
        medAbsDev = np.median(np.abs(data-data0)) / 0.6744897501960817
        if medAbsDev < __epsilon:
            medAbsDev = np.mean(np.abs(data-data0)) / 0.8000
        if medAbsDev < __epsilon:
            sigma = 0.0
            return sigma
        
        u = (data-data0) / 6.0 / medAbsDev
        u2 = u**2.0
        good = np.where( u2 <= 1.0 )
        good = good[0]
        if len(good) < 3:
            print("WARNING:  Distribution is too strange to compute standard deviation")
            sigma = -1.0
            return sigma
        
        numerator = ((data[good]-data0)**2.0 * (1.0-u2[good])**2.0).sum()
        nElements = (data.ravel()).shape[0]
        denominator = ((1.0-u2[good])*(1.0-5.0*u2[good])).sum()
        sigma = nElements*numerator / (denominator*(denominator-1.0))
        if sigma > 0:
            sigma = np.sqrt(sigma)
        else:
            sigma = 0.0
        
    return sigma


def checkfit(inputData, inputFit, epsilon, delta, BisquareLimit=6.0):
    """
    Determine the quality of a fit and biweights.  Returns a tuple
    with elements:
    
      0. Robust standard deviation analog
      1. Fractional median absolute deviation of the residuals
      2. Number of input points given non-zero weight in the calculation
      3. Bisquare weights of the input points
      4. Residual values scaled by sigma

    This function is based on the rob_checkfit routine from the AstroIDL 
    User's Library.
    """

    data = inputData.ravel()
    fit = inputFit.ravel()
    if type(data).__name__ == "MaskedArray":
        data = data.compressed()
    if type(fit).__name__ == "MaskedArray":
        fit = fit.compressed()

    deviation = data - fit
    sigma = std(deviation, Zero=True)
    if sigma < epsilon:
        return (sigma, 0.0, 0, 0.0, 0.0)

    toUse = (np.where( np.abs(fit) > epsilon ))[0]
    if len(toUse) < 3:
        fracDev = 0.0
    else:
        fracDev = np.median(np.abs(deviation[toUse]/fit[toUse]))
    if fracDev < delta:
        return (sigma, fracDev, 0, 0.0, 0.0)
    
    biweights = np.abs(deviation)/(BisquareLimit*sigma)
    toUse = (np.where(biweights > 1))[0]
    if len(toUse) > 0:
        biweights[toUse] = 1.0
    nGood = len(data) - len(toUse)

    scaledResids = (1.0 - biweights**2.0)
    scaledResids = scaledResids / scaledResids.sum()

    return (sigma, fracDev, nGood, biweights, scaledResids)


def linefit(inputX, inputY, iterMax=25, Bisector=False, BisquareLimit=6.0, CloseFactor=0.03):
    """
    Outlier resistance two-variable linear regression function.

    Based on the robust_linefit routine in the AstroIDL User's Library.
    """

    xIn = inputX.ravel()
    yIn = inputY.ravel()
    if type(yIn).__name__ == "MaskedArray":
        xIn = xIn.compress(np.logical_not(yIn.mask))
        yIn = yIn.compressed()
    n = len(xIn)

    np.logical_not(yIn.mask)

    x0 = xIn.sum() / n
    y0 = yIn.sum() / n
    x = xIn - x0
    y = yIn - y0

    cc = np.zeros(2)
    ss = np.zeros(2)
    sigma = 0.0
    yFit = yIn
    badFit = 0
    nGood = n

    lsq = 0.0
    yp = y
    if n > 5:
        s = np.argsort(x)
        u = x[s]
        v = y[s]
        nHalf = n//2 -1
        x1 = np.median(u[0:nHalf])
        x2 = np.median(u[nHalf:])
        y1 = np.median(v[0:nHalf])
        y2 = np.median(v[nHalf:])
        if np.abs(x2-x1) < __epsilon:
            x1 = u[0]
            x2 = u[-1]
            y1 = v[0]
            y2 = v[-1]
        cc[1] = (y2-y1)/(x2-x1)
        cc[0] = y1 - cc[1]*x1
        yFit = cc[0] + cc[1]*x
        sigma, fracDev, nGood, biweights, scaledResids = checkfit(yp, yFit, __epsilon, __delta)
        if nGood < 2:
            lsq = 1.0
        
    if lsq == 1 or n < 6:
        sx = x.sum()
        sy = y.sum()
        sxy = (x*y).sum()
        sxx = (x*x).sum()
        d = sxx - sx*sx
        if np.abs(d) < __epsilon:
            return (0.0, 0.0)
        ySlope = (sxy - sx*sy) / d
        yYInt = (sxx*sy - sx*sxy) / d
    
        if Bisector:
            syy = (y*y).sum()
            d = syy - sy*sy
            if np.abs(d) < __epsilon:
                return (0.0, 0.0)
            tSlope = (sxy - sy*sx) / d
            tYInt = (syy*sx - sy*sxy) / d
            if np.abs(tSlope) < __epsilon:
                return (0.0, 0.0)
            xSlope = 1.0/tSlope
            xYInt = -tYInt / tSlope
            if ySlope > xSlope:
                a1 = yYInt
                b1 = ySlope
                r1 = np.sqrt(1.0+ySlope**2.0)
                a2 = xYInt
                b2 = xSlope
                r2 = np.sqrt(1.0+xSlope**2.0)
            else:
                a2 = yYInt
                b2 = ySlope
                r2 = np.sqrt(1.0+ySlope**2.0)
                a1 = xYInt
                b1 = xSlope
                r1 = np.sqrt(1.0+xSlope**2.0)
            yInt = (r1*a2 + r2*a1) / (r1 + r2)
            slope = (r1*b2 + r2*b1) / (r1 + r2)
            r = np.sqrt(1.0+slope**2.0)
            if yInt > 0:
                r = -r
            u1 = slope / r
            u2 = -1.0/r
            u3 = yInt / r
            yp = u1*x + u2*y + u3
            yFit = y*0.0
            ss = yp
        else:
            slope = ySlope
            yInt = yYInt
            yFit = yInt + slope*x
        cc[0] = yInt
        cc[1] = slope
        sigma, fracDev, nGood, biweights, scaledResids = checkfit(yp, yFit, __epsilon, __delta)
    
    if nGood < 2:
        cc[0] = cc[0] + y0 - cc[1]*x0
        return cc[::-1]
    
    sigma1 = (100.0*sigma)
    closeEnough = CloseFactor * np.sqrt(0.5/(n-1))
    if closeEnough < __delta:
        closeEnough = __delta
    diff = 1.0e20
    nIter = 0
    while diff > closeEnough:
        nIter = nIter + 1
        if nIter > iterMax:
            break
        sigma2 = sigma1
        sigma1 = sigma
        sx = (biweights*x).sum()
        sy = (biweights*y).sum()
        sxy = (biweights*x*y).sum()
        sxx = (biweights*x*x).sum()
        d = sxx - sx*sx
        if np.abs(d) < __epsilon:
            return (0.0, 0.0)
        ySlope = (sxy - sx*sy) / d
        yYInt = (sxx*sy - sx*sxy) / d
        slope = ySlope
        yInt = yYInt
    
        if Bisector:
            syy = (biweights*y*y).sum()
            d = syy - sy*sy
            if np.abs(d) < __epsilon:
                return (0.0, 0.0)
            tSlope = (sxy - sy*sx) / d
            tYInt = (syy*sx - sy*sxy) / d
            if np.abs(tSlope) < __epsilon:
                return (0.0, 0.0)
            xSlope = 1.0/tSlope
            xYInt = -tYInt / tSlope
            if ySlope > xSlope:
                a1 = yYInt
                b1 = ySlope
                r1 = np.sqrt(1.0+ySlope**2.0)
                a2 = xYInt
                b2 = xSlope
                r2 = np.sqrt(1.0+xSlope**2.0)
            else:
                a2 = yYInt
                b2 = ySlope
                r2 = np.sqrt(1.0+ySlope**2.0)
                a1 = xYInt
                b1 = xSlope
                r1 = np.sqrt(1.0+xSlope**2.0)
            yInt = (r1*a2 + r2*a1) / (r1 + r2)
            slope = (r1*b2 + r2*b1) / (r1 + r2)
            r = np.sqrt(1.0+slope**2.0)
            if yInt > 0:
                r = -r
            u1 = slope / r
            u2 = -1.0/r
            u3 = yInt / r
            yp = u1*x + u2*y + u3
            yFit = y*0.0
            ss = yp
        else:
            yFit = yInt + slope*x
        cc[0] = yInt
        cc[1] = slope
        sigma, fracDev, nGood, biweights, scaledResids = checkfit(yp, yFit, __epsilon, __delta)
    
        if nGood < 2:
            badFit = 1
            break
        diff1 = np.abs(sigma1 - sigma)/sigma
        diff2 = np.abs(sigma2 - sigma)/sigma
        if diff1 < diff2:
            diff = diff1
        else:
            diff = diff2
        
    cc[0] = cc[0] + y0 - cc[1]*x0
    return cc[::-1]


def polyfit(inputX, inputY, order, iterMax=25):
    """
    Outlier resistance two-variable polynomial function fitter.

    Based on the robust_poly_fit routine in the AstroIDL User's 
    Library.

    Unlike robust_poly_fit, two different polynomial fitters are used
    because np.polyfit does not support non-uniform weighting of the
    data.  For the weighted fitting, the SciPy Orthogonal Distance
    Regression module (scipy.odr) is used.
    """

    from scipy import odr

    def polyFunc(B, x, order=order):
        out = x*0.0
        for i in range(order+1):
            out = out + B[i]*x**i

    model = odr.Model(polyFunc)

    x = inputX.ravel()
    y = inputY.ravel()
    if type(y).__name__ == "MaskedArray":
        x = x.compress(np.logical_not(y.mask))
        y = y.compressed()
    n = len(x)

    x0 = x.sum() / n
    y0 = y.sum() / n
    u = x
    v = y

    nSeg = int(order + 2)
    if (nSeg/2)*2 == nSeg:
        nSeg = nSeg + 1
    minPts = nSeg*3
    if n < 1000:
        lsqFit = 1
        cc = np.polyfit(u, v, order)
        yFit = np.polyval(cc, u)
    else:
        lsqfit = 0
        q = np.argsort(u)
        u = u[q]
        v = v[q]
        nPerSeg = np.zeros(nSeg, dtype='int') + n//nSeg
        nLeft = n - nPerSeg[0]*nSeg
        nPerSeg[nSeg//2] = nPerSeg[nSeg//2] + nLeft
        r = np.zeros(nSeg)
        s = np.zeros(nSeg)
        print(nPerSeg)
        r[0] = np.median(u[0:nPerSeg[0]])
        s[0] = np.median(v[0:nPerSeg[0]])
        i2 = nPerSeg[0]-1
        for i in range(1,nSeg):
            i1 = i2
            i2 = i1 + nPerSeg[i]
            r[i] = np.median(u[i1:i2])
            s[i] = np.median(v[i1:i2])
        cc = np.polyfit(r, s, order)
        yFit = np.polyval(cc, u)
    
    sigma, fracDev, nGood, biweights, scaledResids = checkfit(v, yFit, __epsilon, __delta)
    if nGood == 0:
        return cc
    if nGood < minPts:
        if lsqFit == 0:
            cc = np.polyfit(u, v, order)
            yFit = np.polyval(cc, u)
            sigma, fracDev, nGood, biweights, scaledResids = checkfit(yp, yFit, __epsilon, __delta)
            if nGood == 0:
                return __processPoly(x0, y0, order, cc)
            nGood = n - nGood
        if nGood < minPts:
            return 0
        
    closeEnough = 0.03*np.sqrt(0.5/(n-1))
    if closeEnough < __delta:
        closeEnough = __delta
    diff = 1.0e10
    sigma1 = 100.0*sigma
    nIter = 0
    while diff > closeEnough:
        nIter = nIter + 1
        if nIter > iterMax:
            break
        sigma2 = sigma1
        sigma1 = sigma
        g = (np.where(biweights > 0))[0]
        if len(g) < len(biweights):
            u = u[g]
            v = v[g]
            biweights = biweights[g]
        try:
            ## Try the fancy method...
            data = odr.RealData(u, v, sy=1.0/biweights)
            fit = odr.ODR(data, model, beta0=cc[::-1])
            out = fit.run()
            cc = out.beta[::-1]
        except:
            ## And then give up when it doesn't work
            cc = np.polyfit(u, v, order)
        yFit = np.polyval(cc, u)
        sigma, fracDev, nGood, biweights, scaledResids = checkfit(v, yFit, __epsilon, __delta)
        if nGood < minPts:
            return cc
        diff1 = np.abs(sigma1 - sigma)/sigma
        diff2 = np.abs(sigma2 - sigma)/sigma
        if diff1 < diff2:
            diff = diff1
        else:
            diff = diff2
    return cc
