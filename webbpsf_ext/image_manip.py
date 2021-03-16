import numpy as np

from scipy.ndimage import fourier_shift
from scipy.ndimage.interpolation import rotate

from poppy.utils import krebin

###########################################################################
#    Image manipulation
###########################################################################

def pad_or_cut_to_size(array, new_shape, fill_val=0.0, offset_vals=None):
    """
    Resize an array to a new shape by either padding with zeros
    or trimming off rows and/or columns. The ouput shape can
    be of any arbitrary amount.

    Parameters
    ----------
    array : ndarray
        A 1D, 2D, or 3D array. If 3D, then taken to be a stack of images
        that are cropped or expanded in the same fashion.
    new_shape : tuple
        Desired size for the output array. For 2D case, if a single value, 
        then will create a 2-element tuple of the same value.
    fill_val : scalar, optional
        Value to pad borders. Default is 0.0
    offset_vals : tuple
        Option to perform image shift in the (xpix) direction for 1D, 
        or (ypix,xpix) direction for 2D/3D.

    Returns
    -------
    output : ndarray
        An array of size new_shape that preserves the central information 
        of the input array.
    """
    
    ndim = len(array.shape)
    if ndim == 1:
        # is_1d = True
        # Reshape array to a 2D array with nx=1
        array = array.reshape((1,1,-1))
        nz, ny, nx = array.shape
        if isinstance(new_shape, (float,int,np.int,np.int64)):
            nx_new = int(new_shape+0.5)
            ny_new = 1
            new_shape = (ny_new, nx_new)
        elif len(new_shape) < 2:
            nx_new = new_shape[0]
            ny_new = 1
            new_shape = (ny_new, nx_new)
        else:
            ny_new, nx_new = new_shape
        output = np.zeros(shape=(nz,ny_new,nx_new), dtype=array.dtype)
    elif (ndim == 2) or (ndim == 3):
        if ndim==2:
            nz = 1
            ny, nx = array.shape
            array = array.reshape([nz,ny,nx])
        else:
            nz, ny, nx = array.shape

        if isinstance(new_shape, (float,int,np.int,np.int64)):
            ny_new = nx_new = int(new_shape+0.5)
            new_shape = (ny_new, nx_new)
        elif len(new_shape) < 2:
            ny_new = nx_new = new_shape[0]
            new_shape = (ny_new, nx_new)
        else:
            ny_new, nx_new = new_shape
        output = np.zeros(shape=(nz,ny_new,nx_new), dtype=array.dtype)
    else:
        raise ValueError('Input image can only have 1 or 2 or 3 dimensions. \
                          Found {} dimensions.'.format(ndim))
                      
    # Return if no difference in shapes
    # This needs to occur after the above so that new_shape is verified to be a tuple
    # If offset_vals is set, then continue to perform shift function
    if (array.shape == new_shape) and (offset_vals is None):
        return array

    # Input the fill values
    if fill_val != 0:
        output += fill_val
        
    # Pixel shift values
    if offset_vals is not None:
        if ndim == 1:
            ny_off = 0
            if isinstance(offset_vals, (float,int,np.int,np.int64)):
                nx_off = offset_vals
            elif len(offset_vals) < 2:
                nx_off = offset_vals[0]
            else:
                raise ValueError('offset_vals should be a single value.')
        else:
            if len(offset_vals) == 2:
                ny_off, nx_off = offset_vals
            else:
                raise ValueError('offset_vals should have two values.')
    else:
        nx_off = ny_off = 0
                
    if nx_new>nx:
        n0 = (nx_new - nx) / 2
        n1 = n0 + nx
    elif nx>nx_new:
        n0 = (nx - nx_new) / 2
        n1 = n0 + nx_new
    else:
        n0, n1 = (0, nx)
    n0 = int(n0+0.5)
    n1 = int(n1+0.5)

    if ny_new>ny:
        m0 = (ny_new - ny) / 2
        m1 = m0 + ny
    elif ny>ny_new:
        m0 = (ny - ny_new) / 2
        m1 = m0 + ny_new
    else:
        m0, m1 = (0, ny)
    m0 = int(m0+0.5)
    m1 = int(m1+0.5)

    if (nx_new>=nx) and (ny_new>=ny):
        #print('Case 1')
        output[:,m0:m1,n0:n1] = array.copy()
        for i, im in enumerate(output):
            output[i] = fshift(im, delx=nx_off, dely=ny_off, pad=True, cval=fill_val)
    elif (nx_new<=nx) and (ny_new<=ny):
        #print('Case 2')
        if (nx_off!=0) or (ny_off!=0):
            array_temp = array.copy()
            for i, im in enumerate(array_temp):
                array_temp[i] = fshift(im, delx=nx_off, dely=ny_off, pad=True, cval=fill_val)
            output = array_temp[:,m0:m1,n0:n1]
        else:
            output = array[:,m0:m1,n0:n1]
    elif (nx_new<=nx) and (ny_new>=ny):
        #print('Case 3')
        if nx_off!=0:
            array_temp = array.copy()
            for i, im in enumerate(array_temp):
                array_temp[i] = fshift(im, delx=nx_off, pad=True, cval=fill_val)
            output[:,m0:m1,:] = array_temp[:,:,n0:n1]
        else:
            output[:,m0:m1,:] = array[:,:,n0:n1]
        for i, im in enumerate(output):
            output[i] = fshift(im, dely=ny_off, pad=True, cval=fill_val)
    elif (nx_new>=nx) and (ny_new<=ny):
        #print('Case 4')
        if ny_off!=0:
            array_temp = array.copy()
            for i, im in enumerate(array_temp):
                array_temp[i] = fshift(im, dely=ny_off, pad=True, cval=fill_val)
            output[:,:,n0:n1] = array_temp[:,m0:m1,:]
        else:
            output[:,:,n0:n1] = array[:,m0:m1,:]
        for i, im in enumerate(output):
            output[i] = fshift(im, delx=nx_off, pad=True, cval=fill_val)
        
    # Flatten if input and output arrays are 1D
    if (ndim==1) and (ny_new==1):
        output = output.flatten()
    elif ndim==2:
        output = output[0]

    return output

def fshift(image, delx=0, dely=0, pad=False, cval=0.0):
    """ Fractional image shift
    
    Ported from IDL function fshift.pro.
    Routine to shift an image by non-integer values.

    Parameters
    ----------
    image: ndarray
        1D or 2D array to be shifted
    delx : float
        shift in x (same direction as IDL SHIFT function)
    dely: float
        shift in y
    pad : bool
        Should we pad the array before shifting, then truncate?
        Otherwise, the image is wrapped.
    cval : sequence or float, optional
        The values to set the padded values for each axis. Default is 0.
        ((before_1, after_1), ... (before_N, after_N)) unique pad constants for each axis.
        ((before, after),) yields same before and after constants for each axis.
        (constant,) or int is a shortcut for before = after = constant for all axes.

        
    Returns
    -------
    ndarray
        Shifted image
    """
    
    
    if len(image.shape) == 1:
        # Return if delx is 0
        if np.isclose(delx, 0, atol=1e-5):
            return image

        # separate shift into an integer and fraction shift
        intx = np.int(delx)
        fracx = delx - intx
        if fracx < 0:
            fracx += 1
            intx -= 1

        # Pad ends with constant value
        if pad:
            padx = np.abs(intx) + 1
            out = np.pad(image,np.abs(intx),'constant',constant_values=cval)
        else:
            padx = 0
            out = image.copy()

        # shift by integer portion
        out = np.roll(out, intx)
        # if significant fractional shift...
        if not np.isclose(fracx, 0, atol=1e-5):
            out = out * (1.-fracx) + np.roll(out,1) * fracx

        out = out[padx:padx+image.size]
        return out

    elif len(image.shape) == 2:	
        # Return if both delx and dely are 0
        if np.isclose(delx, 0, atol=1e-5) and np.isclose(dely, 0, atol=1e-5):
            return image

        # separate shift into an integer and fraction shift
        intx = np.int(delx)
        inty = np.int(dely)
        fracx = delx - intx
        fracy = dely - inty
        if fracx < 0:
            fracx += 1
            intx -= 1
        if fracy < 0:
            fracy += 1
            inty -= 1

        # Pad ends with constant value
        if pad:
            padx = np.abs(intx) + 1
            pady = np.abs(inty) + 1
            pad_vals = ([pady]*2,[padx]*2)
            out = np.pad(image,pad_vals,'constant',constant_values=cval)
        else:
            padx = 0; pady = 0
            out = image.copy()

        # shift by integer portion
        out = np.roll(np.roll(out, intx, axis=1), inty, axis=0)
    
        # Check if fracx and fracy are effectively 0
        fxis0 = np.isclose(fracx,0, atol=1e-5)
        fyis0 = np.isclose(fracy,0, atol=1e-5)
        # If fractional shifts are significant
        # use bi-linear interpolation between four pixels
        if not (fxis0 and fyis0):
            # Break bi-linear interpolation into four parts
            # to avoid NaNs unnecessarily affecting integer shifted dimensions
            part1 = out * ((1-fracx)*(1-fracy))
            part2 = 0 if fyis0 else np.roll(out,1,axis=0)*((1-fracx)*fracy)
            part3 = 0 if fxis0 else np.roll(out,1,axis=1)*((1-fracy)*fracx)
            part4 = 0 if (fxis0 or fyis0) else np.roll(np.roll(out, 1, axis=1), 1, axis=0) * fracx*fracy
    
            out = part1 + part2 + part3 + part4
    
        out = out[pady:pady+image.shape[0], padx:padx+image.shape[1]]
        return out
            

        #if not np.allclose([fracx,fracy], 0, atol=1e-5):
        #	x = x * ((1-fracx)*(1-fracy)) + \
        #		np.roll(x,1,axis=0) * ((1-fracx)*fracy) + \
        #		np.roll(x,1,axis=1) * (fracx*(1-fracy)) + \
        #		np.roll(np.roll(x, 1, axis=1), 1, axis=0) * fracx*fracy

        #x = x[pady:pady+image.shape[0], padx:padx+image.shape[1]]
        #return x

    else:
        ndim = len(image.shape)
        raise ValueError(f'Input image can only have 1 or 2 dimensions. Found {ndim} dimensions.')

                          
def fourier_imshift(image, xshift, yshift, pad=False, cval=0.0):
    """Fourier shift image
    
    Shift an image by use of Fourier shift theorem

    Parameters
    ----------
    image : nd array
        N x K image
    xshift : float
        Number of pixels to shift image in the x direction
    yshift : float
        Number of pixels to shift image in the y direction
    pad : bool
        Should we pad the array before shifting, then truncate?
        Otherwise, the image is wrapped.
    cval : sequence or float, optional
        The values to set the padded values for each axis. Default is 0.
        ((before_1, after_1), ... (before_N, after_N)) unique pad constants for each axis.
        ((before, after),) yields same before and after constants for each axis.
        (constant,) or int is a shortcut for before = after = constant for all axes.

    Returns
    -------
    ndarray
        Shifted image
    """
    
    # Pad ends with zeros
    if pad:
        padx = np.abs(np.int(xshift)) + 1
        pady = np.abs(np.int(yshift)) + 1
        pad_vals = ([pady]*2,[padx]*2)
        im = np.pad(image,pad_vals,'constant',constant_values=cval)
    else:
        padx = 0; pady = 0
        im = image
    
    offset = fourier_shift( np.fft.fft2(im), (yshift,xshift) )
    offset = np.fft.ifft2(offset).real
    
    offset = offset[pady:pady+image.shape[0], padx:padx+image.shape[1]]
    
    return offset
    


def rotate_offset(data, angle, cen=None, cval=0.0, order=1, 
    reshape=True, recenter=True, **kwargs):
    """Rotate and offset an array.

    Same as `rotate` in `scipy.ndimage.interpolation` except that it
    rotates around a center point given by `cen` keyword.
    The array is rotated in the plane defined by the two axes given by the
    `axes` parameter using spline interpolation of the requested order.
    
    Parameters
    ----------
    data : ndarray
        The input array.
    angle : float
        The rotation angle in degrees (rotates in CW direction).
    cen : tuple
        Center location around which to rotate image.
        Values are expected to be `(xcen, ycen)`.
    recenter : bool
        Do we want to reposition so that `cen` is the image center?
        
    Keyword Args
    ------------
    axes : tuple of 2 ints, optional
        The two axes that define the plane of rotation. Default is the first
        two axes.
    reshape : bool, optional
        If `reshape` is True, the output shape is adapted so that the input
        array is contained completely in the output. Default is True.
    order : int, optional
        The order of the spline interpolation, default is 1.
        The order has to be in the range 0-5.
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').
        Default is 'constant'.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0
    prefilter : bool, optional
        The parameter prefilter determines if the input is pre-filtered with
        `spline_filter` before interpolation (necessary for spline
        interpolation of order > 1).  If False, it is assumed that the input is
        already filtered. Default is True.

    Returns
    -------
    rotate : ndarray or None
        The rotated data.

    """

    # Return input data if angle is set to None 0
    if (angle is None) or (angle==0):
        return data

    ndim = len(data.shape)
    if ndim==2:
        ny, nx = data.shape
        nz = 1
    elif ndim==3:
        nz, ny, nx = data.shape
    else:
        raise ValueError('Input image can only have 2 or 3 dimensions. \
                          Found {} dimensions.'.format(ndim))

    if 'axes' not in kwargs.keys():
        kwargs['axes'] = (2,1)
    kwargs['order'] = order
    kwargs['cval'] = cval

    xcen, ycen = (nx/2, ny/2)
    if cen is None:
        cen = (xcen, ycen)
    xcen_new, ycen_new = cen
    delx, dely = (xcen-xcen_new, ycen-ycen_new)

    # Reshape into a 3D array if nz=1
    data = data.reshape([nz,ny,nx])
    # Return rotate function if rotating about center
    if np.allclose((delx, dely), 0, atol=1e-5):
        return rotate(data, angle, reshape=reshape, **kwargs).squeeze()

    # Pad and then shift array
    new_shape = (int(ny+2*abs(dely)), int(nx+2*abs(delx)))
    images_shift = []
    for im in data:
        im_pad = pad_or_cut_to_size(im, new_shape, fill_val=cval)
        im_new = fshift(im_pad, delx, dely, cval=cval)
        images_shift.append(im_new)
    images_shift = np.array(images_shift)
    
    # Remove additional dimension in the case of single image
    #images_shift = images_shift.squeeze()
    
    # Rotate images
    # TODO: Should reshape=True or reshape=reshape?
    images_shrot = rotate(images_shift, angle, reshape=True, **kwargs)
    
    if reshape:
        return images_shrot.squeeze()
    else:
        # Shift back to it's location
        if recenter:
            images_rot = images_shrot
        else:
            images_rot = []
            for im in images_shrot:
                im_new = fshift(im, -delx, -dely, pad=True, cval=cval)
                images_rot.append(im_new)
            images_rot = np.array(images_rot)
    
        images_fin = []
        for im in images_rot:
            im_new = pad_or_cut_to_size(im, (ny,nx))
            images_fin.append(im_new)
        images_fin = np.array(images_fin)
    
        return images_fin.squeeze()

def frebin(image, dimensions=None, scale=None, total=True):
    """Fractional rebin
    
    Python port from the IDL frebin.pro
    Shrink or expand the size of a 1D or 2D array by an arbitary amount 
    using bilinear interpolation. Conserves flux by ensuring that each 
    input pixel is equally represented in the output array.

    Parameters
    ----------
    image : ndarray
        Input image, 1-d or 2-d ndarray.
    dimensions : tuple or None
        Desired size of output array (take priority over scale).
    scale : tuple or None
        Factor to scale output array size. A scale of 2 will increase
        the number of pixels by 2 (ie., finer pixel scale).
    total : bool
        Conserves the surface flux. If True, the output pixels 
        will be the sum of pixels within the appropriate box of 
        the input image. Otherwise, they will be the average.
    
    Returns
    -------
    ndarray
        The binned ndarray
    """

    if dimensions is not None:
        if isinstance(dimensions, float):
            dimensions = [int(dimensions)] * len(image.shape)
        elif isinstance(dimensions, int):
            dimensions = [dimensions] * len(image.shape)
        elif len(dimensions) != len(image.shape):
            raise RuntimeError("The number of input dimensions don't match the image shape.")
    elif scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            dimensions = list(map(int, map(lambda x: x+0.5, map(lambda x: x*scale, image.shape))))
        elif len(scale) != len(image.shape):
            raise RuntimeError("The number of input dimensions don't match the image shape.")
        else:
            dimensions = [scale[i]*image.shape[i] for i in range(len(scale))]
    else:
        raise RuntimeError('Incorrect parameters to rebin.\n\frebin(image, dimensions=(x,y))\n\frebin(image, scale=a')
    #print(dimensions)


    shape = image.shape
    if len(shape)==1:
        nlout = 1
        nsout = dimensions[0]
        nsout = int(nsout+0.5)
        dimensions = [nsout]
    elif len(shape)==2:
        nlout, nsout = dimensions
        nlout = int(nlout+0.5)
        nsout = int(nsout+0.5)
        dimensions = [nlout, nsout]
    if len(shape) > 2:
        raise ValueError('Input image can only have 1 or 2 dimensions. Found {} dimensions.'.format(len(shape)))
    

    if nlout != 1:
        nl = shape[0]
        ns = shape[1]
    else:
        nl = nlout
        ns = shape[0]

    sbox = ns / float(nsout)
    lbox = nl / float(nlout)
    #print(sbox,lbox)

    # Contract by integer amount
    if (sbox.is_integer()) and (lbox.is_integer()):
        image = image.reshape((nl,ns))
        result = krebin(image, (nlout,nsout))
        if not total: result /= (sbox*lbox)
        if nl == 1:
            return result[0,:]
        else:
            return result

    ns1 = ns - 1
    nl1 = nl - 1

    if nl == 1:
        #1D case
        _log.debug("Rebinning to Dimension: %s" % nsout)
        result = np.zeros(nsout)
        for i in range(nsout):
            rstart = i * sbox
            istart = int(rstart)
            rstop = rstart + sbox

            if int(rstop) < ns1:
                istop = int(rstop)
            else:
                istop = ns1

            frac1 = float(rstart) - istart
            frac2 = 1.0 - (rstop - istop)

            #add pixel values from istart to istop and subtract fraction pixel
            #from istart to rstart and fraction pixel from rstop to istop
            result[i] = np.sum(image[istart:istop + 1]) - frac1 * image[istart] - frac2 * image[istop]

        if total:
            return result
        else:
            return result / (float(sbox) * lbox)
    else:
        _log.debug("Rebinning to Dimensions: %s, %s" % tuple(dimensions))
        #2D case, first bin in second dimension
        temp = np.zeros((nlout, ns))
        result = np.zeros((nsout, nlout))

        #first lines
        for i in range(nlout):
            rstart = i * lbox
            istart = int(rstart)
            rstop = rstart + lbox

            if int(rstop) < nl1:
                istop = int(rstop)
            else:
                istop = nl1

            frac1 = float(rstart) - istart
            frac2 = 1.0 - (rstop - istop)

            if istart == istop:
                temp[i, :] = (1.0 - frac1 - frac2) * image[istart, :]
            else:
                temp[i, :] = np.sum(image[istart:istop + 1, :], axis=0) -\
                             frac1 * image[istart, :] - frac2 * image[istop, :]

        temp = np.transpose(temp)

        #then samples
        for i in range(nsout):
            rstart = i * sbox
            istart = int(rstart)
            rstop = rstart + sbox

            if int(rstop) < ns1:
                istop = int(rstop)
            else:
                istop = ns1

            frac1 = float(rstart) - istart
            frac2 = 1.0 - (rstop - istop)

            if istart == istop:
                result[i, :] = (1. - frac1 - frac2) * temp[istart, :]
            else:
                result[i, :] = np.sum(temp[istart:istop + 1, :], axis=0) -\
                               frac1 * temp[istart, :] - frac2 * temp[istop, :]

        if total:
            return np.transpose(result)
        else:
            return np.transpose(result) / (sbox * lbox)

