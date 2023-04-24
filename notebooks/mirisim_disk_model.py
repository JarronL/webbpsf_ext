#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thu Jun  9 10:08:29 2022: Creation

@author: echoquet from jupyter notebook J. Lensenring


Here we create the basics for a MIRI simulation to observe the HD 141569 system with the FQPM. 
This includes simulating the stellar source behind the center of the phase mask, 
the off-axis stellar companions, and a debris disk model that crosses the mask's quadrant boundaries.


Final outputs will be detector-sampled slope images (counts/sec).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from time import time

import webbpsf_ext
from webbpsf_ext import image_manip, setup_logging, coords
from webbpsf_ext import miri_filter
from webbpsf_ext.coords import jwst_point, plotAxes    #class to setup pointing info
from webbpsf_ext.image_manip import pad_or_cut_to_size

plt.rcParams['image.origin'] = 'lower'
plt.rcParams["image.cmap"] = 'gist_heat'#'hot'#'copper'

# TODO: make sure the central star is removed from the model if flag in dictionary is false
# TODO: Check that the flux matches the simulation (photometry is preserved)
# Simplify the code to keep the smaller FOV instead of full detector size
# Remove or make optional the PSF subtraction and derotation
# convert the code in a pure disk simulation (no star, no psf sibtraction)
# add support for mcfost (?) or convert into a functionto include in another simulation code


#%% Functions
def make_spec(name=None, sptype=None, flux=None, flux_units=None, bp_ref=None, **kwargs):
    """
    Create pysynphot stellar spectrum from input dictionary properties.
    """

    from webbpsf_ext import stellar_spectrum
    
    # Renormalization arguments
    renorm_args = (flux, flux_units, bp_ref)
    
    # Create spectrum
    sp = stellar_spectrum(sptype, *renorm_args, **kwargs)
    if name is not None:
        sp.name = name
    
    return sp


def quick_ref_psf(idl_coord, inst, out_shape, sp=None):
    """
    Create a quick reference PSF for subtraction of the science target.
    """
    
    # Observed SIAF aperture
    siaf_ap = tel_point.siaf_ap_obs
    
    # Location of observation
    xidl, yidl = idl_coord
    
    # Get offset in SCI pixels
    xsci_off, ysci_off = np.array(siaf_ap.convert(xidl, yidl, 'idl', 'sci')) - \
                         np.array(siaf_ap.reference_point('sci'))
    
    # Get oversampled pixels offests
    osamp = inst.oversample
    xsci_off_over, ysci_off_over = np.array([xsci_off, ysci_off]) * osamp
    yx_offset = (ysci_off_over, xsci_off_over)
    
    # Create PSF
    prev_log = webbpsf_ext.conf.logging_level
    setup_logging('WARN', verbose=False)
    hdul_psf_ref = inst.calc_psf_from_coeff(sp=sp, coord_vals=(xidl, yidl), coord_frame='idl')
    setup_logging(prev_log, verbose=False)

    im_psf = pad_or_cut_to_size(hdul_psf_ref[0].data, out_shape, offset_vals=yx_offset)

    return im_psf


#%% Simulation parameters

star_A_Q = True
star_B_Q = False
star_C_Q = False
psf_sub = False
export_Q = False

# Mask information
mask_id = '1140'
filt = f'F{mask_id}C'
mask = f'FQPM{mask_id}'
pupil = 'MASKFQPM'

# Set desired PSF size and oversampling
# MIRI 4QPM: 24" x24" at 0.11 pixels, so 219x219 pixels
# MIRISim synthetic datasets: 224 x 288
fov_pix = 100 #256
osamp = 2


pos_ang = 115            # Position angle is angle of V3 axis rotated towards East
base_offset=(0,0)        # Pointing offsets [arcsec] (BaseX, BaseY) columns in .pointing file
dith_offsets = [(0,0)]   # list of nominal dither offsets [arcsec] (DithX, DithY) columns in .pointing file 

# Information necessary to create pysynphot spectrum of star
star_A_params = {
    'name': 'HD 141569 A', 
    'sptype': 'A2V', 
    'Teff': 10000, 'log_g': 4.28, 'metallicity': -0.5, # Merin et al. 2004
    'dist': 111.6,
    'flux': 64.02, 'flux_units': 'mJy', 'bp_ref': miri_filter('F1065C'),
    'RA_obj'  :  +237.49061772933,     # RA (decimal deg) of source
    'Dec_obj' :  -03.92120600474,      # Dec (decimal deg) of source
}



disk_params = {
    'file': "HD141569_3rings_10.65um.fits",
    'pixscale': 0.027491, 
    'wavelength': 10.65,
    'units': 'Jy/pixel',
    'dist' : 116,
    'cen_star' : True,
}

star_B_params = {
    'name': 'HD 141569 B', 
    'sptype': 'M5V', 
    'Teff': 3000, 'log_g': 4.28, 'metallicity': -0.5, # Merin et al. 2004
    'dist': 111.6,
    'flux': 34.22, 'flux_units': 'mJy', 'bp_ref': miri_filter('F1065C'),
    'RA_obj'  :  237.48904057555,     # RA (decimal deg) of source
    'Dec_obj' :  -03.91981148846,      # Dec (decimal deg) of source
}


star_C_params = {
    'name': 'HD 141569 C', 
    'sptype': 'M5V', 
    'Teff': 3000, 'log_g': 4.28, 'metallicity': -0.5, # Merin et al. 2004
    'dist': 111.6,
    'flux': 45.49, 'flux_units': 'mJy', 'bp_ref': miri_filter('F1065C'),
    'RA_obj'  :  237.48871296031,     # RA (decimal deg) of source
    'Dec_obj' :  -03.9196089225,      # Dec (decimal deg) of source
}

if filt == 'F1550C':
    star_A_params['flux'] = 30.71  
    star_B_params['flux'] = 18.49 
    star_C_params['flux'] = 23.64 

#%% Create the PSF structure


# Initiate instrument class with selected filters, pupil mask, and image mask
inst = webbpsf_ext.MIRI_ext(filter=filt, pupil_mask=pupil, image_mask=mask)

# Set desired PSF size and oversampling
inst.fov_pix = fov_pix
inst.oversample = osamp


# Calculate PSF coefficients
inst.gen_psf_coeff()

# Calculate position-dependent PSFs due to FQPM
# Equivalent to generating a giant library to interpolate over
inst.gen_wfemask_coeff()

#%% Observation setup
'''
Configuring observation settings

Observations consist of nested visit, mosaic tiles, exposures, and dithers. 
In this section, we configure a pointing class that houses information for a single 
observation defined in the APT .pointing file. The primary information includes a 
pointing reference SIAF aperturne name, RA and Dec of the ref aperture, Base X/Y offset
relative to the ref aperture position, and Dith X/Y offsets. From this information, 
along with the V2/V3 position angle, we can determine the orientation and location 
of objects on the detector focal plane.

Note: The reference aperture is not necessarily the same as the observed aperture. 
For instance, you may observe simultaneously with four of NIRCam's SWA detectors, 
so the reference aperture would be the entire SWA channel, while the observed apertures 
are A1, A2, A3, and A4.
'''

# Observed and reference apertures
# ap_obs = inst.aperturename
ap_ref = f'MIRIM_MASK{mask_id}'

# Telescope pointing information
tel_point = jwst_point(inst.aperturename, ap_ref, star_A_params['RA_obj'], star_A_params['Dec_obj'], 
                       pos_ang=pos_ang, base_offset=base_offset, dith_offsets=dith_offsets,
                       base_std=0, dith_std=0)

# Get sci position of center in units of detector pixels
# Elodie: gives the position of the mask center, in pixels 
siaf_ap = tel_point.siaf_ap_obs
x_cen, y_cen = siaf_ap.reference_point('sci')

# Elodie: gives the full frame image size in pixel, inc. oversampling (432x432 with osamp=2)
ny_pix, nx_pix = (siaf_ap.YSciSize, siaf_ap.XSciSize)
shape_new = (ny_pix * osamp, nx_pix * osamp)

print(f"Reference aperture: {tel_point.siaf_ap_ref.AperName}")
print(f"  Nominal RA, Dec = ({tel_point.ra_ref:.6f}, {tel_point.dec_ref:.6f})")
print(f"Observed aperture: {tel_point.siaf_ap_obs.AperName}")
print(f"  Nominal RA, Dec = ({tel_point.ra_obs:.6f}, {tel_point.dec_obs:.6f})")

print("Relative offsets in 'idl' for each dither position (incl. pointing errors)")
for i, offset in enumerate(tel_point.position_offsets_act):
    print(f"  Position {i}: ({offset[0]:.4f}, {offset[1]:.4f}) arcsec")
    
    
#%% Add central source
'''
Here we define the stellar atmosphere parameters for HD 141569, including spectral type, 
optional values for (Teff, log_g, metallicity), normalization flux and bandpass, 
as well as RA and Dec.
Then Computes the PSF, including any offset/dither, using the coefficients.
It includes geometric distortions based on SIAF info. 
'''
# Create stellar spectrum and add to dictionary
sp_star = make_spec(**star_A_params)
star_A_params['sp'] = sp_star
  
if star_A_Q:
  
    # Get `sci` coord positions
    coord_obj = (star_A_params['RA_obj'], star_A_params['Dec_obj'])
    xsci, ysci = tel_point.radec_to_frame(coord_obj, frame_out='sci')
    
    # Create oversampled PSF
    hdul = inst.calc_psf_from_coeff(sp=sp_star, coord_vals=(xsci,ysci), coord_frame='sci')

    
    # Get the shifts from center and oversampled pixel shifts
    xsci_off, ysci_off = (xsci-x_cen, ysci-y_cen)
    delyx = (ysci_off * osamp, xsci_off * osamp)
    print("Image shifts (oversampled pixels):", delyx) #xsci_off_over, ysci_off_over)
    
    # Expand PSF to full frame and offset to proper position
    image_full = pad_or_cut_to_size(hdul[0].data, shape_new, offset_vals=delyx)
    print('Size image (oversampled): {}'.format(image_full.shape))
    
    # fig, ax = plt.subplots(1,1)
    # ax.imshow(image_full, vmin=-0.5,vmax=1)
    
    # Make new HDUList of target (just central source so far)
    hdul_full = fits.HDUList(fits.PrimaryHDU(data=image_full, header=hdul[0].header))
    

#%% Add the stellar companions

if star_B_Q:
    sp_star_B = make_spec(**star_B_params)
    star_B_params['sp'] = sp_star_B
    
    coord_star_B = (star_B_params['RA_obj'], star_B_params['Dec_obj'])
    xstar_B, ystar_B = tel_point.radec_to_frame(coord_star_B, frame_out='sci')
    hdul_B = inst.calc_psf_from_coeff(sp=sp_star_B, coord_vals=(xstar_B, ystar_B), coord_frame='sci')
    
    xstar_B_off, ystar_B_off = (xstar_B-x_cen, ystar_B-y_cen)
    delyx_B = (ystar_B_off * osamp, xstar_B_off * osamp)
    
    image_full_B = pad_or_cut_to_size(hdul_B[0].data, shape_new, offset_vals=delyx_B)
    
    if star_A_Q: 
        hdul_full[0].data += image_full_B
    else:
        hdul_full = fits.HDUList(fits.PrimaryHDU(data=image_full_B, header=hdul_B[0].header))


if star_C_Q:
    sp_star_C = make_spec(**star_C_params)
    star_C_params['sp'] = sp_star_C
    
    coord_star_C = (star_C_params['RA_obj'], star_C_params['Dec_obj'])
    xstar_C, ystar_C = tel_point.radec_to_frame(coord_star_C, frame_out='sci')
    hdul_C = inst.calc_psf_from_coeff(sp=sp_star_C, coord_vals=(xstar_C, ystar_C), coord_frame='sci')
    
    xstar_C_off, ystar_C_off = (xstar_C-x_cen, ystar_C-y_cen)
    delyx_C = (ystar_C_off * osamp, xstar_C_off * osamp)
    
    image_full_C = pad_or_cut_to_size(hdul_C[0].data, shape_new, offset_vals=delyx_C)
    
    if star_A_Q or star_B_Q: 
        hdul_full[0].data += image_full_C
    else:
        hdul_full = fits.HDUList(fits.PrimaryHDU(data=image_full_C, header=hdul_C[0].header))
    

# Print the results
if star_A_Q or star_B_Q or star_C_Q:
    fig, ax = plt.subplots(1,1)
    fig.suptitle('Oversampled image stars '+filt)
    # extent = 0.5 * np.array([-1,1,-1,1]) * inst.fov_pix * inst.pixelscale
    ax.imshow(hdul_full[0].data, vmin=-0.5,vmax=1) #, extent=extent, cmap='magma',
    # ax.set_xlabel('Arcsec')
    # ax.set_ylabel('Arcsec')
    # ax.tick_params(axis='both', color='white', which='both')
    # for k in ax.spines.keys():
    #     ax.spines[k].set_color('white')
    # ax.xaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])
    # ax.yaxis.get_major_locator().set_params(nbins=9, steps=[1, 2, 5, 10])
    fig.tight_layout()
    plt.show()

#%% Convolve extended disk image
'''
Properly including extended objects is a little more complicated than for point sources. 
First, we need properly format the input model to a pixel binning and flux units 
appropriate for the simulations (ie., pixels should be equal to oversampled PSFs with 
flux units of counts/sec). Then, the image needs to be rotated relative to the 'idl' 
coordinate plane and subsequently shifted for any pointing offsets. 
Once in the appropriate 'idl' system
'''


### PSF Grid
# Create grid locations for array of PSFs to generate
apname = inst.psf_coeff_header['APERNAME']
siaf_ap = inst.siaf[apname]

field_rot = 0 if inst._rotation is None else inst._rotation

xyoff_half = 10**(np.linspace(-2,1,10))
xoff = yoff = np.concatenate([-1*xyoff_half[::-1],[0],xyoff_half])

# Mask Offset grid positions in arcsec
xgrid_off, ygrid_off = np.meshgrid(xoff, yoff)
xgrid_off, ygrid_off = xgrid_off.flatten(), ygrid_off.flatten()

# Science positions in detector pixels
xoff_sci_asec, yoff_sci_asec = coords.xy_rot(-1*xgrid_off, -1*ygrid_off, -1*field_rot)
xgrid = xoff_sci_asec / siaf_ap.XSciScale + siaf_ap.XSciRef
ygrid = yoff_sci_asec / siaf_ap.YSciScale + siaf_ap.YSciRef


# Now, create all PSFs, one for each (xgrid, ygrid) location
# Only need to do this once. Can be used for multiple dither positions.
t0 = time()
hdul_psfs = inst.calc_psf_from_coeff(coord_vals=(xgrid, ygrid), coord_frame='sci', return_oversample=True)
t1 = time()
print('PSF Grid Calculation time: {} s'.format(t1-t0))
print('Number of PSFs: {}'.format(len(hdul_psfs)))
print('PSF shape: {}'.format(hdul_psfs[0].data.shape))

psf_grid_summed = np.empty(hdul_psfs[0].data.shape)
for i in range(len(hdul_psfs)):
    psf_grid_summed += hdul_psfs[i].data 

fig1, ax1 = plt.subplots(1,1)
fig1.suptitle('PSF grid for disk')
# extent = 0.5 * np.array([-1,1,-1,1]) * inst.fov_pix * inst.pixelscale
ax1.imshow(psf_grid_summed/len(hdul_psfs) , norm=LogNorm(vmin=0.00001,vmax=0.001))
fig1.tight_layout()
plt.show()



##########################  Disk Model Image
# Open model and rebin to PSF sampling
# Scale to instrument wavelength assuming grey scattering function
# Converts to phot/sec/lambda
t0_disk =time()
hdul_disk_model = image_manip.make_disk_image(inst, disk_params, sp_star=star_A_params['sp'])
print('Input disk model shape: {}'.format(hdul_disk_model[0].data.shape))

fig1, ax1 = plt.subplots(1,1)
fig1.suptitle('disk model input')
ax1.imshow(hdul_disk_model[0].data, vmin=0,vmax=20)
fig1.tight_layout()
plt.show()


# Rotation necessary to go from sky coordinates to 'idl' frame
rotate_to_idl = -1*(tel_point.siaf_ap_obs.V3IdlYAngle + tel_point.pos_ang)




### Dither position
# Select the first dither location offset
delx, dely = tel_point.position_offsets_act[0]
#hdul_out = image_manip.rotate_shift_image(hdul_disk_model, PA_offset=rotate_to_idl,
#                                          delx_asec=delx, dely_asec=dely)

### Modification by Elodie: Warrning message then crash: PA_offset deprecated, replace by angle instead
hdul_out = image_manip.rotate_shift_image(hdul_disk_model, angle=rotate_to_idl,
                                          delx_asec=delx, dely_asec=dely)

# Distort image on 'sci' coordinate grid
im_sci, xsci_im, ysci_im = image_manip.distort_image(hdul_out, ext=0, to_frame='sci', return_coords=True)

# Distort image onto 'tel' (V2, V3) coordinate grid for plot illustration
im_tel, v2_im, v3_im = image_manip.distort_image(hdul_out, ext=0, to_frame='tel', return_coords=True)


# Plot locations for PSFs that we will generate
# Show image in V2/V3 plane
# fig, ax = plt.subplots(1,1)
# extent = [v2_im.min(), v2_im.max(), v3_im.min(), v3_im.max()]
# ax.imshow(im_tel**0.1, extent=extent)
# # Add on SIAF aperture boundaries
# tel_point.plot_inst_apertures(ax=ax, clear=False, label=True)
# tel_point.plot_ref_aperture(ax=ax)
# tel_point.plot_obs_aperture(ax=ax, color='C3')
# # Add PSF location points
# v2, v3 = siaf_ap.convert(xgrid, ygrid, 'sci', 'tel')
# ax.scatter(v2, v3, marker='.', alpha=0.5, color='C2', edgecolors='none', linewidths=0)
# ax.set_title('Model disk image and PSF Locations in SIAF FoV')
# fig.tight_layout()


'''This particular disk image is oversized, so we will need to crop the image after 
convolving PSFs. We may want to consider trimming some of this image prior to convolution,
 depending on how some of the FoV is blocked before reaching the coronagraphic optics.'''
 
# If the image is too large, then this process will eat up much of your computer's RAM
# So, crop image to more reasonable size (20% oversized)
xysize = int(1.2 * np.max([siaf_ap.XSciSize,siaf_ap.YSciSize]) * osamp)
xy_add = osamp - np.mod(xysize, osamp)
xysize += xy_add

im_sci = pad_or_cut_to_size(im_sci, xysize)
hdul_disk_model_sci = fits.HDUList(fits.PrimaryHDU(data=im_sci, header=hdul_out[0].header))
print('Resized disk model shape: {}'.format(im_sci.shape))

fig1, ax1 = plt.subplots(1,1)
fig1.suptitle('disk model rotate & resized')
ax1.imshow(im_sci, vmin=0,vmax=20)
fig1.tight_layout()
plt.show()

# Added by Elodie June 7th after chat with Kim
# Get X and Y indices corresponding to aperture reference
# xref, yref = self.siaf_ap.reference_point('sci')
# hdul_disk_model_sci[0].header['XIND_REF'] = (xref*osamp, "x index of aperture reference")
# hdul_disk_model_sci[0].header['YIND_REF'] = (yref*osamp, "y index of aperture reference")
hdul_disk_model_sci[0].header['CFRAME'] = 'sci'

# Convolve image
t0 = time()
im_conv = image_manip.convolve_image(hdul_disk_model_sci, hdul_psfs)
t1 = time()
print('Disk Convolution calculation time: {} s'.format(t1-t0))
print('Convolved disk shape: {}'.format(im_conv.shape)) #260x260

fig1, ax1 = plt.subplots(1,1)
fig1.suptitle('Convolved disk')
ax1.imshow(im_conv, vmin=0,vmax=20)
fig1.tight_layout()
plt.show()


# Add cropped image to final oversampled image
im_conv = pad_or_cut_to_size(im_conv, shape_new)
if star_A_Q or star_B_Q or star_C_Q:
    hdul_full[0].data += im_conv
else:
    hdul_full = fits.HDUList(fits.PrimaryHDU(data=im_conv, header=hdul_disk_model_sci[0].header))
print('Resized Convolved disk shape: {}'.format(im_conv.shape)) #432x432



# Rebin science data to detector pixels
im_sci = image_manip.frebin(hdul_full[0].data, scale=1/osamp)
print('Detector sampled final image shape: {}'.format(im_sci.shape)) #216x216

t1_disk =time()

print('######################')
print('Disk computation time: {} s'.format(t1_disk-t0_disk))


# Subtract a reference PSF from the science data
if psf_sub:
    coord_vals = tel_point.position_offsets_act[0]
    im_psf = quick_ref_psf(coord_vals, inst, hdul_full[0].data.shape, sp=sp_star)
    im_ref = image_manip.frebin(im_psf, scale=1/osamp)
    imdisk = im_sci - im_ref
else:
    imdisk = im_sci

# De-rotate to sky orientation
imrot = image_manip.rotate_offset(imdisk, rotate_to_idl, reshape=False, cval=np.nan)


# Plot results
xsize_asec = siaf_ap.XSciSize * siaf_ap.XSciScale
ysize_asec = siaf_ap.YSciSize * siaf_ap.YSciScale
extent = [-1*xsize_asec/2, xsize_asec/2, -1*ysize_asec/2, ysize_asec/2]
fig, axes = plt.subplots(2,2, figsize=(8,8), dpi=300)

plmax = 0.05 * imdisk.max()
axes[0,0].imshow(imdisk, extent=extent, vmin=-0.1 * plmax, vmax=plmax)
axes[0,1].imshow(imrot, extent=extent, vmin=-0.1 * plmax, vmax=plmax)
axes[0,0].set_title('Raw Image (lin)')
axes[0,1].set_title("De-Rotated (lin)")

plmax = imdisk.max()
axes[1,0].imshow(imdisk, extent=extent, norm=LogNorm(vmin=plmax * 1e-5, vmax=plmax))
axes[1,1].imshow(imrot, extent=extent, norm=LogNorm(vmin=plmax * 1e-5, vmax=plmax))
axes[1,0].set_title('Raw Image (log)')
axes[1,1].set_title("De-Rotated (log)")

for i in range(2):
    for j in range(2):
        axes[i,j].set_xlabel('XSci (arcsec)')
        axes[i,j].set_ylabel('YSci (arcsec)')
    plotAxes(axes[i,0], angle=-1*siaf_ap.V3SciYAngle)
    plotAxes(axes[i,1], position=(0.95,0.35), label1='E', label2='N')

fig.suptitle(f"HD 141569 ({filt})", fontsize=14)
fig.tight_layout()


#%% Figurre Jarron
# fig, axes = plt.subplots(1,3, figsize=(12,4.5))

# ############################
# # Plot raw image
# ax = axes[0]

# im = im_sci
# mn = np.median(im)
# std = np.std(im)
# vmin = 0
# vmax = mn+10*std

# xsize_asec = siaf_ap.XSciSize * siaf_ap.XSciScale
# ysize_asec = siaf_ap.YSciSize * siaf_ap.YSciScale
# extent = [-1*xsize_asec/2, xsize_asec/2, -1*ysize_asec/2, ysize_asec/2]
# norm = LogNorm(vmin=im.max()/1e5, vmax=im.max())
# ax.imshow(im, extent=extent, norm=norm)

# ax.set_title("Raw Image (log scale)")

# ax.set_xlabel('XSci (arcsec)')
# ax.set_ylabel('YSci (arcsec)')
# plotAxes(ax, angle=-1*siaf_ap.V3SciYAngle)

# ############################
# # Basic PSF subtraction
# # Subtract a near-perfect reference PSF
# ax = axes[1]
# norm = LogNorm(vmin=imdiff.max()/1e5, vmax=imdiff.max())
# ax.imshow(imdiff, extent=extent, norm=norm, cmap='magma')

# ax.set_title("PSF Subtracted (log scale)")

# ax.set_xlabel('XSci (arcsec)')
# ax.set_ylabel('YSci (arcsec)')
# plotAxes(ax, angle=-1*siaf_ap.V3SciYAngle)

# ############################
# # De-rotate to sky orientation

# ax = axes[2]
# ax.imshow(imrot, extent=extent, norm=norm, cmap='magma')

# ax.set_title("De-Rotated (log scale)")

# ax.set_xlabel('RA offset (arcsec)')
# ax.set_ylabel('Dec offset (arcsec)')
# plotAxes(ax, position=(0.95,0.35), label1='E', label2='N')

# fig.suptitle(f"Fomalhaut ({siaf_ap.AperName})", fontsize=14)
# fig.tight_layout()


# hdul_disk_model_sci[0].header



#%% Save image to FITS file
hdu_diff = fits.PrimaryHDU(imdisk)

copy_keys = [
    'PIXELSCL', 'DISTANCE', 
    'INSTRUME', 'FILTER', 'PUPIL', 'CORONMSK',
    'APERNAME', 'MODULE', 'CHANNEL',
    'DET_NAME', 'DET_X', 'DET_Y', 'DET_V2', 'DET_V3'
]

hdr = hdu_diff.header
for head_temp in (inst.psf_coeff_header, hdul_out[0].header):
    for key in copy_keys:
        try:
            hdr[key] = (head_temp[key], head_temp.comments[key])
        except (AttributeError, KeyError):
            pass

hdr['PIXELSCL'] = inst.pixelscale

name = star_A_params['name']

outfile = f'HD141569/{name}_{inst.aperturename}_{inst.filter}.fits'
hdu_diff.writeto(outfile, overwrite=True)
