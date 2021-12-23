# WebbPSF Extensions

## Extending JWST PSF generation

[![PyPI Version](https://img.shields.io/pypi/v/webbpsf_ext.svg)](https://pypi.python.org/pypi/webbpsf_ext)

*Authors*: Jarron Leisenring (U. of Arizona, Steward Observatory)

`webbpsf_ext` provides some enhancements to the [WebbPSF](https://webbpsf.readthedocs.io) package for PSF creation. This follows the [pyNRC](https://github.com/JarronL/pynrc) implementation for storing and retrieving JWST PSFs. In particular, this module generates and saves polynomial coefficients to quickly create unique instrument PSFs as a function of wavelength, focal plane position, wavefront error drift from thermal distortions.

More specifically, `webbpsf_ext` uses WebbPSF to generate a series of monochromatic PSF simulations, then produces polynomial fits to each pixel. Storing the coefficients rather than a library of PSFS allows for quick creation (via matrix multiplication) of PSF images for an arbitrary number of wavelengths (subject to hardware memory limitations, of course). The applications range from quickly creating PSFs for many different stellar types over wide bandpasses to generating a large number of monochromatic PSFs for spectral dispersion.

In addition, each science instrument PSF is dependent on the detector position due to field-dependent wavefront errors. Such changes are tracked in WebbPSF, but it becomes burdensome to generate new PSFs from scratch at each location, especially for large starfields. Instead, these changes can be stored by the fitting the residuals of the PSF coefficients across an instrument's field of view, then interpolating for an arbitrary location. A similar scheme can be achieved for coronagraphic occulters, where the PSF changes as the source position moves with respect to the mask.

JWST's thermal evolution (e.g., changing the angle of the sunshield after slewing to a new target) causes small but significant distortions to the telescope backplane. WebbPSF has tools to modify OPDs, but high-fidelity simulations take time to calculate. Since the change to the PSF coefficients varies smoothly with respect to WFE drift components, it's simple to parameterize the coefficient residuals in a fashion similar to the field-dependence.

## `webbpsf_ext` classes

Currently, only NIRCam and MIRI instrument classes are supported via the `NIRCam_ext` and `MIRI_ext` objects (although other instruments can be similarly implemented), which subclass the native `NIRCam` and `MIRI` objects found in WebbPSF. The key functions to call and generate PSF coefficients are:

1. `gen_psf_coeff`: Creates a set of coefficients that will generate simulated PSFs for any arbitrary wavelength. This function first simulates a number of evenly- spaced PSFs throughout the specified bandpass (or the full channel).  An nth-degree polynomial is then fit to each oversampled pixel using  a linear-least squares fitting routine. The final set of coefficients  for each pixel is returned as an image cube. The returned set of  coefficient are then used to produce PSF via `calc_psf_from_coeff`. Useful for quickly generated imaging and dispersed PSFs for multiple spectral types.
1. `gen_wfedrift_coeff`: This function finds a relationship between PSF coefficients in the presence of WFE drift. For a series of WFE drift values, we generate corresponding PSF coefficients and fit a polynomial relationship to the residual values. This allows us to quickly modify a nominal set of PSF image coefficients to generate a new PSF where the WFE has drifted by some amplitude.
1. `gen_wfefield_coeff`: Science instruments generally have field-dependent wavefront errors, which have been carefully measured in ground-based cryovac test campaigns ([SI WFE](https://webbpsf.readthedocs.io/en/latest/jwst.html#si-wfe)). These aberrations are expected to be static throughout the JWST mission lifetime. Similar to the above function, we generate PSF coefficients at each of the sampled positions and create a `RegularGridInterpolator` function to quickly determine new coefficient residuals at arbitrary locations.
1. `gen_wfemask_coeff`: For coronagraphic masks, slight changes in the PSF location relative to the image plane mask can substantially alter the PSF speckle pattern. This function generates a number of PSF coefficients at a variety of positions, then fits polynomials to the residuals to track how the PSF changes across the mask's field of view. Special care is taken near the 10-20mas region in order to provide accurate sampling of the SGD offsets.

## Installing with pip

Download via pip:

```bash
pip install webbpsf_ext
```

## Installing from source

To get the most up to date version of `webbpsf_ext`, install directly from the GitHub source.

In this case, you will need to clone the git repository:

```bash
git clone https://github.com/JarronL/webbpsf_ext
```

Then install the package with:

```bash
cd webbpsf_ext
pip install .
```

For development purposes, you can use editable installations:

```bash
cd webbpsf_ext
pip install -e .
```

This is useful for helping to develop the code, creating bug reports, switching between branches and submitting pull requests to GitHub. Since the code then lives in your local GitHub directory, this method makes it simple to pull the latest updates straight from the GitHub repo, which are then immediately available to python at runtime without needing to reinstall.

## Create Data Directory

You can define the directory that stores the PSF coefficients by setting the environment variable `WEBBPSF_EXT_PATH` to point to some data directory. All PSF coefficients will be saved here as they are generated to be reused later. For example, in `.bashrc` shell file, add:

```bash
export WEBBPSF_EXT_PATH='$HOME/data/webbpsf_ext_data/'
```

If this is not set, then a `psf_coeff` sub-directory is created in the already existing `WEBBPSF_PATH` directory.
