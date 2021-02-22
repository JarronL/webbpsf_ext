# WebbPSF Extensions

## A Toolset for extending WebbPSF functionality
--------------------- 

*Authors*: Jarron Leisenring (University of Arizona, Steward Observatory)

`webbpsf_ext` provides some enhancements to the [WebbPSF](https://webbpsf.readthedocs.io) package for PSF creation. This is a subset of the [pyNRC](https://github.com/JarronL/pynrc) implementation for storing and retrieving JWST PSFs. In particular, this module generates and saves polynomial coefficients to quickly create unique instrument PSFs as a function of wavelength, focal plane position, wavefront error drift from thermal distortions.

More specifically, `webbpsf_ext` uses WebbPSF to generate a series of monochromatic PSF simulations, then produces polynomial fits to each pixel. Storing the coefficients rather than a library of PSFS allows for quick generation (via matrix multiplication) of PSF images for an arbitrary number of wavelengths (subject to hardware memory limitations, of course). 

The applications range from quickly creating PSFs for many different stellar types over wide bandpasses to generating a large number of monochromatic PSFs for spectral dispersion.

In addition, each science instrument PSF is dependent on the detector position due to field-dependent wavefront errors. Such changes are tracked in WebbPSF, but it becomes burdensome to generate new PSFs from scratch at location, especially for large starfields. Instead, these changes can be tracked by the fitting the residuals of the PSF coefficients across an instrument's field of view.

Similarly, JWST's thermal evolution (e.g., changing angle of the sunshield after slewing to a new target) causes small but significant distortions to the telescope backplane. WebbPSF has tools to modify OPDs, but high-fidelity simulations take time to calculate . Since the change to the PSF coefficients varies smoothly with respect to WFE drift components, it's simple to parameterize the coefficient residuals.
