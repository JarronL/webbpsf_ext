David S. Spiegel and Adam Burrows
6 June 2011

Spectra subdirectory:

This contains 1680 files, one for each of 4 atmosphere types, each of
15 masses, and each of 28 ages.  The file names indicate the
combination.

hy1s = hybrid clouds, solar abundances
hy3s = hybrid clouds, 3x solar abundances
cf1s = cloud-free, solar abundances
cf3s = cloud-free, 3x solar abundances

The first number indicates the mass (in units of Jupiter's) and the
second indicates the age (in Myr).

Each file contains the following rows:

Row #, Value
1      col 1: age (Myr);
       cols 2-601: wavelength (in microns, in range 0.8-15.0)
2-end  col 1: initial S;
       cols 2-601: F_nu (in mJy for a source at 10 pc)

Row 1 contains the wavelength scale (in columns 2-61) for a moderate
resolution (R ~ 204) spectrum.  In rows 2-end, where the spectra
appear, the source is assumed to be at a distance of 10 pc.  Initial
entropies increase from 8.0 to the minimum of 13.0 and the maximum
stable initial entropy that we could calculate, in increments of 0.25.

