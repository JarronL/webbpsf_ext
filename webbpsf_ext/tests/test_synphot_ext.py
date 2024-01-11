import pytest
import numpy as np
import os

from astropy.io import ascii

from webbpsf_ext import synphot_ext

# Option to skip tests if pysynphot is not installed
try:
    import pysynphot as S
    pysynphot_installed = True
except ImportError: 
    pysynphot_installed = False

@pytest.mark.skipif(not pysynphot_installed, reason="pysynphot is not installed")
def test_bandpass_equivalence():
    """
    Test the equivalence of bandpasses between pysynphot and synphot_convert.

    This function compares the wave and throughput arrays of a bandpass created using
    pysynphot and synphot_convert. It checks if the arrays are equal and raises an
    assertion error if they are not.

    Note: This test requires the pysynphot package to be installed. If it is not
    installed, the test will be skipped.
    """

    from webbpsf_ext import bandpasses

    bp_2mass_dir = os.path.join(bandpasses._bp_dir, '2MASS')
    file = '2mass_ks.txt'
    name='Ks-Band'
    tbl = ascii.read(os.path.join(bp_2mass_dir, file), names=['Wave', 'Throughput'])
    wave, th = tbl['Wave'].data*1e4, tbl['Throughput'].data

    # Create a pysynphot.ObsBandpass object
    pysyn_bandpass = S.ArrayBandpass(wave, th, name=name)
    pysyn_bandpass.convert('um')

    # Create a synphot_convert.Bandpass object
    synphot_bandpass = synphot_ext.ArrayBandpass(wave, th, name=name)
    synphot_bandpass.convert('um')

    # Compare the wave and throughput arrays
    pysyn_wave = pysyn_bandpass.wave
    synphot_wave = synphot_bandpass.wave
    pysyn_throughput = pysyn_bandpass.throughput
    synphot_throughput = synphot_bandpass.throughput

    # Check if the wave and throughput arrays are close
    assert np.allclose(pysyn_wave, synphot_wave)
    assert np.allclose(pysyn_throughput, synphot_throughput)

def test_validate_unit():
    
    # Test lowercase jy units
    unit_strings = ['jy', 'njy', 'ujy', 'mjy', 'Mjy']
    for unit_str in unit_strings:
        synphot_ext.validate_unit(unit_str)

    # Test that Jys fails
    with pytest.raises(ValueError):
        synphot_ext.validate_unit('Jys')