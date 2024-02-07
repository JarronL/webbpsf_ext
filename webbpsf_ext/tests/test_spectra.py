import pytest

from astropy import units as u

from webbpsf_ext.spectra import *
from webbpsf_ext import synphot_ext

def test_BOSZ_filename():
    # Test the BOSZ_filename function
    filename = BOSZ_filename(3500, 0.0, 5, 2000, carbon=0, alpha=0.0)
    assert filename == 'amp00cp00op00t3500g50v20modrt0b2000rs.fits'

    filename = BOSZ_filename(3750, +0.5, 5, 10000, carbon=0.25, alpha=0.5)
    assert filename == 'amp05cp03op05t3750g50v20modrt0b10000rs.fits'

    filename = BOSZ_filename(3500, -2.25, 5, 2000, carbon=-0.75, alpha=0.0)
    assert filename == 'amm23cm08op00t3500g50v20modrt0b2000rs.fits'

@pytest.mark.remote_data
def test_download_BOSZ_spectrum(tmp_path_factory):
    # Test the download_BOSZ_spectrum function

    outdir = tmp_path_factory.mktemp('bosz')
    download_BOSZ_spectrum(3500, 0.0, 4.5, 1000, carbon=0, alpha=0.0,
                           outdir=outdir)

@pytest.mark.remote_data
def test_download_votable():
    # Test the download_votable function
    tbl = download_votable('Vega')

@pytest.mark.parametrize('sptype', ['A0V', 'F0V', 'G0V', 'K0V', 'M0V'])
def test_stellar_spectrum(sptype):
    # Test the stellar_spectrum function
    sp = stellar_spectrum(sptype, catname='ck04models')
    sp = stellar_spectrum(sptype, catname='phoenix')
