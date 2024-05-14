import pytest

from astropy import units as u

from webbpsf_ext.bandpasses import *
from webbpsf_ext import synphot_ext

def test_bp_igood():
    # Test the bp_igood function
    wave = [2, 2.2, 2.4, 2.6, 2.8, 3] * u.um
    throughput = [0, 0.001, 0.5, 0.5, 0.001, 0]
    bp = synphot_ext.ArrayBandpass(wave, throughput, name='test')

    # Get good wavelength indices
    igood = bp_igood(bp, min_trans=0.001, fext=0.1)
    wgood = bp.wave[igood]

    assert wgood.min() == 2.2
    assert wgood.max() == 2.8
    

def test_miri_filter():
    # Test the miri_filter function
    bp = miri_filter('F560W')

    bp_avgwave = bp.avgwave().to_value(u.um)
    assert np.allclose(bp_avgwave, 5.6450, atol=0.1)

def test_nircam_com_th():
    # Test the nircam_com_th function
    th = nircam_com_th(wave_out=[2,3,4,5])
    assert np.allclose(th, [0.96465925, 0.9028954 , 0.96137487, 0.74519337])

def test_nircam_com_nd():
    # Test the nircam_com_nd function
    od = nircam_com_nd(wave_out=[2,3,4,5])
    th = 10**(-1 * od)
    assert np.allclose(th, [0.0010834, 0.001729, 0.0022805, 0.00221794], rtol=0.001)

def test_nircam_lyot_th():
    # Test the nircam_lyot_th function

    # SW
    th = nircam_lyot_th('SW', wave_out=[1.5,2,2.2])
    assert np.allclose(th, [0.8855, 0.828, 0.719], rtol=0.001)

    # SW
    th = nircam_lyot_th('LW', wave_out=[2.5,3,3.5,4,4.5,5])
    assert np.allclose(th, [0.977, 0.979, 0.976, 0.98, 0.98, 0.959], rtol=0.001)


def test_nircam_grism_si_ar():
    # Test the nircam_grism_si_ar function
    wdata, tdata = nircam_grism_si_ar('A', return_bp=False)
    bp = nircam_grism_si_ar('A', return_bp=True)

    bp.convert('um')

    assert np.allclose(tdata, bp.throughput)
    assert np.allclose(wdata, bp.wave)

def test_nircam_grism_th():
    # Test the nircam_grism_th function
    wave_out = np.linspace(2.5,5,10)
    th = nircam_grism_th('A', grism_order=1, wave_out=wave_out)
    bp = nircam_grism_th('A', grism_order=1, wave_out=wave_out, return_bp=True)

    assert np.allclose(th, bp.throughput)


@pytest.mark.parametrize('sca', np.arange(481,491))
def test_qe_nircam(sca):
    # Test the qe_nircam function
    bp = qe_nircam(sca)
    assert bp.throughput.max() > 0.8


@pytest.mark.parametrize('filt, result', [
    # Filter, (avgwave, fwhm_left, fwhm_right)
    ('F090W', (0.906, 0.796, 1.006)),
    ('F115W', (1.157, 1.011, 1.282)),
    ('F150W', (1.504, 1.330, 1.668)),
    ('F200W', (1.992, 1.755, 2.226)),
    ('F277W', (2.766, 2.401, 3.125)),
    ('F335M', (3.359, 3.162, 3.541)),
    ('F356W', (3.570, 3.141, 3.973)),
    ('F410M', (4.075, 3.850, 4.297)),
    ('F430M', (4.283, 4.170, 4.399)),
    ('F444W', (4.391, 3.874, 4.963)),
    ('F460M', (4.635, 4.520, 4.753)),
    ('F480M', (4.794, 4.653, 4.944)),
])
def test_nircam_filter_wavelengths(filt, result):
    # Test average wavelength of nircam filters, and half-power points
    bp = nircam_filter(filt, wave_units=u.um)
    avgwave = bp.avgwave().to_value(u.um)
    fwidth = filter_width(bp)

    values = (avgwave, fwidth[0], fwidth[1])
    assert np.allclose(values, result, atol=0.001)

@pytest.mark.parametrize('filt, result', [
    ('j', (1.235, 0.162)),
    ('h', (1.646, 0.251)),
    ('k', (2.159, 0.262)),
])
def test_bp_2mass(filt, result):
    # Test the bp_2mass function
    bp = bp_2mass(filt)
    avgwave = bp.barlam().to_value(u.um)
    bw = bp.rectwidth().to_value(u.um)
    assert np.allclose((avgwave, bw), result, atol=0.01)

@pytest.mark.parametrize('filt, result', [
    ('w1', ( 3.346, 0.682)),
    ('w2', ( 4.595, 1.051)),
    ('w3', (11.553, 6.456)),
    ('w4', (22.078, 3.945)),
])
def test_bp_wise(filt, result):
    # Test the bp_wise function
    bp = bp_wise(filt)
    avgwave = bp.barlam().to_value(u.um)
    bw = bp.rectwidth().to_value(u.um)
    assert np.allclose((avgwave, bw), result, atol=0.01)

@pytest.mark.parametrize('filt, result', [
    ('g',  (0.590, 0.441)),
    ('bp', (0.489, 0.263)),
    ('rp', (0.763, 0.276)),
])
def test_bp_gaia(filt, result):
    # Test the bp_gaia function
    bp = bp_gaia(filt)
    avgwave = bp.barlam().to_value(u.um)
    bw = bp.rectwidth().to_value(u.um)
    assert np.allclose((avgwave, bw), result, atol=0.01)


def test_filter_width():
    # Test the filter_width function
    wave = np.arange(1,5.5,0.5)
    th = np.ones_like(wave)
    th[0] = 0
    th[-1] = 0
    th[1] = 0.5
    th[-2] = 0.5

    bp = synphot_ext.ArrayBandpass(wave, th, waveunits='um', name='test')
    width = filter_width(bp, gsmooth=None)

    assert np.allclose(width, [wave[1], wave[-2]])

