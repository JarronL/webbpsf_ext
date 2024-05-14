import numpy as np

import webbpsf_ext.utils as utils

# Test get_detname returns correct detector name
def test_get_detname():
    """Test case for the `get_detname` function.

    This test case checks the behavior of the get_detname function for different input names.
    It verifies that the function correctly returns the expected detector name when the use_long
    parameter is set to False and True.
    """

    nrca5_name_list = [
        'A5', 
        'NRCA5',
        'ALONG',
        'NRCALONG',
    ]
    
    for name in nrca5_name_list:
        assert utils.get_detname(name, use_long=False) == 'NRCA5'
        assert utils.get_detname(name, use_long=True) == 'NRCALONG'

def test_pix_ang_size():
    """ Test the calculation of angular size of a pixel

    This function tests the `pix_ang_size` correctly calculates the angular size of a pixel
    """

    # Check that squaring 206265 arcsec/pixel outputs 1 radian
    assert np.allclose(utils.pix_ang_size(pixscale=206265), 1)

