import numpy as np
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

from kbackground import Estimator, __version__


def test_version():
    assert __version__ == "0.1.7"


def test_kbackground():
    imgpath = get_pkg_data_filename("test.fits.gz")

    hdu = fits.open(imgpath)
    flux = hdu[1].data["flux"]
    r, c = hdu[1].header["2CRV5P"], hdu[1].header["1CRV5P"]
    column, row = np.meshgrid(
        np.arange(c, flux.shape[2] + c),
        np.arange(r, flux.shape[1] + r),
    )
    time = hdu[1].data["TIME"]
    aper = np.ones(flux.shape[1:], bool)
    bkg = Estimator(time, row[aper], column[aper], flux[:, aper])
    model = bkg.model
    assert model.shape == bkg.flux.shape
    # model = bkg.model([0, 1])
    # assert model.shape == (2, bkg.flux.shape[1])
    # model = bkg.model()
    # assert model.shape == bkg.flux.shape
    assert bkg.mask.sum() != 0
