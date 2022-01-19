from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from patsy import dmatrix
from scipy import sparse


@dataclass
class Estimator:
    """Background Estimator for Kepler/K2

    Parameters
    ----------

    time: np.ndarray
        1D array of times for each frame, shape ntimes
    row: np.ndarray
        1D array of row positions for pixels to calculate the background model of with shape npixels
    column: np.ndarray
        1D array of column positions for pixels to calculate the background model of with shape npixels
    flux : np.ndarray
        2D array of fluxes with shape ntimes x npixels
    """

    time: np.ndarray
    row: np.ndarray
    column: np.ndarray
    flux: np.ndarray

    def __post_init__(self):
        s = np.argsort(self.time)
        self.time, self.flux = self.time[s], self.flux[s]
        self.xknots, self.tknots = (
            np.linspace(20, 1108, 30)[1:-1],
            np.linspace(self.time[0], self.time[-1], 240),
        )
        med_flux = np.median(self.flux, axis=0)[None, :]
        f = self.flux - med_flux
        # Mask out pixels that are particularly bright.
        self.mask = (med_flux[0] - np.percentile(med_flux, 20)) < 30
        if not self.mask.any():
            raise ValueError("All the input pixels are brighter than 30 counts.")
        self.mask &= ~sigma_clip(med_flux[0]).mask
        self.mask &= ~sigma_clip(np.std(f, axis=0)).mask
        self.bf = np.asarray(
            [
                np.median(f[:, self.mask & (self.row == r1)], axis=1)
                for r1 in np.unique(self.row)
            ]
        )

        A1 = self._make_A(np.unique(self.row), self.time)
        self.bad_frames = (
            np.where(np.gradient(np.mean(self.bf, axis=0), axis=0) > 2)[0] + 1
        )
        if len(self.bad_frames) > 0:
            badA = sparse.vstack(
                [
                    sparse.csr_matrix(
                        (
                            np.in1d(np.arange(len(self.time)), b)
                            * np.ones(self.bf.shape, bool)
                        ).ravel()
                    )
                    for b in self.bad_frames
                ]
            ).T
            self.A = sparse.hstack([A1, badA])
        else:
            self.A = A1
        prior_sigma = np.ones(self.A.shape[1]) * 100
        sigma_w_inv = self.A.T.dot(self.A) + np.diag(1 / prior_sigma ** 2)
        B = self.A.T.dot(self.bf.ravel())
        self.w = np.linalg.solve(sigma_w_inv, B)

        self._model = self.A.dot(self.w).reshape(self.bf.shape)
        self.model = self._model[np.unique(self.row, return_inverse=True)[1]].T

    @staticmethod
    def from_mission_bkg(fname):
        hdu = fits.open(fname)
        self = Estimator(
            hdu[2].data["RAWX"],
            hdu[2].data["RAWY"],
            hdu[1].data["FLUX"],
        )
        return self

    def __repr__(self):
        return "KBackground.Estimator"

    @property
    def shape(self):
        return self.flux.shape

    def _make_A(self, x, t):
        """Makes a reasonable design matrix for the rolling band."""
        x_spline = sparse.csr_matrix(
            np.asarray(
                dmatrix(
                    "bs(x, knots=knots, degree=3, include_intercept=True)",
                    {"x": np.hstack([0, x, 1400]), "knots": self.xknots},
                )
            )
        )[1:-1]

        t_spline = sparse.csr_matrix(
            np.asarray(
                dmatrix(
                    "bs(x, knots=knots, degree=3, include_intercept=True)",
                    {"x": t, "knots": self.tknots},
                )
            )
        )
        s = x_spline.shape[0] * t_spline.shape[0]
        A1 = sparse.hstack(
            [
                x_spline[:, idx].multiply(t_spline[:, jdx].T).reshape((s, 1))
                for idx in range(x_spline.shape[1])
                for jdx in range(t_spline.shape[1])
            ],
            "csr",
        )
        return A1
