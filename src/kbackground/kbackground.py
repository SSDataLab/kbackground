import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
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
    tknotspacing: int
        Spacing in cadences between time knots. Default is 10
    xknotspacing: int
        Spacing in pixels between row knots. Default is 20
    """

    time: np.ndarray
    row: np.ndarray
    column: np.ndarray
    flux: np.ndarray
    tknotspacing: int = 10
    xknotspacing: int = 20

    def __post_init__(self):
        s = np.argsort(self.time)
        self.time, self.flux = self.time[s], self.flux[s]
        self.xknots = np.arange(20, 1108, self.xknotspacing)
        if np.median(np.diff(self.time)) < 0.03:
            # Time in JD
            self.tknots = np.arange(self.time[0], self.time[-1], self.tknotspacing / 48)
        else:
            self.tknots = np.arange(self.time[0], self.time[-1], self.tknotspacing)
        time_corr = np.nanpercentile(self.flux, 20, axis=1)[:, None]
        med_flux = np.median(self.flux - time_corr, axis=0)[None, :]
        f = self.flux - med_flux
        # Mask out pixels that are particularly bright.
        self.mask = (f - time_corr).std(axis=0) < 500
        if not self.mask.any():
            raise ValueError("All the input pixels are brighter than 500 counts.")
        self.mask &= (f - time_corr).std(axis=0) < 30
        # self.mask=(med_flux[0] - np.percentile(med_flux, 20)) < 30
        self.mask &= ~sigma_clip(med_flux[0]).mask
        self.mask &= ~sigma_clip(np.std(f - time_corr, axis=0)).mask
        self.unq_row = np.unique(self.row)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.bf = np.asarray(
                [
                    np.mean(f[:, self.mask & (self.row == r1)], axis=1)
                    for r1 in self.unq_row
                ]
            )
        A1 = self._make_A(self.unq_row, self.time)

        self.bad_frames = (
            np.where(
                sigma_clip(
                    np.diff(np.mean(self.bf, axis=0)), sigma_upper=1, sigma_lower=np.inf
                ).mask
            )[0]
            + 1
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
            self.A = sparse.hstack([A1, badA], "csr")
        else:
            self.A = A1.tocsr()
        prior_sigma = np.ones(self.A.shape[1]) * 100
        k = np.isfinite(self.bf.ravel())
        sigma_w_inv = self.A[k].T.dot(self.A[k]) + np.diag(1 / prior_sigma ** 2)
        B = self.A[k].T.dot(self.bf.ravel()[k])
        self.w = np.linalg.solve(sigma_w_inv, B)

        self._model = self.A.dot(self.w).reshape(self.bf.shape)
        self.model = np.zeros((self.flux.shape)) * np.nan
        for idx, u in enumerate(self.unq_row):
            self.model[:, self.row == u] = self._model[idx][:, None]

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

    def plot(self):
        with plt.style.context("seaborn-white"):
            b = np.where(np.diff(self.unq_row) != 1)[0] + 1
            rs = np.array_split(self.unq_row, b)
            bfs = np.array_split(self.bf, b)
            mods = np.array_split(self._model, b)
            v = np.nanpercentile(self.bf, (5, 95))
            fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

            for r, bf, mod in zip(rs, bfs, mods):
                im1 = axs[0].pcolormesh(
                    self.time,
                    r,
                    bf,
                    vmin=v[0],
                    vmax=v[1],
                    cmap="coolwarm",
                )
                im2 = axs[1].pcolormesh(
                    self.time,
                    r,
                    mod,
                    vmin=v[0],
                    vmax=v[1],
                    cmap="coolwarm",
                )
            cbar1 = plt.colorbar(im1, ax=axs[0], orientation="horizontal")
            cbar1.set_label(r"$\delta$ Flux [counts]")
            cbar2 = plt.colorbar(im2, ax=axs[1], orientation="horizontal")
            cbar2.set_label(r"$\delta$ Flux [counts]")
            axs[0].set(
                title="Column-wise Binned Flux Data", xlabel="Time", ylabel="Row"
            )
            axs[1].set(title="Column-wise Binned Model", xlabel="Time", ylabel="Row")
        return fig

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
        X = (
            sparse.hstack([x_spline] * t_spline.shape[0])
            .reshape((x_spline.shape[0] * t_spline.shape[0], x_spline.shape[1]))
            .tocsr()
        )
        T = sparse.vstack([t_spline] * x_spline.shape[0])
        A1 = sparse.hstack([X[:, idx].multiply(T) for idx in range(X.shape[1])]).tocsr()
        A1 = A1[:, np.asarray((A1.sum(axis=0) != 0))[0]]

        return A1
