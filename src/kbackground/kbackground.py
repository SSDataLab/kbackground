import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from patsy import dmatrix
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

log = logging.getLogger(__name__)


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
    mask: Optional np.ndarray
        Mask which is `True` where pixels are "faint" background pixels, `False` where pixels contain sources.
    """

    time: np.ndarray
    row: np.ndarray
    column: np.ndarray
    flux: np.ndarray
    tknotspacing: int = 4
    xknotspacing: int = 6
    mask: Optional[np.ndarray] = None

    def __post_init__(self):
        log.info(
            f"new `kbackground` Object. tknotspacing:{self.tknotspacing}, xknotspacing:{self.tknotspacing}"
        )
        log.info(f"ntimes x npixels : {self.flux.shape}")
        s = np.argsort(self.time)
        self.time, self.flux = self.time[s], self.flux[s]
        self.xknots = np.arange(20, 1108, self.xknotspacing) + 0.5
        if np.median(np.diff(self.time)) < 0.03:
            # Time in JD
            self.tknots = np.arange(
                self.time[0], self.time[-1], self.tknotspacing / 48
            ) + 0.5 * (self.tknotspacing / 48)
        else:
            self.tknots = (
                np.arange(self.time[0], self.time[-1], self.tknotspacing) + 0.5
            )
        time_corr = np.nanpercentile(self.flux, 20, axis=1)[:, None]
        med_flux = np.median(self.flux - time_corr, axis=0)[None, :]

        f = self.flux - med_flux
        f -= np.median(f)

        self.bad_frames = []
        if self.mask is None:
            # Mask out pixels that are particularly bright.
            self.mask = (f - time_corr).std(axis=0) < 500
            if not self.mask.any():
                raise ValueError("All the input pixels are brighter than 500 counts.")
            self.mask &= (f - time_corr).std(axis=0) < 30
            # self.mask=(med_flux[0] - np.percentile(med_flux, 20)) < 30
            self.mask &= ~sigma_clip(med_flux[0]).mask
            self.mask &= ~sigma_clip(np.std(f - time_corr, axis=0)).mask

        ratio = (self.flux[:100].mean(axis=0) - self.flux[-100:].mean(axis=0)) / (
            self.flux.mean(axis=0)
        )
        self.mask &= ~sigma_clip(np.ma.masked_array(ratio, ~self.mask), sigma=2.5).mask

        self.unq_row = np.unique(self.row)
        log.debug(
            f"unq_row : {self.unq_row.min()} ... {self.unq_row.max()} ({len(self.unq_row)} unique rows)"
        )
        log.debug("Binning flux data")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.bf = np.asarray(
                [
                    np.median(f[:, self.mask & (self.row == r1)], axis=1)
                    for r1 in self.unq_row
                ]
            )
        log.debug("Binned flux data")
        log.debug(f"`bf` {self.bf.shape}")

        breaks = (
            np.where(np.diff(self.time) > 10 * np.median(np.diff(self.time)))[0] + 1
        )
        log.info(f"{len(breaks)} break points")
        log.debug(f"{breaks}")

        def get_masks(xlim1, xlim2):
            x = np.arange(xlim1, xlim2, self.tknotspacing * 30)
            a = np.max([x ** 0 * xlim1 - self.tknotspacing * 5, x], axis=0)
            b = np.min([x ** 0 * xlim2, x + self.tknotspacing * 35], axis=0)
            masks = np.asarray(
                [np.in1d(self.time, self.time[a1:b1]) for a1, b1 in np.vstack([a, b]).T]
            )
            if masks.shape[0] >= 2:
                masks[-2] |= masks[-1]
            return masks[:-1]

        time_masks = []
        for xlim1, xlim2 in zip(
            np.hstack([0, breaks]), np.hstack([breaks, len(self.time)])
        ):
            time_masks.append(get_masks(xlim1, xlim2))
        time_masks = np.vstack(time_masks)

        self.model = np.zeros_like(self.flux) * np.nan
        self.weights = np.zeros_like(self.flux)
        self._model = np.zeros_like(self.bf)
        for tdx, tm in enumerate(time_masks):
            log.info(f"Fitting model for segment {tdx + 1}/{len(time_masks)}")
            log.debug(f"{tdx + 1}/{len(time_masks)} Making A")
            A1 = self._make_A(self.unq_row, self.time[tm])
            log.debug(f"{tdx + 1}/{len(time_masks)} Made A")
            bad_frames = (
                np.where(
                    sigma_clip(
                        np.diff(np.nanmean(self.bf[:, tm], axis=0)),
                        sigma_upper=5,
                        sigma_lower=1e8,
                    ).mask
                )[0]
                + 1
            )
            log.debug(
                f"{tdx + 1}/{len(time_masks)} {len(bad_frames)}/{len(self.time[tm])} bad frames"
            )

            if len(bad_frames) > 0:
                log.debug(f"{tdx + 1}/{len(time_masks)} Building bad frame offsets")
                badA = sparse.vstack(
                    [
                        sparse.csr_matrix(
                            (
                                np.in1d(np.arange(len(self.time[tm])), b)
                                * np.ones(self.bf[:, tm].shape, bool)
                            ).ravel()
                        )
                        for b in bad_frames
                    ]
                ).T
                log.debug(f"{tdx + 1}/{len(time_masks)} Appending bad frame offsets")
                self.A = sparse.hstack([A1, badA], "csr")
                log.debug(f"{tdx + 1}/{len(time_masks)} Appended bad frame offsets")
            else:
                self.A = A1.tocsr()
            prior_sigma = np.ones(self.A.shape[1]) * 100
            k = np.isfinite(self.bf[:, tm].ravel())
            log.debug(f"{tdx + 1}/{len(time_masks)} Creating `sigma_w_inv`")
            sigma_w_inv = self.A[k].T.dot(self.A[k]) + np.diag(1 / prior_sigma ** 2)
            log.debug(f"{tdx + 1}/{len(time_masks)} Created `sigma_w_inv`")
            log.debug(f"{tdx + 1}/{len(time_masks)} Creating `B`")
            B = self.A[k].T.dot(self.bf[:, tm].ravel()[k])
            log.debug(f"{tdx + 1}/{len(time_masks)} Created `B`")
            log.debug(f"{tdx + 1}/{len(time_masks)} Solving for `w`")
            # self.w = np.linalg.solve(sigma_w_inv, B)
            self.w = cho_solve(cho_factor(sigma_w_inv), B)
            log.debug(f"{tdx + 1}/{len(time_masks)} Solved for `w`")

            self._model[:, tm] = self.A.dot(self.w).reshape(self.bf[:, tm].shape)
            log.debug(f"{tdx + 1}/{len(time_masks)} Building full model")
            model = np.zeros((tm.sum(), self.flux.shape[1])) * np.nan
            for idx, u in enumerate(self.unq_row):
                model[:, self.row == u] = self._model[:, tm][idx][:, None]
            self.model[tm] = model
            self.weights[tm] = np.ones(model.shape)
            self.bad_frames.append(np.where(tm)[0][bad_frames])
        if len(self.bad_frames) != 0:
            self.bad_frames = np.hstack(self.bad_frames)
        self.model = self.model / self.weights
        self.model -= np.median(self.model)
        log.debug("Built")

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
            fig, axs = plt.subplots(1, 3, figsize=(15, 3.5))

            for r, bf, mod in zip(rs, bfs, mods):
                im1 = axs[0].pcolormesh(
                    self.time,
                    r,
                    bf,
                    vmin=v[0],
                    vmax=v[1],
                    cmap="coolwarm",
                    rasterized=True,
                )
                im2 = axs[1].pcolormesh(
                    self.time,
                    r,
                    mod,
                    vmin=v[0],
                    vmax=v[1],
                    cmap="coolwarm",
                    rasterized=True,
                )
            cbar1 = plt.colorbar(im1, ax=axs[0], orientation="horizontal")
            cbar1.set_label(r"$\delta$ Flux [counts]")
            cbar2 = plt.colorbar(im2, ax=axs[1], orientation="horizontal")
            cbar2.set_label(r"$\delta$ Flux [counts]")
            axs[0].set(
                title="Column-wise Binned Flux Data", xlabel="Time", ylabel="Row"
            )
            axs[1].set(title="Column-wise Binned Model", xlabel="Time", ylabel="Row")

            axs[2].plot(
                self.time,
                np.nanmean(self.bf, axis=0),
                c="k",
                label="Data",
                rasterized=True,
            )
            axs[2].scatter(
                self.time[self.bad_frames],
                np.nanmean(self.bf, axis=0)[self.bad_frames],
                marker="x",
                c="r",
                label="Bad Frames",
                rasterized=True,
            )
            axs[2].legend()
            axs[2].set(xlabel="Time", ylabel="Average Flux")

        return fig

    def _make_A(self, x, t):
        """Makes a reasonable design matrix for the rolling band."""
        x_spline = sparse.csr_matrix(
            np.asarray(
                dmatrix(
                    "bs(x, knots=knots, degree=3, include_intercept=True)",
                    {"x": np.hstack([0, x, 1400]), "knots": self.xknots + 1e-10},
                )
            )
        )[1:-1]
        x_spline = x_spline[:, np.asarray((x_spline.sum(axis=0) != 0))[0]]
        log.debug(f"Made `x_spline` {x_spline.shape}")
        t_spline = sparse.csr_matrix(
            np.asarray(
                dmatrix(
                    "bs(x, knots=knots, degree=3, include_intercept=True)",
                    {
                        "x": np.hstack([self.time[0], t, self.time[-1]]),
                        "knots": self.tknots + 1e-10,
                    },
                )
            )
        )[1:-1]
        t_spline = t_spline[:, np.asarray((t_spline.sum(axis=0) != 0))[0]]
        log.debug(f"Made `t_spline` {t_spline.shape}")
        X = (
            sparse.hstack([x_spline] * t_spline.shape[0])
            .reshape((x_spline.shape[0] * t_spline.shape[0], x_spline.shape[1]))
            .tocsr()
        )
        log.debug(f"Made `X` {X.shape}")
        T = sparse.vstack([t_spline] * x_spline.shape[0])
        log.debug(f"Made `T` {T.shape}")
        A1 = sparse.hstack([X[:, idx].multiply(T) for idx in range(X.shape[1])]).tocsr()
        log.debug("Made `A1`")
        A1 = A1[:, np.asarray((A1.sum(axis=0) != 0))[0]]
        log.debug("Cleaned `A1`")
        return A1
