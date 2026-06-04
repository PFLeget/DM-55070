#!/usr/bin/env python
"""
Analyze PSF size residuals (dT/T) vs local background for DM-55070.

For each visit: 3-panel plot (dT/T focal plane, background focal plane, dT/T vs background scatter)
Across all visits: meanified dT/T map and background map
"""

import numpy as np
import treegp
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars

import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCam
from lsst.daf.butler import Butler
import argparse


camera = LsstCam.getCamera()

import matplotlib.gridspec as gridspec

PARQUET_COLUMNS = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
    'slot_Centroid_x', 'slot_Centroid_y',
    'detector', 'psf_background_value', 'psf_max_value', 'calib_psf_reserved',
    'base_GaussianFlux_instFlux', 'base_GaussianFlux_instFluxErr',
]


def load_visit_data(parquet_path):
    """Load visit data from parquet file and compute derived columns."""
    table = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()

    slot_Shape_xx = table['slot_Shape_xx'].to_numpy()
    slot_Shape_yy = table['slot_Shape_yy'].to_numpy()
    slot_PsfShape_xx = table['slot_PsfShape_xx'].to_numpy()
    slot_PsfShape_yy = table['slot_PsfShape_yy'].to_numpy()

    T_src = slot_Shape_xx + slot_Shape_yy
    T_psf = slot_PsfShape_xx + slot_PsfShape_yy

    # Compute FWHM in arcsec from PSF moments
    # sigma = sqrt(T/2), FWHM = 2.355 * sigma, pixel_scale = 0.2 arcsec/pixel
    pixel_scale = 0.2  # arcsec/pixel
    sigma_psf = np.sqrt(T_psf / 2.0)
    fwhm_arcsec = 2.355 * sigma_psf * pixel_scale

    flux = table['base_GaussianFlux_instFlux'].to_numpy()
    flux_err = table['base_GaussianFlux_instFluxErr'].to_numpy()
    snr = flux / flux_err

    return {
        'dT_T': (T_src - T_psf) / T_psf,
        'fwhm_arcsec': fwhm_arcsec,
        'xCCD': table['slot_Centroid_x'].to_numpy(),
        'yCCD': table['slot_Centroid_y'].to_numpy(),
        'detector': table['detector'].to_numpy(),
        'psf_background_value': table['psf_background_value'].to_numpy(),
        'psf_max_value': table['psf_max_value'].to_numpy(),
        'calib_psf_reserved': table['calib_psf_reserved'].to_numpy(),
        'snr': snr,
    }


def pixel_to_focal(x, y, det):
    """Convert pixel coordinates to focal plane coordinates (mm)."""
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()


def get_focal_plane_coords(data):
    """Add focal plane coordinates to data dict."""
    fpx_all = []
    fpy_all = []
    for i in range(len(data['xCCD'])):
        det = camera[data['detector'][i]]
        fpx, fpy = pixel_to_focal(
            np.array([data['xCCD'][i]]),
            np.array([data['yCCD'][i]]),
            det
        )
        fpx_all.append(fpx[0])
        fpy_all.append(fpy[0])
    data['fpx'] = np.array(fpx_all)
    data['fpy'] = np.array(fpy_all)
    return data


def plot_visit_panels(visit, data, repOutPlot, snr_min=50, dT_T_scale=0.05, bkg_scale=None):
    """
    Make 3-panel plot for a single visit:
    - Left: dT/T in focal plane
    - Middle: background in focal plane
    - Right: dT/T vs background scatter
    """
    filt = (data['snr'] > snr_min) & np.isfinite(data['dT_T']) & np.isfinite(data['psf_background_value'])

    fpx = data['fpx'][filt]
    fpy = data['fpy'][filt]
    dT_T = data['dT_T'][filt]
    bkg = data['psf_background_value'][filt]

    if len(fpx) == 0:
        print(f"  Visit {visit}: no valid data after filtering")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: dT/T focal plane
    sc1 = axes[0].scatter(fpx, fpy, c=dT_T, s=1, cmap='RdBu_r', vmin=-dT_T_scale, vmax=dT_T_scale)
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    axes[0].set_title(f'dT/T')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], label='dT/T')

    # Middle: background focal plane
    if bkg_scale is None:
        bkg_vmin, bkg_vmax = np.percentile(bkg[np.isfinite(bkg)], [5, 95])
    else:
        bkg_vmin, bkg_vmax = -bkg_scale, bkg_scale
    sc2 = axes[1].scatter(fpx, fpy, c=bkg, s=1, cmap='viridis', vmin=bkg_vmin, vmax=bkg_vmax)
    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('y (mm)')
    axes[1].set_title('Background (e-/pixel)')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], label='e-/pixel')

    # Right: dT/T vs background scatter
    axes[2].scatter(bkg, dT_T, s=1, alpha=0.3)
    axes[2].set_xlabel('Background (e-/pixel)')
    axes[2].set_ylabel('dT/T')
    axes[2].set_title('dT/T vs Background')
    axes[2].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[2].set_ylim(-dT_T_scale, dT_T_scale)

    fig.suptitle(f'Visit {visit}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(repOutPlot, f'visit_{visit}_dT_T_vs_background.png'), dpi=150)
    plt.close(fig)


def make_meanified_maps(all_data, repOutPlot, bin_spacing=150, snr_min=50,
                        dT_T_scale=0.05, bkg_scale=None):
    """
    Create meanified focal plane maps for dT/T and background across all visits.
    """
    meanify_dT_T = {}
    meanify_bkg = {}

    for visit, data in tqdm(all_data.items(), desc="Building meanified maps"):
        filt = (data['snr'] > snr_min) & np.isfinite(data['dT_T']) & np.isfinite(data['psf_background_value'])

        ccdIds = set(data['detector'][filt])

        for ccd in ccdIds:
            ccd_filt = filt & (data['detector'] == ccd)
            coord = np.array([data['xCCD'][ccd_filt], data['yCCD'][ccd_filt]]).T

            if ccd not in meanify_dT_T:
                meanify_dT_T[ccd] = treegp.meanify(bin_spacing=bin_spacing, statistics="mean",
                                                    bounds=(0, 4100, 0, 4100))
                meanify_bkg[ccd] = treegp.meanify(bin_spacing=bin_spacing, statistics="mean",
                                                   bounds=(0, 4100, 0, 4100))

            meanify_dT_T[ccd].add_field(coord, data['dT_T'][ccd_filt])
            meanify_bkg[ccd].add_field(coord, data['psf_background_value'][ccd_filt])

    for ccd in meanify_dT_T:
        meanify_dT_T[ccd].meanify()
        meanify_bkg[ccd].meanify()

    ccdIds = list(meanify_dT_T.keys())

    # Compute color scales
    dT_T_values = np.concatenate([meanify_dT_T[ccd]._average.ravel() for ccd in ccdIds])
    dT_T_values = dT_T_values[np.isfinite(dT_T_values)]

    bkg_values = np.concatenate([meanify_bkg[ccd]._average.ravel() for ccd in ccdIds])
    bkg_values = bkg_values[np.isfinite(bkg_values)]

    if bkg_scale is None:
        bkg_vmin, bkg_vmax = np.percentile(bkg_values, [5, 95])
    else:
        bkg_vmin, bkg_vmax = -bkg_scale, bkg_scale

    # Plot dT/T map
    fig, ax = plt.subplots(figsize=(12, 10))
    for ccd in ccdIds:
        x, y = np.meshgrid(meanify_dT_T[ccd]._xedge, meanify_dT_T[ccd]._yedge)
        nBin0, nBin1 = x.shape
        x = x.reshape(nBin0 * nBin1)
        y = y.reshape(nBin0 * nBin1)
        x, y = pixel_to_focal(x, y, camera[ccd])
        x = x.reshape((nBin0, nBin1))
        y = y.reshape((nBin0, nBin1))
        ax.pcolormesh(x, y, meanify_dT_T[ccd]._average, vmin=-dT_T_scale, vmax=dT_T_scale, cmap='RdBu_r')

    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label('dT/T', fontsize=14)
    ax.set_xlabel('x (mm)', fontsize=14)
    ax.set_ylabel('y (mm)', fontsize=14)
    ax.set_title('Meanified dT/T across all visits', fontsize=14)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(repOutPlot, 'meanified_dT_T.png'), dpi=150)
    plt.close(fig)

    # Plot background map
    fig, ax = plt.subplots(figsize=(12, 10))
    for ccd in ccdIds:
        x, y = np.meshgrid(meanify_bkg[ccd]._xedge, meanify_bkg[ccd]._yedge)
        nBin0, nBin1 = x.shape
        x = x.reshape(nBin0 * nBin1)
        y = y.reshape(nBin0 * nBin1)
        x, y = pixel_to_focal(x, y, camera[ccd])
        x = x.reshape((nBin0, nBin1))
        y = y.reshape((nBin0, nBin1))
        ax.pcolormesh(x, y, meanify_bkg[ccd]._average, vmin=bkg_vmin, vmax=bkg_vmax, cmap='viridis')

    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label('Background (e-/pixel)', fontsize=14)
    ax.set_xlabel('x (mm)', fontsize=14)
    ax.set_ylabel('y (mm)', fontsize=14)
    ax.set_title('Meanified background across all visits', fontsize=14)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(repOutPlot, 'meanified_background.png'), dpi=150)
    plt.close(fig)

    print(f"Saved meanified maps to {repOutPlot}")


class meanify1D_wrms:
    """
    Take data, build a 1D average with weighted RMS.
    O(1) memory implementation - keeps running sum/count per bin.
    """
    def __init__(self, bin_spacing=0.3, x_min=0, x_max=1000):
        self.bin_spacing = bin_spacing
        self.x_min = x_min
        self.x_max = x_max

        self.nbin = int((x_max - x_min) / bin_spacing) + 1
        self.binning = np.linspace(x_min, x_max, self.nbin)

        self._sum = np.zeros(self.nbin - 1)
        self._sum_sq = np.zeros(self.nbin - 1)
        self._count = np.zeros(self.nbin - 1)

        self.x0 = self.binning[:-1] + (self.binning[1] - self.binning[0]) / 2.0

    def add_data(self, coord, param):
        """Add new data - accumulates directly into bins."""
        valid = np.isfinite(coord) & np.isfinite(param)
        coord = coord[valid]
        param = param[valid]

        bin_indices = np.digitize(coord, self.binning) - 1

        valid_bins = (bin_indices >= 0) & (bin_indices < self.nbin - 1)
        bin_indices = bin_indices[valid_bins]
        param = param[valid_bins]

        np.add.at(self._sum, bin_indices, param)
        np.add.at(self._sum_sq, bin_indices, param ** 2)
        np.add.at(self._count, bin_indices, 1)

    def meanify(self):
        """Compute final statistics from accumulated sums."""
        with np.errstate(divide='ignore', invalid='ignore'):
            self.average = self._sum / self._count
            variance = (self._sum_sq / self._count) - (self.average ** 2)
            self.std = np.sqrt(variance)
            self.count = self._count


def plot_background_vs_psfmax(all_data, repOutPlot, band, snr_min=50,
                               x_min=500, x_max=80000, bin_spacing=2000):
    """
    Plot background as a function of PSF max value.
    Top panel: histogram of psf_max_value
    Bottom panel: binned average of background vs psf_max_value
    """
    meanify = meanify1D_wrms(bin_spacing=bin_spacing, x_min=x_min, x_max=x_max)

    all_psfmax = []
    all_bkg = []

    for visit, data in all_data.items():
        filt = ((data['snr'] > snr_min)
                & np.isfinite(data['psf_max_value'])
                & np.isfinite(data['psf_background_value']))

        psfmax = data['psf_max_value'][filt]
        bkg = data['psf_background_value'][filt]

        all_psfmax.append(psfmax)
        all_bkg.append(bkg)
        meanify.add_data(psfmax, bkg)

    meanify.meanify()

    all_psfmax = np.concatenate(all_psfmax)
    all_bkg = np.concatenate(all_bkg)

    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.12, bottom=0.08, top=0.95, right=0.95, hspace=0)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # Top panel: psf_max_value distribution
    ax1 = plt.subplot(gs[0])
    ax1.hist(all_psfmax, bins=meanify.binning, color='b', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_yscale('log')
    ax1.set_ylabel('# stars', fontsize=14)
    ax1.set_xlim(x_min, x_max)
    ax1.set_xscale('log')
    ax1.tick_params(labelbottom=False)
    ax1.set_title(f"Band: {band} | N_visits: {len(all_data)} | N_stars: {len(all_psfmax):,}", fontsize=14)

    # Bottom panel: background vs psf_max_value
    ax2 = plt.subplot(gs[1])

    valid_bins = np.isfinite(meanify.average)
    ax2.scatter(meanify.x0[valid_bins], meanify.average[valid_bins], s=50, c='b', zorder=3, label='Binned mean')
    ax2.errorbar(meanify.x0[valid_bins], meanify.average[valid_bins],
                 yerr=meanify.std[valid_bins] / np.sqrt(meanify.count[valid_bins]),
                 fmt='none', c='b', capsize=3, zorder=2)
    ax2.plot([x_min, x_max], [0,0], 'k--')

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(-5, 5)
    ax2.set_xscale('log')
    ax2.set_xlabel('PSF max value (e$^-$)', fontsize=14)
    ax2.set_ylabel('Background (e$^-$/pixel)', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)

    output_file = os.path.join(repOutPlot, f'background_vs_psfmax_{band}.png')
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_file}")


def plot_dT_T_vs_background(all_data, repOutPlot, band, snr_min=50,
                             x_min=-5, x_max=5, bin_spacing=0.5, psf_max_value=30000):
    """
    Plot dT/T as a function of background.
    Top panel: histogram of background
    Bottom panel: binned average of dT/T vs background
    """
    meanify = meanify1D_wrms(bin_spacing=bin_spacing, x_min=x_min, x_max=x_max)

    all_bkg = []
    all_dT_T = []

    for visit, data in all_data.items():
        filt = ((data['snr'] > snr_min)
                & np.isfinite(data['dT_T'])
                & np.isfinite(data['psf_background_value'])
                & (data['psf_max_value'] < psf_max_value) )

        bkg = data['psf_background_value'][filt]
        dT_T = data['dT_T'][filt]

        all_bkg.append(bkg)
        all_dT_T.append(dT_T)
        meanify.add_data(bkg, dT_T)

    meanify.meanify()

    all_bkg = np.concatenate(all_bkg)
    all_dT_T = np.concatenate(all_dT_T)

    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.12, bottom=0.08, top=0.95, right=0.95, hspace=0)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # Top panel: background distribution
    ax1 = plt.subplot(gs[0])
    ax1.hist(all_bkg, bins=meanify.binning, color='b', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_yscale('log')
    ax1.set_ylabel('# stars', fontsize=14)
    ax1.set_xlim(x_min, x_max)
    ax1.tick_params(labelbottom=False)
    ax1.set_title(f"Band: {band} | N_visits: {len(all_data)} | N_stars: {len(all_bkg):,}", fontsize=14)

    # Bottom panel: dT/T vs background
    ax2 = plt.subplot(gs[1])

    valid_bins = np.isfinite(meanify.average)
    ax2.scatter(meanify.x0[valid_bins], meanify.average[valid_bins], s=50, c='b', zorder=3, label='Binned mean')
    ax2.errorbar(meanify.x0[valid_bins], meanify.average[valid_bins],
                 yerr=meanify.std[valid_bins] / np.sqrt(meanify.count[valid_bins]),
                 fmt='none', c='b', capsize=3, zorder=2)

    ax2.axhline(0, color='k', linestyle='--', linewidth=1, zorder=1)
    xlim = (x_min, x_max)
    ax2.fill_between(xlim, -0.004, 0.004, color='g', alpha=0.2, zorder=0, label='0.4% requirement')
    ax2.fill_between(xlim, -0.001, 0.001, color='g', alpha=0.3, zorder=0, label='0.1% goal')

    ax2.set_xlim(xlim)
    ax2.set_ylim(-0.02, 0.02)
    ax2.set_xlabel('Background (e$^-$/pixel)', fontsize=14)
    ax2.set_ylabel('$\\langle \\delta T / T \\rangle$', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)

    output_file = os.path.join(repOutPlot, f'dT_T_vs_background_{band}.png')
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_file}")


def compute_fwhm_bins(all_data, snr_min=50, n_bins=3):
    """
    Compute FWHM percentile boundaries to split data into n_bins equal slices.

    Returns
    -------
    fwhm_edges : array of shape (n_bins + 1,)
        FWHM bin edges in arcsec
    """
    all_fwhm = []
    for visit, data in all_data.items():
        filt = ((data['snr'] > snr_min)
                & np.isfinite(data['fwhm_arcsec'])
                & np.isfinite(data['psf_background_value']))
        all_fwhm.append(data['fwhm_arcsec'][filt])

    all_fwhm = np.concatenate(all_fwhm)
    percentiles = np.linspace(0, 100, n_bins + 1)
    fwhm_edges = np.percentile(all_fwhm, percentiles)
    return fwhm_edges


def plot_dT_T_vs_background_by_iq(all_data, repOutPlot, band, snr_min=50,
                                   x_min=-5, x_max=5, bin_spacing=0.5, n_iq_bins=3):
    """
    Plot dT/T as a function of background, split by IQ (FWHM) slices.
    Creates one plot per IQ slice.
    """
    fwhm_edges = compute_fwhm_bins(all_data, snr_min=snr_min, n_bins=n_iq_bins)
    print(f"FWHM bin edges (arcsec): {fwhm_edges}")

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    for i_iq in range(n_iq_bins):
        fwhm_lo = fwhm_edges[i_iq]
        fwhm_hi = fwhm_edges[i_iq + 1]

        meanify = meanify1D_wrms(bin_spacing=bin_spacing, x_min=x_min, x_max=x_max)

        all_bkg = []
        all_dT_T = []
        n_stars = 0

        for visit, data in all_data.items():
            filt = ((data['snr'] > snr_min)
                    & np.isfinite(data['dT_T'])
                    & np.isfinite(data['psf_background_value'])
                    & np.isfinite(data['fwhm_arcsec'])
                    & (data['fwhm_arcsec'] >= fwhm_lo)
                    & (data['fwhm_arcsec'] < fwhm_hi))

            bkg = data['psf_background_value'][filt]
            dT_T = data['dT_T'][filt]

            all_bkg.append(bkg)
            all_dT_T.append(dT_T)
            meanify.add_data(bkg, dT_T)
            n_stars += np.sum(filt)

        meanify.meanify()

        all_bkg = np.concatenate(all_bkg)
        all_dT_T = np.concatenate(all_dT_T)

        fig = plt.figure(figsize=(12, 10))
        plt.subplots_adjust(left=0.12, bottom=0.08, top=0.95, right=0.95, hspace=0)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

        # Top panel: background distribution
        ax1 = plt.subplot(gs[0])
        ax1.hist(all_bkg, bins=meanify.binning, color=colors[i_iq % len(colors)],
                 alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_yscale('log')
        ax1.set_ylabel('# stars', fontsize=14)
        ax1.set_xlim(x_min, x_max)
        ax1.tick_params(labelbottom=False)
        ax1.set_title(f"Band: {band} | FWHM: [{fwhm_lo:.2f}, {fwhm_hi:.2f}] arcsec | N_stars: {n_stars:,}",
                      fontsize=14)

        # Bottom panel: dT/T vs background
        ax2 = plt.subplot(gs[1])

        valid_bins = np.isfinite(meanify.average)
        ax2.scatter(meanify.x0[valid_bins], meanify.average[valid_bins], s=50,
                    c=colors[i_iq % len(colors)], zorder=3, label='Binned mean')
        ax2.errorbar(meanify.x0[valid_bins], meanify.average[valid_bins],
                     yerr=meanify.std[valid_bins] / np.sqrt(meanify.count[valid_bins]),
                     fmt='none', c=colors[i_iq % len(colors)], capsize=3, zorder=2)

        ax2.axhline(0, color='k', linestyle='--', linewidth=1, zorder=1)
        xlim = (x_min, x_max)
        ax2.fill_between(xlim, -0.004, 0.004, color='g', alpha=0.2, zorder=0, label='0.4% requirement')
        ax2.fill_between(xlim, -0.001, 0.001, color='g', alpha=0.3, zorder=0, label='0.1% goal')

        ax2.set_xlim(xlim)
        ax2.set_ylim(-0.02, 0.02)
        ax2.set_xlabel('Background (e$^-$/pixel)', fontsize=14)
        ax2.set_ylabel('$\\langle \\delta T / T \\rangle$', fontsize=14)
        ax2.legend(loc='upper right', fontsize=10)

        output_file = os.path.join(repOutPlot, f'dT_T_vs_background_{band}_iq{i_iq}.png')
        plt.savefig(output_file, dpi=150)
        plt.close(fig)
        print(f"Saved: {output_file}")


def plot_background_vs_psfmax_by_iq(all_data, repOutPlot, band, snr_min=50,
                                     x_min=500, x_max=80000, bin_spacing=2000, n_iq_bins=3):
    """
    Plot background as a function of PSF max value, split by IQ (FWHM) slices.
    Creates one plot per IQ slice.
    """
    fwhm_edges = compute_fwhm_bins(all_data, snr_min=snr_min, n_bins=n_iq_bins)

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    for i_iq in range(n_iq_bins):
        fwhm_lo = fwhm_edges[i_iq]
        fwhm_hi = fwhm_edges[i_iq + 1]

        meanify = meanify1D_wrms(bin_spacing=bin_spacing, x_min=x_min, x_max=x_max)

        all_psfmax = []
        all_bkg = []
        n_stars = 0

        for visit, data in all_data.items():
            filt = ((data['snr'] > snr_min)
                    & np.isfinite(data['psf_max_value'])
                    & np.isfinite(data['psf_background_value'])
                    & np.isfinite(data['fwhm_arcsec'])
                    & (data['fwhm_arcsec'] >= fwhm_lo)
                    & (data['fwhm_arcsec'] < fwhm_hi))

            psfmax = data['psf_max_value'][filt]
            bkg = data['psf_background_value'][filt]

            all_psfmax.append(psfmax)
            all_bkg.append(bkg)
            meanify.add_data(psfmax, bkg)
            n_stars += np.sum(filt)

        meanify.meanify()

        all_psfmax = np.concatenate(all_psfmax)
        all_bkg = np.concatenate(all_bkg)

        fig = plt.figure(figsize=(12, 10))
        plt.subplots_adjust(left=0.12, bottom=0.08, top=0.95, right=0.95, hspace=0)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

        # Top panel: psf_max_value distribution
        ax1 = plt.subplot(gs[0])
        ax1.hist(all_psfmax, bins=meanify.binning, color=colors[i_iq % len(colors)],
                 alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_yscale('log')
        ax1.set_ylabel('# stars', fontsize=14)
        ax1.set_xlim(x_min, x_max)
        ax1.set_xscale('log')
        ax1.tick_params(labelbottom=False)
        ax1.set_title(f"Band: {band} | FWHM: [{fwhm_lo:.2f}, {fwhm_hi:.2f}] arcsec | N_stars: {n_stars:,}",
                      fontsize=14)

        # Bottom panel: background vs psf_max_value
        ax2 = plt.subplot(gs[1])

        valid_bins = np.isfinite(meanify.average)
        ax2.scatter(meanify.x0[valid_bins], meanify.average[valid_bins], s=50,
                    c=colors[i_iq % len(colors)], zorder=3, label='Binned mean')
        ax2.errorbar(meanify.x0[valid_bins], meanify.average[valid_bins],
                     yerr=meanify.std[valid_bins] / np.sqrt(meanify.count[valid_bins]),
                     fmt='none', c=colors[i_iq % len(colors)], capsize=3, zorder=2)
        ax2.plot([x_min, x_max], [0, 0], 'k--')

        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(-5, 5)
        ax2.set_xscale('log')
        ax2.set_xlabel('PSF max value (e$^-$)', fontsize=14)
        ax2.set_ylabel('Background (e$^-$/pixel)', fontsize=14)
        ax2.legend(loc='upper right', fontsize=10)

        output_file = os.path.join(repOutPlot, f'background_vs_psfmax_{band}_iq{i_iq}.png')
        plt.savefig(output_file, dpi=150)
        plt.close(fig)
        print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze dT/T vs local background for DM-55070")
    parser.add_argument('--repo', type=str, default='dp2_prep', help='Butler repository')
    parser.add_argument('--collection', type=str,
                        default='u/leget/LSSTCam/DM-55070/cosmos_r_band_20260528_colorfit_enable',
                        help='Butler collection')
    parser.add_argument('--band', type=str, default='r', help='Band to analyze')
    parser.add_argument('--repOutPlot', type=str, default='plotsWithColor/', help='Output directory for plots')
    parser.add_argument('--bin_spacing', type=float, default=150, help='Bin spacing for meanify (pixels)')
    parser.add_argument('--snr_min', type=float, default=50, help='Minimum SNR threshold')
    parser.add_argument('--dT_T_scale', type=float, default=0.05, help='Color scale for dT/T')
    parser.add_argument('--bkg_scale', type=float, default=None, help='Color scale for background (auto if None)')
    parser.add_argument('--skip_per_visit', action='store_true', help='Skip per-visit plots')
    parser.add_argument('--max_visits', type=int, default=None, help='Max visits to process (for testing)')
    parser.add_argument('--bkg_x_min', type=float, default=-5, help='Min background for dT/T vs background plot')
    parser.add_argument('--bkg_x_max', type=float, default=5, help='Max background for dT/T vs background plot')
    parser.add_argument('--bkg_bin_spacing', type=float, default=0.5, help='Bin spacing for dT/T vs background plot')
    parser.add_argument('--psfmax_x_min', type=float, default=500, help='Min psf_max for background vs psfmax plot')
    parser.add_argument('--psfmax_x_max', type=float, default=60000, help='Max psf_max for background vs psfmax plot')
    parser.add_argument('--psfmax_bin_spacing', type=float, default=2000, help='Bin spacing for background vs psfmax plot')
    parser.add_argument('--n_iq_bins', type=int, default=3, help='Number of IQ (FWHM) bins for sliced plots')

    args = parser.parse_args()

    os.makedirs(args.repOutPlot, exist_ok=True)

    butler = Butler(args.repo, collections=args.collection)

    # Query all visits for the band
    #dsrefs = list(butler.registry.queryDatasets("refit_psf_star", band=args.band))
    #visits = sorted(set(dsr.dataId["visit"] for dsr in dsrefs))

    with open('visits_cosmos_r_dp2.txt', 'r') as file:
    # Read the file and split it into a list, automatically removing the newlines (\n)
        visits = file.read().splitlines()

    if args.max_visits:
        visits = visits[:args.max_visits]

    print(f"Found {len(visits)} visits for band {args.band}")

    all_data = {}

    for visit in tqdm(visits, desc="Loading visits"):
        uri = butler.getURI("refit_psf_star", instrument="LSSTCam", visit=int(visit))
        parquet_path = uri.geturl()

        data = load_visit_data(parquet_path)
        data = get_focal_plane_coords(data)
        all_data[visit] = data

        if not args.skip_per_visit:
            plot_visit_panels(visit, data, args.repOutPlot,
                              snr_min=args.snr_min, dT_T_scale=args.dT_T_scale, bkg_scale=args.bkg_scale)

    # Make meanified maps
    make_meanified_maps(all_data, args.repOutPlot, bin_spacing=args.bin_spacing,
                        snr_min=args.snr_min, dT_T_scale=args.dT_T_scale, bkg_scale=args.bkg_scale)

    # Make 1D binned plots
    plot_background_vs_psfmax(all_data, args.repOutPlot, args.band, snr_min=args.snr_min,
                               x_min=args.psfmax_x_min, x_max=args.psfmax_x_max,
                               bin_spacing=args.psfmax_bin_spacing)

    plot_dT_T_vs_background(all_data, args.repOutPlot, args.band, snr_min=args.snr_min,
                             x_min=args.bkg_x_min, x_max=args.bkg_x_max,
                             bin_spacing=args.bkg_bin_spacing)

    # Make IQ-sliced plots
    plot_dT_T_vs_background_by_iq(all_data, args.repOutPlot, args.band, snr_min=args.snr_min,
                                   x_min=args.bkg_x_min, x_max=args.bkg_x_max,
                                   bin_spacing=args.bkg_bin_spacing, n_iq_bins=args.n_iq_bins)

    plot_background_vs_psfmax_by_iq(all_data, args.repOutPlot, args.band, snr_min=args.snr_min,
                                     x_min=args.psfmax_x_min, x_max=args.psfmax_x_max,
                                     bin_spacing=args.psfmax_bin_spacing, n_iq_bins=args.n_iq_bins)

    print("Done!")


if __name__ == "__main__":
    main()
