#!/usr/bin/env python
"""
Compute Rho statistics from refit_psf_star tables.

Supports both LSSTCam (dp2_prep) and HSC (main) repositories.
"""

import numpy as np
import treecorr
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars

import pickle
import argparse

from lsst.daf.butler import Butler


# Scale reference lines (arcmin)
LSSTCAM_CCD_SCALE = 13.3
LSSTCAM_FP_SCALE = 210.0
HSC_CCD_SCALE = 6.8
HSC_FP_SCALE = 90.0


PARQUET_COLUMNS = [
    'coord_ra', 'coord_dec', 'detector',
    'shape_Iuu', 'shape_Ivv', 'shape_Iuv',
    'psfShape_Iuu', 'psfShape_Ivv', 'psfShape_Iuv',
    'base_GaussianFlux_instFlux', 'base_GaussianFlux_instFluxErr',
    'calib_psf_used', 'calib_psf_reserved',
]


def load_visit_data(parquet_path, snr_min=None, snr_max=None):
    """Load visit data with sky coordinate moments.

    Parameters
    ----------
    parquet_path : str
        Path to parquet file
    snr_min : float or None
        Minimum SNR cut
    snr_max : float or None
        Maximum SNR cut

    Returns
    -------
    dict with keys for moments, coordinates, and flags for used/reserved
    """
    table = polars.scan_parquet(parquet_path).select(PARQUET_COLUMNS).collect()

    # Filter by SNR
    if snr_min is not None or snr_max is not None:
        flux = table['base_GaussianFlux_instFlux'].to_numpy()
        flux_err = table['base_GaussianFlux_instFluxErr'].to_numpy()
        snr = flux / flux_err
        mask = np.ones(len(snr), dtype=bool)
        if snr_min is not None:
            mask &= snr >= snr_min
        if snr_max is not None:
            mask &= snr <= snr_max
        table = table.filter(polars.Series(mask))

    return {
        'ixx': table['shape_Iuu'].to_numpy(),
        'iyy': table['shape_Ivv'].to_numpy(),
        'ixy': table['shape_Iuv'].to_numpy(),
        'ixx_psf': table['psfShape_Iuu'].to_numpy(),
        'iyy_psf': table['psfShape_Ivv'].to_numpy(),
        'ixy_psf': table['psfShape_Iuv'].to_numpy(),
        'ra': np.degrees(table['coord_ra'].to_numpy()),
        'dec': np.degrees(table['coord_dec'].to_numpy()),
        'calib_psf_used': table['calib_psf_used'].to_numpy().astype(bool),
        'calib_psf_reserved': table['calib_psf_reserved'].to_numpy().astype(bool),
    }


def compute_ellipticity(ixx, iyy, ixy, ellipticity_type='distortion'):
    """Compute ellipticity from second moments."""
    T = ixx + iyy
    if ellipticity_type == 'distortion':
        e1 = (ixx - iyy) / T
        e2 = 2 * ixy / T
    else:  # shear
        denom = T + 2 * np.sqrt(ixx * iyy - ixy**2)
        e1 = (ixx - iyy) / denom
        e2 = 2 * ixy / denom
    return e1, e2


def compute_rho_inputs(data, ellipticity_type='distortion'):
    """Compute the inputs needed for rho statistics."""
    e1, e2 = compute_ellipticity(data['ixx'], data['iyy'], data['ixy'], ellipticity_type)
    e1_psf, e2_psf = compute_ellipticity(data['ixx_psf'], data['iyy_psf'], data['ixy_psf'], ellipticity_type)

    T = data['ixx'] + data['iyy']
    T_psf = data['ixx_psf'] + data['iyy_psf']

    e1_res = e1 - e1_psf
    e2_res = e2 - e2_psf
    size_res = (T_psf - T) / T

    responsivity = 2.0 if ellipticity_type == 'distortion' else 1.0
    e1 /= responsivity
    e2 /= responsivity
    e1_res /= responsivity
    e2_res /= responsivity

    e1_size_res = e1 * size_res
    e2_size_res = e2 * size_res

    return {
        'ra': data['ra'],
        'dec': data['dec'],
        'e1': e1,
        'e2': e2,
        'e1_res': e1_res,
        'e2_res': e2_res,
        'size_res': size_res,
        'e1_size_res': e1_size_res,
        'e2_size_res': e2_size_res,
    }


def compute_rho_statistics(inputs, treecorr_config):
    """Compute all rho statistics."""
    ra, dec = inputs['ra'], inputs['dec']
    e1, e2 = inputs['e1'], inputs['e2']
    e1_res, e2_res = inputs['e1_res'], inputs['e2_res']
    size_res = inputs['size_res']
    e1_size_res, e2_size_res = inputs['e1_size_res'], inputs['e2_size_res']

    # Build catalogs
    cat_e = treecorr.Catalog(ra=ra, dec=dec, g1=e1, g2=e2, ra_units='deg', dec_units='deg')
    cat_de = treecorr.Catalog(ra=ra, dec=dec, g1=e1_res, g2=e2_res, ra_units='deg', dec_units='deg')
    cat_eT = treecorr.Catalog(ra=ra, dec=dec, g1=e1_size_res, g2=e2_size_res, ra_units='deg', dec_units='deg')
    cat_T = treecorr.Catalog(ra=ra, dec=dec, k=size_res, ra_units='deg', dec_units='deg')

    rho_stats = {}

    print("  Computing rho1: <de*, de>")
    rho1 = treecorr.GGCorrelation(config=treecorr_config)
    rho1.process(cat_de)
    rho_stats['rho1'] = rho1

    print("  Computing rho2: <e*, de>")
    rho2 = treecorr.GGCorrelation(config=treecorr_config)
    rho2.process(cat_e, cat_de)
    rho_stats['rho2'] = rho2

    print("  Computing rho3: <e*dT/T, e*dT/T>")
    rho3 = treecorr.GGCorrelation(config=treecorr_config)
    rho3.process(cat_eT)
    rho_stats['rho3'] = rho3

    print("  Computing rho4: <de*, e*dT/T>")
    rho4 = treecorr.GGCorrelation(config=treecorr_config)
    rho4.process(cat_de, cat_eT)
    rho_stats['rho4'] = rho4

    print("  Computing rho5: <e*, e*dT/T>")
    rho5 = treecorr.GGCorrelation(config=treecorr_config)
    rho5.process(cat_e, cat_eT)
    rho_stats['rho5'] = rho5

    print("  Computing rho3alt: <dT/T, dT/T>")
    rho3alt = treecorr.KKCorrelation(config=treecorr_config)
    rho3alt.process(cat_T)
    rho_stats['rho3alt'] = rho3alt

    return rho_stats


def plot_rho_statistics_combined(rho_stats_dict, output_file, title=None, instrument='LSSTCam'):
    """Plot all rho statistics with all/used/reserved overlaid.

    Parameters
    ----------
    rho_stats_dict : dict
        Dictionary with keys 'all', 'used', 'reserved', each containing rho stats
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    rho_labels = {
        'rho1': r"$\rho_{1}(\theta) = \langle \delta e, \delta e \rangle$",
        'rho2': r"$\rho_{2}(\theta) = \langle e, \delta e \rangle$",
        'rho3': r"$\rho_{3}(\theta) = \langle e\frac{\delta T}{T} , e\frac{\delta T}{T} \rangle$",
        'rho4': r"$\rho_{4}(\theta) = \langle \delta e, e\frac{\delta T}{T} \rangle$",
        'rho5': r"$\rho_{5}(\theta) = \langle e, e\frac{\delta T}{T} \rangle$",
        'rho3alt': r"$\rho'_{3}(\theta) = \langle \frac{\delta T}{T}, \frac{\delta T}{T}\rangle$",
    }

    colors = {'all': 'C0', 'used': 'C1', 'reserved': 'C2'}
    markers = {'all': 'o', 'used': 's', 'reserved': '^'}

    if instrument == 'LSSTCam':
        ccd_scale = LSSTCAM_CCD_SCALE
        fp_scale = LSSTCAM_FP_SCALE
    else:
        ccd_scale = HSC_CCD_SCALE
        fp_scale = HSC_FP_SCALE

    plot_order = ['rho1', 'rho2', 'rho3', 'rho4', 'rho5', 'rho3alt']

    for idx, rho_name in enumerate(plot_order):
        ax = axes.flat[idx]

        for selection, rho_stats in rho_stats_dict.items():
            if rho_stats is None:
                continue
            rho = rho_stats[rho_name]
            theta = rho.meanr
            color = colors[selection]
            marker = markers[selection]

            if rho_name == 'rho3alt':
                ax.errorbar(theta, rho.xi, yerr=np.sqrt(rho.varxi),
                            fmt=f'{marker}-', capsize=2, markersize=4, color=color,
                            label=f'{selection}', alpha=0.8)
            else:
                ax.errorbar(theta, rho.xip, yerr=np.sqrt(rho.varxip),
                            fmt=f'{marker}-', capsize=2, markersize=4, color=color,
                            label=f'{selection} (xip)', alpha=0.8)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(ccd_scale, color='k', linestyle='--', alpha=0.5)
        ax.axvline(fp_scale, color='k', linestyle=':', alpha=0.5)

        ax.set_xscale('log')
        if rho_name != 'rho3alt':
            ax.set_yscale('symlog', linthresh=1e-8)

        ax.set_xlabel('Separation [arcmin]')
        ax.set_ylabel(rho_labels[rho_name])
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='best', fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compute Rho statistics from refit_psf_star tables")
    parser.add_argument('--repo', type=str, required=True, help='Butler repository')
    parser.add_argument('--collection', type=str, required=True, help='Butler collection')
    parser.add_argument('--instrument', type=str, required=True, choices=['LSSTCam', 'HSC'],
                        help='Instrument name')
    parser.add_argument('--dataset_type', type=str, default=None,
                        help='Dataset type (default: refit_psf_star for LSSTCam, finalized_src_table for HSC)')
    parser.add_argument('--band', type=str, default=None, help='Band to process (optional filter)')
    parser.add_argument('--repOut', type=str, default='rho_stats/', help='Output directory')
    parser.add_argument('--ellipticityType', type=str, default='distortion',
                        choices=['distortion', 'shear'])
    parser.add_argument('--min_sep', type=float, default=0.1, help='Min separation in arcmin')
    parser.add_argument('--max_sep', type=float, default=300.0, help='Max separation in arcmin')
    parser.add_argument('--nbins', type=int, default=30, help='Number of separation bins')
    parser.add_argument('--max_visits', type=int, default=None, help='Max visits to process')
    parser.add_argument('--snr_min', type=float, default=None, help='Minimum SNR cut')
    parser.add_argument('--snr_max', type=float, default=None, help='Maximum SNR cut')
    parser.add_argument('--visit_file', type=str, default=None,
                        help='Text file with visit IDs (one per line)')
    parser.add_argument('--output_suffix', type=str, default='',
                        help='Suffix for output files')
    args = parser.parse_args()

    print(f"Rho Statistics Computation")
    print(f"  Repository: {args.repo}")
    print(f"  Collection: {args.collection}")
    print(f"  Instrument: {args.instrument}")
    print(f"  Band: {args.band if args.band else 'all'}")
    print(f"  Ellipticity type: {args.ellipticityType}")
    print(f"  SNR range: [{args.snr_min}, {args.snr_max}]")
    # Set default dataset type based on instrument
    if args.dataset_type is None:
        if args.instrument == 'HSC':
            args.dataset_type = 'finalized_src_table'
        else:
            args.dataset_type = 'refit_psf_star'

    print(f"  Dataset type: {args.dataset_type}")
    print(f"  Angular bins: {args.nbins} bins from {args.min_sep} to {args.max_sep} arcmin")

    butler = Butler(args.repo, collections=args.collection)

    # Get visits
    if args.visit_file is not None:
        with open(args.visit_file, 'r') as f:
            visits = [int(line.strip()) for line in f if line.strip()]
        print(f"Loaded {len(visits)} visits from {args.visit_file}")
    else:
        query_kwargs = {}
        if args.band is not None:
            query_kwargs['band'] = args.band
        dsrefs = list(butler.registry.queryDatasets(args.dataset_type, **query_kwargs))
        visits = sorted(set(dsr.dataId["visit"] for dsr in dsrefs))
        print(f"Found {len(visits)} visits in collection")

    if args.max_visits is not None and len(visits) > args.max_visits:
        visits = visits[:args.max_visits]
        print(f"Limited to {args.max_visits} visits")

    # Load all data
    all_keys = ['ixx', 'iyy', 'ixy', 'ixx_psf', 'iyy_psf', 'ixy_psf', 'ra', 'dec',
                'calib_psf_used', 'calib_psf_reserved']
    all_data = {k: [] for k in all_keys}

    for visit in tqdm(visits, desc="Loading visits"):
        try:
            uri = butler.getURI(args.dataset_type, instrument=args.instrument, visit=visit)
            parquet_path = uri.geturl()
            data = load_visit_data(parquet_path, snr_min=args.snr_min, snr_max=args.snr_max)
            for k in all_data:
                all_data[k].append(data[k])
        except Exception as e:
            print(f"Warning: failed visit {visit}: {e}")

    # Concatenate
    for k in all_data:
        all_data[k] = np.concatenate(all_data[k])

    print(f"\nTotal sources: {len(all_data['ra']):,}")

    # Filter NaN
    valid = np.isfinite(all_data['ixx']) & np.isfinite(all_data['iyy']) & np.isfinite(all_data['ixy'])
    valid &= np.isfinite(all_data['ixx_psf']) & np.isfinite(all_data['iyy_psf']) & np.isfinite(all_data['ixy_psf'])
    for k in all_data:
        all_data[k] = all_data[k][valid]
    print(f"After NaN filter: {len(all_data['ra']):,}")

    # TreeCorr config
    treecorr_config = {
        'sep_units': 'arcmin',
        'min_sep': args.min_sep,
        'max_sep': args.max_sep,
        'nbins': args.nbins,
    }

    # Create output directory
    os.makedirs(args.repOut, exist_ok=True)

    # Run for all three selections: all, used, reserved
    all_rho_stats = {}
    n_sources_dict = {}

    for star_selection in ['all', 'used', 'reserved']:
        print(f"\n{'='*60}")
        print(f"Computing rho statistics for: {star_selection}")
        print(f"{'='*60}")

        # Apply selection
        if star_selection == 'all':
            sel_mask = np.ones(len(all_data['ra']), dtype=bool)
        elif star_selection == 'used':
            sel_mask = all_data['calib_psf_used']
        elif star_selection == 'reserved':
            sel_mask = all_data['calib_psf_reserved']

        selected_data = {k: all_data[k][sel_mask] for k in all_data}
        print(f"  Sources: {len(selected_data['ra']):,}")

        if len(selected_data['ra']) == 0:
            print(f"  WARNING: No sources for {star_selection}, skipping")
            all_rho_stats[star_selection] = None
            continue

        # Compute rho inputs
        inputs = compute_rho_inputs(selected_data, ellipticity_type=args.ellipticityType)

        # Compute rho stats
        rho_stats = compute_rho_statistics(inputs, treecorr_config)
        all_rho_stats[star_selection] = rho_stats
        n_sources_dict[star_selection] = len(inputs['ra'])

        # Build output filename
        suffix = f'{args.instrument}_{args.ellipticityType}'
        if args.band:
            suffix += f'_{args.band}'
        suffix += f'_{star_selection}'
        if args.snr_min is not None:
            suffix += f'_snrmin{int(args.snr_min)}'
        if args.snr_max is not None:
            suffix += f'_snrmax{int(args.snr_max)}'
        if args.output_suffix:
            suffix += f'_{args.output_suffix}'

        # Save results
        output_pkl = os.path.join(args.repOut, f'rho_stats_{suffix}.pkl')
        with open(output_pkl, 'wb') as f:
            pickle.dump({
                'rho_stats': {k: {'meanr': v.meanr,
                                  'xip': v.xip if hasattr(v, 'xip') else v.xi,
                                  'xim': v.xim if hasattr(v, 'xim') else None,
                                  'varxip': v.varxip if hasattr(v, 'varxip') else v.varxi,
                                  'varxim': v.varxim if hasattr(v, 'varxim') else None,
                                  'npairs': v.npairs}
                             for k, v in rho_stats.items()},
                'instrument': args.instrument,
                'band': args.band,
                'n_sources': len(inputs['ra']),
                'n_visits': len(visits),
                'treecorr_config': treecorr_config,
                'ellipticity_type': args.ellipticityType,
                'collection': args.collection,
                'star_selection': star_selection,
            }, f)
        print(f"  Saved: {output_pkl}")

    # Make combined plot with all selections
    suffix_combined = f'{args.instrument}_{args.ellipticityType}'
    if args.band:
        suffix_combined += f'_{args.band}'
    if args.snr_min is not None:
        suffix_combined += f'_snrmin{int(args.snr_min)}'
    if args.snr_max is not None:
        suffix_combined += f'_snrmax{int(args.snr_max)}'
    if args.output_suffix:
        suffix_combined += f'_{args.output_suffix}'

    output_plot = os.path.join(args.repOut, f'rho_stats_{suffix_combined}_combined.png')
    title = f"Rho Statistics - {args.instrument}"
    if args.band:
        title += f" {args.band}-band"
    title += f" ({args.ellipticityType})\n{len(visits)} visits"
    for sel, n in n_sources_dict.items():
        title += f" | {sel}: {n:,}"
    plot_rho_statistics_combined(all_rho_stats, output_plot, title=title, instrument=args.instrument)


if __name__ == "__main__":
    main()
