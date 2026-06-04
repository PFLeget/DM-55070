# DM-55070 — Local background on PSF candidates

## Goal
Attach a per-star local background estimate (electrons/pixel) to each
`PsfCandidate` so that PSF fitting and second-moment measurement on Rubin
stars can use the local sky as prior information instead of assuming zero
background.

## Repository layout
This working directory holds LSST Science Pipelines packages, each
checked out on the ticket branch `tickets/DM-55070`:

- `meas_algorithms/` — clone of `https://github.com/lsst/meas_algorithms`
- `meas_base/` — clone of `https://github.com/lsst/meas_base`
- `meas_extensions_shapeHSM/` — clone of `https://github.com/lsst/meas_extensions_shapeHSM`
- `pipe_tasks/` — clone of `https://github.com/lsst/pipe_tasks`

Branches:
```
git -C meas_algorithms branch --show-current          # tickets/DM-55070
git -C meas_base branch --show-current                # tickets/DM-55070
git -C meas_extensions_shapeHSM branch --show-current # tickets/DM-55070
git -C pipe_tasks branch --show-current               # tickets/DM-55070
```

## Changes — `meas_algorithms`

The new field mirrors how `psfColor` is plumbed through `PsfCandidate`. The
default value is `0.0` (zero background) so that any existing caller that
does not set it is unaffected.

- `include/lsst/meas/algorithms/PsfCandidate.h`
  - Added member `_psfBackgroundValue` (default `0.0`).
  - Added `getPsfBackgroundValue()` / `setPsfBackgroundValue()`.
  - Initialized in both constructors.
- `python/lsst/meas/algorithms/psfCandidate/psfCandidate.cc`
  - Bound `getPsfBackgroundValue` / `setPsfBackgroundValue` to Python.
- `python/lsst/meas/algorithms/makePsfCandidates.py`
  - When the input star catalog has a `psf_background_value` column, that
    value is copied onto each candidate via `setPsfBackgroundValue`. If the
    column is missing, the default `0.0` is used.
- `tests/test_psfCandidate.py`
  - Added `testBackgroundPsfCandidates` checking the default value and the
    set/get round-trip.

## Changes — `pipe_tasks`

In `python/lsst/pipe/tasks/finalizeCharacterization.py`:

- Added a `psf_background_value` (`float32`) field to both the output and
  selection schemas, alongside the existing `psf_color_value`.
- Added two config options on `FinalizeCharacterizationConfigBase`:
  - `background_annulus_inner` — inner radius (pixels) of the annulus.
    Default `15.0`. Should be larger than the PSF size.
  - `background_annulus_outer` — outer radius (pixels). Default `25.0`.
- Added `FinalizeCharacterizationTaskBase.add_src_backgrounds(srcCat,
  exposure)`. For each source it builds a `photutils.aperture.CircularAnnulus`
  around the pixel centroid (`slot_Centroid_x/y`, mapped to local array
  coordinates via the exposure bbox) and writes
  `ApertureStats(...).median` into `psf_background_value`. The calexp is
  already in electrons, so the value is electrons/pixel.
- The estimator is called inside `compute_psf_and_ap_corr_map`, right after
  `add_src_colors`, on both the `selected_src` and `measured_src`
  catalogs. `MakePsfCandidatesTask` then propagates the value onto each
  candidate via the `psf_background_value` column added above.

## Changes — `meas_base`

In `python/lsst/meas/base/colorUtilities.py`:

- Added `backgroundExtractor(record, columnName="psf_background_value")` function
  that extracts the local background value from a source record (returns 0.0 if
  not available or not finite).
- Exported via `__all__`.

## Changes — `meas_extensions_shapeHSM`

In `python/lsst/meas/extensions/shapeHSM/_hsm_moments.py`:

- Added `subtractBackground` config option to `HsmSourceMomentsConfig` (default
  `False`). When enabled, the local background from `psf_background_value` is
  subtracted from the image stamp before measuring adaptive moments.
- `HsmSourceMomentsRoundConfig` inherits this option automatically.
- The image array is copied before subtraction to avoid modifying the exposure.

To enable background subtraction in `finalizeCharacterization`, add to your
config override:
```python
config.measurement.plugins['ext_shapeHSM_HsmSourceMoments'].subtractBackground = True
config.measurement.plugins['ext_shapeHSM_HsmSourceMomentsRound'].subtractBackground = True
```

## Assumptions / things to verify on the cluster
- The calexp passed to `compute_psf_and_ap_corr_map` is in electrons (this
  is the documented behavior of the calibrated visit images we ingest, but
  worth a quick check on a real exposure).
- The default annulus `15-25 px` is a reasonable starting point for LSSTCam
  seeing. We may want to make these per-instrument or scale by PSF size in
  a follow-up — for now they are plain config fields.
- Local environment cannot run the science pipelines, so neither package
  has been built or tested here. Build & run on s3df with the standard
  EUPS / `setup -k -r .` flow per https://developer.lsst.io.

## Build / test reminders (run on s3df)
```
# meas_algorithms
cd meas_algorithms
setup -k -r .
scons
pytest tests/test_psfCandidate.py

# pipe_tasks
cd ../pipe_tasks
setup -k -r .
scons
pytest tests/test_finalizeCharacterization.py
```

## Analysis script: analyze_background.py

Script to analyze dT/T vs local background from `refit_psf_star` tables.

**Outputs:**
- Per-visit 3-panel plots: dT/T focal plane, background focal plane, dT/T vs background scatter
- Meanified dT/T map across all visits
- Meanified background map across all visits

**Usage (run on s3df):**
```bash
python analyze_background.py \
    --repo dp2_prep \
    --collection "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2" \
    --band r \
    --repOutPlot plots/ \
    --snr_min 50 \
    --bin_spacing 150

# Skip per-visit plots (faster, just get meanified maps):
python analyze_background.py --skip_per_visit ...

# Test on a few visits:
python analyze_background.py --max_visits 5 ...
```
