
visitids=$(paste -sd, visits_cosmos_r_dp2.txt)

# First model run to estimate background
#    -o u/leget/LSSTCam/DM-55070/cosmos_r_band_20260528 \
# Second model with larger grid
#    -o u/leget/LSSTCam/DM-55070/cosmos_r_band_20260528_pixelgrid_35_pixel 

bps submit /sdf/group/rubin/user/leget/batch/bps_generic_main.yaml \
    -b dp2_prep \
    -i LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2 \
    -o u/leget/LSSTCam/DM-55070/cosmos_r_band_20260528_colorfit_enable \
    -p ${DRP_PIPE_DIR}/pipelines/LSSTCam/DRP.yaml#refitPsfModelDetector,consolidateRefitPsfModelDetector \
    --extra-qgraph-options "--config-file refitPsfModelDetector:finalizeCharacterizationConfig.py" \
    -d "instrument='LSSTCam' AND skymap='lsst_cells_v2' AND visit IN (${visitids})"
