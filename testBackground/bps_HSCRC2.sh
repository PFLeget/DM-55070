bps submit /sdf/group/rubin/user/leget/batch/bps_generic_main.yaml \
    -b main \
    -i HSC/runs/RC2/w_2026_18/DM-54847 \
    -o u/leget/LSSTCam/DM-55070/HSC_RC2_run \
    -p ${DRP_PIPE_DIR}/pipelines/HSC/DRP-RC2.yaml#finalizeCharacterizationDetector,consolidateFinalizeCharacterization \
    --extra-qgraph-options "--config-file finalizeCharacterizationDetector:finalizeCharacterizationConfigHSCRc2.py" \
    -d "instrument='HSC' "
