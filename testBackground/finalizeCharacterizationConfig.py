config.psf_determiner['piff'].spatialOrderPerBand = {
    "u": 2,
    "g": 4,
    "r": 4,
    "i": 4,
    "z": 4,
    "y": 4,
}
config.psf_determiner['piff'].zerothOrderInterpNotEnoughStars = False
config.psf_determiner['piff'].piffBasisPolynomialSolver = "cpp"
config.psf_determiner['piff'].piffPixelGridFitCenter = False
config.do_add_sky_moments = True
config.do_add_fgcm_photometry = True
config.fgcmPhotometryBands = ['u', 'g', 'r', 'i', 'z', 'y']

config.background_annulus_inner = 27
config.background_annulus_outer = 35
config.psf_determiner['piff'].stampSize = 41
config.make_psf_candidates.kernelSize = 41
config.psf_determiner['piff'].modelSize = 25
# config.psf_determiner['piff'].modelSize = 35
#config.psf_determiner['piff'].piffMaxIter = 15
config.measurement.plugins['ext_shapeHSM_HsmSourceMoments'].usePsfStampSize = True
config.measurement.plugins['ext_shapeHSM_HsmSourceMomentsRound'].usePsfStampSize = True

# See DM-45569, but this is not enable but it was the config
# used in the plots.
config.psf_determiner['piff'].useColor = True
config.psf_determiner['piff'].colorOrder = 1
config.psf_determiner['piff'].color = {
    "u": "g-i",
    "g": "g-i",
    "r": "g-i",
    "i": "g-i",
    "z": "g-i",
    "y": "g-i",
}