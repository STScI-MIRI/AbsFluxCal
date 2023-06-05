import numpy as np
import astropy.units as u

from pandeia.engine.instrument_factory import InstrumentFactory


def read_miri():
    """ """

    # set up your wavelengths
    wave = np.logspace(np.log10(1.0), np.log10(40.0), 501)

    filters = [
        "f560w",
        "f770w",
        "f1000w",
        "f1065c",
        "f1140c",
        "f1130w",
        "f1280w",
        "f1500w",
        "f1550c",
        "f1800w",
        "f2100w",
        "f2300c",
        "f2550w",
        "fnd",
    ]

    miri_bandpasses = {}

    for filtername in filters:
        # mock configuration
        conf = {
            "detector": {
                "nexp": 1,
                "ngroup": 10,
                "nint": 1,
                "readout_pattern": "fastr1",
                "subarray": "full",
            },
            "dynamic_scene": True,
            "instrument": {
                "aperture": "imager",
                "filter": filtername,
                "instrument": "miri",
                "mode": "imaging",
            },
        }

        # create a configured instrument
        instrument_factory = InstrumentFactory(config=conf)

        # get the throughput of the instrument over the desired wavelength range
        eff = instrument_factory.get_total_eff(wave)

        # compute the reference wave
        # defined as the pivot wavelength in Gordon et al. (2022)
        inttop = np.trapz(wave * eff, wave)
        intbot = np.trapz(eff / wave, wave)
        ref_wave = np.sqrt(inttop / intbot)

        miri_bandpasses[filtername] = (ref_wave, wave * u.micron, eff)

    return miri_bandpasses
