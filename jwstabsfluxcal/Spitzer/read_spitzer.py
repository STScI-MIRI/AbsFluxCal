import pathlib

from astropy.table import QTable
import astropy.units as u


def read_irac():
    """
    Read in the IRAC bandpass functions

    Returns
    -------
    dictionary giving (name, ref_wave, wave, bandpass) tuples for each filter
    """
    irac_bandpasses = {}

    dpath = pathlib.Path(__file__).parent.resolve()

    # from Reach et al. (2005, PASP, 117, 978)
    irac_ref_waves = [3.550, 4.493, 5.731, 7.872]

    irac_files = [f"{dpath}/irac_201125ch{k}trans_full.txt" for k in range(1, 5)]
    for k, cfile in enumerate(irac_files):
        tab = QTable.read(cfile, format="ascii.commented_header", header_start=-2)
        irac_bandpasses[f"IRAC{k+1}"] = (
            irac_ref_waves[k],
            tab["Wave"].data * u.micron,
            tab["SpectralResponse"].data,
        )

    return irac_bandpasses
