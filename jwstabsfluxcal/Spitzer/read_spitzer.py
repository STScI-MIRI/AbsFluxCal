import pathlib

from astropy.table import QTable
import astropy.units as u


def read_irac():
    """
    Read in the IRAC bandpass functions

    Returns
    -------
    dictionary giving (name, wave, bandpass) tuples for each filter
    """
    irac_bandpasses = []

    dpath = pathlib.Path(__file__).parent.resolve()

    irac_files = [f"{dpath}/irac_201125ch{k}trans_full.txt" for k in range(1, 4)]
    for k, cfile in enumerate(irac_files):
        tab = QTable.read(cfile, format="ascii.commented_header", header_start=-2)
        irac_bandpasses.append(
            (f"IRAC{k+1}", tab["Wave"] * u.micron, tab["SpectralResponse"])
        )

    return irac_bandpasses
