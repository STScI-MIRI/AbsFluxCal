import argparse

import warnings
from astropy.units import UnitsWarning

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import QTable

from jwstabsfluxcal.Webb.read_webb import read_miri


def compute_bandflux(wave, flux_source, bwave, bandpass):
    """
    Compute the band flux given the bandpass, reference spectrum,
    and source spectrum.  Assumes a flat reference spectrum
    (for motivation see Gordon et al. 2022).

    Parameters
    ----------
    wave : nd float array
       the wavelengths of flux_source
    flux_source : nd float array
        source flux density F(lambda) as a function of wave
    bwave : nd float array
        the wavelengths of bandpass
    bandpass : nd float array
        end-to-end, total throughput bandpass of filter in fractional units
    """
    flux_source_bp = np.interp(bwave, wave, flux_source)

    # compute the the integrals
    inttop = np.trapz(bwave * bandpass * flux_source_bp)
    intbot = np.trapz(bwave * bandpass)

    return inttop / intbot


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 14
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=2)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    # get sirius sed

    # surpress the annoying units warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UnitsWarning)
        modspec = QTable.read("jwstabsfluxcal/siriusvega/sirius_stis_005.fits")

    mwave = (modspec["WAVELENGTH"].value * u.angstrom).to(u.micron)
    mflux = modspec["FLUX"].value * u.erg / (u.cm * u.cm * u.s * u.angstrom)
    mflux = mflux.to(u.Jy, equivalencies=u.spectral_density(mwave))

    # shift Sirius fainter by 1.395 mags as recommended by Rieke et al. (2022)
    mflux *= 10 ** (-0.4 * 1.395)

    ax.plot(mwave, mflux, "k-", alpha=0.5)

    # get the MIRI band info
    bandpasses = read_miri()

    bands = []
    bwaves = []
    bfluxes = []
    for cband in bandpasses.keys():
        rwave, cwave, ceff = bandpasses[cband]
        bflux = compute_bandflux(mwave, mflux, cwave, ceff)
        bands.append(cband)
        bwaves.append(rwave)
        bfluxes.append(bflux.value)
        #print(cband, rwave, bflux)
        print(f"{cband.upper()} & {bflux.value:.3f} \\\\")

    bwaves = np.array(bwaves) * u.micron
    bfluxes = np.array(bfluxes) * u.Jy

    ax.plot(bwaves, bfluxes, "bo")

    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$F(\nu)$ [Jy]")

    ax.set_xlim(4.0, 28.0)
    ax.set_ylim(1, 200.)

    ax.set_yscale("log")

    fig.tight_layout()

    fname = f"siriusvega_zmag_fluxes"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
