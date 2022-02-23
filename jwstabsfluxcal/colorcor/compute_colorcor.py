import argparse

import numpy as np
import matplotlib.pyplot as plt

from jwstabsfluxcal.Spitzer.read_spitzer import read_irac


def compute_colorcor(wave, bandpass, flux_ref, wave_ref, flux_source):
    """
    Compute the color correction K given the bandpass, reference spectrum,
    and source spectrum.  To use this color correction, divide the flux density
    for a band by K.  Such color corrections are needed to compute the correct
    flux density at the reference wavelength for a source with the flux_source
    spectral shape in the photometric convention that provides the flux density
    at a reference wavelength (connention B, see Gordon et al. 2022 for details).

    Parameters
    ----------
    wave : nd float array
       the wavelengths of the bandpass, flux_ref, and flux_source
    bandpass : nd float array
        end-to-end, total throughput bandpass of filter in fractional units
    flux_ref : nd float array
        reference flux density F(lambda) as a function of wave
    wave_ref : float
        reference wavelength
    flux_source : nd float array
        source flux density F(lambda) as a function of wave
    """
    # get the flux densities at the reference waveength
    flux_source_lambda_ref = np.interp(wave_ref, wave, flux_source)
    flux_ref_lambda_ref = np.interp(wave_ref, wave, flux_ref)

    # compute the top and bottom integrals
    inttop = np.trapz(wave * bandpass * flux_source / flux_source_lambda_ref, wave)
    intbot = np.trapz(wave * bandpass * flux_ref / flux_ref_lambda_ref, wave)

    return inttop / intbot


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inst",
        choices=["irac"],
        default="irac",
        help="Instrument",
    )
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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    irac_bps = read_irac()

    bps = irac_bps
    for ckey in bps.keys():
        waves = bps[ckey][1].value
        flux_source = np.full((len(waves)), 1.0)
        flux_ref = 1.0 / waves
        cc = compute_colorcor(waves, bps[ckey][2], flux_ref, bps[ckey][0], flux_source)
        print(ckey, cc)

    fname = f"color_corrections_{args.inst}"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
