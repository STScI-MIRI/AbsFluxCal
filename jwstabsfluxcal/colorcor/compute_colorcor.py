import argparse

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import PowerLaw1D, BlackBody
import astropy.units as u
from astropy.table import QTable

from jwstabsfluxcal.Spitzer.read_spitzer import read_irac
from jwstabsfluxcal.Webb.read_webb import read_miri


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inst", choices=["miri", "irac"], default="irac", help="Instrument",
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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    if args.inst == "miri":
        bps = read_miri()
        ref_shape = PowerLaw1D(amplitude=1.0, x_0=1.0, alpha=0.0)
    else:
        bps = read_irac()
        ref_shape = PowerLaw1D(amplitude=1.0, x_0=1.0, alpha=1.0)

    # define shapes
    bbscale = 1.0 * u.erg / (u.cm ** 2 * u.s * u.micron * u.steradian)
    source_shapes = [
        PowerLaw1D(amplitude=1.0, x_0=1.0, alpha=-2.0),
        PowerLaw1D(amplitude=1.0, x_0=1.0, alpha=-1.0),
        PowerLaw1D(amplitude=1.0, x_0=1.0, alpha=0.0),
        PowerLaw1D(amplitude=1.0, x_0=1.0, alpha=1.0),
        PowerLaw1D(amplitude=1.0, x_0=1.0, alpha=2.0),
        BlackBody(scale=bbscale, temperature=10000.0 * u.K),
        BlackBody(scale=bbscale, temperature=5000.0 * u.K),
        BlackBody(scale=bbscale, temperature=1000.0 * u.K),
        BlackBody(scale=bbscale, temperature=500.0 * u.K),
        BlackBody(scale=bbscale, temperature=200.0 * u.K),
    ]
    source_labels = [
        r"$\lambda^2$",
        r"$\lambda^1$",
        r"$\lambda^0$",
        r"$\lambda^{-1}$",
        r"$\lambda^{-2}$",
        r"$BB(T=10000 K)$",
        r"$BB(T=5000 K)$",
        r"$BB(T=1000 K)$",
        r"$BB(T=500 K)$",
        r"$BB(T=200 K)$",
    ]

    collabels = [
        "lam2",
        "lam1",
        "lam0",
        "lam-1",
        "lam-2",
        "BB(T=10000K)",
        "BB(T=5000K)",
        "BB(T=1000K)",
        "BB(T=500K)",
        "BB(T=200K)",
    ]

    # output table
    otab = QTable()
    otab["Bands"] = [ckey for ckey in bps.keys()]

    got_rwaves = False
    rwaves = []
    for cshape, clabel, collab in zip(source_shapes, source_labels, collabels):
        pwaves = []
        bandwidths = []
        effresps = []
        bluewaves = []
        redwaves = []
        pcolcor = []
        for ckey in bps.keys():
            waves = bps[ckey][1].to(u.micron).value
            if isinstance(cshape, BlackBody):
                flux_source = cshape(waves * u.micron).value
            else:
                flux_source = cshape(waves)
            flux_ref = ref_shape(waves)
            if not got_rwaves:
                rwaves.append(bps[ckey][0])
            pwaves.append(bps[ckey][0])
            cc = compute_colorcor(
                waves, bps[ckey][2], flux_ref, bps[ckey][0], flux_source
            )
            pcolcor.append(cc)

            # compute additional stats
            eff = bps[ckey][2]

            # bandwidth
            intresp = np.trapz(eff, waves)
            bwidth = intresp / max(eff)
            bandwidths.append(bwidth)

            # effective response
            pwave = bps[ckey][0]
            bvals = (waves >= (pwave - 0.5 * bwidth)) & (waves <= (pwave + 0.5 * bwidth))
            effresps.append(np.average(eff[bvals]))

            # blue and red wavelengths where transmission drops to 50% of the peak
            norm_eff = eff / max(eff)
            bvals = (waves <= pwave) & (norm_eff > 0.1)
            bwave = np.interp([0.5], norm_eff[bvals], waves[bvals])
            bluewaves.append(bwave[0])
            bvals = (waves >= pwave) & (norm_eff > 0.1)
            rwave = np.interp([0.5], norm_eff[bvals], waves[bvals])
            redwaves.append(rwave[0])

        if not got_rwaves:
            otab["pivotwave"] = rwaves
            otab["pivotwave"].format = "10.6f"
            got_rwaves = True

        otab["bandwidth"] = bandwidths
        otab["bandwidth"].format = "10.6f"
        otab["effresponse"] = effresps
        otab["effresponse"].format = "10.6f"
        otab["bluewave"] = bluewaves
        otab["bluewave"].format = "10.6f"
        otab["redwave"] = redwaves
        otab["redwave"].format = "10.6f"

        ax.plot(pwaves, pcolcor, label=clabel)
        otab[collab] = pcolcor
        otab[collab].format = "10.6f"

    otab.write(
        f"color_corrections_{args.inst}.dat",
        # format="ascii.commented_header",
        format="ascii.fixed_width",
        overwrite=True,
    )

    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel("K")
    ax.set_title(args.inst)

    ax.legend()
    fig.tight_layout()

    fname = f"color_corrections_{args.inst}"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
