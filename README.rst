JWST Absolute Flux Calibration
==============================

JWST absolute flux calibration effort.

In Development!
---------------

Active development.

Contributors
-----------
Karl Gordon

License
-------

This code is licensed under a 3-clause BSD style license (see the
``LICENSE`` file).

Utilities
---------

1. Bandpass response functions: read_webb, read_spitzer
   Webb: using pandeia engine (required to be installed), supports MIRI
   Spitzer: cached ascii tables, supports IRAC

2. Color corrections: python -m jwstabsfluxcal.colorcor.compute_colorcor
   All supported bandpasses.

2. SiriusVega zero mag fluxes: python -m jwstabsfluxcal.siriusvega.compute_siriusvega_zmag
   All supported bandpasses.