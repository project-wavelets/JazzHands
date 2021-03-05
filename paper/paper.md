---
title: 'JazzHands: A Python package for fast and accurate wavelet transformations on unevenly sampled data'

tags:
  - Python
  - astronomy
  - time series
  - wavelets
  - light curves

authors:
  - name: Trevor Z. Dorn-Wallenstein^[corresponding author.]
    orcid: 0000-0003-3601-3180
    affiliation: "1"
  - name: Aditya R. Sengupta
    orcid: 0000-0002-5669-035X
    affiliation: "2"
  - name: James S. Kuszlewicz
    orcid: 0000-0002-3322-5279
    affiliation: "3"
  - name: Athul R. T.
    orcid: 0000-0001-5565-7470
    affiliation: "4"
  - name: Avi Vajpeyi
    orcid: 0000-0002-4146-1132
    affiliation: "5, 6"
  - name: Amogh Desai
    orcid: 0000-0002-6015-9553
    affiliation: "7"

affiliations:
 - name: University of Washington
   index: 1
 - name: University of California, Berkeley
   index: 2
 - name: Heidelberg Institute for Theoretical Studies
   index: 3
 - name: Cochin University of Science and Technology
   index: 4
 - name: School of Physics and Astronomy, Monash University, Vic 3800, Australia
   index: 5
 - name: "OzGrav: The ARC Centre of Excellence for Gravitational Wave Discovery, Clayton VIC 3800, Australia"
   index: 6
 - name: "Dwarkadas J. Sanghvi College of Engineering"
   index: 7


date: XX March 2021
bibliography: paper.bib

---

# Summary

Time series photometry of astronomical objects encodes information about
these objects' physical state, and how they change on human timescales. Recent rapid cadence observations from missions like TESS and *Kepler* have been used to discover exoplanets around nearby stars, measure stellar pulsations, characterize variability in active galactic nuclei, and more. Wavelet analysis is an extension of Fourier analysis that allows astronomers to examine the frequency content of a light curve as a function of time. As such, it is an
ideal tool for studying real data, which often contain information that is
otherwise impossible to extract from the Fourier transform alone.

# Statement of need

`JazzHands` is a Python package for wavelet transformations. Typical wavelet
methods [@torrence98] assume that the data are regularly sampled and have
no gaps. This fact has hindered astronomical applications of wavelets. Data
taken from the ground are often irregularly sampled and have daily and seasonal
gaps due to the rotation and orbit of the Earth. While less problematic, space-
based data still have gaps due to downlink times, and time-stamp adjustments to
account for time-of-arrival differences along the spacecraft orbit induce
sampling irregularities. Interpolative methods may be used to offset these
effects, but introduce their own systematics into the wavelet transform.

[@foster96] introduced a mathematical formulation of the wavelet transform
using the Morlet wavelet that accounts for the effects of irregular sampling
and gaps, and avoids the need for interpolation.
`JazzHands` is the first open-source Python 3 implementation of the @foster96 method,
and has already been used in scientific applications [@dornwallenstein20b]. The
combination of speed, customizability, and helpful features for users who are new to
wavelet analysis will enable exciting new science on the wealth of high precision light
curves that are currently being released.

# Acknowledgements

This project was developed in part at the online.tess.science meeting, which
took place globally in 2020 September.

# References

1. Foster, G. (1996). Wavelets for period analysis of unevenly sampled time series. The Astronomical Journal, 112, 1709. https://doi.org/10.1086/118137
2. Torrence, C., & Compo, G. P. (1998). A Practical Guide to Wavelet Analysis. Bulletin of the American Meteorological Society, 79(1), 61–78. https://doi.org/10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2
3. Dorn-Wallenstein, T. Z., Levesque, E. M., Neugent, K. F., Davenport, J. R. A., Morris, B. M., & Gootkin, K. (2020). Short-term Variability of Evolved Massive Stars with TESS. II. A New Class of Cool, Pulsating Supergiants. The Astrophysical Journal, 902(1), 24. https://doi.org/10.3847/1538-4357/abb318
