***********************
JazzHands Documentation
***********************

``JazzHands`` (a.k.a. little waves, a.k.a. wavelets) is a package for computing 
wavelet transformations on astronomical data, with data from the TESS mission
specifically in mind. Because these data have gaps, and the sampling isn't
precisely two minutes (for the short cadence data), traditional wavelet methods
have a number of fallbacks. `Foster (1996)
<https://ui.adsabs.harvard.edu/abs/1996AJ....112.1709F/abstract>`_ introduced
the Weighted Wavelet Z-transform (WWZ), which works around these drawbacks 
using a weighted projection onto the Morlet wavelet basis functions. 

``JazzHands`` is a (hopefully) quick implementation of the WWZ, with methods 
for automatically choosing the frequency and time grids to minimize computation
costs. You can view the `source code and submit issues via GitHub
<https://github.com/tzdwi/JazzHands>`_.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install
   getting_started
   api

Reference/API
=============

.. automodapi:: jazzhands
