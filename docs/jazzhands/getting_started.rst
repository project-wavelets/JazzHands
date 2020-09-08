.. include:: references.txt

.. _getting_started:

***************
Getting started
***************

Contents
========

* :ref:`getting_started-example`
* :ref:`getting_started-plotting`

.. _getting_started-example:

Using ``JazzHands``
-------------------

Let's go through a basic usage example with some synthetic data. First, we're
going to generate a sin wave with a frequency one 1 and amplitude of 1, 
randomly sampled over 10 periods, with some Gaussian noise added in:

.. code-block:: python

    >>> from jazzhands import *
        import numpy as np
        import matplotlib.pyplot as plt
    
    >>> t_obs = np.random.uniform(0,10,1000).sort() #Generate synthetic data
    
    >>> f_obs = np.sin(2*np.pi*1.0*t_obs) #Sin wave with frequency of 1
    
    >>> f_obs += 0.1*np.random.randn(len(f_obs)) #Add in some Gaussian noise

    >>> plt.scatter(t_obs, f_obs)
    
    >>> plt.xlim('Time')
    
    >>> plt.ylim('Flux')
    
Now let's initialize a WHATEVER WE CALL IT object with our data



.. _getting_started-plotting:

Visualizing the Output
----------------------

Now that you've run the wavelet transform, let's plot and see how it looks:

.. code-block:: python

    >>> #code that plots

