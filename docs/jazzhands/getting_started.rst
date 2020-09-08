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

A basic usage example is as follows:

.. code-block:: python

    >>> from jazzhands import *
        import numpy as np
    
    >>> t_obs = np.random.uniform(0,10,1000).sort() #Generate synthetic data
    
    >>> f_obs = np.sin(2*np.pi*1.0*t_obs) #Sin wave with frequency of 1
    
    >>> f_obs += 0.1*np.random.randn(len(f_obs)) #Add in some Gaussian noise

    >>> print([1,2,3]) #until we have the wavelet code up and running



.. _getting_started-plotting:

Visualizing the Output
----------------------

Now that you've run the wavelet transform, let's plot and see how it looks:

.. code-block:: python

    >>> #code that plots

