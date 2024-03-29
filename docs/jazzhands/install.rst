.. _installation:

Installation
==============

Dependencies
------------
Jazzhands requires ``numpy`` package. It must be installed or the package won't work properly.

Optional dependencies are: 

With conda
----------

To install Jazzhands with released latest version of conda, use the following command::
        
    conda install -c conda-forge jazz-hands
    
Make sure that you have added ``conda-forge`` in your conda environment using the command::

    conda config --add channels conda-forge
    
With pip
--------

To install Jazzhands using pip, run the following code::

    pip install jazz-hands
    
Pip will automatically installs the dependency package ``numpy`` also. To install Jazzhands without dependencies, run the code below::

    pip install --no-deps jazz-hands

