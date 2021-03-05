# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------

__all__ = []
from jazzhands.wavelets import *  # noqa
from jazzhands import wavelets
from jazzhands.utils import *

try:
    from jazzhands.version import version
except ModuleNotFoundError:
    version = '0.0.3'
# Then you can be explicit to control what ends up in the namespace,
# or you can keep everything from the subpackage with the following instead
__all__ += wavelets.__all__

#Version number
__version__ = version
