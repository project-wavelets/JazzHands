import unittest

import numpy as np

from jazzhands import frequency_grid


class FreqGridTest(unittest.TestCase):

    def test_omegas_taus_from_min_max_nug(self):
        res = frequency_grid._omegas_taus_from_min_max_nu(
            1, 1, 1, 1, 1, 1
        )
        self.assertIsNotNone(res)
        self.assertIsInstance(res[0], np.ndarray)


if __name__ == '__main__':
    unittest.main()
