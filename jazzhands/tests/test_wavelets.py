import os
import shutil
import unittest

from jazzhands import wavelets


class WaveletsTest(unittest.TestCase):

    def setUp(self):
        self.outdir = "test"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_correct_constructor(self):
        wavelets.WaveletTransformer(func_list=[0, 0], f1=[0, 0], data=[0, 0],
                                    time=[0, 0], omegas=[0, 0], taus=[0, 0], c=0.0125)


class FreqGridTest(unittest.TestCase):

    def _omegas_taus_from_min_max_nu(self):
        res = wavelets._omegas_taus_from_min_max_nu(1, 1, 1, 1, 1, 1)
        self.assertIsNotNone(res)
        self.assertIsInstance(res[0], np.ndarray)


if __name__ == '__main__':
    unittest.main()
