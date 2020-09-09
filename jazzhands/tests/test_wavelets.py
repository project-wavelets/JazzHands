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

    def test_invalid_constructor(self):
        # with self.assertRaises(ValueError):
        # TODO: this should probs fail and give some form of error...
        wavelets.WaveletTransformer(func_list=1, f1=[0, 0], data=[0, 0],
                                time="[0, 0]", omegas=[0, 0], taus=[0, 0], c=[123123])



if __name__ == '__main__':
    unittest.main()
