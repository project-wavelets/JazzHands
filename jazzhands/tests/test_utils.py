import numpy as np
import pytest
import unittest

from jazzhands import utils

class UtilsTest(unittest.TestCase):
    def test_phi_1(self):
        """Test to make sure the output is an array of ones"""
        phi1 = utils.phi_1([1, 2, 3], 3, 10)
        expected = np.array([1., 1., 1.])
        assert np.allclose(phi1, expected)


    def test_phi_2(self):
        """Test to make sure the function output gives an approximate expected result"""
        time = np.array([1, 2, 3])
        phi2 = utils.phi_2(time, 3, 10)
        expected = np.array([1., -0.99, 0.96])
        assert pytest.approx(phi2, expected)


    def test_phi_3(self):
        """Test to make sure the function output gives an approximate expected result"""
        time = np.array([1, 2, 3])
        phi3 = utils.phi_3(time, 3, 10)
        expected = np.array([0., 0.14, -0.28])
        assert pytest.approx(phi3, expected)

if __name__ == '__main__':
    unittest.main()
