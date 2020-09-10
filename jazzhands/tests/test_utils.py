import numpy as np
import pytest

from jazzhands import utils


class TestUtils(object):
    def test_phi_1(self):
        phi1 = utils.phi_1([1, 2, 3], 3, 10)
        expected = np.array([1., 1., 1.])
        assert np.allclose(phi1, expected)

    def test_phi_2(self):
        time = np.array([1, 2, 3])
        phi2 = utils.phi_2(time, 3, 10)
        expected = np.array([1., -0.99, 0.96])
        assert pytest.approx(phi2, expected)

    def test_phi_3(self):
        time = np.array([1, 2, 3])
        phi3 = utils.phi_3(time, 3, 10)
        expected = np.array([0., 0.14, -0.28])
        assert pytest.approx(phi3, expected)
