
import pytest
import numpy as np
from scipy.stats import norm

from src.models import BlackScholesModel


class TestBlackScholesModel:
    def test_volatility_non_negative(self):
        with pytest.raises(ValueError):
            BlackScholesModel(volatility=-0.1)
    def tst_cdf_growing_with_z(self):
        model = BlackScholesModel(volatility=0.2)
        z_values = np.linspace(0.1, 2.0, 10)
        cdf_values = [model.transition_cdf(z, dt=1.0) for z in z_values]
        assert all(cdf_values[i] <= cdf_values[i + 1] for i in range(len(cdf_values) - 1))
    def test_limit_behavior_z(self):
        model = BlackScholesModel(volatility=0.2)
        assert np.isclose(model.transition_cdf(0.005, dt=1.0), 0)
        assert np.isclose(model.partial_expectation(np.inf, dt=1.0), 1)
    def test_martingale_property(self):
        model = BlackScholesModel(volatility=0.2)
        cdf_value = model.transition_cdf(np.inf, dt=1.0)
        partial_expectation_value = model.partial_expectation(np.inf, dt=1.0)
        assert np.isclose(partial_expectation_value, 1)
    def test_ep_inferior_cdf(self):
        model = BlackScholesModel(volatility=0.2)
        z = 1.0
        dt = 1.0
        cdf_value = model.transition_cdf(z, dt)
        partial_expectation_value = model.partial_expectation(z, dt)
        assert partial_expectation_value <= cdf_value
    def test_diff_in_volatility(self):
        model1 = BlackScholesModel(volatility=0.2)
        model2 = BlackScholesModel(volatility=0.3)
        z = 1.0
        dt = 1.0
        cdf_value1 = model1.transition_cdf(z, dt)
        cdf_value2 = model2.transition_cdf(z, dt)
        assert cdf_value1 < cdf_value2