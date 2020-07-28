# -*- coding: utf-8 -*-

"""Tests for coordinate_system.py."""

import pytest

# Use the default configuration for all tests
from test_load_config import config, sysconfig

from coordinate_system import CoordinateSystem

@pytest.fixture
def cs():
    """Create and return CoordinateSystem instance."""
    return CoordinateSystem(config, sysconfig)

def test_initial_config(cs):
    assert cs.calibration_found

def test_convert_coordinates(cs):
    assert cs.convert_to_s(cs.convert_to_d([0, 0])) == [0, 0]
    assert cs.convert_to_d(cs.convert_to_s([-100, 100])) == pytest.approx([-100, 100])
