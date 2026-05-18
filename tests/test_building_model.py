#!/usr/bin/env python3
"""Unit tests for building settlement / shrinkage kinematics."""

import unittest

import numpy as np

from building_model import (
    BuildingGeometry,
    BuildingRates,
    building_length_scale,
    corner_points,
    deform_points,
    settlement_w,
    summary,
    validate_parameters,
)

try:
    from building_model import make_building_mesh  # noqa: F401

    HAS_PYVISTA = True
except Exception:
    HAS_PYVISTA = False


class TestSettlement(unittest.TestCase):
    def test_uniform_settlement_same_at_all_corners(self):
        geom = BuildingGeometry(40, 20, 200)
        rates = BuildingRates(s0=0.02, gx=0, gy=0, alpha=0)
        T = 10.0
        corners = corner_points(geom.lx, geom.ly, geom.h)
        w = settlement_w(corners[:, 0], corners[:, 1], rates, T)
        self.assertAlmostEqual(w[0], 0.2)
        np.testing.assert_allclose(w[:4], w[4:])

    def test_gx_gradient_corner_difference(self):
        geom = BuildingGeometry(40, 20, 200)
        rates = BuildingRates(s0=0, gx=0.001, gy=0, alpha=0)
        T = 10.0
        w_sw = settlement_w(0, 0, rates, T)
        w_se = settlement_w(geom.lx, 0, rates, T)
        self.assertAlmostEqual(w_se - w_sw, rates.gx * geom.lx * T)

    def test_tilt_roof_is_planar_and_sloped(self):
        geom = BuildingGeometry(40, 20, 200)
        rates = BuildingRates(s0=0.01, gx=0.0005, gy=0.0002, alpha=0)
        T = 10.0
        corners = corner_points(geom.lx, geom.ly, geom.h)
        roof = corners[corners[:, 2] == geom.h]
        out = deform_points(roof, geom, rates, T)
        tx, ty = rates.gx * T, rates.gy * T
        expected_z = geom.h - settlement_w(roof[:, 0], roof[:, 1], rates, T)
        expected_x = roof[:, 0] + geom.h * tx
        expected_y = roof[:, 1] + geom.h * ty
        np.testing.assert_allclose(out[:, 2], expected_z)
        np.testing.assert_allclose(out[:, 0], expected_x)
        np.testing.assert_allclose(out[:, 1], expected_y)

    def test_gx_tilts_building_rigidly(self):
        geom = BuildingGeometry(40, 20, 200)
        rates = BuildingRates(s0=0, gx=0.01, gy=0, alpha=0)
        T = 10.0
        tx = rates.gx * T
        sw_ground = np.array([[0.0, 0.0, 0.0]])
        ne_roof = np.array([[geom.lx, 0.0, geom.h]])
        out_g = deform_points(sw_ground, geom, rates, T)[0]
        out_r = deform_points(ne_roof, geom, rates, T)[0]
        self.assertAlmostEqual(out_g[0], 0.0)
        self.assertAlmostEqual(out_g[2], 0.0)
        self.assertAlmostEqual(out_r[0], geom.lx + geom.h * tx)
        self.assertAlmostEqual(out_r[2], geom.h - geom.lx * tx)


class TestShrinkage(unittest.TestCase):
    def test_ground_unchanged_horizontally(self):
        geom = BuildingGeometry(40, 20, 200)
        rates = BuildingRates(s0=0, gx=0, gy=0, alpha=1e-6)
        T = 10.0
        pt = np.array([[0.0, 0.0, 0.0]])
        out = deform_points(pt, geom, rates, T)
        np.testing.assert_allclose(out, pt)

    def test_footprint_unchanged_at_roof(self):
        geom = BuildingGeometry(40, 20, 200)
        rates = BuildingRates(s0=0, gx=0, gy=0, alpha=1e-5)
        T = 10.0
        ne = np.array([[geom.lx, geom.ly, geom.h]])
        out = deform_points(ne, geom, rates, T)[0]
        self.assertAlmostEqual(out[0], geom.lx)
        self.assertAlmostEqual(out[1], geom.ly)

    def test_building_height_reduced_uniformly(self):
        geom = BuildingGeometry(40, 20, 200)
        rates = BuildingRates(s0=0, gx=0, gy=0, alpha=1e-5)
        T = 10.0
        scale = building_length_scale(geom.h, rates.alpha, T)
        mid = np.array([[20.0, 10.0, 100.0]])
        roof = np.array([[40.0, 20.0, geom.h]])
        out_mid = deform_points(mid, geom, rates, T)[0]
        out_roof = deform_points(roof, geom, rates, T)[0]
        self.assertAlmostEqual(out_mid[2], 100.0 * scale)
        self.assertAlmostEqual(out_roof[2], geom.h * scale)

    def test_sw_corner_fixed_at_ground(self):
        geom = BuildingGeometry(40, 20, 200)
        rates = BuildingRates(s0=0, gx=0, gy=0, alpha=1e-5)
        T = 10.0
        sw = np.array([[0.0, 0.0, 0.0]])
        out = deform_points(sw, geom, rates, T)[0]
        np.testing.assert_allclose(out, sw[0])

class TestCombined(unittest.TestCase):
    def test_hand_computed_vertex(self):
        geom = BuildingGeometry(10, 5, 50)
        rates = BuildingRates(s0=0.01, gx=0.001, gy=0, alpha=2e-7)
        T = 5.0
        x, y, z = 10.0, 5.0, 50.0
        tx = rates.gx * T
        length_scale = building_length_scale(geom.h, rates.alpha, T)
        z_s = z * length_scale
        x_t = x + z_s * tx
        z_t = z_s - rates.s0 * T - x * tx
        expected = np.array([x_t, y, z_t])
        out = deform_points(np.array([[x, y, z]]), geom, rates, T)
        np.testing.assert_allclose(out[0], expected, rtol=1e-9)


class TestValidation(unittest.TestCase):
    def test_invalid_geometry(self):
        geom = BuildingGeometry(0, 20, 200)
        rates = BuildingRates(0.02, 0, 0, 0)
        self.assertTrue(validate_parameters(geom, rates, 10))

    def test_large_shrinkage_still_valid(self):
        geom = BuildingGeometry(40, 20, 200)
        rates = BuildingRates(0, 0, 0, 1.0)
        self.assertEqual(validate_parameters(geom, rates, 10), [])
        s = summary(geom, rates, 10)
        self.assertTrue(s["shrinkage_clamped"])


class TestSummary(unittest.TestCase):
    def test_summary_valid_flag(self):
        geom = BuildingGeometry(40, 20, 200)
        rates = BuildingRates(0.02, 0, 0, 1e-6)
        s = summary(geom, rates, 10)
        self.assertTrue(s["valid"])
        self.assertEqual(s["errors"], [])


@unittest.skipUnless(HAS_PYVISTA, "pyvista not installed")
class TestMesh(unittest.TestCase):
    def test_make_mesh_point_count(self):
        mesh = make_building_mesh(40, 20, 200)
        self.assertGreaterEqual(mesh.n_points, 8)


if __name__ == "__main__":
    unittest.main()
