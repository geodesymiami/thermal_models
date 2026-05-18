"""Kinematic model: building prism under settlement, tilt, and elevation-dependent shrinkage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MIN_LENGTH_SCALE = 0.01


@dataclass(frozen=True)
class BuildingRates:
    """Settlement and shrinkage rates (per year)."""

    s0: float  # uniform settlement, m/yr (positive = downward)
    gx: float  # settlement gradient d(w)/dx, m/yr per m
    gy: float  # settlement gradient d(w)/dy, m/yr per m
    alpha: float  # vertical shrinkage rate (/yr/m); total length strain ε_L = α·H·T


@dataclass(frozen=True)
class BuildingGeometry:
    lx: float  # footprint x, m
    ly: float  # footprint y, m
    h: float  # height, m


def settlement_w(
    x: np.ndarray | float,
    y: np.ndarray | float,
    rates: BuildingRates,
    period_yr: float,
) -> np.ndarray:
    """Foundation settlement w(x,y) in m at z=0 (positive = downward)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (rates.s0 + rates.gx * x + rates.gy * y) * period_yr


def tilt_angles_rad(rates: BuildingRates, period_yr: float) -> tuple[float, float]:
    """Small-angle pitch (about y) and roll (about x) from settlement gradients, radians."""
    return rates.gx * period_yr, rates.gy * period_yr


def footprint_center(lx: float, ly: float) -> tuple[float, float]:
    return lx / 2.0, ly / 2.0


def building_length_scale(
    height_m: float,
    alpha: float,
    period_yr: float,
    min_scale: float = MIN_LENGTH_SCALE,
) -> float:
    """
    Uniform vertical scale for the whole building from shrinkage linear in elevation H.

    Total length strain ε_L = α·H·T; all z coordinates scale by (1 − ε_L).
    """
    strain = alpha * height_m * period_yr
    return float(max(1.0 - strain, min_scale))


def apply_rigid_tilt(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rates: BuildingRates,
    period_yr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rigid tilt from settlement gradients about the SW ground corner (0, 0, 0).

    Small-angle pitch θx = gₓ·T (about y) and roll θy = gᵧ·T (about x), plus uniform s₀·T downward.
    """
    tx, ty = tilt_angles_rad(rates, period_yr)
    w0 = rates.s0 * period_yr
    x_t = x + z * tx
    y_t = y + z * ty
    z_t = z - w0 - x * tx - y * ty
    return x_t, y_t, z_t


def apply_shrinkage(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    height_m: float,
    alpha: float,
    period_yr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shorten building height only; footprint x,y unchanged. Ground z=0 stays fixed."""
    scale = building_length_scale(height_m, alpha, period_yr)
    return x, y, z * scale


def deform_points(
    points: np.ndarray,
    geom: BuildingGeometry,
    rates: BuildingRates,
    period_yr: float,
) -> np.ndarray:
    """Apply vertical length shrinkage, then rigid settlement tilt."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    x_s, y_s, z = apply_shrinkage(x, y, z, geom.h, rates.alpha, period_yr)
    x_t, y_t, z_t = apply_rigid_tilt(x_s, y_s, z, rates, period_yr)

    out = np.empty_like(pts)
    out[:, 0] = x_t
    out[:, 1] = y_t
    out[:, 2] = z_t
    return out


def validate_parameters(
    geom: BuildingGeometry,
    rates: BuildingRates,
    period_yr: float,
) -> list[str]:
    """Return human-readable validation errors (empty if OK)."""
    errors: list[str] = []
    if geom.lx <= 0 or geom.ly <= 0 or geom.h <= 0:
        errors.append("Footprint and height must be positive.")
    if period_yr <= 0:
        errors.append("Time period must be positive.")
    if rates.alpha < 0:
        errors.append("Shrinkage rate α must be ≥ 0.")
    return errors


def corner_points(lx: float, ly: float, h: float) -> np.ndarray:
    """Eight corners of the footprint prism (SW ground → NE roof)."""
    xs = (0.0, lx)
    ys = (0.0, ly)
    zs = (0.0, h)
    return np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)


def summary(
    geom: BuildingGeometry,
    rates: BuildingRates,
    period_yr: float,
) -> dict[str, Any]:
    """Computed totals for GUI display."""
    corners = corner_points(geom.lx, geom.ly, geom.h)
    w_corners = settlement_w(corners[:, 0], corners[:, 1], rates, period_yr)
    ground_w = w_corners[corners[:, 2] == 0]
    roof_w = w_corners[corners[:, 2] == geom.h]
    length_strain = rates.alpha * geom.h * period_yr
    length_scale = building_length_scale(geom.h, rates.alpha, period_yr)
    height_after_m = geom.h * length_scale
    tx, ty = tilt_angles_rad(rates, period_yr)
    errors = validate_parameters(geom, rates, period_yr)
    shrinkage_clamped = length_strain > 1.0 - 1e-9
    return {
        "period_yr": period_yr,
        "settlement_ground_min_m": float(np.min(ground_w)),
        "settlement_ground_max_m": float(np.max(ground_w)),
        "settlement_roof_min_m": float(np.min(roof_w)),
        "settlement_roof_max_m": float(np.max(roof_w)),
        "settlement_corner_sw_ground_m": float(settlement_w(0, 0, rates, period_yr)),
        "settlement_corner_ne_roof_m": float(
            settlement_w(geom.lx, geom.ly, rates, period_yr)
        ),
        "building_length_strain": length_strain,
        "building_length_scale": length_scale,
        "height_after_shrink_m": height_after_m,
        "height_reduction_m": geom.h - height_after_m,
        # legacy keys used by GUI
        "roof_horizontal_strain": length_strain,
        "roof_horizontal_scale": length_scale,
        "tilt_pitch_rad": tx,
        "tilt_roll_rad": ty,
        "tilt_pitch_deg": float(np.degrees(tx)),
        "tilt_roll_deg": float(np.degrees(ty)),
        "shrinkage_clamped": shrinkage_clamped,
        "valid": len(errors) == 0,
        "errors": errors,
    }


def make_building_mesh(lx: float, ly: float, h: float):
    """PyVista surface mesh of a rectangular prism."""
    import pyvista as pv

    return pv.Box(bounds=(0, lx, 0, ly, 0, h))


def make_building_wireframe(lx: float, ly: float, h: float):
    """PyVista line mesh outlining the prism edges."""
    return make_building_mesh(lx, ly, h).extract_all_edges()


def deform_mesh(mesh, geom: BuildingGeometry, rates: BuildingRates, period_yr: float):
    """Return a copy of mesh with deformed vertex positions."""
    deformed = mesh.copy(deep=True)
    deformed.points = deform_points(mesh.points, geom, rates, period_yr)
    return deformed
