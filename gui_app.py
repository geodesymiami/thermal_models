#!/usr/bin/env python3
"""
Interactive 3D viewer: high-rise building before/after settlement, tilt, and shrinkage.

Positive settlement rates move the building downward. Shrinkage shortens building height only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from building_model import (
    BuildingGeometry,
    BuildingRates,
    deform_mesh,
    footprint_center,
    make_building_mesh,
    make_building_wireframe,
    summary,
)

DEFAULTS = {
    "lx": 40.0,
    "ly": 20.0,
    "h": 200.0,
    "period_yr": 10.0,
    "s0": 0.02,
    "gx": 0.0,
    "gy": 0.0,
    "alpha": 5e-5,
}

EQUATIONS_SYMBOLIC = """\
Foundation settlement (positive = downward):
  w(x, y) = (s₀ + gₓ·x + gᵧ·y) · T

Rigid tilt about SW corner (small angles θx = gₓ·T, θy = gᵧ·T):
  x' = x + z·θx ,  y' = y + z·θy
  z' = z − s₀·T − x·θx − y·θy

Shrinkage (building length only; footprint unchanged):
  α in /yr/m — length strain per meter of height per year
  ε_L = α·H·T ,  z_s = z·max(1 − ε_L, 0.01) ,  x,y unchanged
  then apply rigid tilt
"""

HELP_TEXT = EQUATIONS_SYMBOLIC + (
    "\nα (/yr/m): total vertical strain ε_L = α×H×T. "
    "Example α=5×10⁻⁵, H=200 m, T=10 yr → ε_L=10% (height 180 m). "
    "Origin at SW corner, z = 0 at ground."
)


def _decimals_for_value(value: float, max_decimals: int = 6) -> int:
    """Use 0 fraction digits for integers; otherwise show only needed digits."""
    rounded = round(value)
    if abs(value - rounded) < 1e-12:
        return 0
    text = f"{value:.{max_decimals}f}".rstrip("0")
    if "." not in text:
        return 0
    return min(len(text.split(".")[1]), max_decimals)


def _spin_fixed(
    minimum: float,
    maximum: float,
    value: float,
    *,
    decimals: int,
    step: float,
    suffix: str = "",
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    return box


def _spin_adaptive(
    minimum: float,
    maximum: float,
    value: float,
    *,
    step: float = 1.0,
    suffix: str = "",
    max_decimals: int = 4,
) -> QDoubleSpinBox:
    """Show decimals only when the value has a fractional part."""
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    if suffix:
        box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    box.setDecimals(_decimals_for_value(value, max_decimals))
    box.setValue(value)

    def _update_decimals(v: float, b: QDoubleSpinBox = box) -> None:
        b.setDecimals(_decimals_for_value(v, max_decimals))

    box.valueChanged.connect(_update_decimals)
    return box


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Building settlement & shrinkage")
        self.resize(1400, 720)

        self._plotter_before = None
        self._plotter_after = None
        self._camera_sync_block = False
        self._cached_geom_key: tuple[float, float, float] | None = None
        self._cached_before_solid = None
        self._cached_before_edges = None
        self._last_undeformed_faces: bool | None = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(100)
        self._debounce.timeout.connect(self._refresh_scene)

        self._build_controls()
        self._build_menus()

        splitter = QSplitter(Qt.Horizontal)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_inner = QWidget()
        left_inner.setLayout(self._left_layout)
        left_scroll.setWidget(left_inner)
        splitter.addWidget(left_scroll)

        views_splitter = QSplitter(Qt.Horizontal)
        before_panel, self._plotter_before = self._make_view_panel("Undeformed")
        after_panel, self._plotter_after = self._make_view_panel("Deformed")
        views_splitter.addWidget(before_panel)
        views_splitter.addWidget(after_panel)
        views_splitter.setStretchFactor(0, 1)
        views_splitter.setStretchFactor(1, 1)
        splitter.addWidget(views_splitter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1040])
        views_splitter.setSizes([520, 520])

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(splitter)
        layout.setContentsMargins(4, 4, 4, 4)
        self.setCentralWidget(central)

        self._connect_signals()
        self._connect_linked_cameras()
        self._refresh_scene()

    def _build_controls(self) -> None:
        self._lx = _spin_adaptive(1, 500, DEFAULTS["lx"], step=1, suffix=" m")
        self._ly = _spin_adaptive(1, 500, DEFAULTS["ly"], step=1, suffix=" m")
        self._h = _spin_adaptive(1, 1000, DEFAULTS["h"], step=1, suffix=" m")
        self._period = _spin_adaptive(0.1, 200, DEFAULTS["period_yr"], step=1, suffix=" yr")
        # Wide ranges for visualization (e.g. s₀ = 9 m/yr, large tilt gradients)
        self._s0 = _spin_fixed(0, 100, DEFAULTS["s0"], decimals=3, step=0.01, suffix=" m/yr")
        self._gx = _spin_fixed(-10, 10, DEFAULTS["gx"], decimals=4, step=0.001, suffix=" m/yr/m")
        self._gy = _spin_fixed(-10, 10, DEFAULTS["gy"], decimals=4, step=0.001, suffix=" m/yr/m")
        self._alpha = _spin_fixed(0, 0.01, DEFAULTS["alpha"], decimals=6, step=1e-5, suffix=" /yr/m")

        self._show_faces = QCheckBox("Translucent faces")
        self._show_faces.setChecked(True)
        self._link_cameras = QCheckBox("Link camera (both views)")
        self._link_cameras.setChecked(True)

        self._equations_label = QLabel()
        self._equations_label.setWordWrap(True)
        self._equations_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        mono = QFont()
        mono.setFamilies(["Menlo", "Consolas", "Courier New", "monospace"])
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self._equations_label.setFont(mono)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        equations_box = QGroupBox("Equations")
        equations_layout = QVBoxLayout(equations_box)
        equations_layout.addWidget(self._equations_label)

        geom_box = QGroupBox("Geometry")
        geom_form = QFormLayout(geom_box)
        geom_form.addRow("Footprint Lx", self._lx)
        geom_form.addRow("Footprint Ly", self._ly)
        geom_form.addRow("Height H", self._h)

        time_box = QGroupBox("Time")
        time_form = QFormLayout(time_box)
        time_form.addRow("Period T", self._period)

        settle_box = QGroupBox("Settlement rates")
        settle_form = QFormLayout(settle_box)
        settle_form.addRow("Uniform s₀", self._s0)
        settle_form.addRow("Gradient gₓ", self._gx)
        settle_form.addRow("Gradient gᵧ", self._gy)

        shrink_box = QGroupBox("Shrinkage")
        shrink_form = QFormLayout(shrink_box)
        shrink_form.addRow("Rate α (/yr/m)", self._alpha)

        view_box = QGroupBox("Display")
        view_layout = QVBoxLayout(view_box)
        view_layout.addWidget(self._show_faces)
        view_layout.addWidget(self._link_cameras)

        summary_box = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.addWidget(self._summary_label)

        self._left_layout = QVBoxLayout()
        self._left_layout.addWidget(equations_box)
        self._left_layout.addWidget(geom_box)
        self._left_layout.addWidget(time_box)
        self._left_layout.addWidget(settle_box)
        self._left_layout.addWidget(shrink_box)
        self._left_layout.addWidget(view_box)
        self._left_layout.addWidget(summary_box)
        self._left_layout.addStretch(1)

    def _make_view_panel(self, title: str) -> tuple[QWidget, QtInteractor]:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel(title)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = heading.font()
        font.setBold(True)
        font.setPointSize(11)
        heading.setFont(font)
        layout.addWidget(heading)
        plotter = QtInteractor(panel)
        layout.addWidget(plotter.interactor, stretch=1)
        return panel, plotter

    def _build_menus(self) -> None:
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction("Reset camera", self._reset_camera)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("Model equations", self._show_help)

    def _connect_signals(self) -> None:
        for w in (
            self._lx, self._ly, self._h, self._period,
            self._s0, self._gx, self._gy, self._alpha,
            self._show_faces, self._link_cameras,
        ):
            if isinstance(w, QDoubleSpinBox):
                w.valueChanged.connect(self._schedule_refresh)
            else:
                w.toggled.connect(self._schedule_refresh)

    def _connect_linked_cameras(self) -> None:
        for plotter in (self._plotter_before, self._plotter_after):
            iren = plotter.iren
            for event in ("InteractionEvent", "EndInteractionEvent"):
                iren.add_observer(event, lambda _o, _e, src=plotter: self._on_view_camera_changed(src))

    def _schedule_refresh(self) -> None:
        self._debounce.start()

    def _read_inputs(self) -> tuple[BuildingGeometry, BuildingRates, float]:
        geom = BuildingGeometry(self._lx.value(), self._ly.value(), self._h.value())
        rates = BuildingRates(
            s0=self._s0.value(),
            gx=self._gx.value(),
            gy=self._gy.value(),
            alpha=self._alpha.value(),
        )
        return geom, rates, self._period.value()

    def _format_equations(self, geom: BuildingGeometry, rates: BuildingRates, period: float) -> str:
        xc, yc = footprint_center(geom.lx, geom.ly)
        s = summary(geom, rates, period)
        length_scale = s["building_length_scale"]
        length_strain = s["building_length_strain"]
        return (
            EQUATIONS_SYMBOLIC
            + "\n── With current inputs ──\n"
            + f"  T = {period:g} yr\n"
            + f"  x_c = {xc:g} m ,  y_c = {yc:g} m\n"
            + f"  w(0,0) = (s₀)·T = {rates.s0 * period:.6g} m\n"
            + f"  w(Lx,Ly) = (s₀ + gₓ·Lx + gᵧ·Ly)·T "
            + f"= {(rates.s0 + rates.gx * geom.lx + rates.gy * geom.ly) * period:.6g} m\n"
            + f"  ε_L = α·H·T = {length_strain:.6g}\n"
            + f"  length scale = {length_scale:.6f}\n"
            + f"  θx = gₓ·T = {s['tilt_pitch_deg']:.4f}° ,  "
            + f"θy = gᵧ·T = {s['tilt_roll_deg']:.4f}°"
        )

    def _format_summary(self, geom: BuildingGeometry, rates: BuildingRates, period: float) -> str:
        s = summary(geom, rates, period)
        lines = [
            f"Period: {s['period_yr']:.2f} yr",
            f"Ground settlement: {s['settlement_ground_min_m']:.4f} – "
            f"{s['settlement_ground_max_m']:.4f} m",
            f"Roof settlement: {s['settlement_roof_min_m']:.4f} – "
            f"{s['settlement_roof_max_m']:.4f} m",
            f"SW ground: {s['settlement_corner_sw_ground_m']:.4f} m",
            f"NE roof: {s['settlement_corner_ne_roof_m']:.4f} m",
            f"Tilt: pitch {s['tilt_pitch_deg']:.4f}°, roll {s['tilt_roll_deg']:.4f}°",
            f"Length shrinkage ε_L: {100 * s['building_length_strain']:.2f}%",
            f"Height after shrink: {s['height_after_shrink_m']:.2f} m "
            f"(Δ {s['height_reduction_m']:.2f} m)",
        ]
        if s.get("shrinkage_clamped"):
            lines.append("⚠ α·H·T > 1: horizontal scale clamped to 0.01 at roof")
        if s["errors"]:
            lines.append("")
            lines.append("⚠ " + "; ".join(s["errors"]))
        return "\n".join(lines)

    def _draw_building(
        self,
        plotter: QtInteractor,
        solid,
        edges,
        color: str,
        *,
        show_faces: bool,
    ) -> None:
        if show_faces:
            plotter.add_mesh(
                solid,
                color=color,
                opacity=0.2,
                show_edges=False,
            )
        plotter.add_mesh(edges, color=color, line_width=2)

    def _geom_key(self, geom: BuildingGeometry) -> tuple[float, float, float]:
        return (geom.lx, geom.ly, geom.h)

    def _get_before_meshes(self, geom: BuildingGeometry):
        key = self._geom_key(geom)
        if key != self._cached_geom_key:
            self._cached_before_solid = make_building_mesh(geom.lx, geom.ly, geom.h)
            self._cached_before_edges = make_building_wireframe(geom.lx, geom.ly, geom.h)
            self._cached_geom_key = key
        return self._cached_before_solid, self._cached_before_edges

    def _populate_plotter(
        self,
        plotter: QtInteractor,
        solid,
        edges,
        color: str,
        *,
        show_faces: bool,
        error: str | None = None,
        reset_camera: bool = True,
    ) -> None:
        saved_camera = None if reset_camera else plotter.camera_position
        plotter.clear()
        plotter.set_background("white")
        plotter.add_axes()
        if error:
            plotter.add_text(error, position="upper_left", color="red", font_size=10)
            if saved_camera is not None:
                plotter.camera_position = saved_camera
            return
        self._draw_building(plotter, solid, edges, color, show_faces=show_faces)
        plotter.show_bounds(grid="back", location="outer", all_edges=True)
        if reset_camera:
            plotter.reset_camera()
        elif saved_camera is not None:
            plotter.camera_position = saved_camera

    def _on_view_camera_changed(self, source: QtInteractor) -> None:
        if self._camera_sync_block or not self._link_cameras.isChecked():
            return
        target = (
            self._plotter_after
            if source is self._plotter_before
            else self._plotter_before
        )
        self._copy_camera(source, target)

    def _copy_camera(self, source: QtInteractor, target: QtInteractor) -> None:
        if not source or not target or source is target:
            return
        self._camera_sync_block = True
        try:
            target.camera_position = source.camera_position
            target.render()
        finally:
            self._camera_sync_block = False

    def _sync_cameras(self) -> None:
        if not self._link_cameras.isChecked():
            return
        if self._plotter_before and self._plotter_after:
            self._copy_camera(self._plotter_before, self._plotter_after)

    def _refresh_undeformed(self, geom: BuildingGeometry, show_faces: bool, *, reset_camera: bool) -> None:
        before_solid, before_edges = self._get_before_meshes(geom)
        self._populate_plotter(
            self._plotter_before,
            before_solid,
            before_edges,
            "steelblue",
            show_faces=show_faces,
            reset_camera=reset_camera,
        )

    def _refresh_deformed(
        self,
        geom: BuildingGeometry,
        rates: BuildingRates,
        period: float,
        show_faces: bool,
    ) -> None:
        s = summary(geom, rates, period)
        before_solid, before_edges = self._get_before_meshes(geom)
        if not s["valid"]:
            self._populate_plotter(
                self._plotter_after,
                before_solid,
                before_edges,
                "crimson",
                show_faces=False,
                error="Invalid parameters — adjust inputs",
                reset_camera=False,
            )
        else:
            after_solid = deform_mesh(before_solid, geom, rates, period)
            after_edges = deform_mesh(before_edges, geom, rates, period)
            self._populate_plotter(
                self._plotter_after,
                after_solid,
                after_edges,
                "crimson",
                show_faces=show_faces,
                reset_camera=False,
            )

    def _refresh_scene(self) -> None:
        geom, rates, period = self._read_inputs()
        self._equations_label.setText(self._format_equations(geom, rates, period))
        self._summary_label.setText(self._format_summary(geom, rates, period))

        show_faces = self._show_faces.isChecked()
        geom_key = self._geom_key(geom)
        geom_changed = geom_key != self._cached_geom_key
        faces_changed = show_faces != self._last_undeformed_faces
        first_draw = self._cached_geom_key is None

        if first_draw or geom_changed or faces_changed:
            self._refresh_undeformed(
                geom,
                show_faces,
                reset_camera=first_draw or geom_changed,
            )
            self._last_undeformed_faces = show_faces

        self._refresh_deformed(geom, rates, period, show_faces)

        if self._link_cameras.isChecked() and (first_draw or geom_changed):
            self._sync_cameras()

    def _reset_camera(self) -> None:
        for plotter in (self._plotter_before, self._plotter_after):
            if plotter:
                plotter.reset_camera()
        self._sync_cameras()

    def _show_help(self) -> None:
        QMessageBox.information(self, "Model", HELP_TEXT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D GUI: building settlement, tilt, and elevation-dependent shrinkage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python tools/thermal_models/simulator_falk/gui_app.py\n"
        "  python tools/thermal_models/simulator_falk/gui_app.py --fullscreen\n",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Open window fullscreen",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QApplication(sys.argv)
    window = MainWindow()
    if args.fullscreen:
        window.showFullScreen()
    else:
        window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
