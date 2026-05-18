# Building settlement & shrinkage simulator (Falk)

Interactive 3D visualization (side-by-side **undeformed** and **deformed** views) of a rectangular high-rise under:

- **Settlement** — uniform rate plus gradients in x and y (tilt)
- **Shrinkage** — shortens building height only (footprint unchanged); rate α in `/yr/m`

All inputs are **rates**; total effect = rate × time period **T** (default 10 years).

## Install

```bash
python3 -m venv .venv-simulator-falk
source .venv-simulator-falk/bin/activate
pip install -r tools/thermal_models/simulator_falk/requirements.txt
```

## Run GUI

```bash
python tools/thermal_models/simulator_falk/gui_app.py
python tools/thermal_models/simulator_falk/gui_app.py --fullscreen
python tools/thermal_models/simulator_falk/gui_app.py --help
```

## Tests

```bash
cd tools/thermal_models/simulator_falk
python -m unittest discover -s tests -v
```

## Model (summary)

- Origin at south-west corner of footprint, z = 0 at ground.
- Settlement: `w = (s₀ + gₓ·x + gᵧ·y)·T`, then `z' = z − w` (positive rates → downward).
- Shrinkage: `ε_L = α·H·T`, height scales by `1 − ε_L` (α in `/yr/m`, strain per meter height per year).

See `PLAN.md` for full specification.
