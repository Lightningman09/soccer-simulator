# PHY 221 Soccer Free-Kick Simulator

This Python project builds a soccer free-kick simulation one physics model at a time. It now compares **ideal projectile motion** with a numerical model that includes **aerodynamic drag**, using the same defensive wall and regulation-size goal for both.

## Model 1: ideal projectile motion

The ball follows

```text
x(t) = v0 cos(theta) t
y(t) = y0 + v0 sin(theta) t - (1/2) g t^2
```

The simulator calculates the ball's height where it reaches:

- an adjustable defensive wall (default: 9.15 m away and 2.0 m tall), and
- a regulation goal (7.32 m wide and 2.44 m high, default distance: 20 m).

The ball is modeled with a 0.11 m radius. A kick clears the wall only when the **bottom of the ball** is above it. It scores only when the **entire ball** fits between the posts and below the crossbar at the goal line.

Model 1 has no air resistance or spin, so sideways motion is represented by one constant lateral offset.

## Model 2: aerodynamic drag

The drag model adds a force opposite the ball's velocity:

```text
F_drag = (1/2) rho C_d A v^2
```

Default physical values are:

| Quantity | Value |
|---|---:|
| Ball mass | 0.43 kg |
| Ball radius | 0.11 m |
| Air density | 1.225 kg/m^3 |
| Drag coefficient | 0.25 |
| Time step | 0.001 s |

Because the drag force changes as the ball slows down, Model 2 uses a semi-implicit Euler numerical calculation rather than one equation for the whole flight. The program linearly interpolates the wall, goal, and ground crossings between time steps.

## Run it

Python 3.10 or newer is recommended. The calculations use only the Python standard library.

```powershell
python soccer_simulation.py
```

The original 25 m/s, 30-degree baseline clears the wall but passes over a goal 20 m away. A lower launch angle demonstrates a scoring kick:

```powershell
python soccer_simulation.py --speed 20 --angle 20
```

Run Model 2 to print a side-by-side table of flight time, maximum height, range, wall height, goal height, and outcomes:

```powershell
python drag_simulation.py
```

Change the physical inputs if desired:

```powershell
python drag_simulation.py --speed 25 --angle 30 --mass 0.43 --air-density 1.225 --drag-coefficient 0.25
```

Adjust the geometry:

```powershell
python soccer_simulation.py --speed 20 --angle 20 --wall-distance 9.15 --wall-height 2.0 --goal-distance 20
```

To display a trajectory plot with the wall, goal plane, and evaluation points, first install Matplotlib and add `--plot`:

```powershell
python -m pip install -r requirements.txt
python soccer_simulation.py --speed 20 --angle 20 --plot
python drag_simulation.py --plot
```

## Test it

```powershell
python -m unittest discover -s tests -v
```

The tests verify the analytical baseline values, wall and goal decisions, the direction of the drag force, convergence to the ideal solution when drag is zero, and the expected reductions in maximum height and range.

## Example results

| Speed | Angle | Wall result | Goal result |
|---:|---:|---|---|
| 20 m/s | 20 degrees | Clears | Scores |
| 25 m/s | 30 degrees | Clears | Over crossbar |
| 25 m/s | 10 degrees | Hits wall | Does not score |

## Model sequence

1. Ideal projectile motion — complete.
2. Aerodynamic drag — complete.
3. Drag plus ball spin and the Magnus force — next.
