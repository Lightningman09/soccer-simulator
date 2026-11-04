# PHY 221 Soccer Free-Kick Simulator

This Python project builds a soccer free-kick simulation one physics model at a time. The current version uses **ideal projectile motion** (gravity only) and adds a defensive wall and a regulation-size goal.

## Current model

The ball follows

```text
x(t) = v0 cos(theta) t
y(t) = y0 + v0 sin(theta) t - (1/2) g t^2
```

The simulator calculates the ball's height where it reaches:

- an adjustable defensive wall (default: 9.15 m away and 2.0 m tall), and
- a regulation goal (7.32 m wide and 2.44 m high, default distance: 20 m).

The ball is modeled with a 0.11 m radius. A kick clears the wall only when the **bottom of the ball** is above it. It scores only when the **entire ball** fits between the posts and below the crossbar at the goal line.

This version has no air resistance or spin, so sideways motion is represented by one constant lateral offset. Later models can replace that with a calculated curved trajectory.

## Run it

Python 3.10 or newer is recommended. The calculations use only the Python standard library.

```powershell
python soccer_simulation.py
```

The original 25 m/s, 30-degree baseline clears the wall but passes over a goal 20 m away. A lower launch angle demonstrates a scoring kick:

```powershell
python soccer_simulation.py --speed 20 --angle 20
```

Adjust the geometry:

```powershell
python soccer_simulation.py --speed 20 --angle 20 --wall-distance 9.15 --wall-height 2.0 --goal-distance 20
```

To display a trajectory plot with the wall, goal plane, and evaluation points, first install Matplotlib and add `--plot`:

```powershell
python -m pip install -r requirements.txt
python soccer_simulation.py --speed 20 --angle 20 --plot
```

## Test it

```powershell
python -m unittest discover -s tests -v
```

The tests verify the analytical baseline values, a goal, a wall collision, a shot over the crossbar, a shot outside the post, finite ball-size clearance, and invalid field geometry.

## Example results

| Speed | Angle | Wall result | Goal result |
|---:|---:|---|---|
| 20 m/s | 20 degrees | Clears | Scores |
| 25 m/s | 30 degrees | Clears | Over crossbar |
| 25 m/s | 10 degrees | Hits wall | Does not score |

## Next model

Add aerodynamic drag and compare its trajectory, range, wall clearance, and goal result with this ideal baseline.
