# PHY 221 Soccer Free-Kick Simulator

This Python project builds a soccer free-kick simulation one physics model at a time. It compares **ideal projectile motion**, **aerodynamic drag**, and a three-dimensional **drag-plus-spin** model using the same defensive wall and regulation-size goal.

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

## Model 3: spin and the Magnus force

Model 3 uses three coordinates: x points toward the goal, y points sideways, and z points upward. It adds the Magnus force

```text
F_M = (1/2) rho C_M A r (omega x v)
```

where `omega x v` is the cross product of angular velocity and ball velocity. Spin about the vertical z-axis bends the ball sideways; spin about a horizontal axis can make it rise or dip. The default is 600 rpm of sidespin with a Magnus coefficient of 0.25.

The wall now has adjustable width and lateral position. Model 3 checks whether the full ball goes above or around the wall, then whether it fits between both posts, below the crossbar, and above the ground.

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

Run all three models and print their comparison:

```powershell
python spin_simulation.py
```

Reverse the curve by using a negative spin rate, or change the spin axis:

```powershell
python spin_simulation.py --spin-rpm -600
python spin_simulation.py --spin-rpm 600 --spin-axis-y -1 --spin-axis-z 0
```

This verified example bends a lower kick around the left edge of an offset wall and into the goal:

```powershell
python spin_simulation.py --speed 20 --angle 20 --spin-rpm -600 --wall-center 1.5
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
python spin_simulation.py --plot
```

## Test it

```powershell
python -m unittest discover -s tests -v
```

The tests verify the analytical baseline values, wall and goal decisions, drag behavior, Magnus-force direction, reversed-spin symmetry, zero-spin agreement with Model 2, and three-dimensional wall and goal geometry.

## Example results

| Speed | Angle | Wall result | Goal result |
|---:|---:|---|---|
| 20 m/s | 20 degrees | Clears | Scores |
| 25 m/s | 30 degrees | Clears | Over crossbar |
| 25 m/s | 10 degrees | Hits wall | Does not score |

## Model sequence

1. Ideal projectile motion — complete.
2. Aerodynamic drag — complete.
3. Drag plus ball spin and the Magnus force — complete.
4. Adjustable controls, experiments, validation, and final presentation material — next.
