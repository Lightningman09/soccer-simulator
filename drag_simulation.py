"""Model 2: soccer free-kick motion with quadratic aerodynamic drag.

Unlike the ideal model, drag acceleration changes with the ball's velocity, so
there is no simple constant-acceleration formula for the whole flight. This
module advances the motion in short time steps and compares the numerical
result with the analytical ideal-projectile baseline.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass

from soccer_simulation import (
    Field,
    Kick,
    ShotResult,
    evaluate_shot,
    flight_time,
    horizontal_range,
    judge_shot,
    trajectory,
)


@dataclass(frozen=True)
class Aerodynamics:
    """Physical constants for a size-5 soccer ball in air."""

    mass: float = 0.43
    air_density: float = 1.225
    drag_coefficient: float = 0.25


@dataclass(frozen=True)
class DragSimulation:
    """Complete numerical trajectory and its measured results."""

    times: tuple[float, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]
    vx: tuple[float, ...]
    vy: tuple[float, ...]
    flight_time: float
    horizontal_range: float
    maximum_height: float
    shot: ShotResult


def drag_acceleration(
    vx: float,
    vy: float,
    field: Field | None = None,
    aerodynamics: Aerodynamics | None = None,
    gravity: float = 9.81,
) -> tuple[float, float]:
    """Return acceleration from gravity and quadratic drag.

    Drag magnitude is (1/2) rho C_d A v^2. Resolving the force opposite the
    velocity makes each component proportional to ``v * vx`` or ``v * vy``.
    """

    if field is None:
        field = Field()
    if aerodynamics is None:
        aerodynamics = Aerodynamics()
    _validate_drag_inputs(field, aerodynamics, gravity, 0.001)
    speed = math.hypot(vx, vy)
    area = math.pi * field.ball_radius**2
    drag_factor = (
        0.5
        * aerodynamics.air_density
        * aerodynamics.drag_coefficient
        * area
        / aerodynamics.mass
    )
    return -drag_factor * speed * vx, -gravity - drag_factor * speed * vy


def _validate_drag_inputs(
    field: Field,
    aerodynamics: Aerodynamics,
    gravity: float,
    time_step: float,
) -> None:
    if min(aerodynamics.mass, aerodynamics.air_density) <= 0:
        raise ValueError("Ball mass and air density must be positive.")
    if aerodynamics.drag_coefficient < 0:
        raise ValueError("Drag coefficient cannot be negative.")
    if field.ball_radius <= 0:
        raise ValueError("Ball radius must be positive.")
    if gravity <= 0:
        raise ValueError("Gravity must be positive.")
    if not 0 < time_step <= 0.02:
        raise ValueError("Time step must be greater than 0 and at most 0.02 s.")


def _interpolate_at_x(
    x_values: list[float], y_values: list[float], target_x: float
) -> tuple[float, bool]:
    """Linearly interpolate height where a sampled path crosses ``target_x``."""

    for index in range(1, len(x_values)):
        if x_values[index] >= target_x:
            x0, x1 = x_values[index - 1], x_values[index]
            fraction = (target_x - x0) / (x1 - x0)
            return y_values[index - 1] + fraction * (
                y_values[index] - y_values[index - 1]
            ), True
    return 0.0, False


def simulate_drag(
    kick: Kick,
    field: Field | None = None,
    aerodynamics: Aerodynamics | None = None,
    gravity: float = 9.81,
    time_step: float = 0.001,
) -> DragSimulation:
    """Numerically integrate the flight using the semi-implicit Euler method.

    Updating velocity before position is simple to explain and more stable than
    ordinary forward Euler. A 1 ms default step makes the numerical error small
    compared with the uncertainty in real ball and air properties.
    """

    if field is None:
        field = Field()
    if aerodynamics is None:
        aerodynamics = Aerodynamics()

    # Reuse the ideal model's validation of the kick and field geometry.
    evaluate_shot(kick, field, gravity)
    _validate_drag_inputs(field, aerodynamics, gravity, time_step)

    theta = math.radians(kick.angle_degrees)
    t = 0.0
    x = 0.0
    y = kick.initial_height
    vx = kick.speed * math.cos(theta)
    vy = kick.speed * math.sin(theta)

    times = [t]
    x_values = [x]
    y_values = [y]
    vx_values = [vx]
    vy_values = [vy]

    # A valid upward kick should land in seconds; this guard prevents an
    # accidental infinite loop if the model is modified later.
    maximum_steps = math.ceil(60.0 / time_step)
    for _ in range(maximum_steps):
        ax, ay = drag_acceleration(vx, vy, field, aerodynamics, gravity)
        next_vx = vx + ax * time_step
        next_vy = vy + ay * time_step
        next_x = x + next_vx * time_step
        next_y = y + next_vy * time_step
        next_t = t + time_step

        if next_y < 0 and next_t > time_step:
            # Replace the below-ground sample with the interpolated impact.
            fraction = y / (y - next_y)
            times.append(t + fraction * time_step)
            x_values.append(x + fraction * (next_x - x))
            y_values.append(0.0)
            vx_values.append(vx + fraction * (next_vx - vx))
            vy_values.append(vy + fraction * (next_vy - vy))
            break

        t, x, y, vx, vy = next_t, next_x, next_y, next_vx, next_vy
        times.append(t)
        x_values.append(x)
        y_values.append(y)
        vx_values.append(vx)
        vy_values.append(vy)
    else:
        raise RuntimeError(
            "Drag simulation did not reach the ground within 60 seconds."
        )

    wall_y, reaches_wall = _interpolate_at_x(x_values, y_values, field.wall_distance)
    goal_y, reaches_goal = _interpolate_at_x(x_values, y_values, field.goal_distance)
    shot = judge_shot(
        kick,
        field,
        wall_y,
        goal_y,
        reaches_wall=reaches_wall,
        reaches_goal=reaches_goal,
    )

    return DragSimulation(
        tuple(times),
        tuple(x_values),
        tuple(y_values),
        tuple(vx_values),
        tuple(vy_values),
        times[-1],
        x_values[-1],
        max(y_values),
        shot,
    )


def ideal_maximum_height(kick: Kick, gravity: float = 9.81) -> float:
    """Return the ideal model's analytical peak height."""

    vertical_speed = kick.speed * math.sin(math.radians(kick.angle_degrees))
    return kick.initial_height + vertical_speed**2 / (2 * gravity)


def comparison_rows(
    kick: Kick, field: Field, drag: DragSimulation, gravity: float = 9.81
) -> list[tuple[str, float, float]]:
    """Return metrics used by the CLI comparison table."""

    ideal_shot = evaluate_shot(kick, field, gravity)
    return [
        ("Flight time (s)", flight_time(kick, gravity), drag.flight_time),
        (
            "Maximum height (m)",
            ideal_maximum_height(kick, gravity),
            drag.maximum_height,
        ),
        ("Range (m)", horizontal_range(kick, gravity), drag.horizontal_range),
        ("Height at wall (m)", ideal_shot.wall_height, drag.shot.wall_height),
        ("Height at goal (m)", ideal_shot.goal_height, drag.shot.goal_height),
    ]


def print_comparison(
    kick: Kick, field: Field, drag: DragSimulation, gravity: float = 9.81
) -> None:
    """Print ideal and drag measurements in a readable table."""

    print(f"{'Measurement':<24} {'Ideal':>10} {'With drag':>12} {'Change':>10}")
    print("-" * 60)
    for label, ideal_value, drag_value in comparison_rows(kick, field, drag, gravity):
        change = drag_value - ideal_value
        print(f"{label:<24} {ideal_value:>10.2f} {drag_value:>12.2f} {change:>+10.2f}")
    print()
    print(f"Ideal outcome: {evaluate_shot(kick, field, gravity).outcome}")
    print(f"Drag outcome:  {drag.shot.outcome}")


def plot_comparison(
    kick: Kick, field: Field, drag: DragSimulation, gravity: float = 9.81
) -> None:
    """Plot the ideal and drag trajectories on the same axes."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib: pip install matplotlib"
        ) from error

    ideal_x, ideal_y = trajectory(kick, gravity, points=400)
    plt.plot(ideal_x, ideal_y, linestyle="--", label="Model 1: ideal")
    plt.plot(drag.x, drag.y, label="Model 2: quadratic drag")
    plt.plot(
        [field.wall_distance, field.wall_distance],
        [0, field.wall_height],
        linewidth=5,
        color="tab:red",
        label="Defensive wall",
    )
    plt.plot(
        [field.goal_distance, field.goal_distance],
        [0, field.goal_height],
        linewidth=5,
        color="tab:green",
        label="Goal plane",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.xlim(
        0, min(max(ideal_x[-1], drag.horizontal_range) + 2, field.goal_distance + 40)
    )
    plt.ylim(bottom=0)
    plt.xlabel("Horizontal distance (m)")
    plt.ylabel("Height (m)")
    plt.title("Soccer Free Kick: Ideal Motion vs. Aerodynamic Drag")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=25.0, help="launch speed in m/s")
    parser.add_argument(
        "--angle", type=float, default=30.0, help="launch angle in degrees"
    )
    parser.add_argument(
        "--wall-distance", type=float, default=9.15, help="distance to wall in m"
    )
    parser.add_argument(
        "--wall-height", type=float, default=2.0, help="wall height in m"
    )
    parser.add_argument(
        "--goal-distance", type=float, default=20.0, help="distance to goal in m"
    )
    parser.add_argument(
        "--lateral-offset", type=float, default=0.0, help="sideways offset at goal in m"
    )
    parser.add_argument("--mass", type=float, default=0.43, help="ball mass in kg")
    parser.add_argument(
        "--air-density", type=float, default=1.225, help="air density in kg/m^3"
    )
    parser.add_argument(
        "--drag-coefficient",
        type=float,
        default=0.25,
        help="dimensionless drag coefficient",
    )
    parser.add_argument(
        "--time-step", type=float, default=0.001, help="numerical step in seconds"
    )
    parser.add_argument(
        "--plot", action="store_true", help="plot ideal and drag trajectories"
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    kick = Kick(args.speed, args.angle, lateral_offset=args.lateral_offset)
    field = Field(args.wall_distance, args.wall_height, args.goal_distance)
    aerodynamics = Aerodynamics(args.mass, args.air_density, args.drag_coefficient)
    drag = simulate_drag(kick, field, aerodynamics, time_step=args.time_step)
    print_comparison(kick, field, drag)
    if args.plot:
        plot_comparison(kick, field, drag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
