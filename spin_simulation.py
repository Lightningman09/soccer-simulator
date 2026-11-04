"""Model 3: three-dimensional soccer-ball motion with drag and spin.

Coordinates use x toward the goal, y sideways, and z vertically upward. The
Magnus force is based on the cross product omega x velocity, so changing the
spin axis changes whether the ball bends sideways, rises, or dips.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass

from drag_simulation import (
    Aerodynamics,
    DragSimulation,
    ideal_maximum_height,
    simulate_drag,
)
from soccer_simulation import (
    Field,
    Kick,
    evaluate_shot,
    flight_time,
    horizontal_range,
    trajectory,
)


@dataclass(frozen=True)
class Spin:
    """Ball spin rate and axis in three dimensions."""

    rate_rpm: float = 600.0
    axis_x: float = 0.0
    axis_y: float = 0.0
    axis_z: float = 1.0

    def angular_velocity(self) -> tuple[float, float, float]:
        """Return the angular-velocity vector in radians per second."""

        axis_length = math.sqrt(self.axis_x**2 + self.axis_y**2 + self.axis_z**2)
        if axis_length == 0:
            if self.rate_rpm == 0:
                return 0.0, 0.0, 0.0
            raise ValueError("A nonzero spin rate requires a nonzero spin axis.")
        omega = self.rate_rpm * 2 * math.pi / 60
        return (
            omega * self.axis_x / axis_length,
            omega * self.axis_y / axis_length,
            omega * self.axis_z / axis_length,
        )


@dataclass(frozen=True)
class SpinShotResult:
    """Wall and goal measurements for a three-dimensional shot."""

    wall_lateral: float
    wall_height: float
    goal_lateral: float
    goal_height: float
    reaches_wall: bool
    avoids_wall: bool
    reaches_goal: bool
    inside_goal_opening: bool
    scores: bool
    outcome: str


@dataclass(frozen=True)
class SpinSimulation:
    """Complete 3D numerical trajectory and summary measurements."""

    times: tuple[float, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]
    z: tuple[float, ...]
    vx: tuple[float, ...]
    vy: tuple[float, ...]
    vz: tuple[float, ...]
    flight_time: float
    horizontal_range: float
    maximum_height: float
    maximum_sideways_displacement: float
    shot: SpinShotResult


def magnus_acceleration(
    vx: float,
    vy: float,
    vz: float,
    spin: Spin,
    field: Field | None = None,
    aerodynamics: Aerodynamics | None = None,
    magnus_coefficient: float = 0.25,
) -> tuple[float, float, float]:
    """Return Magnus acceleration from ``omega x velocity``.

    The force model is

        F_M = (1/2) rho C_M A r (omega x v)

    where ``C_M`` is an adjustable dimensionless coefficient. The cross product
    automatically makes the force perpendicular to both spin and velocity.
    """

    if field is None:
        field = Field()
    if aerodynamics is None:
        aerodynamics = Aerodynamics()
    _validate_inputs(field, aerodynamics, spin, magnus_coefficient, 0.001)

    omega_x, omega_y, omega_z = spin.angular_velocity()
    cross_x = omega_y * vz - omega_z * vy
    cross_y = omega_z * vx - omega_x * vz
    cross_z = omega_x * vy - omega_y * vx
    area = math.pi * field.ball_radius**2
    factor = (
        0.5
        * aerodynamics.air_density
        * magnus_coefficient
        * area
        * field.ball_radius
        / aerodynamics.mass
    )
    return factor * cross_x, factor * cross_y, factor * cross_z


def _validate_inputs(
    field: Field,
    aerodynamics: Aerodynamics,
    spin: Spin,
    magnus_coefficient: float,
    time_step: float,
) -> None:
    if min(aerodynamics.mass, aerodynamics.air_density, field.ball_radius) <= 0:
        raise ValueError("Ball mass, air density, and ball radius must be positive.")
    if aerodynamics.drag_coefficient < 0 or magnus_coefficient < 0:
        raise ValueError("Drag and Magnus coefficients cannot be negative.")
    if field.wall_width <= 0:
        raise ValueError("Wall width must be positive.")
    if not 0 < time_step <= 0.02:
        raise ValueError("Time step must be greater than 0 and at most 0.02 s.")
    spin.angular_velocity()


def _acceleration(
    vx: float,
    vy: float,
    vz: float,
    field: Field,
    aerodynamics: Aerodynamics,
    spin: Spin,
    magnus_coefficient: float,
    gravity: float,
) -> tuple[float, float, float]:
    speed = math.sqrt(vx**2 + vy**2 + vz**2)
    area = math.pi * field.ball_radius**2
    drag_factor = (
        0.5
        * aerodynamics.air_density
        * aerodynamics.drag_coefficient
        * area
        / aerodynamics.mass
    )
    drag_x = -drag_factor * speed * vx
    drag_y = -drag_factor * speed * vy
    drag_z = -drag_factor * speed * vz
    magnus_x, magnus_y, magnus_z = magnus_acceleration(
        vx, vy, vz, spin, field, aerodynamics, magnus_coefficient
    )
    return drag_x + magnus_x, drag_y + magnus_y, drag_z + magnus_z - gravity


def _interpolate_at_x(
    x_values: list[float],
    y_values: list[float],
    z_values: list[float],
    target_x: float,
) -> tuple[float, float, bool]:
    for index in range(1, len(x_values)):
        if x_values[index] >= target_x:
            fraction = (target_x - x_values[index - 1]) / (
                x_values[index] - x_values[index - 1]
            )
            lateral = y_values[index - 1] + fraction * (
                y_values[index] - y_values[index - 1]
            )
            height = z_values[index - 1] + fraction * (
                z_values[index] - z_values[index - 1]
            )
            return lateral, height, True
    return 0.0, 0.0, False


def judge_spin_shot(
    kick: Kick,
    field: Field,
    wall_lateral: float,
    wall_height: float,
    goal_lateral: float,
    goal_height: float,
    *,
    reaches_wall: bool,
    reaches_goal: bool,
) -> SpinShotResult:
    """Judge wall collision and goal entry using the ball's full radius."""

    wall_left = field.wall_center - field.wall_width / 2
    wall_right = field.wall_center + field.wall_width / 2
    overlaps_wall_sideways = (
        wall_lateral + field.ball_radius >= wall_left
        and wall_lateral - field.ball_radius <= wall_right
    )
    overlaps_wall_vertically = wall_height - field.ball_radius <= field.wall_height
    hits_wall = reaches_wall and overlaps_wall_sideways and overlaps_wall_vertically
    avoids_wall = reaches_wall and not hits_wall

    inside_posts = abs(goal_lateral) + field.ball_radius <= field.goal_width / 2
    below_crossbar = goal_height + field.ball_radius <= field.goal_height
    above_ground = goal_height - field.ball_radius >= 0
    inside_goal = reaches_goal and inside_posts and below_crossbar and above_ground
    scores = avoids_wall and inside_goal

    if not reaches_wall:
        outcome = "The ball lands before reaching the wall."
    elif hits_wall:
        outcome = "The ball hits the defensive wall."
    elif not reaches_goal:
        outcome = "The ball avoids the wall but lands before the goal."
    elif not inside_posts:
        outcome = "The ball avoids the wall but passes outside a goalpost."
    elif not below_crossbar:
        outcome = "The ball avoids the wall but passes over the crossbar."
    elif not above_ground:
        outcome = (
            "The ball avoids the wall but reaches the ground before the goal line."
        )
    else:
        outcome = "GOAL: the spinning ball avoids the wall and enters the goal."

    return SpinShotResult(
        wall_lateral,
        wall_height,
        goal_lateral,
        goal_height,
        reaches_wall,
        avoids_wall,
        reaches_goal,
        inside_goal,
        scores,
        outcome,
    )


def simulate_spin(
    kick: Kick,
    field: Field | None = None,
    aerodynamics: Aerodynamics | None = None,
    spin: Spin | None = None,
    magnus_coefficient: float = 0.25,
    gravity: float = 9.81,
    time_step: float = 0.001,
    ground_height: float = 0.0,
) -> SpinSimulation:
    """Numerically integrate 3D motion with drag and the Magnus force."""

    if field is None:
        field = Field()
    if aerodynamics is None:
        aerodynamics = Aerodynamics()
    if spin is None:
        spin = Spin()
    evaluate_shot(kick, field, gravity)
    _validate_inputs(field, aerodynamics, spin, magnus_coefficient, time_step)
    if ground_height < 0:
        raise ValueError("Ground height cannot be negative.")
    if kick.initial_height < ground_height:
        raise ValueError("Initial ball height cannot be below the ground height.")

    theta = math.radians(kick.angle_degrees)
    t = 0.0
    x = 0.0
    y = kick.lateral_offset
    z = kick.initial_height
    vx = kick.speed * math.cos(theta)
    vy = 0.0
    vz = kick.speed * math.sin(theta)

    times = [t]
    x_values = [x]
    y_values = [y]
    z_values = [z]
    vx_values = [vx]
    vy_values = [vy]
    vz_values = [vz]

    if z <= ground_height and vz <= 0:
        shot = judge_spin_shot(
            kick,
            field,
            0.0,
            0.0,
            0.0,
            0.0,
            reaches_wall=False,
            reaches_goal=False,
        )
        return SpinSimulation(
            tuple(times),
            tuple(x_values),
            tuple(y_values),
            tuple(z_values),
            tuple(vx_values),
            tuple(vy_values),
            tuple(vz_values),
            0.0,
            0.0,
            z,
            0.0,
            shot,
        )

    maximum_steps = math.ceil(60.0 / time_step)
    for _ in range(maximum_steps):
        ax, ay, az = _acceleration(
            vx,
            vy,
            vz,
            field,
            aerodynamics,
            spin,
            magnus_coefficient,
            gravity,
        )
        next_vx = vx + ax * time_step
        next_vy = vy + ay * time_step
        next_vz = vz + az * time_step
        next_x = x + next_vx * time_step
        next_y = y + next_vy * time_step
        next_z = z + next_vz * time_step
        next_t = t + time_step

        if next_z < ground_height:
            fraction = (z - ground_height) / (z - next_z)
            times.append(t + fraction * time_step)
            x_values.append(x + fraction * (next_x - x))
            y_values.append(y + fraction * (next_y - y))
            z_values.append(ground_height)
            vx_values.append(vx + fraction * (next_vx - vx))
            vy_values.append(vy + fraction * (next_vy - vy))
            vz_values.append(vz + fraction * (next_vz - vz))
            break

        t, x, y, z = next_t, next_x, next_y, next_z
        vx, vy, vz = next_vx, next_vy, next_vz
        times.append(t)
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        vx_values.append(vx)
        vy_values.append(vy)
        vz_values.append(vz)
    else:
        raise RuntimeError(
            "Spin simulation did not reach the ground within 60 seconds."
        )

    wall_y, wall_z, reaches_wall = _interpolate_at_x(
        x_values, y_values, z_values, field.wall_distance
    )
    goal_y, goal_z, reaches_goal = _interpolate_at_x(
        x_values, y_values, z_values, field.goal_distance
    )
    shot = judge_spin_shot(
        kick,
        field,
        wall_y,
        wall_z,
        goal_y,
        goal_z,
        reaches_wall=reaches_wall,
        reaches_goal=reaches_goal,
    )

    return SpinSimulation(
        tuple(times),
        tuple(x_values),
        tuple(y_values),
        tuple(z_values),
        tuple(vx_values),
        tuple(vy_values),
        tuple(vz_values),
        times[-1],
        x_values[-1],
        max(z_values),
        max(abs(value - kick.lateral_offset) for value in y_values),
        shot,
    )


def print_comparison(
    kick: Kick,
    field: Field,
    drag: DragSimulation,
    spin_result: SpinSimulation,
    gravity: float = 9.81,
) -> None:
    """Print important measurements for all three physics models."""

    ideal_shot = evaluate_shot(kick, field, gravity)
    rows = [
        (
            "Flight time (s)",
            flight_time(kick, gravity),
            drag.flight_time,
            spin_result.flight_time,
        ),
        (
            "Maximum height (m)",
            ideal_maximum_height(kick, gravity),
            drag.maximum_height,
            spin_result.maximum_height,
        ),
        (
            "Range (m)",
            horizontal_range(kick, gravity),
            drag.horizontal_range,
            spin_result.horizontal_range,
        ),
        (
            "Goal lateral (m)",
            kick.lateral_offset,
            kick.lateral_offset,
            spin_result.shot.goal_lateral,
        ),
    ]
    print(f"{'Measurement':<24} {'Ideal':>10} {'Drag':>10} {'Drag + spin':>14}")
    print("-" * 62)
    for label, ideal_value, drag_value, spun_value in rows:
        print(
            f"{label:<24} {ideal_value:>10.2f} {drag_value:>10.2f} {spun_value:>14.2f}"
        )
    print()
    print(f"Ideal:      {ideal_shot.outcome}")
    print(f"Drag:       {drag.shot.outcome}")
    print(f"Drag+spin:  {spin_result.shot.outcome}")


def plot_comparison_3d(
    kick: Kick,
    field: Field,
    drag: DragSimulation,
    spin_result: SpinSimulation,
    gravity: float = 9.81,
) -> None:
    """Plot ideal, drag, and drag-plus-spin trajectories in three dimensions."""

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib: pip install matplotlib"
        ) from error

    ideal_x, ideal_z = trajectory(kick, gravity, points=400)
    ideal_y = [kick.lateral_offset] * len(ideal_x)
    drag_y = [kick.lateral_offset] * len(drag.x)

    figure = plt.figure(figsize=(10, 7))
    axes = figure.add_subplot(111, projection="3d")
    axes.plot(ideal_x, ideal_y, ideal_z, linestyle="--", label="Model 1: ideal")
    axes.plot(drag.x, drag_y, drag.y, label="Model 2: drag")
    axes.plot(spin_result.x, spin_result.y, spin_result.z, label="Model 3: drag + spin")

    wall_left = field.wall_center - field.wall_width / 2
    wall_right = field.wall_center + field.wall_width / 2
    axes.plot_surface(
        np.array([[field.wall_distance, field.wall_distance]] * 2),
        np.array([[wall_left, wall_right], [wall_left, wall_right]]),
        np.array([[0, 0], [field.wall_height, field.wall_height]]),
        color="tab:red",
        alpha=0.35,
    )

    left_post = -field.goal_width / 2
    right_post = field.goal_width / 2
    axes.plot(
        [field.goal_distance, field.goal_distance],
        [left_post, left_post],
        [0, field.goal_height],
        color="tab:green",
        linewidth=4,
    )
    axes.plot(
        [field.goal_distance, field.goal_distance],
        [right_post, right_post],
        [0, field.goal_height],
        color="tab:green",
        linewidth=4,
    )
    axes.plot(
        [field.goal_distance, field.goal_distance],
        [left_post, right_post],
        [field.goal_height, field.goal_height],
        color="tab:green",
        linewidth=4,
        label="Goal",
    )
    axes.set_xlabel("Toward goal, x (m)")
    axes.set_ylabel("Sideways, y (m)")
    axes.set_zlabel("Height, z (m)")
    axes.set_title("Soccer Free Kick: Ideal, Drag, and Magnus Force")
    axes.legend()
    figure.tight_layout()
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=25.0, help="launch speed in m/s")
    parser.add_argument(
        "--angle", type=float, default=30.0, help="launch angle in degrees"
    )
    parser.add_argument(
        "--lateral-offset", type=float, default=0.0, help="launch y position in m"
    )
    parser.add_argument(
        "--spin-rpm", type=float, default=600.0, help="signed spin rate in rpm"
    )
    parser.add_argument("--spin-axis-x", type=float, default=0.0)
    parser.add_argument("--spin-axis-y", type=float, default=0.0)
    parser.add_argument("--spin-axis-z", type=float, default=1.0)
    parser.add_argument("--magnus-coefficient", type=float, default=0.25)
    parser.add_argument("--drag-coefficient", type=float, default=0.25)
    parser.add_argument("--wall-distance", type=float, default=9.15)
    parser.add_argument("--wall-height", type=float, default=2.0)
    parser.add_argument("--wall-width", type=float, default=3.0)
    parser.add_argument("--wall-center", type=float, default=0.0)
    parser.add_argument("--goal-distance", type=float, default=20.0)
    parser.add_argument("--time-step", type=float, default=0.001)
    parser.add_argument("--plot", action="store_true", help="display the 3D comparison")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    kick = Kick(args.speed, args.angle, lateral_offset=args.lateral_offset)
    field = Field(
        wall_distance=args.wall_distance,
        wall_height=args.wall_height,
        goal_distance=args.goal_distance,
        wall_width=args.wall_width,
        wall_center=args.wall_center,
    )
    aerodynamics = Aerodynamics(drag_coefficient=args.drag_coefficient)
    spin = Spin(
        args.spin_rpm,
        args.spin_axis_x,
        args.spin_axis_y,
        args.spin_axis_z,
    )
    drag = simulate_drag(kick, field, aerodynamics, time_step=args.time_step)
    spin_result = simulate_spin(
        kick,
        field,
        aerodynamics,
        spin,
        args.magnus_coefficient,
        time_step=args.time_step,
    )
    print_comparison(kick, field, drag, spin_result)
    if args.plot:
        plot_comparison_3d(kick, field, drag, spin_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
