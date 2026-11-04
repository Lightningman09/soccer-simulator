"""Ideal-projectile soccer free-kick simulator.

The model intentionally includes gravity only.  It is the PHY 221 baseline that
later models can extend with aerodynamic drag and the Magnus force from spin.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Kick:
    """Initial conditions for a kick."""

    speed: float = 25.0
    angle_degrees: float = 30.0
    initial_height: float = 0.0
    lateral_offset: float = 0.0


@dataclass(frozen=True)
class Field:
    """Geometry used to judge the free kick, in metres."""

    wall_distance: float = 9.15
    wall_height: float = 2.0
    goal_distance: float = 20.0
    goal_width: float = 7.32
    goal_height: float = 2.44
    ball_radius: float = 0.11
    wall_width: float = 3.0
    wall_center: float = 0.0


@dataclass(frozen=True)
class ShotResult:
    """Measurements and decisions at the wall and goal planes."""

    wall_height: float
    goal_height: float
    clears_wall: bool
    reaches_goal: bool
    inside_goal_opening: bool
    scores: bool
    outcome: str


def _validate(kick: Kick, field: Field, gravity: float) -> None:
    """Reject geometry or initial conditions that are not physically useful."""

    if kick.speed <= 0:
        raise ValueError("Kick speed must be positive.")
    if not 0 <= kick.angle_degrees < 90:
        raise ValueError("Launch angle must be at least 0 and below 90 degrees.")
    if gravity <= 0:
        raise ValueError("Gravity must be positive.")
    if field.wall_distance <= 0 or field.goal_distance <= 0:
        raise ValueError("Wall and goal distances must be positive.")
    if field.wall_distance >= field.goal_distance:
        raise ValueError("The wall must be closer to the ball than the goal.")
    if (
        min(field.wall_height, field.goal_width, field.goal_height, field.ball_radius)
        <= 0
    ):
        raise ValueError("Wall, goal, and ball dimensions must be positive.")
    if field.ball_radius * 2 >= min(field.goal_width, field.goal_height):
        raise ValueError("The ball must fit through the goal opening.")


def height_at_distance(kick: Kick, distance: float, gravity: float = 9.81) -> float:
    """Return the ball-centre height at horizontal distance ``distance``.

    Combining x(t) = v cos(theta)t with
    y(t) = y0 + v sin(theta)t - (1/2)gt^2 removes time and gives the
    familiar parabolic trajectory used here.
    """

    if distance < 0:
        raise ValueError("Distance cannot be negative.")
    if kick.speed <= 0 or gravity <= 0:
        raise ValueError("Kick speed and gravity must be positive.")

    theta = math.radians(kick.angle_degrees)
    time = distance / (kick.speed * math.cos(theta))
    return (
        kick.initial_height
        + kick.speed * math.sin(theta) * time
        - 0.5 * gravity * time**2
    )


def flight_time(kick: Kick, gravity: float = 9.81) -> float:
    """Return the positive time when the ball centre reaches ground level."""

    theta = math.radians(kick.angle_degrees)
    vertical_speed = kick.speed * math.sin(theta)
    return (
        vertical_speed
        + math.sqrt(vertical_speed**2 + 2 * gravity * kick.initial_height)
    ) / gravity


def horizontal_range(kick: Kick, gravity: float = 9.81) -> float:
    """Return horizontal distance travelled before reaching the ground."""

    theta = math.radians(kick.angle_degrees)
    return kick.speed * math.cos(theta) * flight_time(kick, gravity)


def trajectory(
    kick: Kick, gravity: float = 9.81, points: int = 200
) -> tuple[list[float], list[float]]:
    """Sample the ideal trajectory from launch until ground contact."""

    if points < 2:
        raise ValueError("At least two trajectory points are required.")
    total_time = flight_time(kick, gravity)
    theta = math.radians(kick.angle_degrees)
    vx = kick.speed * math.cos(theta)
    vy = kick.speed * math.sin(theta)
    times = [total_time * index / (points - 1) for index in range(points)]
    x_values = [vx * time for time in times]
    y_values = [
        max(0.0, kick.initial_height + vy * time - 0.5 * gravity * time**2)
        for time in times
    ]
    return x_values, y_values


def evaluate_shot(
    kick: Kick, field: Field | None = None, gravity: float = 9.81
) -> ShotResult:
    """Decide whether the entire ball clears the wall and enters the goal.

    The comparisons use the ball's radius, rather than treating it as a point.
    A clearance therefore requires the bottom of the ball to be above the wall,
    and a goal requires the whole ball to fit between both posts and below the
    crossbar at the goal line.
    """

    if field is None:
        field = Field()
    _validate(kick, field, gravity)
    range_metres = horizontal_range(kick, gravity)
    wall_y = height_at_distance(kick, field.wall_distance, gravity)
    goal_y = height_at_distance(kick, field.goal_distance, gravity)

    return judge_shot(
        kick,
        field,
        wall_y,
        goal_y,
        reaches_wall=range_metres >= field.wall_distance,
        reaches_goal=range_metres >= field.goal_distance,
    )


def judge_shot(
    kick: Kick,
    field: Field,
    wall_y: float,
    goal_y: float,
    *,
    reaches_wall: bool,
    reaches_goal: bool,
) -> ShotResult:
    """Apply the same wall and goal rules to any trajectory model.

    Model 1 supplies analytical heights, while numerical models such as the
    drag simulation supply interpolated heights. Keeping the decision here
    ensures every model is compared against identical rules.
    """

    clears_wall = reaches_wall and wall_y - field.ball_radius > field.wall_height
    vertical_fit = field.ball_radius <= goal_y <= field.goal_height - field.ball_radius
    horizontal_fit = (
        abs(kick.lateral_offset) <= field.goal_width / 2 - field.ball_radius
    )
    inside_goal = reaches_goal and vertical_fit and horizontal_fit
    scores = clears_wall and inside_goal

    if not reaches_wall:
        outcome = "The ball lands before reaching the wall."
    elif not clears_wall:
        outcome = "The ball hits the defensive wall."
    elif not reaches_goal:
        outcome = "The ball clears the wall but lands before the goal."
    elif goal_y > field.goal_height - field.ball_radius:
        outcome = "The ball clears the wall but passes over the crossbar."
    elif goal_y < field.ball_radius:
        outcome = "The ball clears the wall but reaches the ground before crossing the goal line."
    elif not horizontal_fit:
        outcome = "The ball clears the wall but passes outside a goalpost."
    else:
        outcome = "GOAL: the ball clears the wall and enters the goal."

    return ShotResult(
        wall_y, goal_y, clears_wall, reaches_goal, inside_goal, scores, outcome
    )


def plot_shot(
    kick: Kick, field: Field, result: ShotResult, gravity: float = 9.81
) -> None:
    """Display the trajectory together with the wall and goal."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib: pip install matplotlib"
        ) from error

    x_values, y_values = trajectory(kick, gravity)
    plt.plot(x_values, y_values, label="Ball centre")
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
    plt.scatter(
        [field.wall_distance, field.goal_distance],
        [result.wall_height, result.goal_height],
        color="black",
        zorder=3,
        label="Evaluation points",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.xlim(
        0,
        max(
            field.goal_distance + 3,
            min(horizontal_range(kick, gravity) + 2, field.goal_distance + 15),
        ),
    )
    plt.ylim(bottom=0)
    plt.xlabel("Horizontal distance (m)")
    plt.ylabel("Height (m)")
    plt.title(f"Ideal Free Kick: {result.outcome}")
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
        "--goal-distance", type=float, default=20.0, help="distance to goal line in m"
    )
    parser.add_argument(
        "--lateral-offset", type=float, default=0.0, help="sideways offset at goal in m"
    )
    parser.add_argument(
        "--plot", action="store_true", help="display the trajectory graph"
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    kick = Kick(args.speed, args.angle, lateral_offset=args.lateral_offset)
    field = Field(args.wall_distance, args.wall_height, args.goal_distance)
    result = evaluate_shot(kick, field)

    print(f"Height at wall: {result.wall_height:.2f} m")
    print(f"Clears wall: {'yes' if result.clears_wall else 'no'}")
    print(f"Height at goal: {result.goal_height:.2f} m")
    print(f"Scores: {'yes' if result.scores else 'no'}")
    print(result.outcome)

    if args.plot:
        plot_shot(kick, field, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
