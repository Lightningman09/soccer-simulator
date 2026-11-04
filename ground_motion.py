"""Post-impact bounce, sliding, and rolling motion for a soccer ball."""

from __future__ import annotations

import math
from dataclasses import dataclass

from drag_simulation import Aerodynamics
from soccer_simulation import Field


@dataclass(frozen=True)
class GroundParameters:
    """Grass-contact parameters used after the first landing."""

    kinetic_friction: float = 0.35
    rolling_resistance: float = 0.025
    restitution: float = 0.35
    slip_tolerance: float = 0.05
    stop_speed: float = 0.05
    minimum_bounce_speed: float = 0.5
    maximum_time: float = 30.0


@dataclass(frozen=True)
class GroundResult:
    """Trajectory after first impact, including bounce and ground phases."""

    times: tuple[float, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]
    z: tuple[float, ...]
    vx: tuple[float, ...]
    vy: tuple[float, ...]
    vz: tuple[float, ...]
    omega_x: tuple[float, ...]
    omega_y: tuple[float, ...]
    omega_z: tuple[float, ...]
    phases: tuple[str, ...]
    path_distance: float
    hit_wall: bool
    reaches_goal: bool
    scores: bool
    outcome: str


def _air_acceleration(
    vx: float,
    vy: float,
    vz: float,
    omega_x: float,
    omega_y: float,
    omega_z: float,
    radius: float,
    mass: float,
    aerodynamics: Aerodynamics,
    magnus_coefficient: float,
    gravity: float,
) -> tuple[float, float, float]:
    speed = math.sqrt(vx**2 + vy**2 + vz**2)
    area = math.pi * radius**2
    drag_factor = (
        0.5 * aerodynamics.air_density * aerodynamics.drag_coefficient * area / mass
    )
    magnus_factor = (
        0.5 * aerodynamics.air_density * magnus_coefficient * area * radius / mass
    )
    cross_x = omega_y * vz - omega_z * vy
    cross_y = omega_z * vx - omega_x * vz
    cross_z = omega_x * vy - omega_y * vx
    return (
        -drag_factor * speed * vx + magnus_factor * cross_x,
        -drag_factor * speed * vy + magnus_factor * cross_y,
        -gravity - drag_factor * speed * vz + magnus_factor * cross_z,
    )


def _validate(
    parameters: GroundParameters,
    radius: float,
    mass: float,
    time_step: float,
) -> None:
    if radius <= 0 or mass <= 0:
        raise ValueError("Ball radius and mass must be positive.")
    if parameters.kinetic_friction < 0 or parameters.rolling_resistance < 0:
        raise ValueError("Friction coefficients cannot be negative.")
    if not 0 <= parameters.restitution <= 1:
        raise ValueError("Restitution must be between 0 and 1.")
    if (
        min(
            parameters.slip_tolerance,
            parameters.stop_speed,
            parameters.minimum_bounce_speed,
            parameters.maximum_time,
        )
        <= 0
    ):
        raise ValueError("Ground tolerances and maximum time must be positive.")
    if not 0 < time_step <= 0.02:
        raise ValueError("Time step must be greater than 0 and at most 0.02 s.")


def simulate_ground_motion(
    *,
    impact_x: float,
    impact_y: float,
    impact_vx: float,
    impact_vy: float,
    impact_vz: float,
    omega_x: float,
    omega_y: float,
    omega_z: float,
    radius: float,
    mass: float,
    field: Field,
    aerodynamics: Aerodynamics | None = None,
    magnus_coefficient: float = 0.25,
    parameters: GroundParameters | None = None,
    gravity: float = 9.81,
    time_step: float = 0.002,
) -> GroundResult:
    """Simulate bouncing, kinetic sliding, rolling, and stopping.

    The kick affects friction indirectly through the impact velocity and spin.
    While sliding, friction has magnitude mu_k*m*g and opposes the velocity of
    the bottom point relative to the grass. Once slip is negligible, rolling
    resistance has magnitude mu_r*m*g and the no-slip constraint is enforced.
    """

    if aerodynamics is None:
        aerodynamics = Aerodynamics()
    if parameters is None:
        parameters = GroundParameters()
    _validate(parameters, radius, mass, time_step)
    if gravity <= 0:
        raise ValueError("Gravity must be positive.")

    moment_of_inertia = 2.0 / 3.0 * mass * radius**2
    t = 0.0
    x = impact_x
    y = impact_y
    z = radius
    vx = impact_vx
    vy = impact_vy
    rebound_speed = max(0.0, -impact_vz * parameters.restitution)
    if rebound_speed >= parameters.minimum_bounce_speed:
        vz = rebound_speed
        phase = "bounce"
    else:
        vz = 0.0
        phase = "sliding"

    times = [t]
    x_values = [x]
    y_values = [y]
    z_values = [z]
    vx_values = [vx]
    vy_values = [vy]
    vz_values = [vz]
    omega_x_values = [omega_x]
    omega_y_values = [omega_y]
    omega_z_values = [omega_z]
    phases = [phase]
    path_distance = 0.0
    hit_wall = False
    reaches_goal = False
    scores = False
    outcome = "The ball stopped on the field."
    bounce_count = 0

    maximum_steps = math.ceil(parameters.maximum_time / time_step)
    for _ in range(maximum_steps):
        phase_at_start = phase
        previous_x, previous_y, previous_z = x, y, z
        ax_air, ay_air, az_air = _air_acceleration(
            vx,
            vy,
            vz,
            omega_x,
            omega_y,
            omega_z,
            radius,
            mass,
            aerodynamics,
            magnus_coefficient,
            gravity,
        )

        if phase == "bounce":
            ax, ay, az = ax_air, ay_air, az_air
            next_omega_x, next_omega_y, next_omega_z = (
                omega_x,
                omega_y,
                omega_z,
            )
        else:
            z = radius
            vz = 0.0
            az = 0.0
            if phase == "sliding":
                slip_x = vx - radius * omega_y
                slip_y = vy + radius * omega_x
                slip_speed = math.hypot(slip_x, slip_y)
                if slip_speed <= parameters.slip_tolerance:
                    phase = "rolling"
                    omega_x = -vy / radius
                    omega_y = vx / radius
                    slip_speed = 0.0
                if phase == "sliding":
                    friction_force = parameters.kinetic_friction * mass * gravity
                    force_x = -friction_force * slip_x / slip_speed
                    force_y = -friction_force * slip_y / slip_speed
                    ax = ax_air + force_x / mass
                    ay = ay_air + force_y / mass
                    next_omega_x = (
                        omega_x + (radius * force_y / moment_of_inertia) * time_step
                    )
                    next_omega_y = (
                        omega_y + (-radius * force_x / moment_of_inertia) * time_step
                    )
                    next_omega_z = omega_z
                else:
                    ax, ay = ax_air, ay_air
                    next_omega_x, next_omega_y, next_omega_z = (
                        omega_x,
                        omega_y,
                        omega_z,
                    )
            if phase == "rolling":
                horizontal_speed = math.hypot(vx, vy)
                if horizontal_speed > 0:
                    resistance = parameters.rolling_resistance * gravity
                    ax = ax_air - resistance * vx / horizontal_speed
                    ay = ay_air - resistance * vy / horizontal_speed
                else:
                    ax, ay = 0.0, 0.0
                next_omega_x, next_omega_y, next_omega_z = (
                    omega_x,
                    omega_y,
                    omega_z,
                )

        next_vx = vx + ax * time_step
        next_vy = vy + ay * time_step
        next_vz = vz + az * time_step if phase == "bounce" else 0.0
        if phase == "rolling" and vx * next_vx + vy * next_vy <= 0:
            next_vx = 0.0
            next_vy = 0.0
        next_x = x + next_vx * time_step
        next_y = y + next_vy * time_step
        next_z = z + next_vz * time_step if phase == "bounce" else radius
        next_t = t + time_step

        if phase == "bounce" and next_z <= radius and next_vz < 0:
            fraction = (z - radius) / (z - next_z) if z != next_z else 0.0
            next_x = x + fraction * (next_x - x)
            next_y = y + fraction * (next_y - y)
            next_z = radius
            next_t = t + fraction * time_step
            next_vx = vx + fraction * (next_vx - vx)
            next_vy = vy + fraction * (next_vy - vy)
            impact_vertical_speed = vz + fraction * (next_vz - vz)
            bounce_count += 1
            rebound_speed = -parameters.restitution * impact_vertical_speed
            if rebound_speed >= parameters.minimum_bounce_speed and bounce_count < 8:
                next_vz = rebound_speed
            else:
                next_vz = 0.0
                phase = "sliding"

        event_outcome: str | None = None
        event_fraction = 1.0
        phase_verb = {
            "bounce": "bounces",
            "sliding": "slides",
            "rolling": "rolls",
        }[phase]
        if previous_x < field.wall_distance <= next_x:
            fraction = (field.wall_distance - previous_x) / (next_x - previous_x)
            wall_y = previous_y + fraction * (next_y - previous_y)
            wall_z = previous_z + fraction * (next_z - previous_z)
            overlaps_sideways = (
                wall_y + radius >= field.wall_center - field.wall_width / 2
                and wall_y - radius <= field.wall_center + field.wall_width / 2
            )
            overlaps_vertically = wall_z - radius <= field.wall_height
            if overlaps_sideways and overlaps_vertically:
                hit_wall = True
                event_fraction = fraction
                event_outcome = f"The ball {phase_verb} into the defensive wall."

        if event_outcome is None and previous_x < field.goal_distance <= next_x:
            fraction = (field.goal_distance - previous_x) / (next_x - previous_x)
            goal_y = previous_y + fraction * (next_y - previous_y)
            goal_z = previous_z + fraction * (next_z - previous_z)
            reaches_goal = True
            inside_posts = abs(goal_y) + radius <= field.goal_width / 2
            below_crossbar = goal_z + radius <= field.goal_height
            above_ground = goal_z - radius >= -1e-9
            scores = inside_posts and below_crossbar and above_ground
            event_fraction = fraction
            if scores:
                event_outcome = f"GOAL: the ball {phase_verb} across the goal line."
            elif not inside_posts:
                event_outcome = f"The ball {phase_verb} outside a goalpost."
            else:
                event_outcome = f"The ball {phase_verb} over the crossbar."

        if event_outcome is not None:
            next_x = previous_x + event_fraction * (next_x - previous_x)
            next_y = previous_y + event_fraction * (next_y - previous_y)
            next_z = previous_z + event_fraction * (next_z - previous_z)
            next_t = t + event_fraction * (next_t - t)
            outcome = event_outcome

        step_distance = math.sqrt(
            (next_x - previous_x) ** 2
            + (next_y - previous_y) ** 2
            + (next_z - previous_z) ** 2
        )
        path_distance += step_distance
        t, x, y, z = next_t, next_x, next_y, next_z
        vx, vy, vz = next_vx, next_vy, next_vz
        omega_x, omega_y, omega_z = (
            next_omega_x,
            next_omega_y,
            next_omega_z,
        )

        if phase == "sliding" and phase_at_start == "sliding":
            new_slip_x = vx - radius * omega_y
            new_slip_y = vy + radius * omega_x
            if slip_x * new_slip_x + slip_y * new_slip_y <= 0:
                phase = "rolling"
                omega_x = -vy / radius
                omega_y = vx / radius
        elif phase == "rolling":
            omega_x = -vy / radius
            omega_y = vx / radius

        times.append(t)
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        vx_values.append(vx)
        vy_values.append(vy)
        vz_values.append(vz)
        omega_x_values.append(omega_x)
        omega_y_values.append(omega_y)
        omega_z_values.append(omega_z)
        phases.append(phase)

        if event_outcome is not None:
            break
        if phase != "bounce" and math.hypot(vx, vy) <= parameters.stop_speed:
            outcome = "The ball rolls to a stop on the field."
            break
    else:
        outcome = "The ground simulation reached its time limit."

    return GroundResult(
        tuple(times),
        tuple(x_values),
        tuple(y_values),
        tuple(z_values),
        tuple(vx_values),
        tuple(vy_values),
        tuple(vz_values),
        tuple(omega_x_values),
        tuple(omega_y_values),
        tuple(omega_z_values),
        tuple(phases),
        path_distance,
        hit_wall,
        reaches_goal,
        scores,
        outcome,
    )
