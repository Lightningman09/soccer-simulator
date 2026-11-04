"""Interactive kick-contact simulator linked to the 3D flight model.

Click the back-facing view of the ball to choose where the foot applies force.
The simulator converts force and contact time into linear impulse, then uses
the off-centre impulse to calculate angular impulse and ball spin.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from drag_simulation import Aerodynamics
from ground_motion import GroundParameters, GroundResult, simulate_ground_motion
from soccer_simulation import Field, Kick, trajectory
from spin_simulation import Spin, SpinSimulation, simulate_spin


@dataclass(frozen=True)
class ContactParameters:
    """Force, contact point, and ball properties for one kick."""

    force_newtons: float = 850.0
    contact_time_seconds: float = 0.010
    foot_lift_degrees: float = 0.0
    contact_y: float = 0.030
    contact_z: float = -0.040
    ball_mass: float = 0.43
    ball_radius: float = 0.11
    manual_force_elevation_degrees: float | None = None


@dataclass(frozen=True)
class ContactResult:
    """Linear and rotational initial conditions produced by the impulse."""

    contact_x: float
    impulse_x: float
    impulse_y: float
    impulse_z: float
    contact_derived_elevation_degrees: float
    applied_force_elevation_degrees: float
    ground_reaction_impulse_z: float
    net_impulse_x: float
    net_impulse_y: float
    net_impulse_z: float
    launch_speed: float
    launch_angle_degrees: float
    angular_impulse_x: float
    angular_impulse_y: float
    angular_impulse_z: float
    omega_x: float
    omega_y: float
    omega_z: float
    spin_rate_rpm: float
    spin_axis_x: float
    spin_axis_y: float
    spin_axis_z: float
    translational_energy: float
    rotational_energy: float

    def kick(self, initial_height: float = 0.0) -> Kick:
        return Kick(
            self.launch_speed,
            self.launch_angle_degrees,
            initial_height=initial_height,
        )

    def spin(self) -> Spin:
        return Spin(
            self.spin_rate_rpm,
            self.spin_axis_x,
            self.spin_axis_y,
            self.spin_axis_z,
        )


@dataclass(frozen=True)
class ContactFlight:
    """Contact calculation together with its predicted flight."""

    parameters: ContactParameters
    contact: ContactResult
    flight: SpinSimulation
    ground: GroundResult | None

    @property
    def final_outcome(self) -> str:
        if self.ground is not None:
            return self.ground.outcome
        return self.flight.shot.outcome


def calculate_contact(parameters: ContactParameters) -> ContactResult:
    """Convert a force applied at a surface point into speed and spin.

    The ball is treated as a thin spherical shell, with moment of inertia
    I = (2/3)mR^2. The selected y-z point determines the corresponding x point
    on the back hemisphere, facing the kicker.
    """

    if parameters.force_newtons <= 0:
        raise ValueError("Force must be positive.")
    if parameters.contact_time_seconds <= 0:
        raise ValueError("Contact time must be positive.")
    if parameters.ball_mass <= 0 or parameters.ball_radius <= 0:
        raise ValueError("Ball mass and radius must be positive.")
    radial_square = parameters.contact_y**2 + parameters.contact_z**2
    radius_square = parameters.ball_radius**2
    if radial_square > radius_square:
        raise ValueError("The selected contact point lies outside the ball.")

    contact_x = -math.sqrt(max(0.0, radius_square - radial_square))
    contact_derived_elevation = math.degrees(
        math.atan2(-parameters.contact_z, -contact_x)
    )
    if parameters.manual_force_elevation_degrees is None:
        applied_force_elevation = (
            contact_derived_elevation + parameters.foot_lift_degrees
        )
    else:
        applied_force_elevation = parameters.manual_force_elevation_degrees
    if not -89 < applied_force_elevation < 89:
        raise ValueError(
            "Applied force elevation must remain between -89 and 89 degrees."
        )

    angle = math.radians(applied_force_elevation)
    impulse = parameters.force_newtons * parameters.contact_time_seconds
    impulse_x = impulse * math.cos(angle)
    impulse_y = 0.0
    impulse_z = impulse * math.sin(angle)

    # The ball begins on the ground. A downward foot impulse is opposed by an
    # upward ground-reaction impulse, so it cannot launch through the field.
    ground_reaction_impulse_z = max(0.0, -impulse_z)
    net_impulse_x = impulse_x
    net_impulse_y = impulse_y
    net_impulse_z = impulse_z + ground_reaction_impulse_z

    # Angular impulse is r x J.
    angular_x = parameters.contact_y * impulse_z - parameters.contact_z * impulse_y
    angular_y = parameters.contact_z * impulse_x - contact_x * impulse_z
    angular_z = contact_x * impulse_y - parameters.contact_y * impulse_x

    moment_of_inertia = 2.0 / 3.0 * parameters.ball_mass * parameters.ball_radius**2
    omega_x = angular_x / moment_of_inertia
    omega_y = angular_y / moment_of_inertia
    omega_z = angular_z / moment_of_inertia
    omega_magnitude = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    if omega_magnitude == 0:
        spin_axis = (0.0, 0.0, 0.0)
    else:
        spin_axis = (
            omega_x / omega_magnitude,
            omega_y / omega_magnitude,
            omega_z / omega_magnitude,
        )

    net_impulse_magnitude = math.sqrt(
        net_impulse_x**2 + net_impulse_y**2 + net_impulse_z**2
    )
    launch_speed = net_impulse_magnitude / parameters.ball_mass
    launch_angle = math.degrees(math.atan2(net_impulse_z, net_impulse_x))
    spin_rate_rpm = omega_magnitude * 60 / (2 * math.pi)
    translational_energy = 0.5 * parameters.ball_mass * launch_speed**2
    rotational_energy = 0.5 * moment_of_inertia * omega_magnitude**2
    return ContactResult(
        contact_x,
        impulse_x,
        impulse_y,
        impulse_z,
        contact_derived_elevation,
        applied_force_elevation,
        ground_reaction_impulse_z,
        net_impulse_x,
        net_impulse_y,
        net_impulse_z,
        launch_speed,
        launch_angle,
        angular_x,
        angular_y,
        angular_z,
        omega_x,
        omega_y,
        omega_z,
        spin_rate_rpm,
        spin_axis[0],
        spin_axis[1],
        spin_axis[2],
        translational_energy,
        rotational_energy,
    )


def calculate_contact_flight(
    parameters: ContactParameters,
    field: Field | None = None,
    aerodynamics: Aerodynamics | None = None,
    magnus_coefficient: float = 0.25,
    ground_parameters: GroundParameters | None = None,
    time_step: float = 0.003,
) -> ContactFlight:
    """Calculate contact mechanics and pass the result to Model 3."""

    if field is None:
        field = Field(wall_center=1.5)
    if aerodynamics is None:
        aerodynamics = Aerodynamics()
    if ground_parameters is None:
        ground_parameters = GroundParameters()
    contact = calculate_contact(parameters)
    flight = simulate_spin(
        contact.kick(initial_height=parameters.ball_radius),
        field,
        aerodynamics,
        contact.spin(),
        magnus_coefficient,
        time_step=time_step,
        ground_height=parameters.ball_radius,
    )
    blocked_in_air = flight.shot.reaches_wall and not flight.shot.avoids_wall
    crossed_goal_in_air = flight.shot.reaches_goal
    if blocked_in_air or crossed_goal_in_air:
        ground = None
    else:
        angular_speed = contact.spin().angular_velocity()
        ground = simulate_ground_motion(
            impact_x=flight.x[-1],
            impact_y=flight.y[-1],
            impact_vx=flight.vx[-1],
            impact_vy=flight.vy[-1],
            impact_vz=flight.vz[-1],
            omega_x=angular_speed[0],
            omega_y=angular_speed[1],
            omega_z=angular_speed[2],
            radius=parameters.ball_radius,
            mass=parameters.ball_mass,
            field=field,
            aerodynamics=aerodynamics,
            magnus_coefficient=magnus_coefficient,
            parameters=ground_parameters,
            time_step=min(time_step, 0.003),
        )
    return ContactFlight(parameters, contact, flight, ground)


def contact_summary(result: ContactFlight) -> str:
    """Return a concise summary for the interactive interface."""

    contact = result.contact
    flight = result.flight
    return "\n".join(
        [
            "KICK RESULT",
            "",
            (
                f"Contact y/z: {result.parameters.contact_y * 100:+.1f} / "
                f"{result.parameters.contact_z * 100:+.1f} cm"
            ),
            f"Impulse: {math.hypot(contact.impulse_x, contact.impulse_z):.2f} N s",
            (
                f"Natural/applied force angle: "
                f"{contact.contact_derived_elevation_degrees:+.1f} / "
                f"{contact.applied_force_elevation_degrees:+.1f} deg"
            ),
            f"Ground reaction: {contact.ground_reaction_impulse_z:.2f} N s upward",
            (
                f"Actual launch: {contact.launch_speed:.2f} m/s at "
                f"{contact.launch_angle_degrees:.1f} deg"
            ),
            f"Spin: {contact.spin_rate_rpm:.0f} rpm",
            (
                f"Spin axis: ({contact.spin_axis_x:+.2f}, "
                f"{contact.spin_axis_y:+.2f}, {contact.spin_axis_z:+.2f})"
            ),
            (
                f"Angular impulse: ({contact.angular_impulse_x:+.3f}, "
                f"{contact.angular_impulse_y:+.3f}, "
                f"{contact.angular_impulse_z:+.3f}) N m s"
            ),
            (
                f"Energy, translation/rotation: {contact.translational_energy:.1f} / "
                f"{contact.rotational_energy:.1f} J"
            ),
            "",
            "FLIGHT RESULT",
            result.final_outcome,
            (
                f"At wall y/z: {flight.shot.wall_lateral:+.2f} / "
                f"{flight.shot.wall_height:.2f} m"
            ),
            (
                f"At goal y/z: {flight.shot.goal_lateral:+.2f} / "
                f"{flight.shot.goal_height:.2f} m"
            ),
            (
                f"Ground path after landing: {result.ground.path_distance:.2f} m"
                if result.ground is not None
                else "Ground phase: not reached before the result"
            ),
        ]
    )


def math_report(
    result: ContactFlight,
    field: Field | None = None,
    aerodynamics: Aerodynamics | None = None,
    magnus_coefficient: float = 0.25,
    ground_parameters: GroundParameters | None = None,
    gravity: float = 9.81,
) -> str:
    """Build a readable report with equations and substituted values."""

    if field is None:
        field = Field(wall_center=1.5)
    if aerodynamics is None:
        aerodynamics = Aerodynamics()
    if ground_parameters is None:
        ground_parameters = GroundParameters()

    parameters = result.parameters
    contact = result.contact
    radius = parameters.ball_radius
    mass = parameters.ball_mass
    impulse = parameters.force_newtons * parameters.contact_time_seconds
    inertia = 2.0 / 3.0 * mass * radius**2
    initial_vx = contact.net_impulse_x / mass
    initial_vy = contact.net_impulse_y / mass
    initial_vz = contact.net_impulse_z / mass
    speed = contact.launch_speed
    area = math.pi * radius**2
    gravity_force = mass * gravity

    drag_scale = 0.5 * aerodynamics.air_density * aerodynamics.drag_coefficient * area
    drag_x = -drag_scale * speed * initial_vx
    drag_y = -drag_scale * speed * initial_vy
    drag_z = -drag_scale * speed * initial_vz

    cross_x = contact.omega_y * initial_vz - contact.omega_z * initial_vy
    cross_y = contact.omega_z * initial_vx - contact.omega_x * initial_vz
    cross_z = contact.omega_x * initial_vy - contact.omega_y * initial_vx
    magnus_scale = 0.5 * aerodynamics.air_density * magnus_coefficient * area * radius
    magnus_x = magnus_scale * cross_x
    magnus_y = magnus_scale * cross_y
    magnus_z = magnus_scale * cross_z

    sliding_force = ground_parameters.kinetic_friction * mass * gravity
    sliding_acceleration = ground_parameters.kinetic_friction * gravity
    rolling_force = ground_parameters.rolling_resistance * mass * gravity
    rolling_acceleration = ground_parameters.rolling_resistance * gravity

    report = [
        "PHY 221 SOCCER KICK CALCULATION REPORT",
        "======================================",
        "",
        "INPUTS",
        f"Applied force F                 = {parameters.force_newtons:.3f} N",
        (f"Contact time dt                = {parameters.contact_time_seconds:.6f} s"),
        f"Ball mass m                     = {mass:.4f} kg",
        f"Ball radius R                   = {radius:.4f} m",
        (
            f"Contact y, z                    = ({parameters.contact_y:+.5f}, "
            f"{parameters.contact_z:+.5f}) m"
        ),
        f"Gravity g                       = {gravity:.3f} m/s^2",
        f"Air density rho                 = {aerodynamics.air_density:.4f} kg/m^3",
        f"Drag coefficient Cd             = {aerodynamics.drag_coefficient:.4f}",
        f"Magnus coefficient Cm           = {magnus_coefficient:.4f}",
        f"Sliding friction coefficient    = {ground_parameters.kinetic_friction:.4f}",
        f"Rolling resistance coefficient  = {ground_parameters.rolling_resistance:.4f}",
        f"Bounce restitution e            = {ground_parameters.restitution:.4f}",
        "",
        "1. CONTACT GEOMETRY AND FORCE DIRECTION",
        "x_contact = -sqrt(R^2 - y^2 - z^2)",
        (
            f"          = -sqrt({radius:.4f}^2 - "
            f"({parameters.contact_y:+.5f})^2 - "
            f"({parameters.contact_z:+.5f})^2)"
        ),
        f"          = {contact.contact_x:+.6f} m",
        "theta_natural = atan2(-z, -x_contact)",
        f"              = {contact.contact_derived_elevation_degrees:+.3f} deg",
        "theta_applied = theta_natural + foot-lift adjustment",
        f"              = {contact.applied_force_elevation_degrees:+.3f} deg",
        "",
        "2. LINEAR IMPULSE AND LAUNCH",
        "J = F dt",
        (
            f"  = ({parameters.force_newtons:.3f} N)"
            f"({parameters.contact_time_seconds:.6f} s) = {impulse:.6f} N s"
        ),
        "Jx = J cos(theta), Jy = 0, Jz = J sin(theta)",
        (
            f"J_foot = ({contact.impulse_x:+.6f}, {contact.impulse_y:+.6f}, "
            f"{contact.impulse_z:+.6f}) N s"
        ),
        "J_ground,z = max(0, -Jz)",
        f"J_ground,z = {contact.ground_reaction_impulse_z:.6f} N s",
        "v0 = J_net / m",
        (f"v0 = ({initial_vx:+.6f}, {initial_vy:+.6f}, {initial_vz:+.6f}) m/s"),
        "|v0| = sqrt(vx^2 + vy^2 + vz^2)",
        f"|v0| = {contact.launch_speed:.6f} m/s",
        "launch angle = atan2(vz, vx)",
        f"launch angle = {contact.launch_angle_degrees:.6f} deg",
        "",
        "3. ANGULAR IMPULSE AND SPIN",
        "L = r x J_foot",
        (
            f"L = ({contact.angular_impulse_x:+.6f}, "
            f"{contact.angular_impulse_y:+.6f}, "
            f"{contact.angular_impulse_z:+.6f}) N m s"
        ),
        "I = (2/3) m R^2",
        (f"I = (2/3)({mass:.4f})({radius:.4f}^2) = {inertia:.8f} kg m^2"),
        "omega = L / I",
        (
            f"omega = ({contact.omega_x:+.6f}, {contact.omega_y:+.6f}, "
            f"{contact.omega_z:+.6f}) rad/s"
        ),
        "rpm = |omega| 60 / (2 pi)",
        f"rpm = {contact.spin_rate_rpm:.3f}",
        "",
        "4. ENERGY",
        "K_translation = (1/2) m |v0|^2",
        f"K_translation = {contact.translational_energy:.6f} J",
        "K_rotation = (1/2) I |omega|^2",
        f"K_rotation = {contact.rotational_energy:.6f} J",
        "",
        "5. INITIAL FLIGHT FORCES",
        f"Cross-sectional area A = pi R^2 = {area:.8f} m^2",
        "Weight = m g",
        f"Weight magnitude = {gravity_force:.6f} N downward",
        "F_drag = -(1/2) rho Cd A |v| v",
        f"F_drag(0) = ({drag_x:+.6f}, {drag_y:+.6f}, {drag_z:+.6f}) N",
        "F_Magnus = (1/2) rho Cm A R (omega x v)",
        (f"F_Magnus(0) = ({magnus_x:+.6f}, {magnus_y:+.6f}, {magnus_z:+.6f}) N"),
        (
            "The flight is advanced numerically because drag and Magnus force "
            "change with velocity."
        ),
        "",
        "6. GROUND MOTION",
        "Sliding friction magnitude Ff = mu_k m g",
        f"Ff = {sliding_force:.6f} N; corresponding mu_k g = {sliding_acceleration:.6f} m/s^2",
        "Bottom-point slip: ux = vx - R omega_y, uy = vy + R omega_x",
        "Rolling begins when the bottom-point slip is nearly zero: v = omega R",
        "Rolling resistance magnitude Fr = mu_r m g",
        f"Fr = {rolling_force:.6f} N; corresponding mu_r g = {rolling_acceleration:.6f} m/s^2",
        "Bounce vertical speed: vz_after = -e vz_before",
        "",
        "7. SIMULATION RESULT",
        f"Wall plane x                   = {field.wall_distance:.3f} m",
        f"Goal plane x                   = {field.goal_distance:.3f} m",
        (
            f"Ball center at wall y, z       = ({result.flight.shot.wall_lateral:+.4f}, "
            f"{result.flight.shot.wall_height:+.4f}) m"
        ),
        (
            f"Ball center at goal y, z       = ({result.flight.shot.goal_lateral:+.4f}, "
            f"{result.flight.shot.goal_height:+.4f}) m"
        ),
        f"Maximum airborne height        = {result.flight.maximum_height:.4f} m",
    ]
    if result.ground is not None:
        report.extend(
            [
                f"Ground path distance            = {result.ground.path_distance:.4f} m",
                f"Ground-motion duration          = {result.ground.times[-1]:.4f} s",
                f"Ground phases                   = {', '.join(dict.fromkeys(result.ground.phases))}",
            ]
        )
    else:
        report.append("Ground motion                   = not reached before the result")
    report.extend([f"Outcome                         = {result.final_outcome}", ""])
    return "\n".join(report)


def export_math_report(
    result: ContactFlight,
    path: str | Path,
    field: Field | None = None,
    aerodynamics: Aerodynamics | None = None,
    magnus_coefficient: float = 0.25,
    ground_parameters: GroundParameters | None = None,
) -> Path:
    """Write the equations, substituted values, and results to a text file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        math_report(
            result,
            field,
            aerodynamics,
            magnus_coefficient,
            ground_parameters,
        ),
        encoding="utf-8",
    )
    return destination


def export_contact_csv(result: ContactFlight, path: str | Path) -> Path:
    """Export contact inputs, calculated initial conditions, and outcome."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, object] = {}
    row.update(
        {f"input_{key}": value for key, value in asdict(result.parameters).items()}
    )
    row.update(
        {f"result_{key}": value for key, value in asdict(result.contact).items()}
    )
    row.update(
        {
            "wall_lateral_m": result.flight.shot.wall_lateral,
            "wall_height_m": result.flight.shot.wall_height,
            "goal_lateral_m": result.flight.shot.goal_lateral,
            "goal_height_m": result.flight.shot.goal_height,
            "airborne_outcome": result.flight.shot.outcome,
            "ground_path_m": (
                result.ground.path_distance if result.ground is not None else 0.0
            ),
            "scores": (
                result.ground.scores
                if result.ground is not None
                else result.flight.shot.scores
            ),
            "outcome": result.final_outcome,
        }
    )
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return destination


class KickContactSimulator:
    """Interactive Matplotlib interface for selecting the force contact point."""

    def __init__(self) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
        from matplotlib.widgets import Button, Slider

        self.plt = plt
        self.Circle = Circle
        self.Line2D = Line2D
        self.np = np
        self.figure = plt.figure(figsize=(15, 9))
        self.figure.canvas.manager.set_window_title("PHY 221 Kick Contact Simulator")
        self.contact_axes = self.figure.add_axes([0.04, 0.43, 0.31, 0.52])
        self.flight_axes = self.figure.add_axes(
            [0.36, 0.37, 0.36, 0.58], projection="3d"
        )
        self.result_text = self.figure.text(
            0.75, 0.92, "", va="top", family="monospace", fontsize=9, wrap=True
        )
        self.status_text = self.figure.text(
            0.75, 0.04, "", fontsize=9, color="tab:blue"
        )

        slider_color = "#d9e8fb"
        self.force = Slider(
            self.figure.add_axes([0.10, 0.28, 0.22, 0.025], facecolor=slider_color),
            "Force (N)",
            200,
            2000,
            valinit=850,
            valstep=25,
        )
        self.contact_time = Slider(
            self.figure.add_axes([0.48, 0.28, 0.20, 0.025], facecolor=slider_color),
            "Contact time (ms)",
            3,
            20,
            valinit=10,
            valstep=0.5,
        )
        self.foot_lift = Slider(
            self.figure.add_axes([0.10, 0.22, 0.22, 0.025], facecolor=slider_color),
            "Foot lift adjust. (deg)",
            -20,
            20,
            valinit=0,
            valstep=0.5,
        )
        self.wall_center = Slider(
            self.figure.add_axes([0.48, 0.22, 0.20, 0.025], facecolor=slider_color),
            "Wall center y (m)",
            -3,
            3,
            valinit=1.5,
            valstep=0.1,
        )
        self.magnus = Slider(
            self.figure.add_axes([0.10, 0.16, 0.22, 0.025], facecolor=slider_color),
            "Magnus coeff.",
            0,
            0.5,
            valinit=0.25,
            valstep=0.01,
        )
        self.drag = Slider(
            self.figure.add_axes([0.48, 0.16, 0.20, 0.025], facecolor=slider_color),
            "Drag coeff.",
            0,
            0.6,
            valinit=0.25,
            valstep=0.01,
        )
        self.kinetic_friction = Slider(
            self.figure.add_axes([0.10, 0.10, 0.22, 0.025], facecolor=slider_color),
            "Sliding friction",
            0.05,
            0.8,
            valinit=0.35,
            valstep=0.01,
        )
        self.rolling_resistance = Slider(
            self.figure.add_axes([0.48, 0.10, 0.20, 0.025], facecolor=slider_color),
            "Rolling resistance",
            0.005,
            0.1,
            valinit=0.025,
            valstep=0.005,
        )
        self.restitution = Slider(
            self.figure.add_axes([0.10, 0.04, 0.22, 0.025], facecolor=slider_color),
            "Bounce restitution",
            0,
            0.8,
            valinit=0.35,
            valstep=0.05,
        )
        self.save_graph_button = Button(
            self.figure.add_axes([0.74, 0.12, 0.075, 0.045]), "Save graph"
        )
        self.save_csv_button = Button(
            self.figure.add_axes([0.825, 0.12, 0.075, 0.045]), "Export CSV"
        )
        self.save_math_button = Button(
            self.figure.add_axes([0.91, 0.12, 0.075, 0.045]), "Export math"
        )

        self.contact_y = 0.030
        self.contact_z = -0.040
        self.field = Field(wall_center=self.wall_center.val)
        self.result = self._calculate()
        self.figure.canvas.mpl_connect("button_press_event", self._select_contact)
        for slider in (
            self.force,
            self.contact_time,
            self.foot_lift,
            self.wall_center,
            self.magnus,
            self.drag,
            self.kinetic_friction,
            self.rolling_resistance,
            self.restitution,
        ):
            slider.on_changed(self._update)
        self.save_graph_button.on_clicked(self._save_graph)
        self.save_csv_button.on_clicked(self._save_csv)
        self.save_math_button.on_clicked(self._save_math)
        self._draw()

    def _parameters(self) -> ContactParameters:
        return ContactParameters(
            force_newtons=self.force.val,
            contact_time_seconds=self.contact_time.val / 1000,
            foot_lift_degrees=self.foot_lift.val,
            contact_y=self.contact_y,
            contact_z=self.contact_z,
        )

    def _calculate(self) -> ContactFlight:
        self.field = Field(wall_center=self.wall_center.val)
        self.aerodynamics = Aerodynamics(drag_coefficient=self.drag.val)
        self.ground_parameters = GroundParameters(
            kinetic_friction=self.kinetic_friction.val,
            rolling_resistance=self.rolling_resistance.val,
            restitution=self.restitution.val,
        )
        return calculate_contact_flight(
            self._parameters(),
            self.field,
            self.aerodynamics,
            self.magnus.val,
            self.ground_parameters,
        )

    def _draw_contact(self) -> None:
        axes = self.contact_axes
        axes.clear()
        radius = self.result.parameters.ball_radius
        axes.add_patch(
            self.Circle((0, 0), radius, facecolor="#e8eef7", edgecolor="black", lw=2)
        )
        axes.axhline(0, color="0.55", linewidth=1)
        axes.axvline(0, color="0.55", linewidth=1)
        axes.scatter(
            [self.contact_y],
            [self.contact_z],
            s=130,
            color="tab:red",
            edgecolor="black",
            zorder=4,
            label="Force contact",
        )
        contact = self.result.contact
        force_angle = math.radians(contact.applied_force_elevation_degrees)
        force_vertical = 0.075 * math.sin(force_angle)
        axes.annotate(
            "",
            xy=(self.contact_y, self.contact_z + force_vertical),
            xytext=(self.contact_y, self.contact_z),
            arrowprops={"arrowstyle": "-|>", "color": "tab:red", "lw": 2.2},
            zorder=5,
        )
        force_label_z = self.contact_z + force_vertical / 2
        axes.text(
            self.contact_y + 0.008,
            force_label_z,
            "Applied force",
            color="tab:red",
            fontsize=8,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "tab:red", "alpha": 0.85},
            zorder=6,
        )

        launch_angle = math.radians(contact.launch_angle_degrees)
        launch_vertical = 0.080 * math.sin(launch_angle)
        axes.annotate(
            "",
            xy=(0.0, launch_vertical),
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "-|>", "color": "#1155cc", "lw": 2.2},
            zorder=5,
        )
        axes.text(
            0.007,
            max(0.012, launch_vertical * 0.65),
            "Launch velocity",
            color="#1155cc",
            fontsize=8,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "#1155cc", "alpha": 0.85},
            zorder=6,
        )

        if abs(contact.omega_x) > 1e-6:
            if contact.omega_x > 0:
                spin_start, spin_end, curve = (-0.025, -0.035), (-0.025, 0.050), -0.65
            else:
                spin_start, spin_end, curve = (-0.025, 0.050), (-0.025, -0.035), 0.65
            axes.annotate(
                "",
                xy=spin_end,
                xytext=spin_start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "tab:purple",
                    "lw": 2.2,
                    "connectionstyle": f"arc3,rad={curve}",
                },
                zorder=5,
            )
            axes.text(
                -0.067,
                0.005,
                "Spin",
                color="tab:purple",
                fontsize=8,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "tab:purple",
                    "alpha": 0.85,
                },
                zorder=6,
            )
        axes.text(
            0.0,
            radius * 1.18,
            (
                "Vertical projections; motion also points into view (+x); "
                f"vertical angle {contact.applied_force_elevation_degrees:+.1f}°"
            ),
            fontsize=7.5,
            ha="center",
            va="top",
        )
        axes.text(-0.105, 0.095, "upper left", fontsize=8, color="0.35")
        axes.text(0.045, 0.095, "upper right", fontsize=8, color="0.35")
        axes.text(-0.105, -0.105, "lower left", fontsize=8, color="0.35")
        axes.text(0.045, -0.105, "lower right", fontsize=8, color="0.35")
        axes.set_xlim(-radius * 1.25, radius * 1.25)
        axes.set_ylim(-radius * 1.25, radius * 1.25)
        axes.set_aspect("equal")
        axes.set_xlabel("Sideways contact, y (m)")
        axes.set_ylabel("Vertical contact, z (m)")
        axes.set_title("Click Where the Foot Contacts the Ball")
        axes.grid(alpha=0.2)

    def _draw_flight(self) -> None:
        axes = self.flight_axes
        axes.clear()
        flight = self.result.flight
        kick = self.result.contact.kick(
            initial_height=self.result.parameters.ball_radius
        )
        ideal_x, ideal_z = trajectory(kick, points=250)
        axes.plot(
            ideal_x,
            [0.0] * len(ideal_x),
            ideal_z,
            linestyle="--",
            label="No aerodynamics",
        )
        axes.plot(flight.x, flight.y, flight.z, label="Flight")
        phase_colors = {
            "bounce": "tab:purple",
            "sliding": "tab:red",
            "rolling": "tab:brown",
        }
        phase_labels = {
            "bounce": "Bounce",
            "sliding": "Slide",
            "rolling": "Roll",
        }
        if self.result.ground is not None:
            ground = self.result.ground
            plotted_phases: set[str] = set()
            segment_start = 0
            for index in range(1, len(ground.phases) + 1):
                segment_finished = (
                    index == len(ground.phases)
                    or ground.phases[index] != ground.phases[segment_start]
                )
                if not segment_finished:
                    continue
                phase = ground.phases[segment_start]
                start = max(0, segment_start - 1)
                stop = index
                axes.plot(
                    ground.x[start:stop],
                    ground.y[start:stop],
                    ground.z[start:stop],
                    color=phase_colors[phase],
                    linewidth=2.5,
                    label=phase_labels[phase] if phase not in plotted_phases else None,
                )
                plotted_phases.add(phase)
                segment_start = index

        wall_left = self.field.wall_center - self.field.wall_width / 2
        wall_right = self.field.wall_center + self.field.wall_width / 2
        axes.plot_surface(
            self.np.array([[self.field.wall_distance, self.field.wall_distance]] * 2),
            self.np.array([[wall_left, wall_right], [wall_left, wall_right]]),
            self.np.array([[0, 0], [self.field.wall_height, self.field.wall_height]]),
            color="tab:red",
            alpha=0.35,
        )
        left_post = -self.field.goal_width / 2
        right_post = self.field.goal_width / 2
        axes.plot(
            [self.field.goal_distance, self.field.goal_distance],
            [left_post, left_post],
            [0, self.field.goal_height],
            color="tab:green",
            linewidth=4,
        )
        axes.plot(
            [self.field.goal_distance, self.field.goal_distance],
            [right_post, right_post],
            [0, self.field.goal_height],
            color="tab:green",
            linewidth=4,
        )
        axes.plot(
            [self.field.goal_distance, self.field.goal_distance],
            [left_post, right_post],
            [self.field.goal_height, self.field.goal_height],
            color="tab:green",
            linewidth=4,
            label="Goal",
        )
        lateral_limit = max(
            self.field.goal_width / 2 + 1,
            max(abs(value) for value in flight.y) + 1,
            (
                max(abs(value) for value in self.result.ground.y) + 1
                if self.result.ground is not None
                else 0
            ),
        )
        axes.set_xlim(0, max(self.field.goal_distance + 3, 25))
        axes.set_ylim(-lateral_limit, lateral_limit)
        axes.set_zlim(0, max(max(ideal_z), flight.maximum_height, 3) + 0.5)
        axes.set_xlabel("Toward goal, x (m)")
        axes.set_ylabel("Sideways, y (m)")
        axes.set_zlabel("Height, z (m)")
        axes.set_title("Predicted Flight from This Contact")
        handles, labels = axes.get_legend_handles_labels()
        phase_key = [
            self.Line2D([0], [0], color="tab:orange", lw=2, label="Flight"),
            self.Line2D([0], [0], color="tab:purple", lw=2, label="Bounce"),
            self.Line2D([0], [0], color="tab:red", lw=2, label="Slide"),
            self.Line2D([0], [0], color="tab:brown", lw=2, label="Roll"),
        ]
        existing_labels = set(labels)
        for phase_handle in phase_key:
            if phase_handle.get_label() not in existing_labels:
                handles.append(phase_handle)
                labels.append(phase_handle.get_label())
        axes.legend(handles, labels, loc="upper right", fontsize=8)

    def _draw(self) -> None:
        self._draw_contact()
        self._draw_flight()
        self.result_text.set_text(contact_summary(self.result))
        self.figure.canvas.draw_idle()

    def _select_contact(self, event: object) -> None:
        if getattr(event, "inaxes", None) is not self.contact_axes:
            return
        selected_y = getattr(event, "xdata", None)
        selected_z = getattr(event, "ydata", None)
        if selected_y is None or selected_z is None:
            return
        radius = self.result.parameters.ball_radius
        if selected_y**2 + selected_z**2 > radius**2:
            self.status_text.set_text("Contact point must be inside the ball.")
            self.figure.canvas.draw_idle()
            return
        self.contact_y = selected_y
        self.contact_z = selected_z
        self._update(event)

    def _update(self, _value: object) -> None:
        try:
            self.result = self._calculate()
            self.status_text.set_text("")
            self._draw()
        except ValueError as error:
            self.status_text.set_text(str(error))
            self.figure.canvas.draw_idle()

    def _output_path(self, suffix: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return Path("outputs") / f"kick-contact-{timestamp}.{suffix}"

    def _save_graph(self, _event: object) -> None:
        destination = self._output_path("png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(destination, dpi=200, bbox_inches="tight")
        self.status_text.set_text(f"Saved {destination}")
        self.figure.canvas.draw_idle()

    def _save_csv(self, _event: object) -> None:
        destination = export_contact_csv(self.result, self._output_path("csv"))
        self.status_text.set_text(f"Saved {destination}")
        self.figure.canvas.draw_idle()

    def _save_math(self, _event: object) -> None:
        destination = export_math_report(
            self.result,
            self._output_path("txt"),
            self.field,
            self.aerodynamics,
            self.magnus.val,
            self.ground_parameters,
        )
        self.status_text.set_text(f"Saved calculations to {destination}")
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        self.plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show", action="store_true", help="build without opening a window"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    simulator = KickContactSimulator()
    if not args.no_show:
        simulator.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
