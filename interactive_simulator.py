"""Interactive control panel for all three soccer free-kick models.

Run ``python interactive_simulator.py`` to open the interface. Sliders update
the trajectories and wall/goal decisions immediately; buttons save a graph or
a CSV table for use in the PHY 221 report.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from drag_simulation import (
    Aerodynamics,
    DragSimulation,
    ideal_maximum_height,
    simulate_drag,
)
from soccer_simulation import (
    Field,
    Kick,
    ShotResult,
    evaluate_shot,
    flight_time,
    horizontal_range,
)
from spin_simulation import Spin, SpinSimulation, simulate_spin


@dataclass(frozen=True)
class ModelResults:
    """Inputs and outputs for one three-model comparison."""

    kick: Kick
    field: Field
    aerodynamics: Aerodynamics
    spin: Spin
    ideal_shot: ShotResult
    drag: DragSimulation
    spun: SpinSimulation


def calculate_models(
    kick: Kick,
    field: Field,
    aerodynamics: Aerodynamics,
    spin: Spin,
    magnus_coefficient: float = 0.25,
    time_step: float = 0.003,
) -> ModelResults:
    """Calculate every model using identical kick and field inputs."""

    ideal_shot = evaluate_shot(kick, field)
    drag = simulate_drag(kick, field, aerodynamics, time_step=time_step)
    spun = simulate_spin(
        kick,
        field,
        aerodynamics,
        spin,
        magnus_coefficient,
        time_step=time_step,
    )
    return ModelResults(kick, field, aerodynamics, spin, ideal_shot, drag, spun)


def comparison_rows(results: ModelResults) -> list[dict[str, str | float | bool]]:
    """Return report-ready measurements for ideal, drag, and spin models."""

    kick = results.kick
    return [
        {
            "model": "Ideal",
            "flight_time_s": flight_time(kick),
            "maximum_height_m": ideal_maximum_height(kick),
            "range_m": horizontal_range(kick),
            "wall_lateral_m": kick.lateral_offset,
            "wall_height_m": results.ideal_shot.wall_height,
            "goal_lateral_m": kick.lateral_offset,
            "goal_height_m": results.ideal_shot.goal_height,
            "scores": results.ideal_shot.scores,
            "outcome": results.ideal_shot.outcome,
        },
        {
            "model": "Drag",
            "flight_time_s": results.drag.flight_time,
            "maximum_height_m": results.drag.maximum_height,
            "range_m": results.drag.horizontal_range,
            "wall_lateral_m": kick.lateral_offset,
            "wall_height_m": results.drag.shot.wall_height,
            "goal_lateral_m": kick.lateral_offset,
            "goal_height_m": results.drag.shot.goal_height,
            "scores": results.drag.shot.scores,
            "outcome": results.drag.shot.outcome,
        },
        {
            "model": "Drag + spin",
            "flight_time_s": results.spun.flight_time,
            "maximum_height_m": results.spun.maximum_height,
            "range_m": results.spun.horizontal_range,
            "wall_lateral_m": results.spun.shot.wall_lateral,
            "wall_height_m": results.spun.shot.wall_height,
            "goal_lateral_m": results.spun.shot.goal_lateral,
            "goal_height_m": results.spun.shot.goal_height,
            "scores": results.spun.shot.scores,
            "outcome": results.spun.shot.outcome,
        },
    ]


def format_results(results: ModelResults) -> str:
    """Build the compact comparison displayed beside the graph."""

    lines = [
        "MODEL COMPARISON",
        "",
        "                 peak    range   goal y/z",
    ]
    for row in comparison_rows(results):
        score_mark = "GOAL" if row["scores"] else "MISS"
        lines.append(
            f"{row['model']:<12} "
            f"{row['maximum_height_m']:>5.2f} m "
            f"{row['range_m']:>6.2f} m  "
            f"{row['goal_lateral_m']:>5.2f}/{row['goal_height_m']:.2f}  {score_mark}"
        )
    lines.extend(
        [
            "",
            "OUTCOMES",
            f"Ideal: {results.ideal_shot.outcome}",
            f"Drag: {results.drag.shot.outcome}",
            f"Spin: {results.spun.shot.outcome}",
        ]
    )
    return "\n".join(lines)


def export_results_csv(results: ModelResults, path: str | Path) -> Path:
    """Write the current inputs and three-model results to a CSV file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(results)
    fieldnames = list(rows[0])
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


class InteractiveSimulator:
    """Matplotlib figure, controls, and callbacks for the live simulator."""

    def __init__(self) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.widgets import Button, RadioButtons, Slider

        self.plt = plt
        self.np = np
        self.figure = plt.figure(figsize=(15, 9))
        self.figure.canvas.manager.set_window_title(
            "PHY 221 Soccer Free-Kick Simulator"
        )
        self.plot_axes = self.figure.add_axes([0.04, 0.31, 0.62, 0.65], projection="3d")

        slider_color = "#d9e8fb"
        self.speed = Slider(
            self.figure.add_axes([0.09, 0.24, 0.19, 0.025], facecolor=slider_color),
            "Speed (m/s)",
            10,
            40,
            valinit=20,
            valstep=0.5,
        )
        self.angle = Slider(
            self.figure.add_axes([0.45, 0.24, 0.18, 0.025], facecolor=slider_color),
            "Angle (deg)",
            5,
            60,
            valinit=20,
            valstep=0.5,
        )
        self.spin_rate = Slider(
            self.figure.add_axes([0.09, 0.19, 0.19, 0.025], facecolor=slider_color),
            "Spin (rpm)",
            -1200,
            1200,
            valinit=-600,
            valstep=25,
        )
        self.wall_distance = Slider(
            self.figure.add_axes([0.45, 0.19, 0.18, 0.025], facecolor=slider_color),
            "Wall x (m)",
            5,
            15,
            valinit=9.15,
            valstep=0.05,
        )
        self.wall_height = Slider(
            self.figure.add_axes([0.09, 0.14, 0.19, 0.025], facecolor=slider_color),
            "Wall height (m)",
            1,
            3,
            valinit=2,
            valstep=0.05,
        )
        self.wall_width = Slider(
            self.figure.add_axes([0.45, 0.14, 0.18, 0.025], facecolor=slider_color),
            "Wall width (m)",
            1,
            6,
            valinit=3,
            valstep=0.1,
        )
        self.wall_center = Slider(
            self.figure.add_axes([0.09, 0.09, 0.19, 0.025], facecolor=slider_color),
            "Wall center y (m)",
            -3,
            3,
            valinit=1.5,
            valstep=0.1,
        )
        self.goal_distance = Slider(
            self.figure.add_axes([0.45, 0.09, 0.18, 0.025], facecolor=slider_color),
            "Goal x (m)",
            16,
            35,
            valinit=20,
            valstep=0.25,
        )
        self.magnus = Slider(
            self.figure.add_axes([0.09, 0.04, 0.19, 0.025], facecolor=slider_color),
            "Magnus coeff.",
            0,
            0.5,
            valinit=0.25,
            valstep=0.01,
        )
        self.drag = Slider(
            self.figure.add_axes([0.45, 0.04, 0.18, 0.025], facecolor=slider_color),
            "Drag coeff.",
            0,
            0.6,
            valinit=0.25,
            valstep=0.01,
        )

        self.model_selector = RadioButtons(
            self.figure.add_axes([0.69, 0.77, 0.13, 0.17], facecolor="#f2f2f2"),
            ("Compare all", "Ideal", "Drag", "Drag + spin"),
            active=0,
        )
        self.spin_axis_selector = RadioButtons(
            self.figure.add_axes([0.84, 0.77, 0.14, 0.17], facecolor="#f2f2f2"),
            ("Sidespin (z)", "Backspin (-y)", "Topspin (+y)"),
            active=0,
        )
        self.results_text = self.figure.text(
            0.69,
            0.72,
            "",
            va="top",
            family="monospace",
            fontsize=9,
            wrap=True,
        )
        self.status_text = self.figure.text(
            0.69, 0.035, "", fontsize=9, color="tab:blue"
        )
        self.save_graph_button = Button(
            self.figure.add_axes([0.69, 0.08, 0.13, 0.045]), "Save graph"
        )
        self.save_csv_button = Button(
            self.figure.add_axes([0.84, 0.08, 0.13, 0.045]), "Export CSV"
        )

        for slider in (
            self.speed,
            self.angle,
            self.spin_rate,
            self.wall_distance,
            self.wall_height,
            self.wall_width,
            self.wall_center,
            self.goal_distance,
            self.magnus,
            self.drag,
        ):
            slider.on_changed(self._update)
        self.model_selector.on_clicked(self._update)
        self.spin_axis_selector.on_clicked(self._update)
        self.save_graph_button.on_clicked(self._save_graph)
        self.save_csv_button.on_clicked(self._save_csv)

        self.results = self._calculate_current()
        self._draw()

    def _selected_spin(self) -> Spin:
        axes = {
            "Sidespin (z)": (0.0, 0.0, 1.0),
            "Backspin (-y)": (0.0, -1.0, 0.0),
            "Topspin (+y)": (0.0, 1.0, 0.0),
        }
        axis_x, axis_y, axis_z = axes[self.spin_axis_selector.value_selected]
        return Spin(self.spin_rate.val, axis_x, axis_y, axis_z)

    def _calculate_current(self) -> ModelResults:
        kick = Kick(self.speed.val, self.angle.val)
        field = Field(
            wall_distance=self.wall_distance.val,
            wall_height=self.wall_height.val,
            goal_distance=max(self.goal_distance.val, self.wall_distance.val + 0.25),
            wall_width=self.wall_width.val,
            wall_center=self.wall_center.val,
        )
        aerodynamics = Aerodynamics(drag_coefficient=self.drag.val)
        return calculate_models(
            kick,
            field,
            aerodynamics,
            self._selected_spin(),
            self.magnus.val,
        )

    def _draw(self) -> None:
        from soccer_simulation import trajectory

        axes = self.plot_axes
        axes.clear()
        results = self.results
        kick = results.kick
        field = results.field
        selection = self.model_selector.value_selected

        ideal_x, ideal_z = trajectory(kick, points=300)
        if selection in ("Compare all", "Ideal"):
            axes.plot(
                ideal_x,
                [kick.lateral_offset] * len(ideal_x),
                ideal_z,
                linestyle="--",
                label="Ideal",
            )
        if selection in ("Compare all", "Drag"):
            axes.plot(
                results.drag.x,
                [kick.lateral_offset] * len(results.drag.x),
                results.drag.y,
                label="Drag",
            )
        if selection in ("Compare all", "Drag + spin"):
            axes.plot(
                results.spun.x, results.spun.y, results.spun.z, label="Drag + spin"
            )

        wall_left = field.wall_center - field.wall_width / 2
        wall_right = field.wall_center + field.wall_width / 2
        axes.plot_surface(
            self.np.array([[field.wall_distance, field.wall_distance]] * 2),
            self.np.array([[wall_left, wall_right], [wall_left, wall_right]]),
            self.np.array([[0, 0], [field.wall_height, field.wall_height]]),
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
        lateral_limit = max(
            field.goal_width / 2 + 1, max(abs(value) for value in results.spun.y) + 1
        )
        vertical_limit = (
            max(
                ideal_maximum_height(kick),
                results.drag.maximum_height,
                results.spun.maximum_height,
                field.wall_height,
                field.goal_height,
            )
            + 1
        )
        axes.set_xlim(0, max(field.goal_distance + 3, 25))
        axes.set_ylim(-lateral_limit, lateral_limit)
        axes.set_zlim(0, vertical_limit)
        axes.set_xlabel("Toward goal, x (m)")
        axes.set_ylabel("Sideways, y (m)")
        axes.set_zlabel("Height, z (m)")
        axes.set_title("Interactive Soccer Free-Kick Physics Simulator")
        axes.legend(loc="upper right")
        self.results_text.set_text(format_results(results))
        self.figure.canvas.draw_idle()

    def _update(self, _value: object) -> None:
        self.status_text.set_text("Calculating...")
        self.figure.canvas.draw_idle()
        try:
            self.results = self._calculate_current()
            self._draw()
            self.status_text.set_text("")
        except ValueError as error:
            self.status_text.set_text(str(error))

    def _output_path(self, suffix: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return Path("outputs") / f"soccer-simulation-{timestamp}.{suffix}"

    def _save_graph(self, _event: object) -> None:
        destination = self._output_path("png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(destination, dpi=200, bbox_inches="tight")
        self.status_text.set_text(f"Saved {destination}")
        self.figure.canvas.draw_idle()

    def _save_csv(self, _event: object) -> None:
        destination = export_results_csv(self.results, self._output_path("csv"))
        self.status_text.set_text(f"Saved {destination}")
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        self.plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="build the interface without opening a window (for testing)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    interface = InteractiveSimulator()
    if not args.no_show:
        interface.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
