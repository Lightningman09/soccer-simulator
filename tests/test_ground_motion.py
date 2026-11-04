import unittest

from ground_motion import GroundParameters, simulate_ground_motion
from soccer_simulation import Field


def run_ground(
    *,
    vx: float = 8.0,
    vy: float = 0.0,
    vz: float = 0.0,
    omega_x: float = 0.0,
    omega_y: float = 0.0,
    omega_z: float = 0.0,
    field: Field | None = None,
    parameters: GroundParameters | None = None,
):
    return simulate_ground_motion(
        impact_x=0.0,
        impact_y=0.0,
        impact_vx=vx,
        impact_vy=vy,
        impact_vz=vz,
        omega_x=omega_x,
        omega_y=omega_y,
        omega_z=omega_z,
        radius=0.11,
        mass=0.43,
        field=field or Field(wall_distance=100.0, goal_distance=200.0),
        parameters=parameters,
    )


class GroundMotionTests(unittest.TestCase):
    def test_sliding_friction_builds_rolling_spin(self):
        result = run_ground(vx=8.0, omega_y=0.0)
        self.assertIn("sliding", result.phases)
        self.assertIn("rolling", result.phases)
        self.assertGreater(max(result.omega_y), 0.0)
        self.assertLessEqual((result.vx[-1] ** 2 + result.vy[-1] ** 2) ** 0.5, 0.06)

    def test_faster_kick_rolls_farther_under_same_friction(self):
        slower = run_ground(vx=5.0)
        faster = run_ground(vx=10.0)
        self.assertGreater(faster.path_distance, slower.path_distance)

    def test_more_rolling_resistance_shortens_ground_path(self):
        low_resistance = run_ground(
            vx=5.0,
            parameters=GroundParameters(rolling_resistance=0.05),
        )
        high_resistance = run_ground(
            vx=5.0,
            parameters=GroundParameters(rolling_resistance=0.10),
        )
        self.assertGreater(low_resistance.path_distance, high_resistance.path_distance)

    def test_ball_can_roll_into_wall(self):
        result = run_ground(vx=12.0, field=Field())
        self.assertTrue(result.hit_wall)
        self.assertFalse(result.scores)
        self.assertIn("wall", result.outcome)
        self.assertAlmostEqual(result.x[-1], 9.15, places=2)

    def test_ball_can_roll_around_offset_wall_and_score(self):
        field = Field(wall_center=2.0, wall_width=3.0)
        result = run_ground(vx=15.0, field=field)
        self.assertFalse(result.hit_wall)
        self.assertTrue(result.reaches_goal)
        self.assertTrue(result.scores)
        self.assertIn("GOAL", result.outcome)

    def test_restitution_creates_a_bounce(self):
        result = run_ground(
            vx=8.0,
            vz=-5.0,
            parameters=GroundParameters(restitution=0.5),
        )
        self.assertIn("bounce", result.phases)
        self.assertGreater(max(result.z), 0.11)

    def test_invalid_friction_is_rejected(self):
        with self.assertRaises(ValueError):
            run_ground(parameters=GroundParameters(kinetic_friction=-0.1))


if __name__ == "__main__":
    unittest.main()
