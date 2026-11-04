import unittest

from drag_simulation import Aerodynamics, drag_acceleration, simulate_drag
from soccer_simulation import (
    Field,
    Kick,
    evaluate_shot,
    flight_time,
    height_at_distance,
    horizontal_range,
)


class DragForceTests(unittest.TestCase):
    def test_drag_acceleration_opposes_velocity(self):
        ax, ay = drag_acceleration(20.0, 5.0)
        self.assertLess(ax, 0)
        self.assertLess(ay, -9.81)

    def test_zero_velocity_has_only_gravity(self):
        self.assertEqual(drag_acceleration(0.0, 0.0), (0.0, -9.81))

    def test_invalid_drag_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            simulate_drag(Kick(), aerodynamics=Aerodynamics(mass=0.0))
        with self.assertRaises(ValueError):
            simulate_drag(Kick(), time_step=0.1)


class DragTrajectoryTests(unittest.TestCase):
    field = Field()
    kick = Kick(speed=25.0, angle_degrees=30.0)

    def test_zero_drag_converges_to_ideal_solution(self):
        result = simulate_drag(
            self.kick,
            self.field,
            Aerodynamics(drag_coefficient=0.0),
            time_step=0.001,
        )
        self.assertAlmostEqual(result.flight_time, flight_time(self.kick), delta=0.002)
        self.assertAlmostEqual(
            result.horizontal_range, horizontal_range(self.kick), delta=0.03
        )
        self.assertAlmostEqual(
            result.shot.wall_height,
            height_at_distance(self.kick, self.field.wall_distance),
            delta=0.02,
        )

    def test_drag_reduces_height_and_range(self):
        result = simulate_drag(self.kick, self.field)
        ideal_peak = (self.kick.speed * 0.5) ** 2 / (2 * 9.81)
        self.assertLess(result.maximum_height, ideal_peak)
        self.assertLess(result.horizontal_range, horizontal_range(self.kick))
        self.assertLess(
            result.shot.goal_height,
            height_at_distance(self.kick, self.field.goal_distance),
        )

    def test_drag_model_uses_same_scoring_rules(self):
        scoring_kick = Kick(speed=20.0, angle_degrees=20.0)
        ideal = evaluate_shot(scoring_kick, self.field)
        result = simulate_drag(scoring_kick, self.field)
        self.assertTrue(ideal.scores)
        self.assertFalse(result.shot.scores)
        self.assertIn("wall", result.shot.outcome)

    def test_trajectory_ends_at_ground(self):
        result = simulate_drag(self.kick, self.field)
        self.assertEqual(result.y[-1], 0.0)
        self.assertAlmostEqual(result.x[-1], result.horizontal_range)
        self.assertAlmostEqual(result.times[-1], result.flight_time)


if __name__ == "__main__":
    unittest.main()
