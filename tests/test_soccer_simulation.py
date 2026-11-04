import math
import unittest

from soccer_simulation import Field, Kick, evaluate_shot, flight_time, height_at_distance, horizontal_range


class IdealProjectileTests(unittest.TestCase):
    def test_original_baseline_values(self):
        kick = Kick(speed=25.0, angle_degrees=30.0)
        self.assertAlmostEqual(flight_time(kick), 2.548, places=3)
        self.assertAlmostEqual(horizontal_range(kick), 55.17, places=2)

        expected_maximum_height = (25.0 * math.sin(math.radians(30.0))) ** 2 / (2 * 9.81)
        self.assertAlmostEqual(expected_maximum_height, 7.96, places=2)

    def test_height_at_launch_is_initial_height(self):
        self.assertEqual(height_at_distance(Kick(initial_height=0.4), 0), 0.4)


class WallAndGoalTests(unittest.TestCase):
    field = Field()

    def test_scoring_kick_clears_wall_and_fits_below_crossbar(self):
        result = evaluate_shot(Kick(speed=20.0, angle_degrees=20.0), self.field)
        self.assertTrue(result.clears_wall)
        self.assertTrue(result.inside_goal_opening)
        self.assertTrue(result.scores)
        self.assertAlmostEqual(result.wall_height, 2.17, places=2)
        self.assertAlmostEqual(result.goal_height, 1.72, places=2)

    def test_baseline_thirty_degree_kick_goes_over_crossbar(self):
        result = evaluate_shot(Kick(speed=25.0, angle_degrees=30.0), self.field)
        self.assertTrue(result.clears_wall)
        self.assertFalse(result.scores)
        self.assertIn("crossbar", result.outcome)

    def test_low_kick_hits_wall(self):
        result = evaluate_shot(Kick(speed=25.0, angle_degrees=10.0), self.field)
        self.assertFalse(result.clears_wall)
        self.assertFalse(result.scores)
        self.assertIn("wall", result.outcome)

    def test_lateral_miss_does_not_score(self):
        result = evaluate_shot(Kick(speed=20.0, angle_degrees=20.0, lateral_offset=4.0), self.field)
        self.assertTrue(result.clears_wall)
        self.assertFalse(result.inside_goal_opening)
        self.assertFalse(result.scores)
        self.assertIn("goalpost", result.outcome)

    def test_ball_radius_is_included_in_wall_clearance(self):
        point_ball_height = height_at_distance(Kick(speed=25.0, angle_degrees=18.0), self.field.wall_distance)
        nearly_as_tall_wall = Field(wall_height=point_ball_height - 0.05)
        self.assertFalse(evaluate_shot(Kick(speed=25.0, angle_degrees=18.0), nearly_as_tall_wall).clears_wall)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_shot(Kick(), Field(wall_distance=25.0, goal_distance=20.0))


if __name__ == "__main__":
    unittest.main()
