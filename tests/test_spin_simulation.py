import unittest

from drag_simulation import simulate_drag
from soccer_simulation import Field, Kick
from spin_simulation import Spin, judge_spin_shot, magnus_acceleration, simulate_spin


class MagnusForceTests(unittest.TestCase):
    def test_zero_spin_produces_no_magnus_acceleration(self):
        self.assertEqual(
            magnus_acceleration(20.0, 0.0, 5.0, Spin(rate_rpm=0.0)),
            (0.0, 0.0, 0.0),
        )

    def test_vertical_spin_bends_forward_motion_sideways(self):
        ax, ay, az = magnus_acceleration(20.0, 0.0, 0.0, Spin(rate_rpm=600.0))
        self.assertAlmostEqual(ax, 0.0)
        self.assertGreater(ay, 0.0)
        self.assertAlmostEqual(az, 0.0)

    def test_reversing_spin_reverses_magnus_force(self):
        positive = magnus_acceleration(20.0, 0.0, 5.0, Spin(rate_rpm=600.0))
        negative = magnus_acceleration(20.0, 0.0, 5.0, Spin(rate_rpm=-600.0))
        for positive_component, negative_component in zip(
            positive, negative, strict=True
        ):
            self.assertAlmostEqual(positive_component, -negative_component)

    def test_nonzero_spin_requires_an_axis(self):
        with self.assertRaises(ValueError):
            Spin(600.0, 0.0, 0.0, 0.0).angular_velocity()


class SpinTrajectoryTests(unittest.TestCase):
    field = Field()
    kick = Kick(speed=25.0, angle_degrees=30.0)

    def test_zero_spin_matches_drag_model(self):
        drag = simulate_drag(self.kick, self.field)
        spun = simulate_spin(self.kick, self.field, spin=Spin(rate_rpm=0.0))
        self.assertAlmostEqual(spun.flight_time, drag.flight_time, delta=0.002)
        self.assertAlmostEqual(spun.horizontal_range, drag.horizontal_range, delta=0.03)
        self.assertAlmostEqual(spun.maximum_height, drag.maximum_height, delta=0.02)
        self.assertAlmostEqual(spun.shot.goal_height, drag.shot.goal_height, delta=0.02)
        self.assertAlmostEqual(spun.shot.goal_lateral, 0.0, delta=0.001)

    def test_sidespin_curves_trajectory_and_reverses_symmetrically(self):
        right = simulate_spin(self.kick, self.field, spin=Spin(rate_rpm=600.0))
        left = simulate_spin(self.kick, self.field, spin=Spin(rate_rpm=-600.0))
        self.assertGreater(right.shot.goal_lateral, 0.0)
        self.assertLess(left.shot.goal_lateral, 0.0)
        self.assertAlmostEqual(
            right.shot.goal_lateral,
            -left.shot.goal_lateral,
            delta=0.02,
        )

    def test_trajectory_ends_on_ground(self):
        result = simulate_spin(self.kick, self.field)
        self.assertEqual(result.z[-1], 0.0)
        self.assertAlmostEqual(result.x[-1], result.horizontal_range)
        self.assertAlmostEqual(result.times[-1], result.flight_time)

    def test_wall_can_be_avoided_sideways(self):
        field = Field(wall_height=3.0, wall_width=1.0, wall_center=1.0)
        result = judge_spin_shot(
            self.kick,
            field,
            wall_lateral=-0.5,
            wall_height=1.0,
            goal_lateral=0.0,
            goal_height=1.0,
            reaches_wall=True,
            reaches_goal=True,
        )
        self.assertTrue(result.avoids_wall)
        self.assertTrue(result.scores)

    def test_sidespin_can_bend_around_offset_wall_and_score(self):
        kick = Kick(speed=20.0, angle_degrees=20.0)
        field = Field(wall_center=1.5)
        result = simulate_spin(kick, field, spin=Spin(rate_rpm=-600.0))
        self.assertLess(result.shot.wall_lateral, -field.ball_radius)
        self.assertTrue(result.shot.avoids_wall)
        self.assertTrue(result.shot.inside_goal_opening)
        self.assertTrue(result.shot.scores)

    def test_ball_radius_detects_wall_edge_collision(self):
        field = Field(wall_height=3.0, wall_width=1.0, wall_center=0.0)
        result = judge_spin_shot(
            self.kick,
            field,
            wall_lateral=0.6,
            wall_height=1.0,
            goal_lateral=0.0,
            goal_height=1.0,
            reaches_wall=True,
            reaches_goal=True,
        )
        self.assertFalse(result.avoids_wall)
        self.assertFalse(result.scores)


if __name__ == "__main__":
    unittest.main()
