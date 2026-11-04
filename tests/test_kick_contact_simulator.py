import csv
import tempfile
import unittest
from pathlib import Path

from kick_contact_simulator import (
    ContactParameters,
    calculate_contact,
    calculate_contact_flight,
    contact_summary,
    export_contact_csv,
)


class ContactMechanicsTests(unittest.TestCase):
    def test_central_straight_impulse_produces_speed_without_spin(self):
        parameters = ContactParameters(
            force_newtons=860,
            contact_time_seconds=0.01,
            contact_y=0,
            contact_z=0,
        )
        result = calculate_contact(parameters)
        self.assertAlmostEqual(result.launch_speed, 20.0)
        self.assertAlmostEqual(result.spin_rate_rpm, 0.0)

    def test_impulse_momentum_sets_launch_speed(self):
        parameters = ContactParameters(force_newtons=1000, contact_time_seconds=0.01)
        result = calculate_contact(parameters)
        self.assertAlmostEqual(result.launch_speed, 10 / parameters.ball_mass)

    def test_right_side_contact_produces_negative_vertical_spin(self):
        right = calculate_contact(ContactParameters(contact_y=0.03, contact_z=0))
        left = calculate_contact(ContactParameters(contact_y=-0.03, contact_z=0))
        self.assertLess(right.omega_z, 0)
        self.assertGreater(left.omega_z, 0)
        self.assertAlmostEqual(right.omega_z, -left.omega_z)

    def test_force_line_through_center_produces_negligible_spin(self):
        parameters = ContactParameters(contact_y=0, contact_z=-0.04)
        result = calculate_contact(parameters)
        self.assertAlmostEqual(result.spin_rate_rpm, 0.0, places=8)

    def test_high_contact_does_not_create_an_upward_launch(self):
        result = calculate_contact(ContactParameters(contact_y=-0.048, contact_z=0.082))
        self.assertLess(result.applied_force_elevation_degrees, 0)
        self.assertGreater(result.ground_reaction_impulse_z, 0)
        self.assertEqual(result.launch_angle_degrees, 0)

    def test_below_center_contact_creates_an_upward_launch(self):
        result = calculate_contact(ContactParameters(contact_y=0, contact_z=-0.04))
        self.assertGreater(result.applied_force_elevation_degrees, 0)
        self.assertAlmostEqual(result.ground_reaction_impulse_z, 0)
        self.assertGreater(result.launch_angle_degrees, 0)

    def test_foot_lift_adjustment_changes_force_angle_and_adds_spin(self):
        natural = calculate_contact(ContactParameters(contact_y=0, contact_z=-0.04))
        adjusted = calculate_contact(
            ContactParameters(
                contact_y=0,
                contact_z=-0.04,
                foot_lift_degrees=10,
            )
        )
        self.assertAlmostEqual(
            adjusted.applied_force_elevation_degrees,
            natural.applied_force_elevation_degrees + 10,
        )
        self.assertGreater(adjusted.spin_rate_rpm, natural.spin_rate_rpm)

    def test_contact_outside_ball_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_contact(ContactParameters(contact_y=0.11, contact_z=0.11))


class ContactFlightTests(unittest.TestCase):
    def setUp(self):
        self.result = calculate_contact_flight(ContactParameters())

    def test_default_contact_generates_realistic_initial_conditions(self):
        self.assertAlmostEqual(self.result.contact.launch_speed, 19.77, places=2)
        self.assertGreater(self.result.contact.spin_rate_rpm, 500)
        self.assertLess(self.result.contact.spin_axis_z, 0)
        self.assertTrue(self.result.flight.shot.reaches_goal)

    def test_summary_contains_contact_and_flight_results(self):
        summary = contact_summary(self.result)
        self.assertIn("KICK RESULT", summary)
        self.assertIn("Spin axis", summary)
        self.assertIn("FLIGHT RESULT", summary)

    def test_csv_export_contains_contact_and_outcome(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "contact.csv"
            export_contact_csv(self.result, destination)
            with destination.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.DictReader(csv_file))
        self.assertIn("input_contact_y", row)
        self.assertIn("result_spin_rate_rpm", row)
        self.assertIn("outcome", row)

    def test_high_contact_transitions_to_ground_motion_and_hits_wall(self):
        result = calculate_contact_flight(
            ContactParameters(contact_y=-0.058, contact_z=0.060)
        )
        self.assertIsNotNone(result.ground)
        assert result.ground is not None
        self.assertTrue(result.ground.hit_wall)
        self.assertIn("wall", result.final_outcome)


if __name__ == "__main__":
    unittest.main()
