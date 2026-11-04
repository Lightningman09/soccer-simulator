import csv
import math
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
            elevation_degrees=0,
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
        right = calculate_contact(
            ContactParameters(elevation_degrees=0, contact_y=0.03, contact_z=0)
        )
        left = calculate_contact(
            ContactParameters(elevation_degrees=0, contact_y=-0.03, contact_z=0)
        )
        self.assertLess(right.omega_z, 0)
        self.assertGreater(left.omega_z, 0)
        self.assertAlmostEqual(right.omega_z, -left.omega_z)

    def test_force_line_through_center_produces_negligible_spin(self):
        radius = 0.11
        elevation = 20.0
        contact_z = -radius * math.sin(math.radians(elevation))
        parameters = ContactParameters(
            elevation_degrees=elevation,
            contact_y=0,
            contact_z=contact_z,
        )
        result = calculate_contact(parameters)
        self.assertAlmostEqual(result.spin_rate_rpm, 0.0, places=8)

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


if __name__ == "__main__":
    unittest.main()
