import csv
import tempfile
import unittest
from pathlib import Path

from drag_simulation import Aerodynamics
from interactive_simulator import (
    calculate_models,
    comparison_rows,
    export_results_csv,
    format_results,
)
from soccer_simulation import Field, Kick
from spin_simulation import Spin


class InteractiveModelTests(unittest.TestCase):
    def setUp(self):
        self.results = calculate_models(
            Kick(speed=20.0, angle_degrees=20.0),
            Field(wall_center=1.5),
            Aerodynamics(),
            Spin(rate_rpm=-600.0),
        )

    def test_calculates_all_three_models(self):
        rows = comparison_rows(self.results)
        self.assertEqual(
            [row["model"] for row in rows], ["Ideal", "Drag", "Drag + spin"]
        )
        self.assertTrue(rows[0]["scores"])
        self.assertFalse(rows[1]["scores"])
        self.assertTrue(rows[2]["scores"])

    def test_display_text_contains_measurements_and_outcomes(self):
        text = format_results(self.results)
        self.assertIn("MODEL COMPARISON", text)
        self.assertIn("Drag + spin", text)
        self.assertIn("OUTCOMES", text)

    def test_csv_export_contains_three_models(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "results.csv"
            export_results_csv(self.results, destination)
            with destination.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]["model"], "Drag + spin")
        self.assertEqual(rows[-1]["scores"], "True")


if __name__ == "__main__":
    unittest.main()
