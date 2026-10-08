import unittest
import os
import json


class TestPlacementMLPipeline(unittest.TestCase):

    def test_model_exists(self):
        self.assertTrue(
            os.path.exists("placement_model.pkl")
        )

    def test_metrics_exists(self):
        self.assertTrue(
            os.path.exists("metrics.json")
        )

    def test_predictions_exist(self):
        self.assertTrue(
            os.path.exists("placement_predictions.csv")
        )

    def test_accuracy(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(
            metrics["accuracy"],
            0.45
        )


if __name__ == "__main__":
    unittest.main()
