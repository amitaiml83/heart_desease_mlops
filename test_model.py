import unittest
import pandas as pd
import joblib
import os

class TestHeartDiseaseModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load model & files
        cls.model_path = "model.pkl"
        cls.test_data_path = "test.csv"
        cls.output_csv = "unit_test_output.csv"

        if not os.path.exists(cls.model_path):
            raise FileNotFoundError("model.pkl not found!")

        if not os.path.exists(cls.test_data_path):
            raise FileNotFoundError("unit_test_data.csv not found!")

        # Load model
        cls.model = joblib.load(cls.model_path)

        # Load test data
        cls.test_df = pd.read_csv(cls.test_data_path)

        # Expected model input columns
        cls.expected_columns = [
            "age","gender","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak",
            "slope","ca","thal"
        ]

    def test_model_loaded(self):
        """Test if the model loads correctly"""
        self.assertIsNotNone(self.model)

    def test_columns_match(self):
        """Check if input columns match expected"""
        self.assertListEqual(list(self.test_df.columns), self.expected_columns)

    def test_prediction_output(self):
        """Check if prediction runs successfully"""
        preds = self.model.predict(self.test_df)
        self.assertEqual(len(preds), len(self.test_df))

    def test_prediction_values(self):
        """Ensure predictions are only 0 or 1"""
        preds = self.model.predict(self.test_df)
        for p in preds:
            self.assertIn(p, [0, 1])


# -----------------------------
#  Add CSV output after tests
# -----------------------------
def save_predictions():
    """Generate predictions CSV after tests are executed."""
    try:
        model = joblib.load("model.pkl")
        df = pd.read_csv("test.csv")

        preds = model.predict(df)

        df["prediction"] = preds  # append predictions

        df.to_csv("unit_test_output.csv", index=False)

        print("\n==============================")
        print("Predictions saved to unit_test_output.csv")
        print("Generated Predictions:")
        print(df.head(10))
        print("==============================\n")

    except Exception as e:
        print("Error while saving predictions:", str(e))


if __name__ == "__main__":
    # Run tests
    unittest.main(exit=False)

    # After all tests → Save predictions
    save_predictions()

