
import unittest

import joblib
import numpy as np
import pandas as pd

from shap_explainer import explain_churn_prediction


#Random forest and shape validation
class TestModelAndSHAP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Load the actual trained model and preprocessing
        artifacts once for the entire test class.
        """

        cls.model = joblib.load("churn_model.pkl")

        cls.scaler = joblib.load("scaler.pkl")

        cls.feature_columns = joblib.load(
            "feature_columns.pkl"
        )

        # Create one sample customer
        cls.customer_input = pd.DataFrame(
            [[0] * len(cls.feature_columns)],
            columns=cls.feature_columns
        )

        cls.customer_input["tenure"] = 6
        cls.customer_input["MonthlyCharges"] = 95.0
        cls.customer_input["TotalCharges"] = 570.0

        # Apply the same preprocessing used by the app
        cls.input_scaled = cls.scaler.transform(
            cls.customer_input
        )

        # Obtain model predictions
        cls.prediction = cls.model.predict(
            cls.input_scaled
        )

        cls.probability = cls.model.predict_proba(
            cls.input_scaled
        )

        # Generate actual SHAP explanation
        cls.shap_result = explain_churn_prediction(

            model=cls.model,

            model_input=cls.input_scaled,

            feature_names=cls.feature_columns,

            top_n=8
        )


    def test_model_has_expected_classes(self):

        self.assertIn(0, self.model.classes_)

        self.assertIn(1, self.model.classes_)


    def test_prediction_probability_sums_to_one(self):

        probability_sum = np.sum(
            self.probability[0]
        )

        self.assertAlmostEqual(
            probability_sum,
            1.0,
            places=6
        )


    def test_feature_count_matches_model_input(self):

        self.assertEqual(
            len(self.feature_columns),
            self.input_scaled.shape[1]
        )


    def test_customer_feature_order_is_preserved(self):

        self.assertEqual(
            list(self.customer_input.columns),
            list(self.feature_columns)
        )


    def test_shap_explanation_contains_expected_columns(self):

        explanation = self.shap_result[
            "feature_explanation"
        ]

        expected_columns = [
            "Feature",
            "SHAP_Value",
            "Direction",
            "Absolute_Impact"
        ]

        for column in expected_columns:

            self.assertIn(
                column,
                explanation.columns
            )


    def test_shap_values_are_finite(self):

        explanation = self.shap_result[
            "feature_explanation"
        ]

        shap_values = explanation["SHAP_Value"].to_numpy()

        self.assertTrue(
            np.isfinite(shap_values).all()
        )


    def test_shap_additivity_validation_passes(self):

        self.assertTrue(
            self.shap_result["additivity_check"],
            "SHAP contributions did not reconstruct "
            "the model's churn probability."
        )


#Run automated tests
if __name__ == "__main__":

    unittest.main()
