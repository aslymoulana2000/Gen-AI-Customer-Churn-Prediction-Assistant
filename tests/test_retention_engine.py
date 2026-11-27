import unittest
import pandas as pd

from retention_engine import (generate_retention_recommendations)

#Automated retention engine tests
class TestRetentionEngine(unittest.TestCase):

    def setUp(self):

        """
        Prepare a sample customer and SHAP evidence
        before each test.
        """

        self.customer_input = pd.DataFrame([{
            "tenure": 6,

            "MonthlyCharges": 95.0,

            "Contract_One year": 0,

            "Contract_Two year": 0,

            "PaymentMethod_Electronic check": 1,

            "InternetService_Fiber optic": 1,

            "OnlineSecurity_Yes": 0,

            "TechSupport_Yes": 0,

            "InternetService_No": 0

        }])

        self.shap_explanations = pd.DataFrame({
            "Feature": [
                "Contract_Two year",
                "MonthlyCharges",
                "PaymentMethod_Electronic check",
                "tenure"
            ],

            "SHAP_Value": [
                0.12,
                0.08,
                0.05,
                -0.04
            ],

            "Direction": [
                "Increases churn probability",
                "Increases churn probability",
                "Increases churn probability",
                "Decreases churn probability"
            ]

        })

    def test_high_risk_classifications(self):

        result = generate_retention_recommendations(

            customer_input=self.customer_input,

            shap_explanations=self.shap_explanations,

            churn_probability=0.85

        )

        self.assertEqual(
            result["risk_level"],
            "High"
        )

    def test_medium_risk_classification(self):
        result = generate_retention_recommendations(

            customer_input=self.customer_input,

            shap_explanations=self.shap_explanations,

            churn_probability=0.50

        )

        self.assertEqual(
            result["risk_level"],
            "Medium"
        )

    def test_low_risk_classification(self):
        result = generate_retention_recommendations(

            customer_input=self.customer_input,

            shap_explanations=self.shap_explanations,

            churn_probability=0.20

        )

        self.assertEqual(
            result["risk_level"],
            "Low"
        )

    def test_positive_shap_drivers_are_identified(self):
        result = generate_retention_recommendations(

            customer_input=self.customer_input,

            shap_explanations=self.shap_explanations,

            churn_probability=0.85

        )

        drivers = result["positive_shap_drivers"]

        driver_names = [
            driver["feature"]
            for driver in drivers
        ]

        self.assertIn(
            "contracttwoyear",
            driver_names
        )

        self.assertIn(
            "monthlycharges",
            driver_names
        )

    def test_month_to_month_contract_recommendation(self):
        result = generate_retention_recommendations(

            customer_input=self.customer_input,

            shap_explanations=self.shap_explanations,

            churn_probability=0.85

        )

        recommendations = result["recommendations"]

        actions = [
            item["recommended_action"]
            for item in recommendations
        ]

        self.assertTrue(
            any(
                "contract" in action.lower()
                for action in actions
            )
        )

    def test_recommendation_limit(self):
        result = generate_retention_recommendations(

            customer_input=self.customer_input,

            shap_explanations=self.shap_explanations,

            churn_probability=0.85,

            max_recommendations=2

        )

        self.assertLessEqual(
            len(result["recommendations"]),
            2
        )

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_retention_recommendations(

                customer_input=self.customer_input,

                shap_explanations=self.shap_explanations,

                churn_probability=1.5

            )

    # Run automated tests
    if __name__ == "__main__":
        unittest.main()



