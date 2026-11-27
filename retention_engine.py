import numpy as np
import pandas as pd

#Intelligent customer retention engine
def normalize_feature_name(feature_name):
    """Normalize feature names for reliable matching."""

    return "".join(
        character.lower()
        for character in str(feature_name)
        if character.isalnum()
    )

def get_customer_value(customer_input, *feature_names):
    """
    Retrieve a value from the original, unscaled
    customer input DataFrame.
    """

    if not isinstance(customer_input, pd.DataFrame):

        raise TypeError(
            "Customer input must be a pandas DataFrame."
        )

    if len(customer_input) != 1:

        raise ValueError(
            "Exactly one customer record is required."
        )

    column_lookup = {
        normalize_feature_name(column): column
        for column in customer_input.columns
    }

    for feature_name in feature_names:

        normalize_name = normalize_feature_name(feature_name)

        actual_column = column_lookup.get(normalize_name)

        if actual_column is not None:
            return customer_input.iloc[0][actual_column]

    return None

def is_feature_active(value):
    """Check whether an encoded feature is active."""

    if value is None or pd.isna(value):
        return False

    return float(value) >= 0.5

def get_positive_shap_drivers(shap_explanation):
    """
    Extract features that positively contribute
    to the model's churn probability.

    Positive SHAP values indicate an upward
    contribution relative to the model baseline.
    """

    if not isinstance(shap_explanation, pd.DataFrame):
        raise TypeError(
            "SHAP explanations must be a DataFrame."
        )

    required_columns = ["Feature", "SHAP_Value"]

    if not all(
        column in shap_explanation.columns
        for column in required_columns
    ):
        raise ValueError(
            "SHAP DataFrame is missing required columns."
        )

    positive_drivers = {}

    for _, row in shap_explanation.iterrows():

        feature = str(row["Feature"])
        shap_value = float(row["SHAP_Value"])

        if not np.isfinite(shap_value):
            raise ValueError(
                "SHAP values must contain finite numbers."
            )

        if shap_value > 0:
            positive_drivers[
                normalize_feature_name(feature)
            ] = shap_value

    return positive_drivers

def has_positive_shap(positive_drivers, *feature_names):
    """Check whether a feature has a positive SHAP contribution."""

    for feature_name in feature_names:

        normalized_name = normalize_feature_name(
            feature_name
        )

        if any(
            normalized_name in feature
            for feature in positive_drivers
        ):
            return True

    return False

def generate_retention_recommendations(
    customer_input,
    shap_explanations,
    churn_probability,
    max_recommendations=4
):
    """
    Generate evidence-based customer retention actions.

    Parameters:
        customer_input:
            Original, unscaled encoded customer DataFrame.

        shap_explanations:
            SHAP feature contribution DataFrame.

        churn_probability:
            Model's churn probability between 0 and 1.

        max_recommendations:
            Maximum number of actions to return.

    Returns:
        Dictionary containing risk assessment,
        positive SHAP drivers and retention actions.
    """

    # Validate inputs
    churn_probability = float(churn_probability)

    if not 0 <= churn_probability <= 1:
        raise ValueError(
            "Churn probability must be between 0 and 1."
        )

    if not isinstance(max_recommendations, int):
        raise TypeError(
            "max_recommendations must be an integer."
        )

    if max_recommendations < 1:
        raise ValueError(
            "At least one recommendation must be allowed."
        )

    #Determine customer risk category
    if churn_probability >= 0.70:
        risk_level = "High"

    elif churn_probability >= 0.30:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    #Extract positive SHAP drivers
    positive_drivers = get_positive_shap_drivers(
        shap_explanations
    )

    #Read original customer characteristics
    tenure = get_customer_value(
        customer_input,
        "tenure",
        "Tenure"
    )

    monthly_charges = get_customer_value(
        customer_input,
        "MonthlyCharges",
        "monthly_charges"
    )

    contract_one_year = get_customer_value(
        customer_input,
        "Contract_One year"
    )

    contract_two_year = get_customer_value(
        customer_input,
        "Contract_Two year"
    )

    electronic_check = get_customer_value(
        customer_input,
        "PaymentMethod_Electronic check"
    )

    fiber_optic = get_customer_value(
        customer_input,
        "InternetService_Fiber optic"
    )

    online_security = get_customer_value(
        customer_input,
        "OnlineSecurity_Yes"
    )

    tech_support = get_customer_value(
        customer_input,
        "TechSupport_Yes"
    )

    no_internet_service = get_customer_value(
        customer_input,
        "InternetService_No"
    )

    #Build evidence-based retention actions
    recommendations = []

    def add_recommendation(trigger, action):

        recommendations.append({
            "trigger": trigger,
            "recommended_action": action,
            "priority": risk_level
        })

    # Contract-related recommendation
    contract_shap = has_positive_shap(
        positive_drivers,
        "Contract_One year",
        "Contract_Two year"
    )

    if contract_shap:

        if (
                contract_one_year is not None
                and contract_two_year is not None
                and not is_feature_active(contract_one_year)
                and not is_feature_active(contract_two_year)
        ):

            add_recommendation(
                "Month-to-month contract characteristics "
                "contribute positively to churn risk.",

                "Review whether an optional longer-term "
                "contract with suitable pricing or benefits "
                "could improve customer retention."
            )

        else:

            add_recommendation(
                "Contract-related features contribute positively "
                "to the model's churn probability.",

                "Review the customer's current contract, "
                "upcoming renewal and overall satisfaction. "
                "Explore suitable alternatives where appropriate."
            )

    # Customer tenure recommendation
    if has_positive_shap(
            positive_drivers,
            "tenure"
    ):

        if tenure is not None and float(tenure) <= 12:

            add_recommendation(
                "Tenure contributes positively to churn risk "
                "and the customer has relatively short tenure.",

                "Provide proactive onboarding support, "
                "a customer check-in and guidance on "
                "getting the most value from available services."
            )

        else:

            add_recommendation(
                "Tenure contributes positively to the model's "
                "churn probability.",

                "Conduct a relationship review and identify "
                "any unresolved service issues or changing "
                "customer expectations."
            )

    # Monthly charges recommendation
    if has_positive_shap(
            positive_drivers,
            "MonthlyCharges"
    ):
        add_recommendation(
            "Monthly charges contribute positively "
            "to the predicted churn probability.",

            "Review the customer's bill, current package "
            "and service usage. Consider suitable plan "
            "alternatives that provide clearer value."
        )

    # Electronic payment recommendation
    if (
            is_feature_active(electronic_check)
            and has_positive_shap(
        positive_drivers,
        "PaymentMethod_Electronic check"
    )
    ):
        add_recommendation(
            "Electronic check payment contributes positively "
            "to the model's churn probability.",

            "Review the payment experience and offer "
            "convenient alternative payment methods "
            "where available."
        )

    # Fiber optic service recommendation
    if (
            is_feature_active(fiber_optic)
            and has_positive_shap(
        positive_drivers,
        "InternetService_Fiber optic"
    )
    ):
        add_recommendation(
            "Fiber optic service contributes positively "
            "to the model's churn probability.",

            "Check internet service satisfaction, "
            "connection quality, reliability and "
            "whether the current package meets expectations."
        )

    # Online security recommendation
    if (
            not is_feature_active(no_internet_service)
            and not is_feature_active(online_security)
            and online_security is not None
            and has_positive_shap(
        positive_drivers,
        "OnlineSecurity_Yes"
    )
    ):
        add_recommendation(
            "The online security feature contributes positively "
            "to churn risk, and the customer is not recorded "
            "as having this service.",

            "Explore whether the customer would benefit "
            "from an optional online security service."
        )

    # Technical support recommendation
    if (
            not is_feature_active(no_internet_service)
            and not is_feature_active(tech_support)
            and tech_support is not None
            and has_positive_shap(
        positive_drivers,
        "TechSupport_Yes"
    )
    ):
        add_recommendation(
            "The technical support feature contributes positively "
            "to churn risk, and the customer is not recorded "
            "as having this service.",

            "Offer information about available technical "
            "support options and assess whether additional "
            "assistance would improve the experience."
        )

    #Add general recommendations when necessary
    if not recommendations:

        if risk_level == "High":

            add_recommendation(
                "The model estimates a high churn probability.",

                "Prioritize a human-led retention conversation. "
                "Identify the customer's main concern and "
                "develop an appropriate service recovery plan."
            )

        elif risk_level == "Medium":

            add_recommendation(
                "The model estimates a medium churn probability.",

                "Schedule a proactive customer check-in, "
                "review satisfaction and identify opportunities "
                "to improve the customer experience."
            )

        else:

            add_recommendation(
                "The model estimates a low churn probability.",

                "Maintain good service quality, monitor "
                "customer satisfaction and continue "
                "normal relationship management."
            )

    #Limit and return recommendations
    recommendations = recommendations[
        :max_recommendations
    ]

    return {
        "risk_level": risk_level,

        "churn_probability_percentage": round(
            churn_probability * 100, 2
        ),

        "positive_shap_drivers": [
            {
                "feature": feature,
                "impact_percentage_points": round(
                    shap_value * 100, 2
                )
            }
            for feature, shap_value in sorted(
                positive_drivers.items(),
                key=lambda item: abs(item[1]),
                reverse=True
            )
        ],

        "recommendations": recommendations
    }


































