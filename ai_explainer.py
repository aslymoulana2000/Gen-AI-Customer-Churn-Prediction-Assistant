import json
import pandas as pd

#GEN-AI CUSTOMER CHURN EXPLANATION
def generate_churn_explanation(client, prediction_label, churn_probability, shap_explanation, model_name="openai/gpt-oss-20b"):

    """
        Generate a natural-language explanation of an
        existing machine learning churn prediction.

        Parameters:
            client: Initialized Groq client.
            prediction_label: Model prediction (Churn or Stay).
            churn_probability: Churn probability from 0 to 1.
            shap_explanations: DataFrame containing SHAP results.
            model_name: Groq model identifier.

        Returns:
            AI-generated explanation as a string.
        """

    #Validate prediction probability
    churn_probability = float(churn_probability)

    if not 0 <= churn_probability <= 1:

        raise ValueError(
            "Churn probability must be between 0 and 1."
        )

    #Validate prediction label
    if prediction_label not in ["Likely to Churn", "Likely to Stay"]:
        raise ValueError("Invalid prediction label.")

    #Validate SHAP explanation data
    if not isinstance(shap_explanation, pd.DataFrame):
        raise ValueError(
            "SHAP explanations must be a pandas DataFrame."
        )

    #Validate SHAP explanation columns
    required_columns = [
        "Feature",
        "Prediction",
        "Direction"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in shap_explanation.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing SHAP columns: {missing_columns}"
        )

    if shap_explanation.empty:
        raise ValueError(
            "No SHAP explanations available."
        )

    #Prepare SHAP evidence for the LLM
    shap_evidence = []

    for _, row in shap_explanation.iterrows():

        shap_value = float(row["Shap_Value"])

        if not pd.notna(shap_value):
            raise ValueError("SHAP values contain invalid numbers.")

        shap_evidence.append({
            "Feature": str(row["Feature"]),
            "direction": str(row["Direction"]),
            "impact_percentage_points": round(shap_value * 100, 2)
        })

    #Prepare the prediction information
    prediction_information = {
        "model_prediction": prediction_label,
        "churn_probability_percentage": round(churn_probability * 100, 2),
        "shap_feature_contributions": shap_evidence
    }

    context = json.dumps(
        prediction_information,
        indent=2
    )

    #Define instructions for the AI assistant
    system_prompt = """
    You are an experienced customer retention analyst
    explaining machine learning predictions to business users.

    Your task is to explain an existing customer churn prediction.

    IMPORTANT RULES:

    1. The machine learning model has already made the prediction.
       Never change, override or recalculate it.

    2. Use only the prediction and SHAP evidence supplied.
       Never invent customer information or contributing factors.

    3. Positive SHAP contributions increase the model's
       churn probability relative to its baseline.

    4. Negative SHAP contributions decrease the model's
       churn probability relative to its baseline.

    5. SHAP contributions describe model behavior.
       Do not claim that they prove causation.

    6. A churn probability is an estimate, not a certainty.

    7. Do not invent discounts, promotions, company policies
       or retention actions. Those will be handled separately.

    RESPONSE FORMAT:

    Customer Prediction:
    Briefly explain the model's predicted outcome and probability.

    Explanation:
    Explain the most important contributing factors in
    simple business language.

    Overall Assessment:
    Provide a short, balanced conclusion.

    Use clear, professional and understandable language.
    Avoid unnecessary technical terminology.
    """

    #Send prediction evidence to Groq
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                        "Explain the following customer churn "
                        "prediction using the supplied evidence:\n\n"
                        + context
                )
            }
        ],

        max_completion_tokens=500
    )

    #Extract the generated explanation
    explanation = response.choices[0].message.content

    if not explanation or not explanation.strip():

        raise ValueError(
            "The AI model returned an empty explanation."
        )

    return explanation.strip()