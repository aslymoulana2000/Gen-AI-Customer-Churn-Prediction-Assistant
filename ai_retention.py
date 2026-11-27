import json

#Gen-AI Customer retention strategy
def generate_ai_retention_strategy(
        client,
        prediction_label,
        retention_result,
        model_name="openai/gpt-oss-20b"
):

    """
        Generate personalized customer retention strategies
    using the GroqCloud LLM API.

    Parameters:
        client:
            Initialized Groq client.

        prediction_label:
            Existing Random Forest prediction.

        retention_result:
            Structured output from retention_engine.py.

        model_name:
            Groq model identifier.

    Returns:
        AI-generated customer retention strategy.
    """

    #Validate prediction
    valid_predictions = [
        "Likely to Churn",
        "Likely to Stay"
    ]

    if prediction_label not in valid_predictions:
        raise ValueError(
            "Invalid customer prediction."
        )

    #Validate retention engine output
    if not isinstance(retention_result, dict):
        raise ValueError(
            "Retention result must be a dictionary."
        )

    required_fields = [
        "risk_level",
        "churn_probability_percentage",
        "positive_shap_drivers",
        "recommendations"
    ]

    for field in required_fields:

        if field not in retention_result:
            raise ValueError(
                f"Missing retention result field: {field}"
            )

    #Validate risk information
    risk_level = retention_result["risk_level"]

    if risk_level not in ["Low", "Medium", "High"]:
        raise ValueError(
            "Invalid customer risk category."
        )

    churn_probability = float(
        retention_result["churn_probability_percentage"]
    )

    if not 0 <= churn_probability <= 100:
        raise ValueError(
            "Churn probability must be between 0 and 100."
        )

    #Validate SHAP evidence
    positive_drivers = retention_result[
        "positive_shap_drivers"
    ]

    if not isinstance(positive_drivers, list):
        raise TypeError(
            "Positive SHAP drivers must be a list."
        )

    validated_drivers = []

    for driver in positive_drivers:

        if not isinstance(driver, dict):
            raise TypeError(
                "Each SHAP driver must be a dictionary."
            )

        feature = str(driver["feature"])

        impact = float(
            driver["impact_percentage_points"]
        )

        if not feature.strip():
            raise ValueError(
                "SHAP feature names cannot be empty."
            )

        validated_drivers.append({

            "feature": feature,

            "impact_percentage_points": round(
                impact, 2
            )

        })

    #Validate retention recommendations
    recommendations = retention_result[
        "recommendations"
    ]

    if not isinstance(recommendations, list) or not recommendations:
        raise ValueError(
            "No retention recommendations were supplied."
        )

    validated_recommendations = []

    for recommendation in recommendations:

        if not isinstance(recommendation, dict):
            raise TypeError(
                "Each recommendation must be a dictionary."
            )

        trigger = str(
            recommendation["trigger"]
        )

        action = str(
            recommendation["recommended_action"]
        )

        if not trigger.strip() or not action.strip():
            raise ValueError(
                "Recommendation evidence and actions cannot be empty."
            )

        validated_recommendations.append({

            "evidence": trigger,

            "recommended_action": action

        })

    #Prepare structured information for the LLM
    customer_analysis = {

        "existing_machine_learning_prediction": prediction_label,

        "churn_probability_percentage": round(
            churn_probability, 2
        ),

        "customer_risk_level": risk_level,

        "positive_shap_contributions": validated_drivers,

        "retention_engine_recommendations": validated_recommendations

    }

    context = json.dumps(
        customer_analysis,
        indent=2
    )

    #Define AI assistant instructions
    system_prompt = """
        You are an experienced customer retention strategist
        working with a telecommunications business.

        Your responsibility is to transform structured
        customer churn analysis into practical retention strategies.

        IMPORTANT RULES:

        1. The Random Forest model has already generated
           the prediction. Never change or override it.

        2. Use only the customer evidence and recommendations
           supplied in the input.

        3. Never invent customer characteristics,
           services, complaints or historical activities.

        4. Positive SHAP contributions indicate features
           that push the model's churn probability upward
           relative to its baseline.

        5. SHAP contributions describe model behavior.
           They do not establish real-world causation.

        6. Prioritize the retention actions supplied by
           the existing recommendation engine.

        7. You may explain and organize those actions
           into practical business steps.

        8. Never invent company policies, specific discounts,
           promotional amounts, service guarantees or offers
           that have not been provided.

        9. Present retention actions as suggestions for
           a human retention team to review.

        10. Never claim that a recommended action will
            definitely prevent customer churn.

        RESPONSE FORMAT:

        1. Customer Risk Overview

           Summarize the existing prediction, churn probability
           and risk category.

        2. Key Risk Indicators

           Explain the important positive SHAP contributions
           in understandable business language.

        3. Personalized Retention Strategy

           Present the most relevant recommended actions.
           Explain why each action is appropriate based
           on the supplied evidence.

        4. Suggested Execution Plan

           Provide practical steps for a retention team
           to consider when approaching the customer.

        5. Expected Business Benefits

           Describe potential benefits without inventing
           numerical improvements or guaranteeing outcomes.

        6. Follow-up and Monitoring

           Suggest appropriate ways to evaluate whether
           the retention actions improve the customer experience.

        Use professional, clear and concise language.

        Do not produce generic advice unrelated to
        the supplied customer evidence.
        """

    #Send information to GroqCloud
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
                        "Develop a personalized customer retention "
                        "strategy using the following verified "
                        "machine learning and recommendation evidence:\n\n"
                        + context
                )
            }

        ],

        max_completion_tokens=900
    )

    #Extract AI-generated strategy
    ai_strategy = response.choices[0].message.content

    if not ai_strategy or not ai_strategy.strip():
        raise ValueError(
            "The AI model returned an empty retention strategy."
        )

    return ai_strategy.strip()

