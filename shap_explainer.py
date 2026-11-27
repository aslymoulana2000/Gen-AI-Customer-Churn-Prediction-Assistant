import numpy as np
import pandas as pd
import shap

#SHAP - Individual Customer Churn Explanation
def explain_churn_prediction(model, model_input, feature_names, top_n=8):

    """
    Explain one customer's churn prediction using SHAP.

    Parameters:
        model: Trained Random Forest classifier.
        model_input: Processed input passed to the model.
        feature_names: Original ordered feature names.
        top_n: Number of important factors to return.

    Returns:
        Dictionary containing SHAP explanations
        Dictionary containing SHAP explanations
        and prediction validation information.
    """

    #Prepare and validate customer input
    feature_names = list(feature_names)
    input_array = np.asarray(model_input, dtype=float)

    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)

    if input_array.shape[0] != 1:
        raise ValueError(
            "SHAP explanation currently supports one customer at a time."
        )

    if input_array.shape[1] != len(feature_names):
        raise ValueError(
            "Input feature count does not match feature names."
        )

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The supplied model must support predict_proba()."
        )

    #Identify the churn class (encoded as 1)
    classes = np.asarray(model.classes_)
    churn_indices = np.flatnonzero(classes == 1)

    if len(churn_indices) != 1:
        raise ValueError(
            "Could not identify churn class 1 in the model."
        )

    churn_class_index = int(churn_indices[0])

    #Create SHAP TreeExplainer
    explainer = shap.TreeExplainer(
        model,
        feature_names=feature_names,
        feature_perturbation="tree_path_dependent"
    )

    #Calculate SHAP values
    shap_output = explainer.shap_values(input_array)

    #Support different SHAP return formats
    if isinstance(shap_output, list):

        churn_values = np.asarray(shap_output[churn_class_index])

    else:
        shap_array = np.asarray(shap_output)

        if shap_array.ndim == 3:

            churn_values = shap_array[:, :, churn_class_index]

        elif shap_array.ndim == 2:

            churn_values = shap_array

        elif shap_array.ndim == 1:

            churn_values = shap_array.reshape(1, -1)

        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {shap_array.shape}"
            )

    churn_values = churn_values.reshape(-1)

    if len(churn_values) != len(feature_names):

        raise ValueError(
            "SHAP output does not match the feature count."
        )

    #Get the expected model output for churn
    expected_value = np.asarray(explainer.expected_value).reshape(-1)

    if len(expected_value) > churn_class_index:

        base_value = float(expected_value[churn_class_index])

    else:
        base_value = float(expected_value[0])

    #Obtain the actual churn probability
    churn_probability = float(model.predict_proba(input_array)[0, churn_class_index])

    #Verify SHAP additivity
    reconstructed_probability = (base_value + float(np.sum(churn_values)))
    additivity_check = bool(np.isclose(reconstructed_probability, churn_probability, atol=1e-4, rtol=1e-3))

    #Create feature contribution table
    explanation = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": churn_values
    })

    #Positive: Increases churn probability
    #Negative: Decreases churn probability
    explanation["Direction"] = np.where(
        explanation["SHAP_Value"] > 0,
        "Increases churn probability",
        np.where(
            explanation["SHAP_Value"] < 0,
            "Decreases churn probability",
            "No measurable contribution"
        )
    )

    #Rank features by absolute contribution
    explanation["Absolute_Impact"] = (explanation["SHAP_Value"].abs())

    explanation =(explanation.sort_values(by="Absolute_Impact", ascending=False).head(top_n).reset_index(drop=True))

    #Return explanation results
    return {
        "churn_probability": churn_probability,
        "base_value": base_value,
        "reconstructed_probability": reconstructed_probability,
        "additivity_check": additivity_check,
        "feature_explanation": explanation
    }
