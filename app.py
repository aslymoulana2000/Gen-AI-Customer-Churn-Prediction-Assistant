import streamlit as st
import pandas as pd
import joblib
from groq import Groq

from shap_explainer import explain_churn_prediction
from ai_explainer import generate_churn_explanation
from retention_engine import generate_retention_recommendations
from ai_retention import generate_ai_retention_strategy
from churn_assistant import answer_customer_churn_question

#Project Page Description
st.set_page_config(
    page_title="AI-Powered Customer Retention Intelligence System",
    page_icon="📊",
    layout="wide"
)

#Load saved files
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

#Title
def render_page_header(title, subtitle):

    st.markdown(
        f"""
<div style="background: linear-gradient(115deg, #FFFFFF 0%, #7cb3d8 175%); border: 1px solid #DDE8E7; border-left: 10px solid #5495c0; border-radius: 12px; padding: 22px 26px; margin-bottom: 24px;">
<div style="color: #5495c0; font-size: 11px; font-weight: 700; letter-spacing: 1.4px; margin-bottom: 8px;">RETENTION INTELLIGENCE / ANALYTICS WORKSPACE</div>
<h1 style="color: #172033; font-size: 29px; line-height: 1.2; margin: 0;">{title}</h1>
<p style="color: #536175; font-size: 14px; margin: 9px 0 0 0;">{subtitle}</p>
</div>
""",
        unsafe_allow_html=True
    )

render_page_header(
"AI-Powered Customer Retention Intelligence System Using Machine Learning",
"Explore The Machine Learning Power!"
)

st.subheader("Model Loaded Successfully!")

st.markdown("---")

st.markdown("""This system predicts customer churn risk using Machine Learning.

The model analyzes customer demographics, contract details,
billing information, and subscribed services to estimate
the probability of churn and support customer retention strategies.""")

st.subheader("Customer Information")

#Create Sidebar Header & Info
st.sidebar.markdown(
    """
<div style="background: linear-gradient(135deg, #5495c0, #7cb3d8); padding: 18px 15px; border-radius: 12px; margin-bottom: 15px; width: 100%; box-sizing: border-box;">
<div style="color: white; font-size: 32px; font-weight: 700; margin-top: 2px;">About Project</div>
<div style="color: white; font-size: 13px; margin-top: 6px; margin-bottom: 6px; line-height: 1;">Prediction Model & Features:</div>
</div>
""",
    unsafe_allow_html=True
)

st.sidebar.info("""
Machine Learning Model:     
✔ Random Forest

Tech Stacks:    
✔ Python    
✔ Pandas    
✔ NumPy   
✔ Scaling    
✔ SMOTE  
✔ Scikit-Learn  
✔ SHAP  
✔ LLM   
✔ Streamlit     

Objective:      
Predict customer churn and support
customer retention strategies.
""")

#Input - Numbers
tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=0.0
)

#Shows values entered
st.write("Tenure:", tenure)
st.write("Monthly Charges:", monthly_charges)
st.write("Total Charges:", total_charges)

#Inputs - Drop-downs - (One hot encoding ones)
gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    ["Yes", "No"]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["Fiber optic", "DSL", "No"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

#New inputs of drop-downs (One hot encoding ones)
phone_service = st.selectbox(
    "Phone Services",
    ["No", "Yes"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone services"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]
)


#Shows values entered
#st.write("Gender: ", gender)
#st.write("Senior Citizen: ", senior_citizen)
#st.write("Partner: ", partner)
#st.write("Dependents: ", dependents)
#st.write("Contract: ", contract)
#st.write("Internet Service: ", internet_service)
#st.write("Paperless Billing: ", paperless_billing)

#Shows values entered - New set
#st.write("Phone Service: ", phone_service)
#st.write("Multiple Lines: ", multiple_lines)
#st.write("Online Security: ", online_security)
#st.write("Online Backup: ", online_backup)
#st.write("Device Protection: ", device_protection)
#st.write("Tech Support: ", tech_support)
#st.write("Streaming TV: ", streaming_TV)
#st.write("Streaming Movies: ", streaming_movies)
#st.write("Payment Method: ", payment_method)

#Add a Predict Button
predict_button = st.button("Predict Churn")

# Preserve prediction results across Streamlit reruns
if predict_button:
    st.session_state["prediction_requested"] = True

# Continue displaying the prediction after other interactions
if st.session_state.get("prediction_requested", False):

    input_data = pd.DataFrame(
        [[0] * len(feature_columns)],
        columns=feature_columns
    )
    # Input block for each number input
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges

    #If block for each drop-down inputs
    #Senior Citizen
    if senior_citizen == "Yes":
        input_data["SeniorCitizen"] = 1
    else:
        input_data["SeniorCitizen"] = 0

    #Gender
    if gender == "Male":
        input_data["gender_Male"] = 1
    else:
        input_data["gender_Male"] = 0

    #Partner
    if partner == "Yes":
        input_data["Partner_Yes"] = 1
    else:
        input_data["Partner_Yes"] = 0

    #Dependents
    if dependents == "Yes":
        input_data["Dependents_Yes"] = 1
    else:
        input_data["Dependents_Yes"] = 0

    #Phone Service
    if phone_service == "Yes":
        input_data["PhoneService_Yes"] = 1
    else:
        input_data["PhoneService_Yes"] = 0

    #Multiple Lines
    if multiple_lines == "Yes":
        input_data["MultipleLines_Yes"] = 1
    elif multiple_lines == "No phone services":
        input_data["MultipleLines_No phone service"] = 1
    else:
        input_data["MultipleLines_Yes"] = 0
        input_data["MultipleLines_No phone service"] = 0

    #Internet Service
    if internet_service == "No":
        input_data["InternetService_No"] = 1
    elif internet_service == "Fiber optic":
        input_data["InternetService_Fiber optic"] = 1
    else:
        input_data["InternetService_No"] = 0
        input_data["InternetService_Fiber optic"] = 0

    #Online Security
    if online_security== "Yes":
        input_data["OnlineSecurity_Yes"] = 1
    elif online_security == "No internet service":
        input_data["OnlineSecurity_No internet service"] = 1
    else:
        input_data["OnlineSecurity_Yes"] = 0
        input_data["OnlineSecurity_No internet service"] = 0

    #Online Backup
    if online_backup == "Yes":
        input_data["OnlineBackup_Yes"] = 1
    elif online_backup == "No internet service":
        input_data["OnlineBackup_No internet service"] = 1
    else:
        input_data["OnlineBackup_Yes"] = 0
        input_data["OnlineBackup_No internet service"] = 0

    #Device Protection
    if device_protection == "No internet service":
        input_data["DeviceProtection_No internet service"] = 1
    elif device_protection == "Yes":
        input_data["DeviceProtection_Yes"] = 1
    else:
        input_data["DeviceProtection_Yes"] = 0
        input_data["DeviceProtection_No internet service"] = 0

    #Tech Support
    if tech_support == "Yes":
        input_data["TechSupport_Yes"] = 1
    elif tech_support == "No internet service":
        input_data["TechSupport_No internet service"] = 1
    else:
        input_data["TechSupport_Yes"] = 0
        input_data["TechSupport_No internet service"] = 0

    #Streaming TV
    if streaming_tv == "Yes":
        input_data["StreamingTV_Yes"] = 1
    elif streaming_tv == "No internet service":
        input_data["StreamingTV_No internet service"] = 1
    else:
        input_data["StreamingTV_Yes"] = 0
        input_data["StreamingTV_No internet service"] = 0

    #Streaming Movies
    if streaming_movies == "Yes":
        input_data["StreamingMovies_Yes"] = 1
    elif streaming_movies == "No internet service":
        input_data["StreamingMovies_No internet service"] = 1
    else:
        input_data["StreamingMovies_No internet service"] = 0
        input_data["StreamingMovies_Yes"] = 0

    #Contract
    if contract == "One year":
        input_data["Contract_One year"] = 1
    elif contract == "Two year":
        input_data["Contract_Two year"] = 1
    else:
        input_data["Contract_One year"] = 0
        input_data["Contract_Two year"] = 0

    #Paperless Billing
    if paperless_billing == "Yes":
        input_data["PaperlessBilling_Yes"] = 1
    else:
        input_data["PaperlessBilling_Yes"] = 0

    #Payment Method
    if payment_method == "Credit card (automatic)":
        input_data["PaymentMethod_Credit card (automatic)"] = 1
    elif payment_method == "Electronic check":
        input_data["PaymentMethod_Electronic check"] = 1
    elif payment_method == "Mailed check":
        input_data["PaymentMethod_Mailed check"] = 1
    else:
        input_data["PaymentMethod_Credit card (automatic)"] = 0
        input_data["PaymentMethod_Electronic check"] = 0
        input_data["PaymentMethod_Mailed check"] = 0

    #Scale the Input Data - Make Prediction - Get Probability
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    #SHAP - Individual Customer Explanation
    shap_result = explain_churn_prediction(
        model=model,
        model_input=input_scaled,
        feature_names=feature_columns,
        top_n=8
    )

    #Extract the most influential features
    shap_explanation = shap_result["feature_explanation"]

    st.divider()

    #SHAP - Prediction Explanation Display
    st.subheader("🔍 Why Did the Model Make This Prediction?")

    st.write(
        "The following factors show how individual customer "
        "characteristics influenced the model's churn prediction."
    )

    #Display SHAP validation
    if shap_result["additivity_check"]:

        st.success(
            "SHAP validation passed. Feature contributions "
            "reconstruct the model's churn probability."
        )

    else:

        st.warning(
            "SHAP validation did not pass. "
            "Review the explanation output before using it."
        )

    #Prepare explanation table
    shap_display = shap_explanation.copy()
    shap_display["SHAP Impact (Percentage Points)"] = (shap_display["SHAP_Value"] * 100).round(2)
    shap_display = shap_display[["Feature", "Direction", "SHAP Impact (Percentage Points)"]]

    #Display feature contributions
    st.dataframe(shap_display, use_container_width=True, hide_index=True)

    st.divider()

    #Visualize positive and negative contributions
    with st.container(border=True):

        st.subheader("Feature Contribution Analysis")

        chart_data = (shap_display.set_index("Feature")[["SHAP Impact (Percentage Points)"]])
        st.bar_chart(chart_data)

    st.divider()

    #Groq AI Customer churn explanation
    st.subheader("🤖 AI-Powered Churn Explanation")

    st.write(
        "Generate a personalized explanation of this customer's "
        "churn prediction using our Groq-powered AI assistant."
    )

    #Create a signature for the current customer input
    current_input_signature = input_data.to_json(orient="split", default_handler=str)

    #Clear an old explanation if the customer input changes
    if st.session_state.get("ai_explanation_input") != current_input_signature:

        st.session_state.pop("ai_churn_explanation", None)
        st.session_state["ai_explanation_input"] = current_input_signature

    #Display the generated explanation
    if st.session_state.get("ai_churn_explanation"):

        st.markdown("---")

        st.subheader("🧠 AI Assistant Analysis")

        st.markdown(st.session_state["ai_churn_explanation"])

    #Display the Result
    st.subheader("Prediction Results")

    if prediction[0] == 1:
        st.error("Customer Likely To Churn!")
    else:
        st.success("Customer Likely To Stay!")

    #Display Churn Probability
    stay_probability = probability[0][0] * 100
    churn_probability = probability[0][1] * 100

    #Dashboard Metrics
    col1, col2, col3 = st.columns(3)

    #KPI Cards
    #Card 1: Customer Status
    with col1:
        if prediction[0] == 1:
            st.metric("Customer Status", "At Risk")
        else:
            st.metric("Customer Status", "Stable")

    #Churn Risk
    with col2:

        st.metric("Churn Risk", f"{churn_probability:.2f}%")

    #Retention Score
    with col3:

        retention_score = 100 - churn_probability

        st.metric("Retention Score", f"{retention_score:.2f}%")

    #Churn Risk Meter
    st.progress(churn_probability / 100)

    #Probabilities
    #st.write(f"Churn Probability: {churn_probability:.2f}%")
    #st.write(f"Stay Probability: {stay_probability:.2f}%")

    #Risk Levels
    if churn_probability < 30:
        st.success("Low Churn Risk!")
    elif churn_probability < 70:
        st.warning("Medium Churn Risk!")
    else:
        st.error("High Churn Risk!")

    #Business Insights Panel
    st.subheader("Business Insights")

    risk_factors = []

    if contract == "Month-to-month":
        risk_factors.append("Month-To-Month Contract")

    if internet_service == "Fiber optic":
        risk_factors.append("Fiber Optic Internet")

    if tenure is not None and tenure < 12:
        risk_factors.append("Short Customer Tenure")

    if monthly_charges is not None and monthly_charges > 70:
        risk_factors.append("High Monthly Charges")

    if tech_support == "No":
        risk_factors.append("No Tech Support Subscription")

    if online_security == "No":
        risk_factors.append("No Online Security Service")

    #Display Them Business Insights Panel
    if len(risk_factors) > 0:
        st.write("Top Risk Factors Detected")

        for factor in risk_factors:
            st.write(f"✓ {factor}")

    #Top Factors Influencing Churn - Explainable AI (XAI)
    st.subheader("Model Insights")
    st.write("""Top Features Influencing Churn Prediction""")

    #Display Top Factors
    top_features = [
        "Total Charges",
        "Monthly Charges",
        "Tenure",
        "Fiber Optic Internet",
        "Electronic Check Payment"
    ]

    for i, feature in enumerate(top_features, start=1):
        st.write(f"{i}. {feature}")

    #let's create dynamic explanations - (Churn Analysis).
    st.subheader("Churn Analysis")

    reasons = []

    if contract == "Month-to-month":
        reasons.append("Month-to-month contracts are associated with higher churn.")

    if internet_service == "Fiber optic":
        reasons.append("Fiber optic customers historically showed higher churn.")

    if monthly_charges is not None and monthly_charges > 70:
        reasons.append("Higher monthly charges may increase churn risk.")

    if tenure is not None and tenure < 12:
        reasons.append("Short tenure customers are more likely to churn.")

    if tech_support == "No":
        reasons.append('Customers with "**No Tech Support**" increase the churn risk.')

    if online_security == "No":
        reasons.append('Customers with "**No Online Security**" increase the churn risk.')

    if payment_method == "Electronic check":
        reasons.append("Electronic check users exhibited higher churn rates.")

    #Display dynamic explanations - (Churn Analysis)
    if len(reasons) > 0:
        st.write("Dynamic Explanations")

        for reason in reasons:
            st.write(f"✓ {reason}")

    st.divider()

    #Recommendation Engine
    st.subheader("🎯 Intelligent Retention Recommendations")

    st.write(
        "Our recommendation engine analyzes the customer's "
        "churn probability, SHAP contributions and service "
        "information to identify suitable retention actions."
    )

    # Generate recommendations only when SHAP validation passes
    if shap_result["additivity_check"]:

        try:

            # Generate personalized retention recommendations
            retention_result = generate_retention_recommendations(

                # Original customer data, not scaled data
                customer_input=input_data,

                # Actual SHAP feature contributions
                shap_explanations=shap_explanation,

                # Churn probability as a decimal between 0 and 1
                churn_probability=float(probability[0][1]),

                max_recommendations=4
            )

            # Display customer risk assessment
            risk_level = retention_result["risk_level"]

            st.markdown("### Customer Risk Assessment")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Churn Risk Level",
                    risk_level
                )

            with col2:

                st.metric(
                    "Churn Probability",
                    f"{retention_result['churn_probability_percentage']:.2f}%"
                )

            # Display positive SHAP drivers
            positive_drivers = retention_result[
                "positive_shap_drivers"
            ]

            with st.expander("View Contributing Risk Factors"):

                if positive_drivers:

                    for driver in positive_drivers:
                        st.write(
                            f"• {driver['feature']}: "
                            f"+{driver['impact_percentage_points']:.2f} "
                            "percentage points"
                        )

                else:

                    st.write(
                        "No positive SHAP contributions were "
                        "identified among the supplied features."
                    )

            # Display retention recommendations
            st.markdown("### Recommended Retention Actions")

            recommendations = retention_result["recommendations"]

            # Display recommendations inside a loop
            for index, recommendation in enumerate(
                    recommendations,
                    start=1
            ):
                with st.container(border=True):
                    st.markdown(
                        f"**Recommendation {index}**"
                    )

                    st.write(
                        recommendation["recommended_action"]
                    )

                    st.caption(
                        "Evidence: "
                        + recommendation["trigger"]
                    )

            st.divider()

            #Gen-AI customer retention strategist (ai_retention)
            st.subheader("🤖 AI-Powered Retention Strategist")

            st.write(
                "Generate a personalized retention strategy using "
                "the customer's churn prediction, SHAP evidence "
                "and recommended retention actions."
            )

            # Track the current customer input
            current_input_signature = input_data.to_json(
                orient="split",
                default_handler=str
            )

            # Remove an old strategy when customer details change
            if (
                    st.session_state.get("ai_retention_input")
                    != current_input_signature
            ):
                st.session_state.pop(
                    "ai_retention_strategy",
                    None
                )

                st.session_state["ai_retention_input"] = (
                    current_input_signature
                )

            # Generate AI retention strategy
            if st.button(
                    "✨ Generate AI Retention Strategy",
                    use_container_width=True,
                    key="generate_ai_retention_strategy"
            ):

                try:

                    with st.spinner(
                            "AI strategist is developing a personalized retention plan..."
                    ):

                        groq_client = Groq(
                            api_key=st.secrets["GROQ_API_KEY"]
                        )

                        prediction_label = (
                            "Likely to Churn"
                            if int(prediction[0]) == 1
                            else "Likely to Stay"
                        )

                        ai_strategy = generate_ai_retention_strategy(

                            client=groq_client,

                            prediction_label=prediction_label,

                            retention_result=retention_result,

                            model_name="openai/gpt-oss-20b"
                        )

                        st.session_state[
                            "ai_retention_strategy"
                        ] = ai_strategy

                    st.success(
                        "AI retention strategy generated successfully!"
                    )

                except Exception as error:

                    st.error(
                        "Unable to generate the AI retention strategy. "
                        "Please check your API configuration and try again."
                    )

                    st.caption(
                        f"Error details: {type(error).__name__}: {error}"
                    )

            # Display generated AI retention strategy
            if st.session_state.get("ai_retention_strategy"):
                st.markdown("---")

                with st.container(border=True):
                    st.subheader(
                        "📋 Personalized Customer Retention Plan"
                    )

                    st.markdown(
                        st.session_state["ai_retention_strategy"]
                    )

                    st.divider()

                    #Interactive customer churn AI assistant (churn_assistant)
                    st.subheader("💬 Interactive Customer Churn Assistant")

                    st.write(
                        "Ask questions about the customer's churn prediction, "
                        "risk factors, SHAP explanations and retention strategies."
                    )

                    #Track the current customer profile
                    chat_customer_signature = input_data.to_json(
                        orient="split",
                        default_handler=str
                    )

                    #Reset chat when a different customer is entered
                    if (
                        st.session_state.get("churn_chat_customer_signature")
                        != chat_customer_signature
                    ):

                        st.session_state["churn_chat_history"] = []
                        st.session_state["churn_chat_customer_signature"] = (
                            chat_customer_signature
                        )

                    #Initialize chat history if it doesn't exist
                    if "churn_chat_history" not in st.session_state:

                        st.session_state["churn_chat_history"] = []

                    #Prepare current customer analysis
                    chat_prediction_label = (
                        "Likely to Churn"
                        if int(prediction[0]) == 1
                        else "Likely to Stay"
                    )

                    customer_context = {
                        "model_prediction": chat_prediction_label,
                        "churn_probability_percentage": round(
                            float(probability[0][1]) * 100, 2
                        ),
                        "stay_probability_percentage": round(
                            float(probability[0][0]) * 100, 2
                        ),

                        "customer_profile": {
                            "tenure_months": int(tenure),

                            "monthly_charges": float(monthly_charges),

                            "total_charges": float(total_charges),

                            "contract": contract,

                            "internet_service": internet_service,

                            "payment_method": payment_method,

                            "tech_support": tech_support,

                            "online_security": online_security
                        },

                        "shap_validation_passed": bool(
                            shap_result["additivity_check"]
                        ),

                        "shap_feature_contributions": [
                            {
                                "feature": str(row["Feature"]),

                                "shap_value": float(row["SHAP_Value"]),

                                "direction": str(row["Direction"])
                            }

                            for _, row in shap_explanation.iterrows()
                        ],

                        "retention_analysis": retention_result,

                        "previous_ai_explanation": (
                            st.session_state.get("ai_churn_explanation")
                        ),

                        "previous_ai_retention_strategy": (
                            st.session_state.get("ai_retention_strategy")
                        )
                    }

                    #Display previous chat messages
                    for message in st.session_state["churn_chat_history"]:

                        with st.chat_message(message["role"]):

                            st.markdown(message["content"])

                    #Accept a new user question
                    user_question = st.chat_input(
                        "Ask something about this customer's churn risk...",
                        key="churn_assistant_chat_input"
                    )

                    #Generate an AI response
                    if user_question:

                        #Save the user's question
                        st.session_state["churn_chat_history"].append({
                            "role": "user",
                            "content": user_question
                        })

                        #Display the question
                        with st.chat_message("user"):

                            st.markdown(user_question)

                        #Generate assistant response
                        with st.chat_message("assistant"):

                            try:

                                with st.spinner(
                                        "AI assistant is analyzing your question..."
                                ):
                                    # Initialize Groq client securely
                                    groq_client = Groq(
                                        api_key=st.secrets["GROQ_API_KEY"]
                                    )

                                    # Generate response
                                    assistant_response = (
                                        answer_customer_churn_question(

                                            client=groq_client,

                                            question=user_question,

                                            customer_context=customer_context,

                                            chat_history=(
                                                st.session_state[
                                                    "churn_chat_history"
                                                ][:-1]
                                            ),

                                            model_name="openai/gpt-oss-20b"

                                        )
                                    )

                                #Display response
                                st.markdown(assistant_response)

                                #Save assistant response
                                st.session_state["churn_chat_history"].append({

                                "role": "assistant",

                                "content": assistant_response

                                })

                                #Keep a maximum of 20 stored messages
                                if len(st.session_state["churn_chat_history"]) > 20:
                                    st.session_state["churn_chat_history"] = (
                                        st.session_state["churn_chat_history"][-20:]
                                    )

                            except Exception as error:

                                st.error(
                                    "The AI assistant could not generate a response. "
                                    "Please try again."
                                )

                                st.caption(
                                    f"Error details: {type(error).__name__}: {error}"
                                )

            st.success(
                "Retention recommendations generated successfully!"
            )

        except (ValueError, TypeError, KeyError) as error:

            st.error(
                "Unable to generate retention recommendations."
            )

            st.caption(f"Error details: {error}")

    else:

        st.warning(
            "Retention recommendations are unavailable because "
            "SHAP validation did not pass."
        )

st.markdown("___")
st.caption("AI-Powered Customer Retention Intelligence System Using Machine Learning | Version 1.0")

#st.write(input_data)
#st.write(feature_columns)
