
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tempfile
import os

# LOAD SAVED MODEL AND EVALUATION RESULTS
model = joblib.load("online_purchase_model.pkl")
assets = joblib.load("dashboard_assets.pkl")

metrics_df = assets["metrics"]
cm = assets["confusion_matrix"]

# DATASET FEATURES
FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

# CONFUSION MATRIX
def create_confusion_matrix():
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Purchase", "Purchase"],
        yticklabels=["No Purchase", "Purchase"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Purchase Prediction Confusion Matrix")
    plt.tight_layout()

    return fig


# CSV PREDICTION
def predict_csv(file_path):

    if file_path is None:
        raise gr.Error("Please upload a CSV file.")

    data = pd.read_csv(file_path)

    if data.empty:
        raise gr.Error("The uploaded CSV is empty.")

    data = data.drop(columns=["Revenue"], errors="ignore")

    missing = [col for col in FEATURES if col not in data.columns]

    if missing:
        raise gr.Error(f"Missing columns: {missing}")

    customer_data = data[FEATURES]

    predictions = model.predict(customer_data)
    probabilities = model.predict_proba(customer_data)[:, 1]

    results = customer_data.copy()

    results["Predicted_Revenue"] = predictions
    results["Prediction"] = [
        "Purchase" if p == 1 else "No Purchase"
        for p in predictions
    ]

    results["Purchase_Probability_%"] = (
        probabilities * 100
    ).round(2)

    summary = (
        f"Total Customers: {len(results)}\n"
        f"Predicted Purchases: {(predictions == 1).sum()}\n"
        f"Predicted No Purchases: {(predictions == 0).sum()}"
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False
    ) as f:
        results.to_csv(f.name, index=False)
        output_path = f.name

    return summary, results, output_path


# MANUAL CUSTOMER PREDICTION
def predict_manual(*values):

    customer_data = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    prediction = model.predict(customer_data)[0]
    probabilities = model.predict_proba(customer_data)[0]

    result = "PURCHASE" if prediction == 1 else "NO PURCHASE"

    details = pd.DataFrame({
        "Prediction": [result],
        "No Purchase Probability (%)": [
            round(probabilities[0] * 100, 2)
        ],
        "Purchase Probability (%)": [
            round(probabilities[1] * 100, 2)
        ]
    })

    return result, details


# DASHBOARD INTERFACE
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Online Purchase Prediction System"
) as demo:

    gr.Markdown("""
    # 🛒 Online Purchase Prediction System

    ### Machine Learning Project using Logistic Regression

    Predict whether an online visitor is likely to make a purchase
    based on their website browsing behaviour.
    """)

    gr.Markdown("---")

    # MODEL PERFORMANCE
    gr.Markdown("## 📊 Model Performance")

    gr.Dataframe(
        value=metrics_df,
        label="Model Evaluation Metrics",
        interactive=False
    )

    gr.Plot(
        value=create_confusion_matrix(),
        label="Confusion Matrix"
    )

    gr.Markdown("---")

    # CSV UPLOAD
    gr.Markdown("## 📁 Predict Using CSV File")

    csv_file = gr.File(
        label="Upload Customer CSV",
        file_types=[".csv"],
        type="filepath"
    )

    csv_button = gr.Button(
        "Predict Customer Purchases",
        variant="primary"
    )

    csv_summary = gr.Textbox(
        label="Prediction Summary",
        lines=3
    )

    csv_results = gr.Dataframe(
        label="Customer Prediction Results",
        interactive=False
    )

    csv_download = gr.File(
        label="Download Prediction Results"
    )

    csv_button.click(
        fn=predict_csv,
        inputs=csv_file,
        outputs=[csv_summary, csv_results, csv_download]
    )

    gr.Markdown("---")

    # MANUAL INPUT
    gr.Markdown("## 👤 Manual Customer Prediction")

    with gr.Row():
        with gr.Column():
            administrative = gr.Number(
                label="Administrative Pages", value=0
            )
            administrative_duration = gr.Number(
                label="Administrative Duration", value=0
            )
            informational = gr.Number(
                label="Informational Pages", value=0
            )
            informational_duration = gr.Number(
                label="Informational Duration", value=0
            )
            product_related = gr.Number(
                label="Product Related Pages", value=1
            )
            product_duration = gr.Number(
                label="Product Related Duration", value=30
            )

        with gr.Column():
            bounce_rates = gr.Number(
                label="Bounce Rates", value=0.02
            )
            exit_rates = gr.Number(
                label="Exit Rates", value=0.05
            )
            page_values = gr.Number(
                label="Page Values", value=0
            )
            special_day = gr.Number(
                label="Special Day", value=0
            )
            month = gr.Dropdown(
                label="Month",
                choices=[
                    "Jan", "Feb", "Mar", "Apr", "May", "June",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
                ],
                value="Nov"
            )
            operating_systems = gr.Number(
                label="Operating Systems", value=2
            )

        with gr.Column():
            browser = gr.Number(
                label="Browser", value=2
            )
            region = gr.Number(
                label="Region", value=1
            )
            traffic_type = gr.Number(
                label="Traffic Type", value=2
            )
            visitor_type = gr.Dropdown(
                label="Visitor Type",
                choices=[
                    "Returning_Visitor",
                    "New_Visitor",
                    "Other"
                ],
                value="Returning_Visitor"
            )
            weekend = gr.Checkbox(
                label="Weekend Visit",
                value=False
            )

    manual_button = gr.Button(
        "Predict Purchase Behaviour",
        variant="primary"
    )

    manual_result = gr.Textbox(
        label="Predicted Purchase Behaviour"
    )

    manual_details = gr.Dataframe(
        label="Purchase Probability Details",
        interactive=False
    )

    manual_button.click(
        fn=predict_manual,
        inputs=[
            administrative,
            administrative_duration,
            informational,
            informational_duration,
            product_related,
            product_duration,
            bounce_rates,
            exit_rates,
            page_values,
            special_day,
            month,
            operating_systems,
            browser,
            region,
            traffic_type,
            visitor_type,
            weekend
        ],
        outputs=[manual_result, manual_details]
    )

    gr.Markdown("""
    ---
    **Project:** Online Purchase Prediction Using Logistic Regression

    **Dataset:** UCI Online Shoppers Purchasing Intention Dataset
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
