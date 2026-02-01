import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Student Dropout Risk Predictor", layout="centered")

MODEL_FILE = "gb_binary_streamlit.joblib"
FEATURES_FILE = "gb_binary_streamlit_features.joblib"


@st.cache_resource
def load_artifacts():
    model = load(MODEL_FILE)
    feature_list = load(FEATURES_FILE)
    return model, feature_list


def yes_no_input(label: str, default_yes: bool = False, help_text: str | None = None) -> int:
    options = ["No", "Yes"]
    idx = 1 if default_yes else 0
    choice = st.selectbox(label, options, index=idx, help=help_text)
    return 1 if choice == "Yes" else 0


def int_input(label: str, min_v: int, max_v: int, default: int, help_text: str | None = None) -> int:
    return int(
        st.number_input(
            label,
            min_value=min_v,
            max_value=max_v,
            value=default,
            step=1,
            help=help_text
        )
    )


def main():
    st.title("Student Dropout Risk Predictor")
    st.caption("Binary prediction: Dropout vs Graduate")

    try:
        model, feature_list = load_artifacts()
    except Exception as e:
        st.error("Could not load model artifacts. Ensure these files exist in the same folder:")
        st.code(f"{MODEL_FILE}\n{FEATURES_FILE}")
        st.exception(e)
        return

    st.subheader("Decision threshold")
    threshold = st.slider(
        "Classify as Dropout if probability ≥",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01
    )

    st.divider()

    # --------------------
    # Inputs
    # --------------------
    inputs = {}

    st.subheader("Academic performance (GPA-style input)")
    st.caption(
        "Singapore uses GPA (0.0–4.0). This app lets you enter GPA, then converts it into the dataset’s format.\n\n"
        "⚠️ Note: In the training dataset, academic performance is represented more like **module pass / not-pass** "
        "(e.g., how many modules were passed, not a true GPA). So GPA here is used as a **front-end proxy**.\n\n"
        "Conversion used: **Pass Rate ≈ GPA ÷ 4.0**."
    )

    col1, col2 = st.columns(2)

    # GPA sliders (frontend-friendly)
    with col1:
        gpa_sem1 = st.slider(
            "Semester 1 GPA (0.0–4.0)",
            0.0, 4.0, 2.8, 0.1,
            help="Enter Semester 1 GPA on a 0.0–4.0 scale."
        )

    with col2:
        gpa_sem2 = st.slider(
            "Semester 2 GPA (0.0–4.0)",
            0.0, 4.0, 2.8, 0.1,
            help="Enter Semester 2 GPA on a 0.0–4.0 scale."
        )

    # Convert GPA -> pass rates (what the model expects)
    # These map to your engineered features used in training
    if "approval_rate_1st" in feature_list:
        inputs["approval_rate_1st"] = gpa_sem1 / 4.0
    if "approval_rate_2nd" in feature_list:
        inputs["approval_rate_2nd"] = gpa_sem2 / 4.0

    # We do NOT ask for overall pass rate (redundant); compute if model expects it
    if "approval_rate_overall" in feature_list:
        a1 = inputs.get("approval_rate_1st", gpa_sem1 / 4.0)
        a2 = inputs.get("approval_rate_2nd", gpa_sem2 / 4.0)
        inputs["approval_rate_overall"] = (a1 + a2) / 2.0

    # Show the converted values for transparency (optional but helps clarity)
    with st.expander("See conversion used (GPA → Pass Rate)", expanded=False):
        st.write(f"Semester 1 pass-rate proxy = {gpa_sem1:.1f} ÷ 4.0 = **{gpa_sem1/4.0:.2f}**")
        st.write(f"Semester 2 pass-rate proxy = {gpa_sem2:.1f} ÷ 4.0 = **{gpa_sem2/4.0:.2f}**")
        if "approval_rate_overall" in feature_list:
            st.write(f"Overall pass-rate proxy = average = **{inputs['approval_rate_overall']:.2f}**")

    st.divider()

    st.subheader("Academic engagement (module-based)")
    st.caption(
        "These fields come from the dataset and behave like **module pass/not-pass + participation signals**.\n\n"
        "• **Assessments Taken** = how many graded assessments were attempted.\n"
        "• **Modules Not Assessed** = modules with **no assessment record** (often indicates absent/withdraw/incomplete)."
    )

    col3, col4 = st.columns(2)

    if "total_enrolled" in feature_list:
        with col3:
            inputs["total_enrolled"] = int_input(
                "Total Modules Enrolled (Sem 1 + Sem 2)",
                min_v=0, max_v=40, default=10,
                help_text="Total number of modules enrolled across the first 2 semesters."
            )

    if "Curricular_units_2nd_sem_evaluations" in feature_list:
        with col4:
            inputs["Curricular_units_2nd_sem_evaluations"] = int_input(
                "Semester 2 Assessments Taken (Count)",
                min_v=0, max_v=60, default=10,
                help_text="Number of graded assessments attempted in Semester 2 (tests/assignments/exams)."
            )

    col5, col6 = st.columns(2)

    if "Curricular_units_1st_sem_without_evaluations" in feature_list:
        with col5:
            inputs["Curricular_units_1st_sem_without_evaluations"] = int_input(
                "Semester 1 Modules Not Assessed (Count)",
                min_v=0, max_v=40, default=0,
                help_text=(
                    "Number of Semester 1 modules with **no assessment record** "
                    "(e.g., absent/withdraw/incomplete). In a pass/not-pass module view, "
                    "a higher count often signals low engagement."
                )
            )

    if "Curricular_units_2nd_sem_without_evaluations" in feature_list:
        with col6:
            inputs["Curricular_units_2nd_sem_without_evaluations"] = int_input(
                "Semester 2 Modules Not Assessed (Count)",
                min_v=0, max_v=40, default=0,
                help_text=(
                    "Number of Semester 2 modules with **no assessment record** "
                    "(e.g., absent/withdraw/incomplete). In a pass/not-pass module view, "
                    "a higher count often signals low engagement."
                )
            )

    st.divider()

    st.subheader("Student profile")
    col7, col8 = st.columns(2)

    if "Age" in feature_list:
        with col7:
            inputs["Age"] = st.slider(
                "Age at Enrollment",
                15, 60, 18, 1,
                help="Student's age when enrolled."
            )

    if "Scholarship_holder" in feature_list:
        with col8:
            inputs["Scholarship_holder"] = yes_no_input(
                "Scholarship Holder",
                default_yes=False,
                help_text="Whether the student has a scholarship."
            )

    st.divider()

    st.subheader("Financial indicators")
    col9, col10 = st.columns(2)

    if "Tuition_fees_up_to_date" in feature_list:
        with col9:
            inputs["Tuition_fees_up_to_date"] = yes_no_input(
                "Tuition Fees Paid Up To Date",
                default_yes=True,
                help_text="Yes = fees are paid. No = fees are overdue."
            )

    if "Debtor" in feature_list:
        with col10:
            inputs["Debtor"] = yes_no_input(
                "Has Outstanding Debt",
                default_yes=False,
                help_text="Yes = student has unpaid debt recorded. No = no debt."
            )

    # Auto compute financial_risk if model expects it
    if "financial_risk" in feature_list:
        tuition = inputs.get("Tuition_fees_up_to_date", 1)
        debtor = inputs.get("Debtor", 0)
        inputs["financial_risk"] = 1 if (tuition == 0 or debtor == 1) else 0
        st.info(f"Financial Risk (auto-computed): **{inputs['financial_risk']}**")

    # Optional macro indicators (only if model expects them)
    macro_feats = [f for f in ["GDP", "Inflation_rate", "Unemployment_rate"] if f in feature_list]
    if macro_feats:
        with st.expander("Optional / External factors (can leave default)", expanded=False):
            if "GDP" in macro_feats:
                inputs["GDP"] = float(st.number_input("GDP (Optional)", value=1.5))
            if "Inflation_rate" in macro_feats:
                inputs["Inflation_rate"] = float(st.number_input("Inflation Rate (Optional)", value=2.0))
            if "Unemployment_rate" in macro_feats:
                inputs["Unemployment_rate"] = float(st.number_input("Unemployment Rate (Optional)", value=2.0))

    st.divider()

    # --------------------
    # Predict
    # --------------------
    if st.button("Predict"):
        # Build row with expected columns
        X_new = pd.DataFrame([{col: inputs.get(col, 0) for col in feature_list}])

        # If model expects Course but we removed it from UI, set it to 0
        if "Course" in feature_list:
            X_new["Course"] = 0

        # Ensure exact column order
        X_new = X_new.reindex(columns=feature_list, fill_value=0)

        try:
            prob_dropout = float(model.predict_proba(X_new)[0][1])
        except Exception as e:
            st.error("Prediction failed (likely feature mismatch).")
            st.write("Expected columns:")
            st.code("\n".join(feature_list))
            st.write("Row sent to model:")
            st.dataframe(X_new)
            st.exception(e)
            return

        pred = 1 if prob_dropout >= threshold else 0

        st.subheader("Prediction result")
        st.metric("Dropout probability", f"{prob_dropout:.3f}")

        if pred == 1:
            st.error("Prediction: **At risk of Dropping Out**")
        else:
            st.success("Prediction: **Likely to Graduate**")

        with st.expander("See input row sent to the model", expanded=False):
            st.dataframe(X_new)


if __name__ == "__main__":
    main()
