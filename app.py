# app.py
import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Hotel Cancellation Predictor", layout="centered")

MODEL_PATH = "rf_model.joblib"
COLS_PATH = "rf_feature_columns.joblib"

# Load model + expected feature columns
rf = load(MODEL_PATH)
feature_cols = load(COLS_PATH)

st.title("Hotel Booking Cancellation Predictor")
st.caption("Predicts cancellation probability using your trained Random Forest model.")

# ----------------------------
# Helpers
# ----------------------------
def engineer_features(raw: dict) -> pd.DataFrame:
    """
    Takes a dict of raw inputs and returns a 1-row DataFrame with engineered features.
    Mirrors your Iteration 2 feature engineering:
      - total_nights
      - has_special_request
      - lead_time_group
      - drops stays_in_* (not added to model)
    """
    df_in = pd.DataFrame([raw])

    # total_nights from stays
    df_in["total_nights"] = df_in["stays_in_week_nights"] + df_in["stays_in_weekend_nights"]

    # has_special_request from total_of_special_requests
    df_in["has_special_request"] = (df_in["total_of_special_requests"] > 0).astype(int)

    # lead_time_group from lead_time
    df_in["lead_time_group"] = pd.cut(
        df_in["lead_time"],
        bins=[-1, 30, 90, 180, 365, 1000],
        labels=["very_short", "short", "medium", "long", "very_long"],
    )

    # Drop raw stay fields (your Iteration 2 model uses total_nights instead)
    df_in = df_in.drop(columns=["stays_in_week_nights", "stays_in_weekend_nights"])

    return df_in


def prepare_for_model(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes and aligns columns to the model's training columns.
    """
    X_new = pd.get_dummies(df_in, drop_first=True)
    X_new = X_new.reindex(columns=feature_cols, fill_value=0)
    return X_new


def predict_one(raw: dict):
    df_in = engineer_features(raw)
    X_new = prepare_for_model(df_in)

    proba = rf.predict_proba(X_new)[0][1]
    pred = int(proba >= threshold)

    return pred, proba


# ----------------------------
# UI
# ----------------------------
mode = st.radio("Input mode", ["Manual form", "Upload CSV"], horizontal=True)
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)

st.divider()

if mode == "Manual form":
    with st.form("booking_form"):
        # Categorical
        hotel = st.selectbox("Hotel", ["Resort Hotel", "City Hotel"])
        arrival_date_month = st.selectbox(
            "Arrival month",
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"],
        )
        meal = st.selectbox("Meal", ["BB", "HB", "FB", "SC", "Undefined"])
        market_segment = st.selectbox(
            "Market segment",
            ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Aviation", "Undefined"],
        )
        distribution_channel = st.selectbox(
            "Distribution channel",
            ["Direct", "Corporate", "TA/TO", "GDS", "Undefined"],
        )
        reserved_room_type = st.selectbox("Reserved room type", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"])
        deposit_type = st.selectbox("Deposit type", ["No Deposit", "Non Refund", "Refundable"])
        customer_type = st.selectbox("Customer type", ["Transient", "Transient-Party", "Contract", "Group"])

        # Numeric
        lead_time = st.number_input("Lead time (days)", min_value=0, max_value=1000, value=60, step=1)
        arrival_date_week_number = st.number_input("Arrival week number", min_value=1, max_value=53, value=27, step=1)
        arrival_date_day_of_month = st.number_input("Arrival day of month", min_value=1, max_value=31, value=15, step=1)

        stays_in_week_nights = st.number_input("Stays in week nights", min_value=0, max_value=60, value=2, step=1)
        stays_in_weekend_nights = st.number_input("Stays in weekend nights", min_value=0, max_value=60, value=1, step=1)

        adults = st.number_input("Adults", min_value=0, max_value=10, value=2, step=1)
        children = st.number_input("Children", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
        babies = st.number_input("Babies", min_value=0, max_value=10, value=0, step=1)

        is_repeated_guest = st.selectbox("Repeated guest", [0, 1], index=0)
        previous_cancellations = st.number_input("Previous cancellations", min_value=0, max_value=50, value=0, step=1)
        previous_bookings_not_canceled = st.number_input(
            "Previous bookings not canceled", min_value=0, max_value=100, value=0, step=1
        )
        booking_changes = st.number_input("Booking changes", min_value=0, max_value=50, value=0, step=1)

        adr = st.number_input("ADR (average daily rate)", min_value=0.0, max_value=10000.0, value=120.0, step=1.0)
        required_car_parking_spaces = st.number_input(
            "Required car parking spaces", min_value=0, max_value=10, value=0, step=1
        )
        total_of_special_requests = st.number_input(
            "Total special requests", min_value=0, max_value=10, value=0, step=1
        )

        submitted = st.form_submit_button("Predict")

    if submitted:
        raw = {
            "hotel": hotel,
            "lead_time": int(lead_time),
            "arrival_date_month": arrival_date_month,
            "arrival_date_week_number": int(arrival_date_week_number),
            "arrival_date_day_of_month": int(arrival_date_day_of_month),
            "stays_in_week_nights": int(stays_in_week_nights),
            "stays_in_weekend_nights": int(stays_in_weekend_nights),
            "adults": int(adults),
            "children": float(children),
            "babies": int(babies),
            "meal": meal,
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "is_repeated_guest": int(is_repeated_guest),
            "previous_cancellations": int(previous_cancellations),
            "previous_bookings_not_canceled": int(previous_bookings_not_canceled),
            "reserved_room_type": reserved_room_type,
            "booking_changes": int(booking_changes),
            "deposit_type": deposit_type,
            "customer_type": customer_type,
            "adr": float(adr),
            "required_car_parking_spaces": int(required_car_parking_spaces),
            "total_of_special_requests": int(total_of_special_requests),
        }

        pred, proba = predict_one(raw)

        st.subheader("Result")
        st.metric("Cancellation probability", f"{proba:.3f}")
        if pred == 1:
            st.error("Prediction: Canceled (1)")
        else:
            st.success("Prediction: Not canceled (0)")

        st.caption(f"Threshold used: {threshold:.2f}")

else:
    st.write("Upload a CSV containing the booking fields (same names as the manual form fields).")
    st.write("Required columns (minimum):")
    st.code(
        "hotel, lead_time, arrival_date_month, arrival_date_week_number, arrival_date_day_of_month, "
        "stays_in_week_nights, stays_in_weekend_nights, adults, children, babies, meal, market_segment, "
        "distribution_channel, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, "
        "reserved_room_type, booking_changes, deposit_type, customer_type, adr, required_car_parking_spaces, "
        "total_of_special_requests"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df_upload = pd.read_csv(uploaded)

        # Basic check
        required = [
            "hotel", "lead_time", "arrival_date_month", "arrival_date_week_number", "arrival_date_day_of_month",
            "stays_in_week_nights", "stays_in_weekend_nights", "adults", "children", "babies", "meal",
            "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations",
            "previous_bookings_not_canceled", "reserved_room_type", "booking_changes", "deposit_type",
            "customer_type", "adr", "required_car_parking_spaces", "total_of_special_requests"
        ]
        missing = [c for c in required if c not in df_upload.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            # Engineer + prepare
            df_eng = df_upload.copy()
            df_eng["total_nights"] = df_eng["stays_in_week_nights"] + df_eng["stays_in_weekend_nights"]
            df_eng["has_special_request"] = (df_eng["total_of_special_requests"] > 0).astype(int)
            df_eng["lead_time_group"] = pd.cut(
                df_eng["lead_time"],
                bins=[-1, 30, 90, 180, 365, 1000],
                labels=["very_short", "short", "medium", "long", "very_long"],
            )
            df_eng = df_eng.drop(columns=["stays_in_week_nights", "stays_in_weekend_nights"])

            X_new = prepare_for_model(df_eng)
            proba = rf.predict_proba(X_new)[:, 1]
            pred = (proba >= threshold).astype(int)

            out = df_upload.copy()
            out["cancel_probability"] = proba
            out["prediction"] = pred

            st.subheader("Predictions")
            st.dataframe(out, use_container_width=True)

            st.download_button(
                "Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
