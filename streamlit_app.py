# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

st.set_page_config(page_title="Walmart Weekly Sales Forecast", page_icon="🛒", layout="centered")

# ---------- Utilities ----------
def read_file_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_image_bytes(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()

@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

@st.cache_data
def load_merged_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Merged data not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def derive_date_features(date_val: datetime) -> dict:
    year = int(date_val.year)
    month = int(date_val.month)
    weekofyear = int(date_val.isocalendar().week)
    dayofweek = int(date_val.weekday())  # Mon=0
    is_weekend = 1 if dayofweek >= 5 else 0
    return {"Year": year, "Month": month, "WeekOfYear": weekofyear, "DayOfWeek": dayofweek, "IsWeekend": is_weekend}

def guess_schema(df: pd.DataFrame):
    cols = df.columns.tolist()
    out = {
        "store": None, "dept": None, "date": None, "holiday": None, "size": None,
        "temp": None, "fuel": None, "cpi": None, "unemp": None, "type": None, "sales": None
    }
    for c in cols:
        cl = c.lower()
        if cl == "store":
            out["store"] = c
        elif cl in ("dept", "department", "deptnum", "dept_number", "departmentnumber"):
            out["dept"] = c
        elif cl in ("date",):
            out["date"] = c
        elif cl in ("isholiday", "is_holiday", "holiday_flag", "holiday"):
            out["holiday"] = c
        elif cl == "size":
            out["size"] = c
        elif cl in ("temperature", "temp"):
            out["temp"] = c
        elif cl in ("fuel_price", "fuelprice"):
            out["fuel"] = c
        elif cl == "cpi":
            out["cpi"] = c
        elif cl in ("unemployment", "unemployment_rate", "unemp"):
            out["unemp"] = c
        elif cl == "type":
            out["type"] = c
        elif cl in ("weekly_sales", "weeklysales"):
            out["sales"] = c
    return out

def align_features_to_model(input_df: pd.DataFrame, reference_columns: list) -> pd.DataFrame:
    for col in reference_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[reference_columns]

# ---------- CSS & Header Injection ----------
def inject_css(css_text: str):
    # Hide default Streamlit chrome to get closer to a custom page look
    st.markdown("""
        <style>
            /* Optional: compact page */
            .block-container { padding-top: 1rem; padding-bottom: 2rem; }
            header[data-testid="stHeader"] { display: none; }
            footer { visibility: hidden; }
        </style>
    """, unsafe_allow_html=True)

    if css_text:
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)

def render_header(logo_bytes: bytes | None):
    # Mimic your templates/index.html structure
    st.markdown("""
        <div class="header">
            <div class="branding">
                <img src="data:image/png;base64,IMG_B64" class="logo" alt="Walmart Logo"/>
                <div class="title-block">
                    <h1 class="title">Walmart Weekly Sales Forecast</h1>
                    <p class="subtitle">Predict weekly sales using Random Forest</p>
                </div>
            </div>
        </div>
    """.replace("IMG_B64", ("" if not logo_bytes else __import__("base64").b64encode(logo_bytes).decode("utf-8"))),
        unsafe_allow_html=True
    )

def render_form_container_open():
    st.markdown("""
        <div class="form-container">
            <div class="card">
                <h2 class="section-title">Enter Inputs</h2>
                <div class="form-grid">
    """, unsafe_allow_html=True)

def render_form_container_close():
    st.markdown("""
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_result_container(pred_value: float | None, error_msg: str | None = None):
    if error_msg:
        st.markdown(f"""
            <div class="result-container error">
                <p class="result-text">Prediction failed: {error_msg}</p>
            </div>
        """, unsafe_allow_html=True)
    elif pred_value is not None:
        st.markdown(f"""
            <div class="result-container success">
                <p class="result-label">Predicted Weekly Sales</p>
                <p class="result-value">${pred_value:,.2f}</p>
            </div>
        """, unsafe_allow_html=True)

# ---------- Main ----------
def main():
    # Load CSS and assets
    css_text = read_file_text("static/css/styles.css")
    logo_bytes = load_image_bytes("static/images/walmarticon.png")
    inject_css(css_text)
    render_header(logo_bytes)

    # Paths
    model_path = "rf_model.pkl"
    merged_data_path = "merged_data.csv"

    # Load model & data
    try:
        model = load_model(model_path)
    except Exception as e:
        render_result_container(None, f"{e}")
        st.stop()

    try:
        merged_df = load_merged_data(merged_data_path)
    except Exception as e:
        render_result_container(None, f"{e}")
        st.stop()

    schema = guess_schema(merged_df)
    if schema["date"] and merged_df[schema["date"]].isnull().all():
        st.warning("Warning: Date parsing failed for merged_data.csv. Date-derived features may be off.")

    # Options
    store_values = sorted(merged_df[schema["store"]].dropna().unique().tolist()) if schema["store"] else []
    dept_values = sorted(merged_df[schema["dept"]].dropna().unique().tolist()) if schema["dept"] else []
    type_values = sorted(merged_df[schema["type"]].dropna().unique().astype(str).tolist()) if schema["type"] else []

    # Render form like your HTML
    render_form_container_open()

    # Use columns but wrap each control with a div to apply your CSS classes
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="form-field"><label>Store</label>', unsafe_allow_html=True)
        store_id = st.selectbox("", store_values, label_visibility="collapsed") if store_values else st.number_input("", min_value=1, step=1, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-field"><label>Department</label>', unsafe_allow_html=True)
        dept_id = st.selectbox("", dept_values, label_visibility="collapsed") if dept_values else st.number_input("", min_value=1, step=1, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-field"><label>Week Date</label>', unsafe_allow_html=True)
        date_input = st.date_input("", value=pd.to_datetime("2012-11-02").date(), label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="form-field"><label>Store Size</label>', unsafe_allow_html=True)
        size_val = st.number_input("", min_value=0, value=int(merged_df[schema["size"]].median()) if schema["size"] else 150000, step=1000, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-field"><label>Temperature (°F)</label>', unsafe_allow_html=True)
        temp_val = st.number_input("", value=float(merged_df[schema["temp"]].median()) if schema["temp"] else 60.0, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-field"><label>Is Holiday?</label>', unsafe_allow_html=True)
        holiday_flag = st.selectbox("", options=[0, 1], index=0, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="section-subtitle">Optional: Economic/External Features</div>
    """, unsafe_allow_html=True)

    c3, c4, c5, c6 = st.columns(4)
    with c3:
        st.markdown('<div class="form-field small"><label>Fuel Price</label>', unsafe_allow_html=True)
        fuel_price = st.number_input("", value=float(merged_df[schema["fuel"]].median()) if schema["fuel"] else 3.0, step=0.01, format="%.2f", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="form-field small"><label>CPI</label>', unsafe_allow_html=True)
        cpi_val = st.number_input("", value=float(merged_df[schema["cpi"]].median()) if schema["cpi"] else 200.0, step=0.1, format="%.1f", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="form-field small"><label>Unemployment Rate</label>', unsafe_allow_html=True)
        unemp_val = st.number_input("", value=float(merged_df[schema["unemp"]].median()) if schema["unemp"] else 7.0, step=0.1, format="%.1f", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="form-field small"><label>Store Type</label>', unsafe_allow_html=True)
        store_type = st.selectbox("", options=type_values if type_values else ["A","B","C"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    # Primary CTA button styled by your CSS (wrap the Streamlit button)
    st.markdown('<div class="actions">', unsafe_allow_html=True)
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    render_form_container_close()

    # Prepare input, predict, and render result
    pred_value, err = None, None
    if predict_clicked:
        try:
            date_dt = pd.to_datetime(date_input)
            derived = derive_date_features(date_dt)

            # Build input row
            row = {}
            row[schema["store"] or "Store"] = store_id
            row[schema["dept"] or "Dept"] = dept_id
            row["Year"] = derived["Year"]
            row["Month"] = derived["Month"]
            row["WeekOfYear"] = derived["WeekOfYear"]
            row["DayOfWeek"] = derived["DayOfWeek"]
            row["IsWeekend"] = derived["IsWeekend"]
            row[schema["size"] or "Size"] = size_val
            row[schema["temp"] or "Temperature"] = temp_val
            row[schema["holiday"] or "IsHoliday"] = holiday_flag
            row[schema["fuel"] or "Fuel_Price"] = fuel_price
            row[schema["cpi"] or "CPI"] = cpi_val
            row[schema["unemp"] or "Unemployment"] = unemp_val

            input_df = pd.DataFrame([row])

            # One-hot type alignment
            if schema["type"]:
                dummy_prefix = schema["type"]
                input_df[dummy_prefix] = str(store_type)
                merged_types = merged_df[schema["type"]].astype(str)
                known_type_cols = pd.get_dummies(merged_types, prefix=dummy_prefix).columns.tolist()
                d_in = pd.get_dummies(input_df[dummy_prefix], prefix=dummy_prefix)
                for col in known_type_cols:
                    if col not in d_in.columns:
                        d_in[col] = 0
                input_df = pd.concat([input_df.drop(columns=[dummy_prefix]), d_in[known_type_cols]], axis=1)

            # Determine model columns
            candidate = merged_df.drop(columns=[c for c in [schema["sales"]] if c in merged_df.columns], errors="ignore").copy()
            if schema["type"]:
                dummy_prefix = schema["type"]
                type_dummies = pd.get_dummies(candidate[schema["type"]].astype(str), prefix=dummy_prefix)
                candidate = pd.concat([candidate.drop(columns=[schema["type"]]), type_dummies], axis=1)
            if hasattr(model, "feature_names_in_"):
                model_cols = list(model.feature_names_in_)
            else:
                model_cols = [c for c in candidate.columns if c != schema["sales"]]

            # Drop any datetime from input_df (model uses derived features)
            for c in input_df.columns:
                if np.issubdtype(input_df[c].dtype, np.datetime64):
                    input_df = input_df.drop(columns=[c])

            # Ensure engineered defaults
            for c in ["Year","Month","WeekOfYear","DayOfWeek","IsWeekend"]:
                if c not in input_df.columns:
                    input_df[c] = derived.get(c, 0)

            X = align_features_to_model(input_df, model_cols)
            pred_value = float(model.predict(X)[0])
        except Exception as e:
            err = str(e)

    render_result_container(pred_value, err)

    # Optional: debug panel styled to match
    with st.expander("Debug: Feature Vector"):
        if "model_cols" in locals():
            st.text("Model expects columns:")
            st.code(str(model_cols))
        if "X" in locals():
            st.dataframe(X)

if __name__ == "__main__":
    main()
