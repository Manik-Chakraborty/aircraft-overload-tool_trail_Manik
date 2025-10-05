import io
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ACR–PCR Overload Classifier", layout="centered")
st.title("ACR–PCR Overload Classifier")
st.write("Upload your trained **pipeline (.joblib)**. Optionally upload your Excel to enable dropdown menus for categorical fields.")

# ---------------------------
# Helpers
# ---------------------------
def get_expected_input_columns(pipeline):
    """Return numeric and categorical column names expected by the ColumnTransformer."""
    prep = pipeline.named_steps["prep"]
    num_cols, cat_cols = [], []
    for name, trans, cols in prep.transformers_:
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)
    return num_cols, cat_cols

def extract_choices_from_excel(file_bytes_or_path):
    """Read Excel and return dict of choices for key categorical columns."""
    try:
        if isinstance(file_bytes_or_path, (bytes, bytearray)):
            df = pd.read_excel(io.BytesIO(file_bytes_or_path), sheet_name="data")
        else:
            df = pd.read_excel(file_bytes_or_path, sheet_name="data")
    except Exception:
        # fallback: first sheet
        if isinstance(file_bytes_or_path, (bytes, bytearray)):
            df = pd.read_excel(io.BytesIO(file_bytes_or_path))
        else:
            df = pd.read_excel(file_bytes_or_path)
    df.columns = [str(c).strip() for c in df.columns]
    choices = {}
    for col in ["Aircraft Name", "Subgrade soil type", "Subgrade Categories (FAA)"]:
        if col in df.columns:
            vals = pd.Series(df[col]).dropna().astype(str).str.strip()
            vals = sorted([v for v in vals.unique().tolist() if v != ""])
            choices[col] = vals
    return choices

def build_input_df(pipeline, row_dict):
    """Coerce numeric fields and reindex columns to match training-time order."""
    num_cols, cat_cols = get_expected_input_columns(pipeline)
    d = dict(row_dict)
    if "Degree of saturation" in d and isinstance(d["Degree of saturation"], str):
        d["Degree of saturation"] = d["Degree of saturation"].replace("%", "").strip()
    for k in ["Gross Wt. (lbs)", "Degree of saturation", "CBR"]:
        if k in d and d[k] not in (None, ""):
            try:
                d[k] = float(d[k])
            except Exception:
                pass
    X = pd.DataFrame([d])
    X = X.reindex(columns=(num_cols + cat_cols))
    return X

# ---------------------------
# 1) Upload trained pipeline
# ---------------------------
up_model = st.file_uploader("Upload trained pipeline (.joblib)", type=["joblib"])
if up_model is None:
    st.info("⬆️ Upload your .joblib pipeline to begin.")
    st.stop()

try:
    pipeline = joblib.load(io.BytesIO(up_model.read()))
    st.success("Model loaded.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

num_cols, cat_cols = get_expected_input_columns(pipeline)

# ---------------------------
# 2) Optional Excel to populate dropdowns
# ---------------------------
st.subheader("Optional: Upload Excel to enable dropdowns")
choices = {}
up_excel = st.file_uploader("Upload training Excel (e.g., traffic_decision_dataset.xlsx)", type=["xlsx", "xls"])
if up_excel is not None:
    try:
        choices = extract_choices_from_excel(up_excel.read())
        st.success("Dropdown choices loaded from Excel.")
    except Exception as e:
        st.warning(f"Could not read Excel: {e}")

# ---------------------------
# 3) Inputs
# ---------------------------
st.subheader("Inputs")
row = {}

# Numeric fields
if "Gross Wt. (lbs)" in num_cols:
    row["Gross Wt. (lbs)"] = st.number_input("Gross Wt. (lbs)", min_value=0.0, value=120000.0, step=1000.0)
if "Degree of saturation" in num_cols:
    row["Degree of saturation"] = st.number_input("Degree of saturation (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
if "CBR" in num_cols:
    row["CBR"] = st.number_input("CBR", min_value=0.0, value=6.0, step=1.0)

# Dropdown categorical fields (if model expects them)
for col in ["Aircraft Name", "Subgrade soil type", "Subgrade Categories (FAA)"]:
    if col in cat_cols:
        if col in choices:
            row[col] = st.selectbox(col, options=[""] + choices[col], index=0)
        else:
            row[col] = st.text_input(col, value="")

# Any other categorical fields → free text
for c in cat_cols:
    if c not in row:
        row[c] = st.text_input(c, value="")

# ---------------------------
# 4) Predict
# ---------------------------
if st.button("Predict"):
    try:
        X = build_input_df(pipeline, row)
        pred = pipeline.predict(X)[0]
        label = "Overload" if int(pred) == 1 else "Safe"
        st.success(f"Prediction: **{label}**")
        try:
            prob = pipeline.predict_proba(X)[0, 1]
            st.write(f"Probability of Overload: **{prob:.3f}**")
        except Exception:
            pass
        with st.expander("Show input row (debug)"):
            st.write(X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
