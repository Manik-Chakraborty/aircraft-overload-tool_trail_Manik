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
# ---------------------------
# 3) Inputs  (MULTI-AIRCRAFT)
# ---------------------------
st.subheader("Inputs")
row_base = {}   # everything except Aircraft Name

# Numeric fields
if "Gross Wt. (lbs)" in num_cols:
    row_base["Gross Wt. (lbs)"] = st.number_input(
        "Gross Wt. (lbs)", min_value=0.0, value=120000.0, step=1000.0
    )
if "Degree of saturation" in num_cols:
    row_base["Degree of saturation"] = st.number_input(
        "Degree of saturation (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0
    )
if "CBR" in num_cols:
    row_base["CBR"] = st.number_input("CBR", min_value=0.0, value=6.0, step=1.0)

# Dropdown categorical fields (soil + FAA categories)
# (Keep these single-select; they will be applied to all chosen aircraft.)
soil_val = ""
faa_cat_val = ""

if "Subgrade soil type" in cat_cols:
    if "Subgrade soil type" in choices:
        soil_val = st.selectbox("Subgrade soil type", options=[""] + choices["Subgrade soil type"])
    else:
        soil_val = st.text_input("Subgrade soil type", value="")
    row_base["Subgrade soil type"] = soil_val

if "Subgrade Categories (FAA)" in cat_cols:
    if "Subgrade Categories (FAA)" in choices:
        faa_cat_val = st.selectbox("Subgrade Categories (FAA)", options=[""] + choices["Subgrade Categories (FAA)"])
    else:
        faa_cat_val = st.text_input("Subgrade Categories (FAA)", value="")
    row_base["Subgrade Categories (FAA)"] = faa_cat_val

# Aircraft Name: MULTISELECT
aircraft_selected = []
if "Aircraft Name" in cat_cols:
    if "Aircraft Name" in choices and choices["Aircraft Name"]:
        aircraft_selected = st.multiselect(
            "Aircraft Name(s)",
            options=choices["Aircraft Name"],
            default=choices["Aircraft Name"][:1]
        )
    else:
        # Fallback: comma-separated input if no Excel choices
        free = st.text_area("Aircraft Name(s) (comma-separated)", value="")
        aircraft_selected = [a.strip() for a in free.split(",") if a.strip()]

# Any other categorical fields (not aircraft/soil/FAA) → free text (applied to all)
for c in cat_cols:
    if c in ("Aircraft Name", "Subgrade soil type", "Subgrade Categories (FAA)"):
        continue
    if c not in row_base:
        row_base[c] = st.text_input(c, value="")

# ---------------------------
# 4) Predict for ALL selected aircraft -> results table
# ---------------------------
if st.button("Predict"):
    try:
        # build one row per selected aircraft
        rows = []
        if "Aircraft Name" in cat_cols:
            if not aircraft_selected:
                st.warning("Please select at least one aircraft.")
                st.stop()
            for ac in aircraft_selected:
                d = dict(row_base)
                d["Aircraft Name"] = ac
                rows.append(d)
        else:
            # model doesn't use aircraft name; just one row
            rows.append(dict(row_base))

        # Coerce numeric fields as before
        def coerce_numeric(d):
            dd = dict(d)
            if "Degree of saturation" in dd and isinstance(dd["Degree of saturation"], str):
                dd["Degree of saturation"] = dd["Degree of saturation"].replace("%", "").strip()
            for k in ["Gross Wt. (lbs)", "Degree of saturation", "CBR"]:
                if k in dd and dd[k] not in (None, ""):
                    try:
                        dd[k] = float(dd[k])
                    except Exception:
                        pass
            return dd

        rows = [coerce_numeric(r) for r in rows]

        # Reindex to match training-time order
        inf_cols = num_cols + cat_cols
        X = pd.DataFrame(rows).reindex(columns=inf_cols)

        # Predict
        pred = pipeline.predict(X)
        labels = np.where(pred.astype(int) == 1, "Overloaded", "Safe")

        # Probabilities if available
        probs = None
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            probs = pipeline.predict_proba(X)[:, 1]

        # Build results table
        out = pd.DataFrame({"Prediction": labels})
        # include Aircraft Name if present
        if "Aircraft Name" in X.columns:
            out.insert(0, "Aircraft", X["Aircraft Name"].fillna("").astype(str))
        # include other context columns if you want
        if "Gross Wt. (lbs)" in X.columns:
            out.insert(1, "Gross Wt. (lbs)", X["Gross Wt. (lbs)"])
        if "Degree of saturation" in X.columns:
            out.insert(2, "Degree of saturation", X["Degree of saturation"])
        if "Subgrade soil type" in X.columns:
            out.insert(3, "Soil", X["Subgrade soil type"])
        if "Subgrade Categories (FAA)" in X.columns:
            out.insert(4, "FAA Category", X["Subgrade Categories (FAA)"])
        # prob column last
        if probs is not None:
            out["P(Overloaded)"] = (probs).round(3)

        st.subheader("Results")
        st.dataframe(out, use_container_width=True)

        with st.expander("Show input rows (debug)"):
            st.write(X)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

