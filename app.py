from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, mean_squared_error,
)

BASE = Path(__file__).resolve().parent
REG_PATH = BASE / "FORWARD_SCORED.csv"
FOREST_PATH = BASE / "FOREST_SCORED.csv"

st.set_page_config(
    page_title="Bike Rental Demand | SAS Best Models",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Exact SAS Predictive Regression Forward model from PROJ17(1).pdf, pp. 2-3.
# The reconstructed score is checked against SAS P_cnt in FORWARD_SCORED.csv.
# This is the predictive-regression model that generated the supplied scored
# output. The separately supplied Forward_score.sas has a different coefficient
# set and does not reproduce this file's P_cnt values.
# -----------------------------------------------------------------------------
REG_INTERCEPT = 76.324240
REG_COEFFICIENTS = {
    "season": {1: -43.259833, 2: -18.703738, 3: -8.925772},
    "yr": {0: -82.712110},
    "mnth": {
        1: -24.344177, 2: -20.806528, 3: -3.483231, 4: -6.109367,
        5: 5.695896, 6: -9.348626, 7: -45.950090, 8: -36.542966,
        9: -7.754670, 10: 2.708040, 11: -3.848921,
    },
    "hr": {
        0: -26.849862, 1: -38.852180, 2: -49.815331, 3: -57.150629,
        4: -60.005337, 5: -44.867410, 6: 1.963969, 7: 111.885184,
        8: 233.417746, 9: 115.830201, 10: 70.306869, 11: 86.187449,
        12: 121.108895, 13: 116.396684, 14: 104.474149, 15: 112.378871,
        16: 163.757322, 17: 286.009931, 18: 274.851047, 19: 174.147939,
        20: 104.207803, 21: 65.268533, 22: 32.989934,
    },
    "holiday": {0: 27.759539},
    "weathersit": {1: 17.081160, 2: 2.739506, 3: -46.213268},
}
REG_CONTINUOUS = {"temp": 239.244270, "hum": -64.622250, "windspeed": -50.931623}

PREDICTOR_COLUMNS = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed",
]
CATEGORICAL_COLUMNS = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"
]
CONTINUOUS_COLUMNS = ["temp", "atemp", "hum", "windspeed"]


def read_csv_source(uploaded, default_path, label):
    if uploaded is not None:
        return pd.read_csv(io.BytesIO(uploaded.getvalue()))
    if default_path.exists():
        return pd.read_csv(default_path)
    st.error(f"{label} is missing. Upload the CSV in the sidebar.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_csv_bytes(data: bytes):
    return pd.read_csv(io.BytesIO(data))


def coerce_numeric(df, columns):
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def score_forward_regression(row: dict) -> float:
    """Reproduce the supplied SAS Forward predictive-regression equation."""
    score = REG_INTERCEPT
    for col, mapping in REG_COEFFICIENTS.items():
        value = int(row[col])
        score += mapping.get(value, 0.0)
    for col, coef in REG_CONTINUOUS.items():
        score += coef * float(row[col])
    return float(score)


def validate_regression_input(row: dict):
    categories = {
        "season": set(range(1, 5)), "yr": {0, 1}, "mnth": set(range(1, 13)),
        "hr": set(range(0, 24)), "holiday": {0, 1},
        "weathersit": {1, 2, 3, 4},
    }
    for col, allowed in categories.items():
        if int(row[col]) not in allowed:
            raise ValueError(f"{col} has an invalid category value.")
    for col in ["temp", "hum", "windspeed"]:
        if not 0 <= float(row[col]) <= 1:
            raise ValueError(f"{col} must be between 0 and 1 (dataset-normalised scale).")


def prepare_forest(raw):
    required = {
        "instant", "Demand", "I_Demand", "P_Demand1", "P_Demand0",
        "_PartInd_", "_SampleFreq_1", *PREDICTOR_COLUMNS,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("Forest CSV is missing required columns: " + ", ".join(missing))

    df = coerce_numeric(raw, [
        "instant", "Demand", "I_Demand", "P_Demand1", "P_Demand0",
        "_PartInd_", "_SampleFreq_1", *PREDICTOR_COLUMNS,
    ])
    df = df.dropna(subset=[
        "instant", "Demand", "I_Demand", "P_Demand1", "P_Demand0", "_PartInd_", *PREDICTOR_COLUMNS
    ]).copy()
    df["instant"] = df["instant"].astype(int)
    df["Demand"] = df["Demand"].astype(int)
    df["I_Demand"] = df["I_Demand"].astype(int)
    df["_PartInd_"] = df["_PartInd_"].astype(int)
    df["_SampleFreq_1"] = df["_SampleFreq_1"].fillna(1).clip(lower=0)

    # SAS Model Studio emits a row for each observation in its partition output.
    # Prefer validation row for any ID appearing in both partitions; otherwise
    # use the available training row only for historical-reference purposes.
    df = df.sort_values(["instant", "_PartInd_"], ascending=[True, True])
    unique = df.drop_duplicates("instant", keep="first").copy()
    unique["Prediction partition"] = np.where(unique["_PartInd_"] == 0, "Validation", "Training fallback")
    return df, unique


def circular_difference(a, b, period):
    raw = np.abs(np.asarray(a, dtype=float) - float(b))
    return np.minimum(raw, period - raw) / (period / 2.0)


def nearest_forest_reference(forest_unique: pd.DataFrame, request: dict):
    """Find closest SAS-scored reference row on the 12 future-available features.

    This is explicitly a nearest-reference lookup, NOT execution of the SAS
    Forest model on a new record. The native ASTORE was not included in the
    uploaded materials, and the CSV/TSV ASTORE export is not a valid model file.
    """
    d = pd.DataFrame(index=forest_unique.index)
    # Categorical features: scaled mismatch/distance; hour/month/weekday are cyclic.
    d["season"] = np.abs(forest_unique["season"].to_numpy(float) - request["season"]) / 3.0
    d["yr"] = np.abs(forest_unique["yr"].to_numpy(float) - request["yr"])
    d["mnth"] = circular_difference(forest_unique["mnth"].to_numpy(), request["mnth"], 12)
    d["hr"] = circular_difference(forest_unique["hr"].to_numpy(), request["hr"], 24)
    d["holiday"] = np.abs(forest_unique["holiday"].to_numpy(float) - request["holiday"])
    d["weekday"] = circular_difference(forest_unique["weekday"].to_numpy(), request["weekday"], 7)
    d["workingday"] = np.abs(forest_unique["workingday"].to_numpy(float) - request["workingday"])
    d["weathersit"] = np.abs(forest_unique["weathersit"].to_numpy(float) - request["weathersit"]) / 3.0

    # Continuous features are standardised by the observed reference-data spread.
    for col in CONTINUOUS_COLUMNS:
        scale = float(forest_unique[col].std(ddof=0))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        d[col] = (forest_unique[col].to_numpy(float) - float(request[col])) / scale
    distances = np.sqrt(np.mean(np.square(d.to_numpy(dtype=float)), axis=1))
    pos = int(np.argmin(distances))
    idx = forest_unique.index[pos]
    result = forest_unique.loc[idx].copy()
    # Exact match means all displayed input predictors match one row.
    exact = True
    for col in PREDICTOR_COLUMNS:
        if col in CONTINUOUS_COLUMNS:
            exact &= bool(np.isclose(float(result[col]), float(request[col]), atol=1e-10))
        else:
            exact &= int(result[col]) == int(request[col])
    return result, float(distances[pos]), bool(exact)


def compute_validation_metrics(forest_rows):
    validation = forest_rows[forest_rows["_PartInd_"] == 0].copy()
    if validation.empty:
        return validation, None
    y = validation["Demand"].astype(int)
    pred = validation["I_Demand"].astype(int)
    prob = validation["P_Demand1"].astype(float).clip(0, 1)
    weights = validation["_SampleFreq_1"].astype(float).clip(lower=0)
    if weights.sum() <= 0:
        weights = None
    metrics = {
        "accuracy": accuracy_score(y, pred, sample_weight=weights),
        "f1": f1_score(y, pred, sample_weight=weights, zero_division=0),
        "precision": precision_score(y, pred, sample_weight=weights, zero_division=0),
        "recall": recall_score(y, pred, sample_weight=weights, zero_division=0),
        "auc": roc_auc_score(y, prob, sample_weight=weights) if y.nunique() == 2 else np.nan,
        "ase": mean_squared_error(y, prob, sample_weight=weights),
        "confusion": confusion_matrix(y, pred, labels=[0, 1], sample_weight=weights),
        "weight_sum": float(weights.sum()) if weights is not None else float(len(validation)),
    }
    return validation, metrics


with st.sidebar:
    st.header("SAS model data")
    st.success("Regression and Forest scored datasets load automatically from the deployed app package.")
    st.caption("Visitors do not need to upload any files.")
    st.divider()

try:
    if not REG_PATH.exists() or not FOREST_PATH.exists():
        raise FileNotFoundError("The bundled SAS scored CSV files are missing from the app folder.")
    reg_raw = pd.read_csv(REG_PATH)
    forest_raw = pd.read_csv(FOREST_PATH)
except Exception as exc:
    st.error(f"Unable to load the bundled SAS output files: {exc}")
    st.stop()

reg_required = {"instant", "cnt", "p_cnt", "r_cnt", *PREDICTOR_COLUMNS}
reg_missing = sorted(reg_required - set(reg_raw.columns))
if reg_missing:
    st.error("Regression scored CSV is missing: " + ", ".join(reg_missing))
    st.stop()
reg = coerce_numeric(reg_raw, ["instant", "cnt", "p_cnt", "r_cnt", *PREDICTOR_COLUMNS])
reg = reg.dropna(subset=["instant", "cnt", "p_cnt", "r_cnt", *PREDICTOR_COLUMNS]).copy()
reg["instant"] = reg["instant"].astype(int)
reg = reg.drop_duplicates("instant")

try:
    forest_rows, forest_unique = prepare_forest(forest_raw)
except Exception as exc:
    st.error(f"Unable to prepare Forest scored output: {exc}")
    st.stop()

validation, forest_metrics = compute_validation_metrics(forest_rows)
combined = reg.merge(
    forest_unique[["instant", "Demand", "I_Demand", "P_Demand1", "P_Demand0", "Prediction partition"]],
    on="instant", how="inner", validate="one_to_one"
)

# Verify the reconstructed regression formula against the supplied SAS output.
reconstructed = np.full(len(reg), REG_INTERCEPT, dtype=float)
for _col, _mapping in REG_COEFFICIENTS.items():
    reconstructed += reg[_col].map(_mapping).fillna(0.0).to_numpy(dtype=float)
for _col, _coef in REG_CONTINUOUS.items():
    reconstructed += _coef * reg[_col].to_numpy(dtype=float)
max_score_diff = float(np.max(np.abs(reconstructed - reg["p_cnt"].to_numpy(dtype=float))))

# Main header
st.title("🚲 Bike Rental Demand Intelligence")
st.caption("SAS Forward Selection regression + SAS Viya Forest classification | Original SAS scored outputs")

st.info(
    "This application automatically loads the SAS observation-level outputs bundled with the deployment. Website visitors do not upload files. The regression model is reconstructed "
    "from the Forward Selection coefficient table in PROJ17 and verified against all rows in FORWARD_SCORED.csv. "
    "For classification, the supplied Forest CSV provides SAS predictions for previously scored observations; "
    "without the native ASTORE, a new input is matched to its nearest SAS-scored reference record rather than "
    "being scored directly by the Forest model."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Regression scored rows", f"{len(reg):,}")
m2.metric("Forest output rows", f"{len(forest_rows):,}")
m3.metric("Unique Forest observations", f"{forest_unique['instant'].nunique():,}")
m4.metric("Matched observations", f"{len(combined):,}")

# Model input form shared by regression scorer and classifier reference lookup.
st.header("1. Enter independent variables")
st.write("Enter the hour's calendar and weather conditions. The regression result is calculated using the SAS Forward Selection equation. A new Forest prediction is shown only when the inputs exactly match a record already scored by SAS; otherwise the app will not substitute a nearest-neighbour result.")
with st.form("scenario_form"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        season_label = st.selectbox("Season", ["Spring (1)", "Summer (2)", "Fall (3)", "Winter (4)"], index=1)
        year_label = st.selectbox("Year", ["2011 (0)", "2012 (1)"], index=1)
        month = st.selectbox("Month", list(range(1, 13)), index=5)
    with c2:
        hour = st.selectbox("Hour of day", list(range(24)), index=17)
        holiday_label = st.selectbox("Holiday", ["No (0)", "Yes (1)"], index=0)
        weekday = st.selectbox("Weekday (0=Sunday … 6=Saturday)", list(range(7)), index=5)
    with c3:
        workingday_label = st.selectbox("Working day", ["No (0)", "Yes (1)"], index=1)
        weather_label = st.selectbox(
            "Weather situation",
            ["Clear / partly cloudy (1)", "Mist / cloudy (2)", "Light rain or snow (3)", "Heavy rain or snow (4)"],
            index=0,
        )
        temp = st.number_input("Temperature (normalised 0–1)", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
    with c4:
        atemp = st.number_input("Feels-like temperature (0–1)", min_value=0.0, max_value=1.0, value=0.62, step=0.01)
        hum = st.number_input("Humidity (0–1)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
        windspeed = st.number_input("Windspeed (0–1)", min_value=0.0, max_value=1.0, value=0.18, step=0.01)
    submitted = st.form_submit_button("Calculate demand estimates", type="primary", use_container_width=True)

if submitted:
    request = {
        "season": int(season_label[-2]),
        "yr": int(year_label[-2]),
        "mnth": int(month),
        "hr": int(hour),
        "holiday": int(holiday_label[-2]),
        "weekday": int(weekday),
        "workingday": int(workingday_label[-2]),
        "weathersit": int(weather_label.split("(")[-1].replace(")", "")),
        "temp": float(temp),
        "atemp": float(atemp),
        "hum": float(hum),
        "windspeed": float(windspeed),
    }
    try:
        validate_regression_input(request)
        regression_prediction = score_forward_regression(request)
        exact_mask = np.ones(len(forest_unique), dtype=bool)
        for col in PREDICTOR_COLUMNS:
            if col in CONTINUOUS_COLUMNS:
                exact_mask &= np.isclose(forest_unique[col].to_numpy(float), float(request[col]), atol=1e-10)
            else:
                exact_mask &= forest_unique[col].to_numpy(int) == int(request[col])
        exact_rows = forest_unique.loc[exact_mask]
        st.session_state["last_request"] = request
        st.session_state["last_regression_prediction"] = regression_prediction
        st.session_state["last_forest_exact_row"] = exact_rows.iloc[0].to_dict() if not exact_rows.empty else None
    except Exception as exc:
        st.error(f"Could not score this input: {exc}")

if "last_request" in st.session_state:
    request = st.session_state["last_request"]
    regression_prediction = st.session_state["last_regression_prediction"]
    forest_reference_data = st.session_state["last_forest_exact_row"]
    forest_reference = pd.Series(forest_reference_data) if forest_reference_data is not None else None

    st.header("2. Demand prediction results")
    c_reg, c_clf = st.columns(2)
    with c_reg:
        st.subheader("Regression — SAS Forward Selection")
        st.metric("Predicted hourly rentals (raw SAS score)", f"{regression_prediction:,.1f}")
        st.metric("Non-negative operational estimate", f"{max(0.0, regression_prediction):,.1f}")
        if regression_prediction < 0:
            st.warning("The linear model produced a negative raw estimate. The raw SAS model score is retained above; bike rentals cannot be negative, so the operational estimate is floored at zero.")
        st.caption("Uses the Forward Selection model from PROJ17: R² 0.6845, adjusted R² 0.6829, Test ASE 7,272.4784, Test RMSE ≈ 85.28 (sqrt of Test ASE).")
    with c_clf:
        st.subheader("Classification — SAS Viya Forest")
        if forest_reference is None:
            st.metric("New-input Forest prediction", "Not available")
            st.warning("These inputs do not exactly match a record in the supplied SAS Forest scored output. The native Forest ASTORE is not available in a usable format, so the original Forest model cannot currently score this new combination. No approximate class or probability is displayed.")
        else:
            predicted_class = int(forest_reference["I_Demand"])
            st.metric("Saved SAS Forest predicted class", f"Class {predicted_class}")
            st.metric("Probability of Class 1", f"{float(forest_reference['P_Demand1']):.2%}")
            st.metric("Probability of Class 0", f"{float(forest_reference['P_Demand0']):.2%}")
            st.success("Exact predictor match found. These are the saved SAS Forest outputs for this existing observation, not a newly executed model score.")
            st.caption(f"Observation ID: {int(forest_reference['instant'])} | Partition: {forest_reference['Prediction partition']}")

    if forest_reference is not None:
        st.subheader("Existing SAS-scored record matched exactly")
        compare = [{"Input variable": col, "Your input": request[col], "SAS record": forest_reference[col]} for col in PREDICTOR_COLUMNS]
        st.dataframe(pd.DataFrame(compare), use_container_width=True, hide_index=True)

    st.warning(
        "Model limitation: the supplied Forest score code declares `casual` and `registered` as model inputs. "
        "These are components of total rentals and are not known before an hour's demand occurs. The native Forest "
        "ASTORE was not provided in usable binary form, so this app does not claim to execute the Forest model on "
        "new inputs. For a valid forward-looking classifier, the Forest should be retrained/scored without "
        "casual` and `registered`, or its valid native ASTORE must be made available."
    )
else:
    st.caption("Enter the calendar and weather variables, then select Calculate demand estimates. Regression scores new inputs; Forest outputs are shown only for exact existing SAS-scored matches until the native ASTORE is available.")

# Historical record look-up based on the actual SAS output tables.
st.divider()
st.header("3. Actual SAS-scored observation lookup")
st.caption("Choose an existing observation ID to see the exact regression and Forest output values already generated by SAS.")
common_ids = np.intersect1d(reg["instant"].unique(), forest_unique["instant"].unique())
if len(common_ids):
    default_idx = min(len(common_ids) - 1, len(common_ids) // 2)
    selected_id = st.selectbox("Observation ID (instant)", common_ids.tolist(), index=default_idx)
    rr = reg.loc[reg["instant"] == int(selected_id)].iloc[0]
    ff = forest_unique.loc[forest_unique["instant"] == int(selected_id)].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Actual cnt", f"{rr['cnt']:,.0f}")
    b.metric("SAS predicted cnt", f"{rr['p_cnt']:,.2f}")
    c.metric("SAS Forest class", f"{int(ff['I_Demand'])}")
    d.metric("Probability Demand=1", f"{float(ff['P_Demand1']):.2%}")
    st.write({
        "Regression residual (cnt - p_cnt)": round(float(rr["r_cnt"]), 4),
        "Actual Demand class": int(ff["Demand"]),
        "Forest predicted class": int(ff["I_Demand"]),
        "Forest partition": ff["Prediction partition"],
        "Probability Demand=0": float(ff["P_Demand0"]),
    })

# Dashboards and validation evidence.
st.divider()
st.header("4. Model performance and visual analytics")
t_overview, t_reg, t_clf, t_data = st.tabs(["Overview", "Regression", "Classification", "Scored data"])
with t_overview:
    l, r = st.columns(2)
    with l:
        st.markdown("**Forward regression — SAS benchmark**")
        st.table(pd.DataFrame([
            ["Selected method", "Forward Selection"],
            ["Selection criterion", "SBC"],
            ["Training observations", "8,680"],
            ["Testing observations", "3,485"],
            ["R²", "0.6845"],
            ["Adjusted R²", "0.6829"],
            ["Root MSE", "86.1109"],
            ["Test ASE", "7,272.4784"],
        ], columns=["Metric", "SAS result"]))
    with r:
        st.markdown("**Forest classification — SAS validation benchmark**")
        if forest_metrics:
            st.metric("Weighted validation accuracy", f"{forest_metrics['accuracy']:.2%}")
            st.metric("Weighted validation ROC AUC", f"{forest_metrics['auc']:.4f}")
            st.metric("Weighted validation F1", f"{forest_metrics['f1']:.4f}")
            st.metric("Weighted validation ASE", f"{forest_metrics['ase']:.4f}")
            st.caption(f"Validation observations: {len(validation):,}; weighted frequency total: {forest_metrics['weight_sum']:,.0f}.")
        else:
            st.warning("No validation rows were found in the Forest output.")
    plot = reg.sample(min(2500, len(reg)), random_state=42)
    fig = px.scatter(plot, x="cnt", y="p_cnt", opacity=0.45,
                     labels={"cnt": "Actual hourly rentals", "p_cnt": "SAS predicted hourly rentals"},
                     title="SAS Forward regression: actual vs predicted (scored observations)")
    lo = float(min(plot["cnt"].min(), plot["p_cnt"].min()))
    hi = float(max(plot["cnt"].max(), plot["p_cnt"].max()))
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
with t_reg:
    reg_rmse = float(np.sqrt(mean_squared_error(reg["cnt"], reg["p_cnt"])))
    reg_mae = float(np.mean(np.abs(reg["cnt"] - reg["p_cnt"])))
    reg_r2 = float(1 - np.sum((reg["cnt"] - reg["p_cnt"])**2) / np.sum((reg["cnt"] - reg["cnt"].mean())**2))
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Scored-row R²", f"{reg_r2:.4f}")
    x2.metric("Scored-row RMSE", f"{reg_rmse:.2f}")
    x3.metric("Scored-row MAE", f"{reg_mae:.2f}")
    x4.metric("Max formula check error", f"{max_score_diff:.2e}")
    st.caption("The reconstructed formula matches the supplied SAS P_cnt column across the full scored dataset (rounding-level difference). The scored-row metrics are descriptive; use the SAS test metrics above for the model-selection benchmark.")
    fig = px.histogram(reg, x="r_cnt", nbins=55, title="SAS regression residual distribution")
    st.plotly_chart(fig, use_container_width=True)
    series = reg.sort_values("instant").head(1200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["instant"], y=series["cnt"], mode="lines", name="Actual cnt"))
    fig.add_trace(go.Scatter(x=series["instant"], y=series["p_cnt"], mode="lines", name="SAS p_cnt"))
    fig.update_layout(title="Actual and SAS-predicted rental counts", xaxis_title="Observation ID", yaxis_title="Bike rentals", height=420)
    st.plotly_chart(fig, use_container_width=True)
with t_clf:
    if forest_metrics:
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Weighted accuracy", f"{forest_metrics['accuracy']:.2%}")
        x2.metric("Weighted AUC", f"{forest_metrics['auc']:.4f}")
        x3.metric("Weighted F1", f"{forest_metrics['f1']:.4f}")
        x4.metric("Weighted ASE", f"{forest_metrics['ase']:.4f}")
        cm = pd.DataFrame(forest_metrics["confusion"], index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        st.markdown("**Weighted confusion matrix**")
        st.dataframe(cm, use_container_width=True)
        fig = px.histogram(validation, x="P_Demand1", color="Demand", nbins=40,
                           barmode="overlay", opacity=0.65,
                           title="SAS Forest validation probability distribution by actual class")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Forest output fields**")
    st.write({
        "Scored output rows": int(len(forest_rows)),
        "Validation rows": int((forest_rows["_PartInd_"] == 0).sum()),
        "Training rows": int((forest_rows["_PartInd_"] == 1).sum()),
        "Unique observation IDs": int(forest_unique["instant"].nunique()),
        "Actual target": "Demand",
        "Predicted class": "I_Demand",
        "Class 1 probability": "P_Demand1",
        "Class 0 probability": "P_Demand0",
    })
with t_data:
    st.write("The table below contains observations matched across both SAS scored outputs.")
    merged = combined.copy()
    st.dataframe(merged.head(200), use_container_width=True)
    st.download_button(
        "Download matched SAS predictions (CSV)",
        merged.to_csv(index=False).encode("utf-8"),
        file_name="BIKE_COMBINED_SAS_PREDICTIONS.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download Forest validation rows (CSV)",
        validation.to_csv(index=False).encode("utf-8"),
        file_name="FOREST_VALIDATION_ONLY.csv",
        mime="text/csv",
    )

st.divider()
st.subheader("Important model-use notes")
st.markdown(
    "- **Regression:** the predictive Forward Selection equation is reconstructed from the coefficient table in `PROJ17(1).pdf` and checked against the SAS `p_cnt` output. The separate `Forward_score.sas` uploaded earlier uses a different coefficient set and does not reproduce this scored CSV.\n"
    "- **Classification:** the Forest CSV contains actual SAS predictions for scored records. The supplied Forest DS2 score code references the ASTORE table `_34WGSCQO4DIP2SECYA3SPICGP_AST`; the CSV/TSV export of that ASTORE is not a usable native model artifact. Therefore this app cannot execute the original Forest model on a wholly new row.\n"
    "- **Leakage review:** the Forest DS2 score code lists `casual` and `registered` as scoring inputs. These are components of `cnt`; confirm predictor roles in SAS before using the classifier for future-demand forecasting.\n"
    "- **Class interpretation:** the available files identify target classes as `Demand=0` and `Demand=1`. Their HIGH/LOW meaning should be confirmed in the SAS project."
)
