import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

# ------------------------------------------------------------
# Bike Rental Demand Prediction | Streamlit application
# ------------------------------------------------------------
st.set_page_config(
    page_title="Bike Rental Demand Intelligence",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed"
]
REQUIRED = FEATURES + ["cnt"]
SEASON_NAMES = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
WEATHER_NAMES = {
    1: "Clear / partly cloudy",
    2: "Mist / cloudy",
    3: "Light snow or rain",
    4: "Severe weather"
}

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(90, 130, 180, 0.08); padding: 14px; border-radius: 12px;}
div[data-testid="stTabs"] button {font-weight: 600;}
.small-note {font-size: 0.88rem; color: #666;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    if file_bytes is not None:
        from io import BytesIO
        df = pd.read_csv(BytesIO(file_bytes))
    elif Path("hour.csv").exists():
        df = pd.read_csv("hour.csv")
    else:
        return None

    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


@st.cache_resource(show_spinner="Training the Random Forest model...")
def train_model(dataframe):
    missing = [c for c in REQUIRED if c not in dataframe.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing)
        )

    data = dataframe[REQUIRED].copy()
    for c in REQUIRED:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 100:
        raise ValueError("At least 100 complete rows are recommended for this prototype.")

    # Target label is derived from cnt. cnt is never included in X.
    threshold = float(data["cnt"].mean())
    X = data[FEATURES].copy()
    y = (data["cnt"] >= threshold).astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "rows": len(data),
        "threshold": threshold,
        "accuracy": accuracy_score(y_valid, predictions),
        "auc": roc_auc_score(y_valid, probabilities),
        "f1": f1_score(y_valid, predictions, zero_division=0),
        "precision": precision_score(y_valid, predictions, zero_division=0),
        "recall": recall_score(y_valid, predictions, zero_division=0),
        "confusion": confusion_matrix(y_valid, predictions),
        "feature_importance": pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False),
        "validation": pd.DataFrame({
            "Actual": y_valid.to_numpy(),
            "Predicted": predictions,
            "Probability_HIGH": probabilities
        }),
        "clean_data": data,
    }
    return model, metrics


def fmt_class(value):
    return "HIGH" if int(value) == 1 else "LOW"


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 🚲 Demand Intelligence")
    st.caption("Bike rental demand classification")
    uploaded = st.file_uploader(
        "Upload hourly bike-sharing CSV",
        type=["csv"],
        help="Use the hourly dataset containing cnt, season, yr, mnth, hr and weather variables."
    )
    st.markdown("---")
    st.markdown("**Expected data**")
    st.code("hour.csv", language="text")
    st.caption("You can also place hour.csv in the same folder as app.py.")
    st.markdown("---")
    st.caption("Prototype model: Random Forest")

uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
uploaded_name = uploaded.name if uploaded is not None else None
df = load_data(uploaded_bytes, uploaded_name)

st.title("Bike Rental Demand Intelligence")
st.write(
    "Explore historical rental patterns and estimate whether demand will be **HIGH** "
    "or **LOW** for a selected hour and set of conditions."
)

if df is None:
    st.info("To start, upload your hourly bike-sharing CSV using the left sidebar.")
    st.markdown("""
    ### What this app includes
    - **Overview:** dataset profile and demand patterns
    - **Predict demand:** an interactive prediction form
    - **Model performance:** validation metrics and feature importance
    """)
    st.stop()

try:
    model, metrics = train_model(df)
except Exception as exc:
    st.error(f"Unable to prepare the dataset: {exc}")
    st.markdown("Check that your CSV contains all required columns and that the column names match.")
    st.stop()

data = metrics["clean_data"]
top_a, top_b, top_c, top_d = st.columns(4)
top_a.metric("Rows analysed", f"{metrics['rows']:,}")
top_b.metric("Validation accuracy", f"{metrics['accuracy']:.1%}")
top_c.metric("Validation ROC AUC", f"{metrics['auc']:.3f}")
top_d.metric("Validation F1 score", f"{metrics['f1']:.3f}")
st.caption(
    f"Label definition: HIGH means hourly rental count (`cnt`) is at or above the "
    f"dataset mean of {metrics['threshold']:.1f}. Validation is a stratified random 20% holdout."
)

overview_tab, predict_tab, performance_tab = st.tabs(
    ["📊 Overview", "🎯 Predict demand", "🧪 Model performance"]
)

# ---------------- Overview ----------------
with overview_tab:
    st.subheader("Historical demand overview")
    left, right = st.columns(2)

    with left:
        by_hour = data.groupby("hr", as_index=False)["cnt"].mean()
        fig = px.line(
            by_hour, x="hr", y="cnt", markers=True,
            title="Average rentals by hour",
            labels={"hr": "Hour of day", "cnt": "Average rental count"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        by_season = data.groupby("season", as_index=False)["cnt"].mean()
        by_season["Season"] = by_season["season"].map(SEASON_NAMES)
        fig = px.bar(
            by_season, x="Season", y="cnt",
            title="Average rentals by season",
            labels={"cnt": "Average rental count", "Season": "Season"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        by_weather = data.groupby("weathersit", as_index=False)["cnt"].mean()
        by_weather["Weather"] = by_weather["weathersit"].map(WEATHER_NAMES)
        fig = px.bar(
            by_weather, x="Weather", y="cnt",
            title="Average rentals by weather",
            labels={"cnt": "Average rental count", "Weather": "Weather situation"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        demand_counts = pd.DataFrame({
            "Demand class": ["LOW", "HIGH"],
            "Rows": [
                int((data["cnt"] < metrics["threshold"]).sum()),
                int((data["cnt"] >= metrics["threshold"]).sum())
            ]
        })
        fig = px.pie(
            demand_counts, names="Demand class", values="Rows",
            title="HIGH vs LOW demand distribution", hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Preview cleaned data"):
        st.dataframe(data.head(20), use_container_width=True, hide_index=True)

# ---------------- Prediction ----------------
with predict_tab:
    st.subheader("Predict demand for a specific hour")
    st.write("Choose calendar and weather conditions, then select **Predict demand**.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            season = st.selectbox(
                "Season", [1, 2, 3, 4],
                format_func=lambda x: f"{x} — {SEASON_NAMES[x]}"
            )
            yr = st.selectbox(
                "Year indicator", [0, 1],
                format_func=lambda x: f"{x} — {'First' if x == 0 else 'Second'} year in dataset"
            )
            mnth = st.selectbox("Month", list(range(1, 13)), index=5)
            hr = st.slider("Hour of day", 0, 23, 17)
        with c2:
            holiday = st.selectbox("Holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            weekday = st.selectbox("Weekday code", list(range(7)))
            workingday = st.selectbox("Working day?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            weathersit = st.selectbox(
                "Weather situation", [1, 2, 3, 4],
                format_func=lambda x: f"{x} — {WEATHER_NAMES[x]}"
            )
        with c3:
            temp = st.slider("Normalized temperature", 0.0, 1.0, 0.50, 0.01)
            atemp = st.slider("Normalized feeling temperature", 0.0, 1.0, 0.48, 0.01)
            hum = st.slider("Normalized humidity", 0.0, 1.0, 0.60, 0.01)
            windspeed = st.slider("Normalized windspeed", 0.0, 1.0, 0.20, 0.01)

        submitted = st.form_submit_button("Predict demand", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "season": season, "yr": yr, "mnth": mnth, "hr": hr,
            "holiday": holiday, "weekday": weekday, "workingday": workingday,
            "weathersit": weathersit, "temp": temp, "atemp": atemp,
            "hum": hum, "windspeed": windspeed
        }], columns=FEATURES)
        prob_high = float(model.predict_proba(row)[0, 1])
        predicted = int(prob_high >= 0.5)

        st.markdown("---")
        res1, res2 = st.columns([1, 1])
        with res1:
            if predicted:
                st.success("## Predicted demand: HIGH")
            else:
                st.warning("## Predicted demand: LOW")
            st.metric("Estimated probability of HIGH demand", f"{prob_high:.1%}")
            st.progress(prob_high)
        with res2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_high * 100,
                number={"suffix": "%"},
                title={"text": "Probability of HIGH demand"},
                gauge={"axis": {"range": [0, 100]}}
            ))
            gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(gauge, use_container_width=True)
        st.write("**Inputs submitted**")
        st.dataframe(row, use_container_width=True, hide_index=True)
        st.caption(
            "This is a model-estimated probability, not a guarantee of actual rental volume. "
            "The model predicts the HIGH/LOW class, not the exact number of rentals."
        )

# ---------------- Performance ----------------
with performance_tab:
    st.subheader("Validation performance")
    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    met2.metric("ROC AUC", f"{metrics['auc']:.3f}")
    met3.metric("Precision", f"{metrics['precision']:.3f}")
    met4.metric("Recall", f"{metrics['recall']:.3f}")

    cm = metrics["confusion"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual LOW", "Actual HIGH"],
        columns=["Predicted LOW", "Predicted HIGH"]
    )
    fig = px.imshow(
        cm_df, text_auto=True, aspect="auto",
        title="Confusion matrix",
        labels={"x": "Predicted class", "y": "Actual class", "color": "Observations"}
    )
    st.plotly_chart(fig, use_container_width=True)

    fi = metrics["feature_importance"].sort_values("Importance", ascending=True)
    fig = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        title="Random Forest feature importance",
        labels={"Importance": "Importance score", "Feature": "Input feature"}
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Classification report"):
        report = classification_report(
            metrics["validation"]["Actual"],
            metrics["validation"]["Predicted"],
            target_names=["LOW", "HIGH"],
            output_dict=True,
            zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

st.markdown("---")
st.markdown(
    "**Method note:** This application trains a Python scikit-learn Random Forest from the uploaded CSV. "
    "It is not automatically connected to the exact champion model stored in SAS Viya. "
    "The target is defined as `cnt >= dataset mean`; `cnt`, `casual`, and `registered` are excluded "
    "from predictors to avoid target leakage. Temperature, apparent temperature, humidity, and windspeed "
    "are expected on the dataset's normalized 0–1 scale."
)
