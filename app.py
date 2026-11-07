from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import confusion_matrix

BASE = Path(__file__).resolve().parent
FEATURES = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]
CAT_FEATURES = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
CONT_FEATURES = ["temp", "atemp", "hum", "windspeed"]
REG_INTERCEPT = 76.324240
REG_COEFFICIENTS = {
    "season": {1:-43.259833, 2:-18.703738, 3:-8.925772},
    "yr": {0:-82.712110},
    "mnth": {1:-24.344177,2:-20.806528,3:-3.483231,4:-6.109367,5:5.695896,6:-9.348626,7:-45.950090,8:-36.542966,9:-7.754670,10:2.708040,11:-3.848921},
    "hr": {0:-26.849862,1:-38.852180,2:-49.815331,3:-57.150629,4:-60.005337,5:-44.867410,6:1.963969,7:111.885184,8:233.417746,9:115.830201,10:70.306869,11:86.187449,12:121.108895,13:116.396684,14:104.474149,15:112.378871,16:163.757322,17:286.009931,18:274.851047,19:174.147939,20:104.207803,21:65.268533,22:32.989934},
    "holiday": {0:27.759539},
    "weathersit": {1:17.081160,2:2.739506,3:-46.213268},
}
REG_CONTINUOUS = {"temp":239.244270,"hum":-64.622250,"windspeed":-50.931623}

st.set_page_config(page_title="Bike Rental Demand Intelligence",page_icon="🚲",layout="wide")

@st.cache_resource
def load_classifier():
    return joblib.load(BASE/"forest_deployment_model.joblib")

@st.cache_data
def load_data():
    reg=pd.read_csv(BASE/"FORWARD_SCORED.csv")
    forest=pd.read_csv(BASE/"FOREST_SAS_SCORED_UNIQUE.csv")
    metrics=json.loads((BASE/"forest_deployment_metrics.json").read_text())
    return reg,forest,metrics

try:
    reg,forest,py_metrics=load_data()
    classifier=load_classifier()
except Exception as e:
    st.error(f"Application data/model could not be loaded: {e}")
    st.stop()

# Validate reconstructed regression score against SAS scored output
reg_cols=["season","yr","mnth","hr","holiday","weathersit","temp","hum","windspeed","p_cnt","cnt","r_cnt"]
if not set(reg_cols).issubset(reg.columns):
    st.error("Forward scored file is missing required regression columns.")
    st.stop()
calc=np.full(len(reg),REG_INTERCEPT,dtype=float)
for c,m in REG_COEFFICIENTS.items(): calc += reg[c].map(m).fillna(0).to_numpy(float)
for c,b in REG_CONTINUOUS.items(): calc += b*reg[c].to_numpy(float)
reg_check=float(np.max(np.abs(calc-reg.p_cnt.to_numpy(float))))

st.title("🚲 Bike Rental Demand Intelligence")
st.caption("Best-model decision support | SAS Forward Selection regression + Forest classification workflow")
st.info("The app loads all bundled data and model files automatically. Website visitors do not need to upload CSV files.")

# Top-level metric cards
c1,c2,c3,c4=st.columns(4)
c1.metric("Regression scored observations",f"{len(reg):,}")
c2.metric("Forest SAS observations",f"{len(forest):,}")
c3.metric("Regression formula check",f"{reg_check:.2e}")
c4.metric("Python deployment classifier accuracy",f"{py_metrics['accuracy']:.1%}")

st.header("1. Enter operating conditions")
st.caption("Enter only variables available before the rental hour. The classifier excludes `casual`, `registered` and `cnt` to avoid target leakage.")
with st.form("predict_form"):
    a,b,c,d=st.columns(4)
    with a:
        season=st.selectbox("Season",[1,2,3,4],index=1,format_func=lambda x:{1:"1 – Spring",2:"2 – Summer",3:"3 – Fall",4:"4 – Winter"}[x])
        yr=st.selectbox("Year",[0,1],index=1,format_func=lambda x:{0:"2011",1:"2012"}[x])
        mnth=st.selectbox("Month",list(range(1,13)),index=5)
    with b:
        hr=st.selectbox("Hour (0–23)",list(range(24)),index=17)
        holiday=st.selectbox("Holiday",[0,1],format_func=lambda x:"Yes" if x else "No")
        weekday=st.selectbox("Weekday (0=Sunday)",list(range(7)),index=5)
    with c:
        workingday=st.selectbox("Working day",[0,1],index=1,format_func=lambda x:"Yes" if x else "No")
        weathersit=st.selectbox("Weather situation",[1,2,3,4],format_func=lambda x:{1:"1 – Clear / partly cloudy",2:"2 – Mist / cloudy",3:"3 – Light rain / snow",4:"4 – Heavy rain / snow"}[x])
        temp=st.number_input("Temperature (normalised 0–1)",0.0,1.0,0.65,0.01)
    with d:
        atemp=st.number_input("Feels-like temperature (0–1)",0.0,1.0,0.62,0.01)
        hum=st.number_input("Humidity (0–1)",0.0,1.0,0.55,0.01)
        windspeed=st.number_input("Windspeed (0–1)",0.0,1.0,0.18,0.01)
    run=st.form_submit_button("Predict hourly demand",type="primary",use_container_width=True)

if run:
    row={"season":season,"yr":yr,"mnth":mnth,"hr":hr,"holiday":holiday,"weekday":weekday,"workingday":workingday,"weathersit":weathersit,"temp":temp,"atemp":atemp,"hum":hum,"windspeed":windspeed}
    # SAS Forward Selection score equation
    reg_pred=REG_INTERCEPT
    for col,mapping in REG_COEFFICIENTS.items(): reg_pred += mapping.get(int(row[col]),0.0)
    for col,coef in REG_CONTINUOUS.items(): reg_pred += coef*float(row[col])
    # New-input classifier scorer: deployable Python RF trained on SAS Demand labels
    x=pd.DataFrame([[row[f] for f in FEATURES]],columns=FEATURES)
    class_pred=int(classifier.predict(x)[0])
    class_prob=classifier.predict_proba(x)[0]
    prob0=float(class_prob[list(classifier.classes_).index(0)])
    prob1=float(class_prob[list(classifier.classes_).index(1)])
    st.session_state["last_result"]={"row":row,"reg":float(reg_pred),"class":class_pred,"p0":prob0,"p1":prob1}

if "last_result" in st.session_state:
    z=st.session_state["last_result"]
    st.header("2. Predictions")
    x,y=st.columns(2)
    with x:
        st.subheader("Regression — SAS Forward Selection")
        st.metric("Predicted hourly rentals",f"{z['reg']:,.1f}")
        st.metric("Non-negative operational estimate",f"{max(0,z['reg']):,.1f}")
        if z['reg']<0: st.warning("The raw linear regression score is negative. The operational estimate is floored at zero.")
        st.caption("SAS Forward Selection: R² 0.6845 | Adjusted R² 0.6829 | Test ASE 7,272.4784 | Test RMSE ≈ 85.28.")
    with y:
        st.subheader("Classification — new-input deployment scorer")
        st.metric("Predicted Demand class",f"Class {z['class']}")
        st.metric("Probability of Class 1",f"{z['p1']:.1%}")
        st.metric("Probability of Class 0",f"{z['p0']:.1%}")
        st.caption("Fresh prediction for the entered inputs. This is a Python Random Forest deployment scorer trained on the SAS Viya Demand labels and the same 12 future-known predictors; it is not execution of the native SAS ASTORE.")
        st.progress(z['p1'],text="Class 1 probability")
    st.dataframe(pd.DataFrame([{"Input variable":k,"Entered value":v} for k,v in z['row'].items()]),hide_index=True,use_container_width=True)

st.divider()
st.header("3. Model performance")
t1,t2,t3=st.tabs(["Regression","SAS Viya Forest benchmark","Live deployment classifier"])
with t1:
    r1,r2,r3,r4=st.columns(4)
    rmse=float(np.sqrt(np.mean((reg.cnt-reg.p_cnt)**2)))
    mae=float(np.mean(np.abs(reg.cnt-reg.p_cnt)))
    r2score=float(1-np.sum((reg.cnt-reg.p_cnt)**2)/np.sum((reg.cnt-reg.cnt.mean())**2))
    r1.metric("R² on scored rows",f"{r2score:.4f}")
    r2.metric("RMSE on scored rows",f"{rmse:.2f}")
    r3.metric("MAE on scored rows",f"{mae:.2f}")
    r4.metric("Max SAS score check",f"{reg_check:.2e}")
    st.caption("Use the reported SAS test metrics above for model selection; scored-row metrics are descriptive and not an independent test estimate.")
    sample=reg.sample(min(3000,len(reg)),random_state=17)
    fig=px.scatter(sample,x="cnt",y="p_cnt",opacity=.4,title="SAS Forward Selection: actual vs predicted rentals",labels={"cnt":"Actual rentals","p_cnt":"Predicted rentals"})
    lo=min(sample.cnt.min(),sample.p_cnt.min()); hi=max(sample.cnt.max(),sample.p_cnt.max())
    fig.add_shape(type="line",x0=lo,y0=lo,x1=hi,y1=hi,line=dict(dash="dash"))
    st.plotly_chart(fig,use_container_width=True)
    st.plotly_chart(px.histogram(reg,x="r_cnt",nbins=50,title="Regression residual distribution"),use_container_width=True)
with t2:
    st.caption("Official SAS Viya Model Comparison validation metrics for the corrected Forest champion model (casual and registered excluded from predictors).")
    sas=pd.DataFrame([
      ["Forest (Champion)",0.9666,0.9947,0.9794,0.0328,0.9219],
      ["Decision Tree",0.8866,0.8664,0.9327,0.0880,0.5847],
      ["Logistic Regression",0.8193,0.7972,0.8947,0.1288,0.4490],
      ["Gradient Boosting",0.9352,0.9729,0.9601,0.0487,0.8337],
    ],columns=["Model","Accuracy","ROC AUC","F1","ASE","KS (Youden)"])
    st.dataframe(sas.style.format({"Accuracy":"{:.2%}","ROC AUC":"{:.4f}","F1":"{:.4f}","ASE":"{:.4f}","KS (Youden)":"{:.4f}"}),use_container_width=True,hide_index=True)
    st.success("SAS Viya selected Forest as champion: validation accuracy 96.66%, ROC AUC 0.9947, F1 0.9794 and KS 0.9219.")
    st.caption("These are SAS Viya's reported benchmark results. They are not represented as the Python deployment model's metrics.")
with t3:
    q1,q2,q3,q4=st.columns(4)
    q1.metric("Holdout accuracy",f"{py_metrics['accuracy']:.2%}")
    q2.metric("Holdout ROC AUC",f"{py_metrics['roc_auc']:.4f}")
    q3.metric("Holdout F1",f"{py_metrics['f1']:.4f}")
    q4.metric("Holdout rows",f"{py_metrics['test_rows']:,}")
    cm=np.array(py_metrics['confusion_matrix'])
    cm_df=pd.DataFrame(cm,index=["Actual 0","Actual 1"],columns=["Predicted 0","Predicted 1"])
    st.markdown("**Python deployment scorer confusion matrix**")
    st.dataframe(cm_df,use_container_width=True)
    st.caption("Holdout split: 80/20 stratified, random seed 42. The Python scorer is a deployable approximation trained on SAS-labelled observations, not the native SAS ASTORE.")

st.divider()
st.header("4. Historical SAS-scored observation lookup")
st.caption("Inspect the actual new Forest outputs produced by SAS Viya for an existing observation ID.")
obs=st.selectbox("Observation ID (instant)",forest.instant.astype(int).tolist(),index=min(8000,len(forest)-1))
frow=forest.loc[forest.instant.astype(int)==int(obs)].iloc[0]
cc1,cc2,cc3,cc4=st.columns(4)
cc1.metric("Actual Demand",int(frow.Demand))
cc2.metric("SAS Forest predicted class",int(frow.I_Demand))
cc3.metric("SAS probability class 1",f"{float(frow.P_Demand1):.2%}")
cc4.metric("SAS probability class 0",f"{float(frow.P_Demand0):.2%}")
st.write({"SAS partition indicator":int(frow["_PartInd_"]),"Sample frequency":int(frow["_SampleFreq_1"])})

st.divider()
st.header("5. Scored data and downloads")
st.download_button("Download regression SAS scored data",reg.to_csv(index=False).encode(),"FORWARD_SCORED.csv","text/csv")
st.download_button("Download unique Forest SAS scored data",forest.to_csv(index=False).encode(),"FOREST_SAS_SCORED_UNIQUE.csv","text/csv")
st.dataframe(forest.head(100),use_container_width=True)

st.divider()
st.subheader("Model governance and deployment note")
st.markdown("""
- **Regression:** the SAS Forward Selection equation is embedded in the app and validated against the supplied SAS `p_cnt` scored output.
- **SAS classification benchmark:** the corrected SAS Viya Forest champion excludes `casual` and `registered`; its validation metrics are shown separately.
- **Live classification:** Streamlit uses a Python Random Forest trained on the SAS `Demand` labels and the same 12 future-known predictors to generate new-input class/probability estimates. This is a deployment-compatible approximation, **not an execution of the native SAS ASTORE**.
- **Native ASTORE:** `FOREST_AST.sasast` and `dmcas_epscorecode.sas` are included in the project package for traceability and future SAS runtime integration. Streamlit Community Cloud does not execute SAS ASTORE files directly.
- **No leakage predictors:** `casual`, `registered` and `cnt` are never used as new-input classifier predictors.
""")
