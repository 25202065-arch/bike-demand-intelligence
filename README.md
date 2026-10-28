# Bike Rental Demand Intelligence App

A Streamlit web app for exploring hourly bike-sharing demand and predicting HIGH/LOW demand using a Random Forest classifier.

## Run on your computer

1. Install Python 3.10 or later.
2. Extract this folder.
3. Upload your dataset in the app sidebar, or save it as `hour.csv` in this folder.
4. Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected columns

`season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, cnt`

The app also works if the CSV contains additional columns; it uses only the listed columns.

## Important model details

- The target label is defined as HIGH if `cnt` is at or above the dataset's mean, otherwise LOW.
- `cnt`, `casual`, and `registered` are not predictors. This avoids target leakage because `cnt` is the source used to create the label.
- Model validation is a stratified random 20% holdout. For a more rigorous time-series evaluation, test on a future time period.
- This application trains a new Python scikit-learn Random Forest. It does not automatically use the exact Random Forest champion model saved in SAS Viya. To use the SAS Viya model itself, export/deploy its scoring code or endpoint and integrate that scoring mechanism.

## Deploy online

This app can be deployed on Streamlit Community Cloud or another Python hosting platform. Upload `app.py` and `requirements.txt` to a repository and configure the app entry point as `app.py`. The dataset can be uploaded through the app, or you can include `hour.csv` in the repository if your course's data-sharing rules allow it.

## Suggested architecture for presentation

User input / CSV → data validation and feature selection → Random Forest classifier → HIGH/LOW prediction + estimated probability → charts and performance metrics.
