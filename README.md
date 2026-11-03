# Bike Rental Demand Intelligence — SAS best-model outputs

This Streamlit app uses the actual SAS scored output files supplied by the student.

## Included files
- `app.py` — Streamlit application.
- `requirements.txt` — Python dependencies.
- `FORWARD_SCORED.csv` — SAS predictive Forward Selection regression output (12,165 rows).
- `FOREST_SCORED.csv` — SAS Viya Forest classification output (34,211 partition rows; 17,379 unique IDs).

## What the app does
1. Accepts independent-variable inputs for season, year, month, hour, holiday, weekday, working day, weather, temperature, feels-like temperature, humidity and windspeed.
2. Calculates a regression estimate using the Forward Selection coefficients in `PROJ17(1).pdf`. The reconstructed equation matches the supplied SAS `p_cnt` values across the 12,165 scored records with rounding-level differences.
3. Retrieves the saved SAS Forest class and probabilities for the nearest reference row in `FOREST_SCORED.csv`. Exact input matches return the saved SAS prediction for that row. Non-exact inputs are explicitly labelled as nearest-reference lookups, not new Forest model scores.
4. Provides existing observation lookup, validation metrics, charts, and downloadable combined SAS predictions.

## Critical limitations that are disclosed in the app
- The separately supplied `Forward_score.sas` has a different coefficient set and does not reproduce the `p_cnt` values in `FORWARD_SCORED.csv`; the app uses the predictive Forward coefficients from `PROJ17(1).pdf`, verified against the CSV.
- The Forest DS2 score-code ZIP references an ASTORE table. The ASTORE was uploaded as CSV/TSV, which does not preserve the native binary analytic store; therefore the app cannot execute the actual SAS Forest scoring package on arbitrary new rows. It uses its SAS-scored observation outputs as references.
- The Forest score code declares `casual` and `registered` as inputs. These are components of total `cnt` and could cause target leakage if used as predictors for future demand. Confirm the predictor roles in SAS before operational use.
- The class meaning of `Demand=1` has not been confirmed in the supplied files. The interface displays class 0/1 by default and lets the user choose a HIGH/LOW label only after confirmation.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
Upload these four files to the root of the GitHub repository:
- `app.py`
- `requirements.txt`
- `FORWARD_SCORED.csv`
- `FOREST_SCORED.csv`

Set the app entry point to `app.py` and deploy. The CSV outputs must be in the same repository root as `app.py`, unless the user uploads them in the app sidebar.

## Presentation diagrams
- `workflow_diagram.png` — analytical workflow from raw data through both SAS models to Streamlit.
- `web_app_architecture.png` — user input, regression scorer, Forest scored-output reference, results dashboard and exports.
