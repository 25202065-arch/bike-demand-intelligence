# Bike Rental Demand Intelligence — corrected deployment package

## Included components
- `app.py`: Streamlit application.
- `FORWARD_SCORED.csv`: SAS Forward Selection regression scored output.
- `FOREST_SAS_SCORED_UNIQUE.csv`: latest SAS Viya Forest scored output, one row per observation.
- `forest_deployment_model.joblib`: Python Random Forest scorer for new inputs.
- `forest_deployment_metrics.json`: holdout metrics for the Python scorer.
- `FOREST_AST.sasast`: native SAS ASTORE artifact (reference / future SAS-runtime integration).
- `dmcas_epscorecode.sas`: SAS-generated Forest score code.

## Important model distinction
The SAS ASTORE is a proprietary SAS analytic-store binary. Streamlit Community Cloud does not execute it natively. Therefore the app clearly separates (1) official SAS Viya Forest champion validation metrics and historical SAS-scored rows from (2) the live Python Random Forest scorer. The Python scorer is trained on the SAS `Demand` labels using the same 12 future-known predictors, excluding `casual`, `registered`, and `cnt`; it is a deployable approximation, not the exact SAS Forest ASTORE. Do not report its holdout metrics as the SAS Forest metrics.

## Deploy
1. Upload all files in this folder to the root of the GitHub repository, replacing previous versions.
2. Commit to `main`.
3. In Streamlit Community Cloud, use repository `25202065-arch/bike-demand-intelligence`, branch `main`, main file `app.py`.
4. `requirements.txt` installs the runtime dependencies. No visitor CSV upload is required.

## Model inputs
`season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed`. Values use the original Bike Sharing dataset's encoding and normalized weather scales.
