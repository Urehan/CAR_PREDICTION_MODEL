# Breast Cancer Diagnosis Predictor

Streamlit app that predicts Benign vs Malignant from tumor measurements,
using a trained `DecisionTreeClassifier`.

## Files
- `app.py` — the Streamlit app (modern, responsive UI with sidebar form + gauge chart)
- `requirements.txt` — pinned dependencies
- `.streamlit/config.toml` — theme colors
- `breast_cancer_model.pkl` — copy your model file here (loaded successfully, no corruption issues)

## About the features
The model expects 16 features in this exact order:
`id`, 8 raw tumor measurements (`radius_mean`, `texture_mean`, `perimeter_mean`,
`area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`,
`concave points_mean`), and 7 engineered features (`shape_irregularity`,
`border_complexity`, `tumor_aggressiveness`, `radius_texture_interaction`,
`radius_concavity_interaction`, `concavity_density`, `malignancy_risk_score`).

Since the exact training formulas for the engineered features weren't
available, the app **auto-calculates estimated values** from the 8 raw
measurements (toggle "Auto-calculate from raw measurements" off in the
sidebar to enter exact values manually instead):

| Feature | Estimated formula |
|---|---|
| `shape_irregularity` | `perimeter_mean² / area_mean` |
| `border_complexity` | `concave_points_mean × perimeter_mean` |
| `tumor_aggressiveness` | `compactness_mean × concavity_mean × (1 + concave_points_mean)` |
| `radius_texture_interaction` | `radius_mean × texture_mean` |
| `radius_concavity_interaction` | `radius_mean × concavity_mean` |
| `concavity_density` | `concavity_mean / area_mean` |
| `malignancy_risk_score` | `(compactness_mean + concavity_mean + concave_points_mean) / 3` |

These are reasonable domain-standard approximations, **not** guaranteed to
match the original training pipeline exactly. If you get the real training
notebook/script later, share the actual formulas and the app can be updated
to match precisely — this affects prediction accuracy since the model was
trained on the true engineered values.

## Deploy on GitHub + Streamlit Community Cloud

1. **Create a GitHub repo**
   ```bash
   cd breast_cancer_app
   git init
   git add .
   git commit -m "Breast cancer diagnosis predictor app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in
   with your GitHub account.

3. Click **"New app"** → pick your repo, branch `main`, and set
   **Main file path** to `app.py`.

4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically and gives you a live URL like
   `https://<your-app-name>.streamlit.app`.
