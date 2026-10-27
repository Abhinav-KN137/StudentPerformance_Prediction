# Student Success — Student Performance Prediction

A Streamlit + Scikit-learn mini project for exploring student data, training a model, evaluating predictions, and generating indicative performance estimates.

## Features
- Dashboard and dataset inspection
- CSV upload for custom datasets
- Synthetic demo data so the app starts without a dataset file
- Regression or classification target selection
- Random Forest / Extra Trees training
- Holdout evaluation (MAE, RMSE, R² or accuracy, weighted F1, confusion matrix)
- Feature importance chart
- Prediction form generated from the selected dataset
- CSV download
- Responsive Streamlit interface

## Run locally
Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. Sign in to Streamlit Community Cloud using your GitHub account.
3. Select **Create app**, choose your repository/branch, and set the main file path to `app.py`.
4. Deploy. Streamlit installs packages from `requirements.txt`.
5. Open the public app URL and test the Overview, Model lab, and Predict tabs.

Account authorization is performed by you in GitHub/Streamlit; never share your password or tokens in chat.

## Dataset notes
The app uses a synthetic dataset if no CSV is uploaded. Synthetic data is for demonstration, not evidence of real predictive performance. For a real project, use a documented dataset that you are permitted to use, describe its source, clean it, and discuss limitations.

When uploading a CSV:
- Each row should represent a student or student assessment.
- Choose the column to predict in **Model lab**.
- Numeric targets with more than 12 unique values are auto-detected as regression; otherwise classification is assumed. You can override this.
- Exclude IDs and columns that leak the target.
- Do not upload identifiable or sensitive student data to a public demo.
- Ensure there are enough examples for the selected task.

## Academic integrity and limitations
This application is a learning prototype, not a validated educational decision system. Predictions can be inaccurate or biased. Do not use predictions as the sole basis for grading, admission, discipline, or other consequential decisions. Model feature importance is associative, not proof of causation.
