# Student Success — Student Performance Prediction

A Streamlit + Scikit-learn mini project for exploring student data, training a model, evaluating predictions, and generating indicative performance estimates.

## Features
- Dashboard and dataset inspection
- CSV upload for custom datasets
- Two built-in samples: a real 480-student dataset (`data/student_academic_performance.csv`) and a 120-row synthetic demo
- Sensitive columns (gender, nationality, place of birth...) are ignored by default, with a warning if you add them back
- Student ID login with a profile page and a per-session prediction history you can download
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
If no CSV is uploaded the app uses the built-in sample chosen in the sidebar. The synthetic sample is generated from a formula, so scores on it are for demonstration only and are not evidence of real predictive performance.

The real sample is the "Students' Academic Performance" dataset (xAPI-Edu-Data; Amrieh, Hamtini & Aljarah, 2016), 480 students with a Low / Medium / High `Class` label. Confirm its licence and terms of use before redistributing the repository or publishing results. For a real project, use a documented dataset that you are permitted to use, describe its source, clean it, and discuss limitations.

When uploading a CSV:
- Each row should represent a student or student assessment.
- Choose the column to predict in **Model lab**.
- Numeric targets with more than 12 unique values are auto-detected as regression; otherwise classification is assumed. You can override this.
- Exclude IDs and columns that leak the target.
- Do not upload identifiable or sensitive student data to a public demo.
- Ensure there are enough examples for the selected task.

## Academic integrity and limitations
This application is a learning prototype, not a validated educational decision system. Predictions can be inaccurate or biased. Do not use predictions as the sole basis for grading, admission, discipline, or other consequential decisions. Model feature importance is associative, not proof of causation.

## Student ID login setup

The app includes student registration and login using a generated ID such as `STU-A1B2C3D4E5F6` plus a password. Accounts are managed by Supabase Authentication. **The login will not work on a fresh deployment until the app owner configures Supabase.**

1. Create a project at [Supabase](https://supabase.com/).
2. In Supabase, open **Project Settings → API** and copy the Project URL and the **anon/public** key.
3. In Streamlit Community Cloud, open the app's **Settings → Secrets** and add:

   ```toml
   SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"
   SUPABASE_ANON_KEY = "YOUR-SUPABASE-ANON-KEY"
   ```

4. In Supabase Authentication settings, disable email confirmation for this prototype. It uses a generated internal `@example.com` address (a reserved example domain, not a real inbox) so students can log in with their ID without supplying an email address. Consequently, email-based password recovery is not available; for a real production system, use verified student email addresses and implement a recovery flow.
5. Restart/redeploy the Streamlit app and test registration with a new account.

Use only the Supabase **anon/public** key in Streamlit Secrets. Never expose a Supabase service-role key in client-facing code. Student profile metadata is stored with the Supabase Auth user. Prediction history is kept in the active Streamlit session (visible and downloadable from **My profile**) and is not yet saved permanently to a database, so it resets when the session ends.
