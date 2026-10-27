import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

st.set_page_config(page_title="Student Success | Performance AI", page_icon="🎓", layout="wide")

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#102a43,#176b87); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin:0; font-size:2.15rem;}
.hero p {color:#d9edf4; margin:.45rem 0 0 0; font-size:1.03rem;}
.small-muted {color:#64748b;font-size:.88rem;}
div[data-testid="stMetric"] {background:#f8fafc; border:1px solid #e2e8f0; padding:14px; border-radius:14px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_data
def make_demo_data(n=700, seed=42):
    rng = np.random.default_rng(seed)
    study = rng.integers(1, 6, n)
    attendance = rng.integers(45, 101, n)
    previous = np.clip(rng.normal(65, 16, n), 15, 100).round(0)
    assignments = rng.integers(30, 101, n)
    sleep = np.clip(rng.normal(7, 1.3, n), 3, 10).round(1)
    support = rng.choice(["Low", "Medium", "High"], n, p=[.25, .5, .25])
    activities = rng.choice(["Yes", "No"], n)
    support_effect = pd.Series(support).map({"Low": -5, "Medium": 0, "High": 4}).to_numpy()
    activity_effect = np.where(activities == "Yes", 1.5, 0)
    noise = rng.normal(0, 7, n)
    score = np.clip(18 + study*5 + attendance*.18 + previous*.22 + assignments*.08 + sleep*.8 + support_effect + activity_effect + noise, 0, 100).round(1)
    return pd.DataFrame({
        "study_hours_per_day": study, "attendance_percent": attendance,
        "previous_score": previous, "assignment_completion": assignments,
        "sleep_hours": sleep, "family_support": support,
        "extracurricular_activities": activities, "final_score": score
    })

def load_data(uploaded):
    if uploaded is None:
        return make_demo_data(), "Built-in demo dataset"
    try:
        data = pd.read_csv(uploaded)
        if data.empty or len(data.columns) < 2:
            raise ValueError("CSV must contain at least two columns and one row.")
        return data, uploaded.name
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return make_demo_data(), "Built-in demo dataset (fallback)"

def build_pipeline(X, task, model_name):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers = []
    if numeric:
        transformers.append(("num", numeric_pipe, numeric))
    if categorical:
        transformers.append(("cat", categorical_pipe, categorical))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    if task == "Regression":
        model = ExtraTreesRegressor(n_estimators=180, random_state=42, min_samples_leaf=2) if model_name == "Extra Trees" else RandomForestRegressor(n_estimators=180, random_state=42, min_samples_leaf=2)
    else:
        model = RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced")
    return Pipeline([("preprocess", prep), ("model", model)])

def infer_task(y):
    return "Regression" if pd.api.types.is_numeric_dtype(y) and y.nunique() > 12 else "Classification"

def feature_names_and_importances(pipe, input_cols):
    try:
        prep = pipe.named_steps["preprocess"]
        names = prep.get_feature_names_out()
        vals = pipe.named_steps["model"].feature_importances_
        frame = pd.DataFrame({"Feature": names, "Importance": vals})
        frame["Feature"] = frame["Feature"].str.replace(r"^(num|cat)__", "", regex=True)
        return frame.sort_values("Importance", ascending=False).head(15)
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])

st.markdown("""
<div class="hero">
  <h1>🎓 Student Success</h1>
  <p>Performance prediction, model insights, and practical academic support — in one dashboard.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Project settings")
    uploaded = st.file_uploader("Upload student dataset (CSV)", type=["csv"])
    st.caption("No file? The app uses a clearly labelled synthetic demo dataset.")
    st.divider()
    st.subheader("Dataset setup")
    st.caption("Choose a target column such as final_score, G3, grade, or result.")
    st.caption("Avoid identifiers and columns that reveal the answer in advance.")
    st.divider()
    st.caption("Privacy: data is processed by this app session. Do not upload sensitive student records to a public demo.")

data, source_name = load_data(uploaded)
st.caption(f"Dataset source: **{source_name}** · {len(data):,} rows · {len(data.columns)} columns")

tab_overview, tab_predict, tab_models, tab_data = st.tabs(["📊 Overview", "🔮 Predict", "🧠 Model lab", "🗂️ Data explorer"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Student records", f"{len(data):,}")
    c2.metric("Input columns", max(0, len(data.columns)-1))
    c3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    c4.metric("Numeric columns", len(numeric_cols))
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Dataset preview")
        st.dataframe(data.head(12), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Missing values by column")
        missing = data.isna().sum().sort_values(ascending=False).head(12).reset_index()
        missing.columns = ["Column", "Missing"]
        fig = px.bar(missing, x="Missing", y="Column", orientation="h", color="Missing",
                     color_continuous_scale="Blues", title="Missing cell count")
        fig.update_layout(height=350, coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    if numeric_cols:
        st.subheader("Explore a numeric variable")
        chosen = st.selectbox("Select column", numeric_cols, key="overview_numeric")
        fig = px.histogram(data, x=chosen, nbins=25, marginal="box", title=f"Distribution of {chosen}")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

with tab_predict:
    st.subheader("Predict an individual student's outcome")
    st.write("Choose the target column in **Model lab** first. The input form is generated from the remaining dataset columns.")
    target_default = "final_score" if "final_score" in data.columns else ("G3" if "G3" in data.columns else data.columns[-1])
    target_for_form = st.session_state.get("target_column", target_default)
    if target_for_form not in data.columns:
        target_for_form = target_default
    feature_cols = [c for c in data.columns if c != target_for_form]
    model_key = f"trained_model_{target_for_form}"
    if model_key not in st.session_state:
        st.info("Train a model in the **Model lab** tab to enable predictions.")
    else:
        trained = st.session_state[model_key]
        input_values = {}
        with st.form("prediction_form"):
            cols = st.columns(2)
            for idx, col in enumerate(feature_cols):
                series = data[col]
                with cols[idx % 2]:
                    if pd.api.types.is_numeric_dtype(series):
                        median = float(series.median()) if series.notna().any() else 0.0
                        minv = float(series.min()) if series.notna().any() else 0.0
                        maxv = float(series.max()) if series.notna().any() else 100.0
                        if minv == maxv:
                            input_values[col] = st.number_input(col.replace("_", " ").title(), value=median)
                        else:
                            input_values[col] = st.number_input(col.replace("_", " ").title(), min_value=minv, max_value=maxv, value=float(np.clip(median, minv, maxv)))
                    else:
                        options = series.dropna().astype(str).value_counts().index.tolist()
                        if not options:
                            options = ["Unknown"]
                        input_values[col] = st.selectbox(col.replace("_", " ").title(), options)
            submitted = st.form_submit_button("Predict performance", type="primary", use_container_width=True)
        if submitted:
            one = pd.DataFrame([{c: input_values[c] for c in feature_cols}])
            for c in one.columns:
                if c in data.columns and not pd.api.types.is_numeric_dtype(data[c]):
                    one[c] = one[c].astype(str)
            prediction = trained.predict(one)[0]
            task = st.session_state.get(f"task_{target_for_form}", infer_task(data[target_for_form].dropna()))
            st.success("Prediction generated")
            if task == "Regression":
                score = float(prediction)
                st.metric("Predicted score", f"{score:.1f}")
                if score >= 80: category = "High performance"
                elif score >= 60: category = "Moderate performance"
                else: category = "Needs support"
                st.info(f"Indicative category: **{category}**. This is an estimate, not a guaranteed outcome.")
            else:
                st.metric("Predicted category", str(prediction))
                st.info("Treat this result as decision support, not as the sole basis for academic decisions.")
            st.subheader("Input summary")
            st.dataframe(one, use_container_width=True, hide_index=True)

with tab_models:
    st.subheader("Train and evaluate a model")
    st.write("Select a target. Numeric targets with many distinct values are treated as regression; otherwise classification.")
    suggested = "final_score" if "final_score" in data.columns else ("G3" if "G3" in data.columns else data.columns[-1])
    target = st.selectbox("Target column to predict", data.columns.tolist(), index=data.columns.tolist().index(suggested), key="target_select")
    st.session_state["target_column"] = target
    model_type = infer_task(data[target].dropna())
    task_choice = st.radio("Prediction task", ["Auto-detect", "Regression", "Classification"], horizontal=True, index=0)
    task = model_type if task_choice == "Auto-detect" else task_choice
    model_name = st.selectbox("Model", ["Random Forest", "Extra Trees"] if task == "Regression" else ["Random Forest"])
    excluded = st.multiselect("Exclude columns from model inputs (recommended for IDs or leakage)", [c for c in data.columns if c != target])
    test_size = st.slider("Test set (%)", 10, 35, 20, 5)
    train_clicked = st.button("🚀 Train model", type="primary")
    if train_clicked:
        clean = data.dropna(subset=[target]).copy()
        features = [c for c in clean.columns if c != target and c not in excluded]
        if not features:
            st.error("Select at least one input column.")
        elif len(clean) < 10:
            st.error("At least 10 labelled rows are recommended for a train/test split.")
        elif clean[target].nunique() < 2:
            st.error("The target must contain at least two distinct values.")
        else:
            X = clean[features].copy()
            y = clean[target].copy()
            if task == "Classification":
                y = y.astype(str)
                stratify = y if y.value_counts().min() >= 2 else None
            else:
                y = pd.to_numeric(y, errors="coerce")
                valid = y.notna()
                X, y = X.loc[valid], y.loc[valid]
                stratify = None
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, stratify=stratify)
                pipe = build_pipeline(X_train, task, model_name)
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)
                st.session_state[f"trained_model_{target}"] = pipe
                st.session_state[f"task_{target}"] = task
                st.session_state[f"features_{target}"] = features
                st.session_state[f"model_name_{target}"] = model_name
                st.success(f"{model_name} trained successfully on {len(X_train)} rows; tested on {len(X_test)} rows.")
                m1, m2, m3 = st.columns(3)
                if task == "Regression":
                    mae = mean_absolute_error(y_test, pred)
                    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                    r2 = r2_score(y_test, pred) if len(y_test) > 1 else float("nan")
                    m1.metric("MAE (lower is better)", f"{mae:.2f}")
                    m2.metric("RMSE (lower is better)", f"{rmse:.2f}")
                    m3.metric("R²", f"{r2:.3f}" if np.isfinite(r2) else "N/A")
                    compare = pd.DataFrame({"Actual": np.asarray(y_test), "Predicted": pred})
                    fig = px.scatter(compare, x="Actual", y="Predicted", title="Actual vs predicted")
                    low = min(compare.min())
                    high = max(compare.max())
                    fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line={"dash":"dash"})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    acc = accuracy_score(y_test, pred)
                    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
                    m1.metric("Accuracy", f"{acc:.3f}")
                    m2.metric("Weighted F1", f"{f1:.3f}")
                    m3.metric("Test records", len(y_test))
                    labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
                    cm = confusion_matrix(y_test, pred, labels=labels)
                    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, aspect="auto", title="Confusion matrix", labels={"x":"Predicted","y":"Actual","color":"Count"})
                    st.plotly_chart(fig, use_container_width=True)
                importance = feature_names_and_importances(pipe, features)
                if not importance.empty:
                    st.subheader("Most influential model features")
                    fig = px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Feature importance (not causal evidence)")
                    st.plotly_chart(fig, use_container_width=True)
                st.caption("Metrics depend on the dataset, target definition, and test split. They are not a guarantee of real-world performance.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.info("Check for mixed target types, too few examples per class, or columns containing unusual values.")

with tab_data:
    st.subheader("Data explorer")
    st.write("Inspect column types and summary statistics before training.")
    schema = pd.DataFrame({"Column": data.columns, "Type": [str(t) for t in data.dtypes], "Missing": [int(data[c].isna().sum()) for c in data.columns], "Unique values": [int(data[c].nunique(dropna=True)) for c in data.columns]})
    st.dataframe(schema, use_container_width=True, hide_index=True)
    with st.expander("Summary statistics"):
        st.dataframe(data.describe(include="all").transpose(), use_container_width=True)
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download current dataset as CSV", csv, file_name="student_dataset.csv", mime="text/csv")
    st.caption("The built-in dataset is synthetic and exists for demonstration only. Replace it with an appropriately licensed, documented dataset for academic evaluation.")
