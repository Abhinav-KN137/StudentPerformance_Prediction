import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Student Success | Performance AI", page_icon="🎓", layout="wide")

CUSTOM_CSS = """
<style>
:root { --ink:#142d4e; --muted:#64748b; --line:#e2e8f0; --surface:#ffffff; --soft:#f5f8fc; }
.block-container {padding-top:1.35rem; padding-bottom:2.5rem; max-width:1380px;}
[data-testid="stAppViewContainer"] {background:#f7f9fc;}
[data-testid="stHeader"] {background:rgba(247,249,252,.88);}
[data-testid="stSidebar"] {background:#f0f5fa; border-right:1px solid #e1e8f0;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {color:var(--ink);}
.hero {padding:1.65rem 1.8rem; border-radius:22px; background:linear-gradient(115deg,#102a43 0%,#155e75 60%,#2b7a78 100%); color:white; margin-bottom:1rem; box-shadow:0 10px 28px rgba(16,42,67,.12);}
.hero h1 {color:white; margin:0; font-size:2.15rem; letter-spacing:-.035em;}
.hero p {color:#e0f2f7; margin:.5rem 0 0 0; font-size:1rem;}
.eyebrow {color:#a9e3e7; font-size:.76rem; font-weight:750; letter-spacing:.13em; text-transform:uppercase; margin-bottom:.45rem;}
div[data-testid="stMetric"] {background:var(--surface); border:1px solid var(--line); padding:15px 16px; border-radius:16px; box-shadow:0 3px 12px rgba(20,45,78,.035);}
div[data-testid="stMetricLabel"] {color:#64748b;}
div[data-testid="stTabs"] button {font-weight:650; padding-top:.75rem; padding-bottom:.75rem;}
div[data-testid="stTabs"] button[aria-selected="true"] {color:#0f766e; border-bottom-color:#0f766e;}
.result-card {padding:1.25rem 1.4rem; border:1px solid #c8e0e5; border-radius:18px; background:linear-gradient(135deg,#edf8f8,#ffffff); margin:.65rem 0 1rem 0; box-shadow:0 5px 16px rgba(15,118,110,.06);}
.result-value {font-size:2.15rem; font-weight:780; color:#12344a; line-height:1.25; margin:.2rem 0;}
.empty-card {padding:1.6rem; text-align:center; border:1px dashed #b8c9d6; border-radius:18px; background:#fff; color:#526579;}
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {border-radius:11px; font-weight:700; min-height:2.7rem;}
[data-testid="stDataFrame"] {border:1px solid #e2e8f0; border-radius:12px; overflow:hidden;}
h1,h2,h3 {letter-spacing:-.02em; color:#18324f;}
hr {border-color:#e2e8f0;}
.small-muted {color:var(--muted); font-size:.88rem;}
@media (max-width: 700px) {.hero {padding:1.2rem; border-radius:16px;} .hero h1 {font-size:1.7rem;}}
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
        model = (ExtraTreesClassifier(n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1)
                 if model_name == "Extra Trees" else
                 RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1))
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
  <div class="eyebrow">Student analytics · ML dashboard</div>
  <h1>🎓 Student Success</h1>
  <p>Understand your data, train a prediction model, and explore student performance — step by step.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Workspace")
    st.caption("Start with the sample data, or upload your own CSV.")
    uploaded = st.file_uploader("Choose a student dataset", type=["csv"], help="Each row should represent one student or assessment.")
    st.divider()
    st.markdown("#### Quick guide")
    st.markdown("**1 · Overview** — check your dataset")
    st.markdown("**2 · Model Lab** — choose a target and train")
    st.markdown("**3 · Predict** — enter details and get an estimate")
    st.divider()
    st.caption("🔒 Privacy tip: avoid uploading names, IDs, or sensitive student records to a public app.")

data, source_name = load_data(uploaded)
st.markdown(f"""
<div style="display:flex;flex-wrap:wrap;gap:.55rem;align-items:center;margin:.4rem 0 1.1rem 0">
<span style="background:#e6f4f1;color:#17685f;border-radius:999px;padding:.32rem .7rem;font-size:.82rem;font-weight:650">● Data loaded</span>
<span style="color:#526579;font-size:.9rem"><b>{source_name}</b> &nbsp;·&nbsp; {len(data):,} rows &nbsp;·&nbsp; {len(data.columns)} columns</span>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_predict, tab_models, tab_data = st.tabs(["🏠 Overview", "✨ Make a prediction", "🧪 Train a model", "📁 Explore data"])

with tab_overview:
    st.subheader("Your dataset at a glance")
    st.caption("A quick health check before you train a model.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Student records", f"{len(data):,}")
    c2.metric("Input columns", max(0, len(data.columns)-1))
    c3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    c4.metric("Numeric columns", len(numeric_cols))
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("A look at your data")
        st.dataframe(data.head(12), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Data completeness")
        st.caption("Columns with the most blank cells")
        missing = data.isna().sum().sort_values(ascending=False).head(12).reset_index()
        missing.columns = ["Column", "Missing"]
        fig = px.bar(missing, x="Missing", y="Column", orientation="h", color="Missing",
                     color_continuous_scale="Blues", title="Missing cell count")
        fig.update_layout(height=350, coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    if numeric_cols:
        st.subheader("Explore a score or other numeric field")
        st.caption("Choose a column to see its distribution.")
        chosen = st.selectbox("Select column", numeric_cols, key="overview_numeric")
        fig = px.histogram(data, x=chosen, nbins=25, marginal="box", title=f"Distribution of {chosen}")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

with tab_predict:
    st.subheader("✨ Make a prediction")
    st.write("Enter a student's details to estimate the selected outcome.")
    st.caption("First train a model in **Train a model**. Your prediction is an estimate, not a guarantee.")
    target_default = "final_score" if "final_score" in data.columns else ("G3" if "G3" in data.columns else data.columns[-1])
    target_for_form = st.session_state.get("target_column", target_default)
    if target_for_form not in data.columns:
        target_for_form = target_default
    # Use exactly the features used for training, respecting excluded columns.
    feature_cols = st.session_state.get(f"features_{target_for_form}", [c for c in data.columns if c != target_for_form])
    feature_cols = [c for c in feature_cols if c in data.columns and c != target_for_form]
    model_key = f"trained_model_{target_for_form}"
    if model_key not in st.session_state:
        st.markdown('<div class="empty-card"><h3>🔮 Your prediction will appear here</h3><p>Train a model in <b>Model lab</b>, then return here to enter student details and generate a prediction.</p></div>', unsafe_allow_html=True)
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
            st.success("Prediction generated successfully")
            if task == "Regression":
                score = float(prediction)
                if score >= 80: category = "High performance"
                elif score >= 60: category = "Moderate performance"
                else: category = "May benefit from support"
                st.markdown(f'<div class="result-card"><div>Predicted outcome</div><div class="result-value">{score:.1f}</div><div>{category}</div></div>', unsafe_allow_html=True)
                st.caption("This is a model estimate, not a guaranteed outcome or a substitute for educator judgment.")
            else:
                st.markdown(f'<div class="result-card"><div>Predicted category</div><div class="result-value">{str(prediction)}</div></div>', unsafe_allow_html=True)
                try:
                    if hasattr(trained.named_steps["model"], "predict_proba"):
                        probabilities = trained.predict_proba(one)[0]
                        confidence = float(np.max(probabilities)) * 100
                        st.progress(min(1.0, max(0.0, confidence / 100)))
                        st.caption(f"Model confidence for this prediction: {confidence:.1f}% (probability estimate, not a guarantee).")
                except Exception:
                    pass
                st.info("Use this result as decision support, not as the sole basis for academic decisions.")
            st.subheader("Input summary")
            st.dataframe(one, use_container_width=True, hide_index=True)

with tab_models:
    st.subheader("🧪 Train and evaluate")
    st.write("Tell the model what you want to predict, choose the inputs, then train and review its results.")
    st.caption("Tip: choose a final score or result column as the target, and exclude IDs or columns that reveal the answer.")
    suggested = "final_score" if "final_score" in data.columns else ("G3" if "G3" in data.columns else data.columns[-1])
    target = st.selectbox("Target column to predict", data.columns.tolist(), index=data.columns.tolist().index(suggested), key="target_select")
    st.session_state["target_column"] = target
    model_type = infer_task(data[target].dropna())
    task_choice = st.radio("Prediction task", ["Auto-detect", "Regression", "Classification"], horizontal=True, index=0)
    task = model_type if task_choice == "Auto-detect" else task_choice
    model_name = st.selectbox("Algorithm", ["Random Forest", "Extra Trees"], help="Both are tree-based models; compare their evaluation metrics.")
    excluded = st.multiselect("Ignore columns (optional)", [c for c in data.columns if c != target], help="Exclude IDs, names, or columns that would leak the answer.")
    test_size = st.slider("Test set size (%)", 10, 50, 20, 5,
                          help="Share of rows held out for evaluation, not the model's accuracy. It cannot be 100% because the model needs training data.")
    st.caption("For a fair score, keep test data separate from model fitting. Classification uses accuracy/F1; regression uses MAE/RMSE/R².")
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
                    m1.metric("Test accuracy", f"{acc * 100:.1f}%")
                    m2.metric("Weighted F1", f"{f1 * 100:.1f}%")
                    correct = int((np.asarray(y_test).astype(str) == np.asarray(pred).astype(str)).sum())
                    m3.metric("Correct predictions", f"{correct}/{len(y_test)}")
                    labels = sorted(set(pd.Series(y_test).astype(str).tolist()) | set(pd.Series(pred).astype(str).tolist()))
                    cm = confusion_matrix(pd.Series(y_test).astype(str), pd.Series(pred).astype(str), labels=labels)
                    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, aspect="auto", title="Confusion matrix", labels={"x":"Predicted","y":"Actual","color":"Count"})
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Per-class precision and recall"):
                        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
                        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
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
    st.subheader("📁 Explore your dataset")
    st.write("Review column types, missing values, and summary statistics before training.")
    schema = pd.DataFrame({"Column": data.columns, "Type": [str(t) for t in data.dtypes], "Missing": [int(data[c].isna().sum()) for c in data.columns], "Unique values": [int(data[c].nunique(dropna=True)) for c in data.columns]})
    st.dataframe(schema, use_container_width=True, hide_index=True)
    with st.expander("Summary statistics"):
        st.dataframe(data.describe(include="all").transpose(), use_container_width=True)
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download current dataset as CSV", csv, file_name="student_dataset.csv", mime="text/csv")
    st.caption("The built-in dataset is synthetic and exists for demonstration only. Replace it with an appropriately licensed, documented dataset for academic evaluation.")
