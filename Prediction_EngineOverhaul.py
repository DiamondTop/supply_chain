import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import StringIO
from openai import OpenAI          # OpenRouter is OpenAI-compatible
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------ #
# API CLIENT SETUP — OpenRouter with Arcee-AI                        #
# OpenRouter uses the same OpenAI SDK, just a different base_url     #
#                                                                     #
# Streamlit Cloud → App Settings → Secrets, add:                     #
#   OPENROUTER_API_KEY = "sk-or-..."                                  #
# ------------------------------------------------------------------ #
try:
    client = OpenAI(
        api_key=st.secrets["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    ai_available = True
except Exception:
    client = None
    ai_available = False

# Model identifier for Arcee-AI
# trinity-large-preview:free = 400B MoE model, currently FREE on OpenRouter
ARCEE_MODEL = "arcee-ai/trinity-large-preview:free"

# Page configuration
st.set_page_config(
    page_title="Engine Overhaul Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Engine Overhaul Prediction Dashboard")
st.markdown("""
This dashboard predicts time until first engine overhaul using linear regression
based on 4 key factors: annual miles driven, average load weight, average driving speed, and oil change intervals.
""")

# ------------------------------------------------------------------ #
# Sidebar                                                             #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("Configuration")

    st.subheader("Data Upload")
    uploaded_file = st.file_uploader(
        "Upload engine overhaul data (tab-separated .txt)",
        type=['txt', 'csv'],
        help="File should contain columns: Time, Miles, Weight, Speed, Oil"
    )

    st.divider()
    st.markdown("""
**Sample Data Format:**
```
Time\tMiles\tWeight\tSpeed\tOil
7.9\t42.8\t19\t46\t15
0.9\t98.5\t25\t46\t29
```
""")

    st.divider()
    st.subheader("AI Status")
    if ai_available:
        st.success("Arcee-AI is ready")
    else:
        st.error("AI unavailable — check Secrets config")


# ------------------------------------------------------------------ #
# Data loading                                                        #
# ------------------------------------------------------------------ #
@st.cache_data
def load_data(file_bytes: bytes | None):
    if file_bytes is not None:
        try:
            df = pd.read_csv(StringIO(file_bytes.decode("utf-8")), sep='\t', engine='python')
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    else:
        sample_data = (
            "Time\tMiles\tWeight\tSpeed\tOil\n"
            "7.9\t42.8\t19\t46\t15\n"
            "0.9\t98.5\t25\t46\t29\n"
            "8.5\t43.4\t21\t64\t14\n"
            "1.3\t110.7\t27\t60\t26\n"
            "1.4\t102.3\t28\t51\t17\n"
            "5.2\t61.2\t22\t55\t18\n"
            "3.1\t75.4\t24\t58\t20\n"
            "6.4\t55.0\t20\t50\t16\n"
            "2.8\t88.3\t26\t62\t23\n"
            "4.7\t68.1\t23\t53\t19\n"
        )
        df = pd.read_csv(StringIO(sample_data), sep='\t', engine='python')

    if df.columns[0] in ['', 'Unnamed: 0']:
        df = df.iloc[:, 1:]

    for col in ['Time', 'Miles', 'Weight', 'Speed', 'Oil']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.dropna()


file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
df = load_data(file_bytes)

if df is not None:
    required_cols = ['Time', 'Miles', 'Weight', 'Speed', 'Oil']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # Fit model once, outside tabs
    y = df['Time']
    X = df[['Miles', 'Weight', 'Speed', 'Oil']]

    model = LinearRegression()
    model.fit(X, y)

    r2       = model.score(X, y)
    y_pred   = model.predict(X)
    mae      = pd.Series(y - y_pred).abs().mean()

    equation = f"Time = {model.intercept_:.3f} "
    for var, coef in zip(X.columns, model.coef_):
        sign = "+" if coef >= 0 else "-"
        equation += f"{sign} {abs(coef):.3f} \\cdot {var} "

    coef_df = pd.DataFrame({
        'Feature':     X.columns,
        'Coefficient': model.coef_,
        'Impact':      ['Negative' if c < 0 else 'Positive' for c in model.coef_]
    })

    # Overview metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Trucks in Dataset", len(df))
    with c2:
        st.metric("Average Time Until Overhaul", f"{df['Time'].mean():.1f} units")
    with c3:
        st.metric("Features Used", "4 (Miles, Weight, Speed, Oil)")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Model Summary", "Visualizations", "Make Prediction", "AI Interpretation"])

    # ------------------------------------------------------------------ #
    # Tab 1 – Model Summary                                               #
    # ------------------------------------------------------------------ #
    with tab1:
        st.subheader("Regression Equation")
        st.latex(equation)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Coefficients")
            st.dataframe(coef_df, use_container_width=True)
        with col2:
            st.subheader("Model Performance")
            st.metric("R-squared (Explained Variance)", f"{r2:.4f}")
            st.metric("Mean Absolute Error", f"{mae:.3f}")
            st.progress(float(r2), text=f"Accuracy: {r2:.1%}")

        st.markdown("**How to Read These Results:**")
        if r2 >= 0.7:
            model_health = "🟢 Strong — this model is reliable for maintenance planning"
        elif r2 >= 0.4:
            model_health = "🟡 Moderate — use as a guide, not a guarantee"
        else:
            model_health = "🔴 Weak — collect more data before relying on predictions"

        st.markdown(f"""
| Metric | What It Means for Your Fleet |
|---|---|
| **Model Accuracy ({r2:.1%})** | {model_health} |
| **Mean Absolute Error ({mae:.2f} units)** | On average, predictions are off by **{mae:.2f} time units** — factor this buffer into your scheduling |
| **Positive coefficient (+)** | 🔼 That factor **extends** engine life — maintain or improve it |
| **Negative coefficient (−)** | 🔽 That factor **shortens** engine life — reduce or monitor it closely |

> 💡 **Practical Takeaway:** Focus maintenance resources on features with the **largest negative
> coefficients** — these are your biggest risk drivers. Features with positive coefficients are
> protective and should be maintained at current levels or improved where cost-effective.
""")

    # ------------------------------------------------------------------ #
    # Tab 2 – Visualizations                                              #
    # ------------------------------------------------------------------ #
    with tab2:
        st.subheader("Actual vs Predicted Time")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y, y_pred, alpha=0.6, edgecolors='k', s=80)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Time Until Overhaul')
        ax.set_ylabel('Predicted Time Until Overhaul')
        ax.set_title('Model Predictions vs Actual Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Residual Analysis")

        import numpy as np

        residuals      = y - y_pred
        residual_std   = residuals.std()
        warn_thresh    = 1.0 * residual_std
        outlier_thresh = 2.0 * residual_std

        fig2, ax2 = plt.subplots(figsize=(9, 5))

        # Assign a colour string per point using c= (matplotlib-compatible)
        color_list = []
        for r in residuals:
            if abs(r) <= warn_thresh:
                color_list.append('#2ecc71')
            elif abs(r) <= outlier_thresh:
                color_list.append('#f39c12')
            else:
                color_list.append('#e74c3c')

        ax2.scatter(
            y_pred, residuals,
            c=color_list, edgecolors='k', linewidths=0.5,
            alpha=0.85, s=90
        )

        # Manual legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markeredgecolor='k', markersize=9, label='🟢 Normal (within 1 std dev) — reliable'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12', markeredgecolor='k', markersize=9, label='🟡 Monitor (1–2 std devs) — slight deviation'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markeredgecolor='k', markersize=9, label='🔴 Outlier (>2 std devs) — flag for inspection'),
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

        ax2.axhline(y=0,                color='#2c3e50', linestyle='--', lw=1.5)
        ax2.axhline(y= warn_thresh,     color='#f39c12', linestyle=':',  lw=1.2, alpha=0.7)
        ax2.axhline(y=-warn_thresh,     color='#f39c12', linestyle=':',  lw=1.2, alpha=0.7)
        ax2.axhline(y= outlier_thresh,  color='#e74c3c', linestyle=':',  lw=1.2, alpha=0.7)
        ax2.axhline(y=-outlier_thresh,  color='#e74c3c', linestyle=':',  lw=1.2, alpha=0.7)

        ax2.axhspan(-warn_thresh,    warn_thresh,                      alpha=0.06, color='#2ecc71')
        ax2.axhspan( warn_thresh,    outlier_thresh,                   alpha=0.06, color='#f39c12')
        ax2.axhspan(-outlier_thresh,-warn_thresh,                      alpha=0.06, color='#f39c12')
        ax2.axhspan( outlier_thresh, residuals.max() + 0.5,            alpha=0.06, color='#e74c3c')
        ax2.axhspan( residuals.min() - 0.5, -outlier_thresh,           alpha=0.06, color='#e74c3c')

        ax2.set_xlabel('Predicted Values', fontsize=11)
        ax2.set_ylabel('Residuals',        fontsize=11)
        ax2.set_title('Residuals vs Predicted Values — Colour-Coded by Severity', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("**How to Read This Chart:**")
        st.markdown("""
| What You See | What It Means Operationally |
|---|---|
| 🟢 **Random scatter around zero** | Model is trustworthy across all truck types |
| 🟡 **Pattern / curve in the dots** | Some truck categories are being mis-scheduled — investigate which ones |
| 🟡 **Widening spread left to right** | Add a larger time buffer for trucks in the high-risk range |
| 🔴 **Outlier dots far from zero** | Specific trucks behaving unexpectedly — flag for manual inspection |
""")

        # Dynamic residual interpretation
        st.markdown("**What Your Residual Chart Is Telling You:**")

        residual_mean = residuals.mean()
        outliers      = (residuals.abs() > outlier_thresh).sum()
        outlier_pct   = outliers / len(residuals) * 100

        median_pred   = y_pred.mean()
        low_mask      = y_pred <= median_pred
        high_mask     = y_pred > median_pred
        spread_low    = residuals[low_mask].std()
        spread_high   = residuals[high_mask].std()
        spread_ratio  = spread_high / spread_low if spread_low > 0 else 1

        resid_sq_corr = pd.Series(y_pred).corr(pd.Series(residuals ** 2))

        flags = []

        if abs(residual_mean) < 0.2 * residual_std:
            flags.append("✅ **No systematic bias detected** — the model is not consistently over- or under-predicting overhaul times across your fleet.")
        elif residual_mean > 0:
            flags.append(f"⚠️ **Slight under-prediction bias** (mean residual = +{residual_mean:.2f}) — the model tends to predict overhaul *earlier* than it actually occurs. Consider scheduling slightly later than the model suggests.")
        else:
            flags.append(f"⚠️ **Slight over-prediction bias** (mean residual = {residual_mean:.2f}) — the model tends to predict overhaul *later* than it actually occurs. Build in earlier maintenance checks to be safe.")

        if spread_ratio > 1.5:
            flags.append(f"⚠️ **Widening spread detected** (spread ratio = {spread_ratio:.1f}x) — predictions become less reliable for trucks with longer predicted overhaul times. Add a larger safety buffer when scheduling these higher-endurance vehicles.")
        else:
            flags.append(f"✅ **Consistent spread** (spread ratio = {spread_ratio:.1f}x) — prediction reliability is even across all truck types, so the same scheduling buffer can be applied fleet-wide.")

        if abs(resid_sq_corr) > 0.3:
            flags.append(f"⚠️ **Possible non-linear pattern detected** (correlation = {resid_sq_corr:.2f}) — some truck categories may be systematically mis-scheduled. Consider grouping trucks by load weight or mileage for separate analysis.")
        else:
            flags.append(f"✅ **No significant pattern detected** (correlation = {resid_sq_corr:.2f}) — residuals are randomly scattered, meaning the linear model is a good fit for your data.")

        if outliers == 0:
            flags.append("✅ **No outliers detected** — all trucks are behaving consistently with the model's expectations.")
        elif outlier_pct <= 10:
            flags.append(f"🔍 **{outliers} truck(s) flagged as outliers** ({outlier_pct:.0f}% of fleet) — a small number of vehicles are behaving unexpectedly. Review their maintenance history or operating conditions for anomalies.")
        else:
            flags.append(f"🔴 **{outliers} trucks flagged as outliers** ({outlier_pct:.0f}% of fleet) — a significant portion of your fleet is not well-captured by this model. The dataset may need more variables (e.g. driver behaviour, route type) to improve accuracy.")

        for flag in flags:
            st.markdown(f"- {flag}")

    # ------------------------------------------------------------------ #
    # Tab 3 – Make Prediction                                             #
    # ------------------------------------------------------------------ #
    with tab3:
        st.subheader("Predict for New Truck")
        st.markdown("Enter values for a new truck to predict time until first engine overhaul:")

        col1, col2 = st.columns(2)
        with col1:
            miles_input  = st.number_input("Annual Miles Driven",          min_value=0.0, max_value=500000.0, value=50.0,  step=1.0)
            weight_input = st.number_input("Average Load Weight (tons)",    min_value=0.0, max_value=100.0,    value=20.0,  step=0.1)
        with col2:
            speed_input  = st.number_input("Average Driving Speed (mph)",   min_value=0.0, max_value=150.0,    value=55.0,  step=1.0)
            oil_input    = st.number_input("Oil Change Interval (k miles)", min_value=0.0, max_value=50.0,     value=15.0,  step=0.1)

        if st.button("Predict Time Until Overhaul", type="primary", use_container_width=True):
            new_data = pd.DataFrame({
                'Miles':  [miles_input],
                'Weight': [weight_input],
                'Speed':  [speed_input],
                'Oil':    [oil_input]
            })
            try:
                prediction = model.predict(new_data)[0]
                st.success(f"### Predicted Time: {prediction:.2f} units")

                contributions = []
                for i, (feature, value) in enumerate(zip(X.columns, new_data.iloc[0])):
                    contributions.append({
                        'Feature':      feature,
                        'Value':        value,
                        'Coefficient':  model.coef_[i],
                        'Contribution': model.coef_[i] * value
                    })
                contrib_df = pd.DataFrame(contributions)
                total = contrib_df['Contribution'].sum() + model.intercept_
                contrib_df['% of Total'] = (contrib_df['Contribution'] / total * 100).round(2)
                st.dataframe(contrib_df, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ------------------------------------------------------------------ #
    # Tab 4 – AI Interpretation — Business Report Format                 #
    # ------------------------------------------------------------------ #
    with tab4:
        st.subheader("📋 Fleet Maintenance Intelligence Report")
        st.caption("Powered by Arcee-AI via OpenRouter — Business Analysis Edition")

        if not ai_available:
            st.error("""
**Arcee-AI is not configured.** To enable it:
1. Sign up at [openrouter.ai](https://openrouter.ai) and copy your API key
2. Go to your Streamlit Cloud app → **Settings → Secrets**
3. Add the following and save:
```toml
OPENROUTER_API_KEY = "sk-or-your-key-here"
```
4. Reboot the app — AI status in the sidebar will turn green
""")
        else:
            st.markdown("""
This report translates the statistical model results into **plain business language**,
covering what the data means for your operations, where the risks are, and what actions to take.
""")

            if st.button("Generate Business Intelligence Report", type="primary", use_container_width=True):
                with st.spinner("Arcee-AI is preparing your business report..."):
                    try:
                        # Identify highest-risk factor (largest negative coefficient)
                        coef_series   = pd.Series(model.coef_, index=X.columns)
                        top_risk      = coef_series.idxmin()
                        top_risk_coef = coef_series.min()
                        top_positive  = coef_series.idxmax()

                        business_prompt = f"""
You are a Fleet Operations Consultant preparing a formal business intelligence report
for a transport company's senior management team. Write in clear, professional business
language — avoid statistical jargon. Use section headers, bullet points, and concrete
operational recommendations.

The company has built a predictive model to forecast when truck engines will require
their first major overhaul. Here is the model data:

FLEET OVERVIEW
- Fleet size analysed: {len(df)} trucks
- Average time until overhaul: {df['Time'].mean():.1f} units
- Factors monitored: Annual mileage, Load weight, Driving speed, Oil change frequency

MODEL RELIABILITY SCORE
- Accuracy (R-squared): {r2:.1%}
- Average prediction error: ±{mae:.2f} time units
- Status: {"RELIABLE — suitable for operational planning" if r2 >= 0.7 else "MODERATE — use as indicative guidance only" if r2 >= 0.4 else "LOW — additional data collection recommended"}

FACTOR IMPACT ON ENGINE LIFE
{coef_df.to_string(index=False)}
(Negative coefficient = shortens engine life | Positive = extends engine life)

HIGHEST RISK FACTOR: {top_risk} (coefficient: {top_risk_coef:.4f})
MOST PROTECTIVE FACTOR: {top_positive} (coefficient: {coef_series[top_positive]:.4f})

MODEL EQUATION: {equation}

SAMPLE DATA (first 5 trucks):
{df.head().to_string()}

---

Please write a business intelligence report with EXACTLY these 5 sections:

## 1. Executive Summary
2-3 sentences. What is the model telling us overall? Is the fleet at risk? Is the model trustworthy?
Mention the single most important finding a CEO should know.

## 2. Key Risk Drivers
For each of the 4 factors (Miles, Weight, Speed, Oil), write ONE bullet point explaining:
- What the factor means in plain English
- Whether it helps or hurts engine life
- A specific operational recommendation (e.g. "Reduce average load by X" or "Shorten oil change intervals")
Rank them from highest risk to lowest risk impact.

## 3. Model Confidence & Reliability
In 2-3 sentences, tell the operations manager how much they can trust these predictions.
Mention the ±{mae:.2f} unit error margin in practical terms (e.g. "this means scheduling
windows should include a X-unit buffer").
Flag any data limitations given the fleet sample size of {len(df)} trucks.

## 4. Immediate Action Plan
Provide a prioritised 3-step action plan the fleet manager can act on this week.
Format as numbered steps with clear owners (e.g. "Fleet Manager", "Drivers", "Maintenance Team").

## 5. Long-Term Strategic Recommendations
2-3 bullet points on how to improve prediction accuracy and fleet maintenance strategy
over the next 6-12 months. Focus on data collection, process changes, and cost savings.
"""

                        response = client.chat.completions.create(
                            model=ARCEE_MODEL,
                            messages=[
                                {"role": "user", "content": business_prompt}
                            ],
                            max_tokens=1500,
                            temperature=0.5,       # lower = more consistent, professional tone
                            extra_headers={
                                "HTTP-Referer": "https://streamlit.io",
                                "X-Title":      "Engine Overhaul Dashboard",
                            }
                        )

                        ai_response = response.choices[0].message.content

                        # Display with a styled container
                        st.divider()
                        st.markdown(ai_response)
                        st.divider()

                        # Download as formatted report
                        report_text = f"""FLEET MAINTENANCE INTELLIGENCE REPORT
Generated: {pd.Timestamp.now().strftime('%d %B %Y')}
Model: Arcee-AI via OpenRouter
Fleet Size: {len(df)} trucks | Model Accuracy: {r2:.1%} | Avg Error: ±{mae:.2f} units
{'='*60}

{ai_response}

{'='*60}
END OF REPORT
"""
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.download_button(
                                label="📥 Download Report (.txt)",
                                data=report_text,
                                file_name=f"fleet_maintenance_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        with col_b:
                            st.download_button(
                                label="📥 Download Report (.md)",
                                data=report_text,
                                file_name=f"fleet_maintenance_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
                                mime="text/markdown",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"Arcee-AI error: {e}")
                        st.code(str(e), language="text")

else:
    st.info("Please upload a data file to begin analysis, or the sample data will be used automatically.")
