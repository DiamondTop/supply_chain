import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import StringIO
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Engine Overhaul Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Engine Overhaul Prediction Dashboard")
st.markdown("""
This dashboard predicts time until first engine overhaul using linear regression
based on 4 key factors: annual miles driven, average load weight, average driving speed, and oil change intervals.
""")

# Sidebar for API key and file upload
with st.sidebar:
    st.header("Configuration")

    # API Key input
    st.subheader("OpenAI API Key")
    api_key = st.text_input(
        "Enter your OpenAI API key",
        type="password",
        help="Required for AI interpretation of results"
    )

    client = None  # FIX: initialise client to None so it's always defined
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            st.success("API key validated")
        except Exception as e:
            st.error(f"Invalid API key: {e}")
            api_key = None

    st.divider()

    # File upload
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


# FIX: Accept bytes instead of UploadedFile so @st.cache_data can hash the argument
@st.cache_data
def load_data(file_bytes: bytes | None):
    if file_bytes is not None:
        try:
            df = pd.read_csv(StringIO(file_bytes.decode("utf-8")), sep='\t', engine='python')
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    else:
        # FIX: clean sample data – no leading index column, consistent tab separation
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

    # Remove accidental leading index column
    if df.columns[0] in ['', 'Unnamed: 0']:
        df = df.iloc[:, 1:]

    # Coerce all required columns to numeric
    for col in ['Time', 'Miles', 'Weight', 'Speed', 'Oil']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.dropna()


# FIX: pass bytes (hashable) to the cached function
file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
df = load_data(file_bytes)

if df is not None:
    required_cols = ['Time', 'Miles', 'Weight', 'Speed', 'Oil']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # ------------------------------------------------------------------ #
    # FIX: Fit model OUTSIDE tabs so all tabs can access these variables  #
    # ------------------------------------------------------------------ #
    y = df['Time']
    X = df[['Miles', 'Weight', 'Speed', 'Oil']]

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    y_pred = model.predict(X)
    mae = pd.Series(y - y_pred).abs().mean()

    equation = f"Time = {model.intercept_:.3f} "
    for var, coef in zip(X.columns, model.coef_):
        sign = "+" if coef >= 0 else "-"
        equation += f"{sign} {abs(coef):.3f} \\cdot {var} "

    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_,
        'Impact': ['Negative' if c < 0 else 'Positive' for c in model.coef_]
    })

    # Overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Trucks in Dataset", len(df))
    with col2:
        st.metric("Average Time Until Overhaul", f"{df['Time'].mean():.1f} units")
    with col3:
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

        st.markdown("**Interpretation Guide:**")
        st.markdown("""
- **R-squared**: % of variance explained by model
- **Positive coefficient**: Feature increases time until overhaul
- **Negative coefficient**: Feature decreases time until overhaul
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
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        residuals = y - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Predicted Values')
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

    # ------------------------------------------------------------------ #
    # Tab 3 – Make Prediction                                             #
    # ------------------------------------------------------------------ #
    with tab3:
        st.subheader("Predict for New Truck")
        st.markdown("Enter values for a new truck to predict time until first engine overhaul:")

        col1, col2 = st.columns(2)
        with col1:
            miles_input = st.number_input("Annual Miles Driven", min_value=0.0, max_value=500000.0, value=50.0, step=1.0)
            weight_input = st.number_input("Average Load Weight (tons)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
        with col2:
            speed_input = st.number_input("Average Driving Speed (mph)", min_value=0.0, max_value=150.0, value=55.0, step=1.0)
            oil_input = st.number_input("Oil Change Interval (k miles)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)

        if st.button("Predict Time Until Overhaul", type="primary", use_container_width=True):
            new_data = pd.DataFrame({
                'Miles': [miles_input],
                'Weight': [weight_input],
                'Speed': [speed_input],
                'Oil': [oil_input]
            })

            try:
                prediction = model.predict(new_data)[0]
                st.success(f"### Predicted Time: {prediction:.2f} units")

                contributions = []
                for i, (feature, value) in enumerate(zip(X.columns, new_data.iloc[0])):
                    effect = model.coef_[i] * value
                    contributions.append({
                        'Feature': feature,
                        'Value': value,
                        'Coefficient': model.coef_[i],
                        'Contribution': effect
                    })

                contrib_df = pd.DataFrame(contributions)
                total = contrib_df['Contribution'].sum() + model.intercept_
                contrib_df['% of Total'] = (contrib_df['Contribution'] / total * 100).round(2)
                st.dataframe(contrib_df, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ------------------------------------------------------------------ #
    # Tab 4 – AI Interpretation                                           #
    # ------------------------------------------------------------------ #
    with tab4:
        st.subheader("AI-Powered Analysis")

        if not api_key or client is None:
            st.warning("Please enter your OpenAI API key in the sidebar to enable AI interpretation.")
        else:
            if st.button("Generate AI Analysis", type="primary"):
                with st.spinner("Generating AI analysis..."):
                    try:
                        context = f"""
Linear Regression Model for Engine Overhaul Prediction

Dataset: {len(df)} trucks
Features: Miles (annual miles), Weight (load tons), Speed (mph), Oil (oil change interval in 1k miles)
Target: Time until first engine overhaul (units)

Model Performance:
- R-squared: {r2:.4f} ({r2:.1%} of variance explained)
- Mean Absolute Error: {mae:.3f}

Coefficients:
{coef_df.to_string(index=False)}

Equation: {equation}

Data Sample:
{df.head().to_string()}

Please provide:
1. Business interpretation of each coefficient
2. Assessment of model reliability given sample size
3. Practical recommendations for fleet management
4. Potential limitations and improvements
5. How to use these predictions for preventive maintenance scheduling
"""
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a senior data scientist specialising in predictive maintenance for transportation fleets."},
                                {"role": "user", "content": context}
                            ],
                            temperature=0.7,
                            max_tokens=800
                        )

                        ai_response = response.choices[0].message.content
                        st.markdown(ai_response)

                        st.download_button(
                            label="Download AI Analysis",
                            data=ai_response,
                            file_name=f"engine_overhaul_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )

                    except Exception as e:
                        st.error(f"AI service error: {e}")
                        st.code(str(e), language="text")

else:
    st.info("Please upload a data file to begin analysis, or the sample data will be used automatically.")
