import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create cache directory
cache_dir = Path("./cache")
cache_dir.mkdir(exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="🍷 Red Wine Quality Predictor & Analytics",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (adapted from real estate app)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #F3F4F6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
        border-bottom: 2px solid #2563EB;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Theme toggle
st.sidebar.markdown('<div style="text-align: center; padding: 12px 0;"><h2 style="color: #1E3A8A;">🍷 Wine Quality Analytics</h2><p style="font-size: 0.9rem; color: #6B7280;">Advanced Wine Quality Analysis & Predictions</p><hr style="margin: 10px 0;"></div>', unsafe_allow_html=True)
theme = st.sidebar.selectbox("Interface Theme", ["Light", "Dark", "Modern Blue"])
if theme == "Dark":
    st.markdown("""
    <style>
    body { background-color: #111827; color: #F9FAFB; }
    .stApp { background-color: #111827; }
    .card { background-color: #1F2937; }
    .metric-card { background-color: #111827; border-left: 4px solid #3B82F6; }
    .main-header { color: #60A5FA; }
    .sub-header { color: #93C5FD; }
    .stTabs [data-baseweb="tab"] { background-color: #374151; }
    .stTabs [aria-selected="true"] { background-color: #1F2937; border-bottom: 2px solid #3B82F6; }
    </style>
    """, unsafe_allow_html=True)
elif theme == "Modern Blue":
    st.markdown("""
    <style>
    body { background-color: #F0F9FF; color: #0F172A; }
    .stApp { background-color: #F0F9FF; }
    .card { background-color: #FFFFFF; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); }
    .metric-card { background-color: #DBEAFE; border-left: 4px solid #2563EB; }
    .main-header { color: #1E40AF; }
    .sub-header { color: #1D4ED8; }
    .stTabs [data-baseweb="tab"] { background-color: #EFF6FF; }
    .stTabs [aria-selected="true"] { background-color: #BFDBFE; border-bottom: 2px solid #1D4ED8; }
    </style>
    """, unsafe_allow_html=True)

# Data loading and preprocessing
@st.cache_data(ttl=3600)
def load_data():
    try:
        start_time = time.time()
        processed_file = cache_dir / "processed_wine_data.pkl"
        if processed_file.exists():
            data = pd.read_pickle(processed_file)
            logger.info(f"Loaded cached data in {time.time() - start_time:.2f} seconds")
            return data

        # Load dataset (replace with actual path)
        data = pd.read_csv('winequality-red.csv')
        data.columns = [col.strip().replace(' ', '_') for col in data.columns]
        data = data.dropna()
        data['alcohol_to_acidity'] = data['alcohol'] / data['fixed_acidity']
        data['sulfur_ratio'] = data['free_sulfur_dioxide'] / data['total_sulfur_dioxide']
        data.to_pickle(processed_file)
        logger.info(f"Processed dataset with {len(data)} rows in {time.time() - start_time:.2f} seconds")
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data loading failed: {str(e)}")
        return None

# Load data first
with st.spinner('Loading and optimizing data...'):
    data = load_data()

if data is None or data.empty:
    st.markdown('<p class="error-message">No valid data available. Please check the dataset.</p>', unsafe_allow_html=True)
    st.stop()

# Sidebar filters (defined after data loading)
st.sidebar.markdown('<div class="sub-header">Filter Options</div>', unsafe_allow_html=True)
with st.sidebar.expander("Quality Filters", expanded=True):
    quality_range = st.slider("Quality Range", int(data['quality'].min()), int(data['quality'].max()), (int(data['quality'].min()), int(data['quality'].max())))
with st.sidebar.expander("Physicochemical Filters", expanded=True):
    alcohol_range = st.slider("Alcohol (%)", float(data['alcohol'].min()), float(data['alcohol'].max()), (float(data['alcohol'].min()), float(data['alcohol'].max())), step=0.1)
    volatile_acidity_range = st.slider("Volatile Acidity (g/L)", float(data['volatile_acidity'].min()), float(data['volatile_acidity'].max()), (float(data['volatile_acidity'].min()), float(data['volatile_acidity'].max())), step=0.01)
    sulphates_range = st.slider("Sulphates (g/L)", float(data['sulphates'].min()), float(data['sulphates'].max()), (float(data['sulphates'].min()), float(data['sulphates'].max())), step=0.01)
    citric_acid_range = st.slider("Citric Acid (g/L)", float(data['citric_acid'].min()), float(data['citric_acid'].max()), (float(data['citric_acid'].min()), float(data['citric_acid'].max())), step=0.01)
with st.sidebar.expander("Advanced Filters", expanded=False):
    pH_range = st.slider("pH", float(data['pH'].min()), float(data['pH'].max()), (float(data['pH'].min()), float(data['pH'].max())), step=0.01)
    density_range = st.slider("Density (g/cm³)", float(data['density'].min()), float(data['density'].max()), (float(data['density'].min()), float(data['density'].max())), step=0.0001)

if st.sidebar.button("Reset Filters", use_container_width=True):
    quality_range = (int(data['quality'].min()), int(data['quality'].max()))
    alcohol_range = (float(data['alcohol'].min()), float(data['alcohol'].max()))
    volatile_acidity_range = (float(data['volatile_acidity'].min()), float(data['volatile_acidity'].max()))
    sulphates_range = (float(data['sulphates'].min()), float(data['sulphates'].max()))
    citric_acid_range = (float(data['citric_acid'].min()), float(data['citric_acid'].max()))
    pH_range = (float(data['pH'].min()), float(data['pH'].max()))
    density_range = (float(data['density'].min()), float(data['density'].max()))

# Apply filters
if st.sidebar.button("Apply Filters", type="primary", use_container_width=True):
    with st.spinner('Applying filters...'):
        filtered_data = data[
            (data['quality'].between(quality_range[0], quality_range[1])) &
            (data['alcohol'].between(alcohol_range[0], alcohol_range[1])) &
            (data['volatile_acidity'].between(volatile_acidity_range[0], volatile_acidity_range[1])) &
            (data['sulphates'].between(sulphates_range[0], sulphates_range[1])) &
            (data['citric_acid'].between(citric_acid_range[0], citric_acid_range[1])) &
            (data['pH'].between(pH_range[0], pH_range[1])) &
            (data['density'].between(density_range[0], density_range[1]))
        ]
        if filtered_data.empty:
            st.markdown('<p class="error-message">No data matches the selected filters.</p>', unsafe_allow_html=True)
            st.stop()
        st.session_state['filtered_data'] = filtered_data
        logger.info(f"Filtered dataset to {len(filtered_data)} rows")

filtered_data = st.session_state.get('filtered_data', data)

# Model preparation
@st.cache_resource
def train_model(data):
    try:
        start_time = time.time()
        X = data.drop('quality', axis=1)
        y = data['quality']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)
        logger.info(f"Model trained in {time.time() - start_time:.2f} seconds")
        return model, scaler, X_train, X_test, y_train, y_test, y_pred, report, feature_importances, shap_values
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        logger.error(f"Model training failed: {str(e)}")
        return None, None, None, None, None, None, None, None, None, None

# Main page header
st.markdown('<h1 class="main-header">🍷 Red Wine Quality Predictor & Analytics</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <p style="text-align: center;">Explore insights, analysis, visualizations, and recommendations for red wine quality using advanced machine learning. Ideal for winemakers, sommeliers, and enthusiasts.</p>
</div>
""", unsafe_allow_html=True)

# Insights section
st.markdown('<div class="sub-header">Insights</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Wines", f"{len(filtered_data):,}")
col2.metric("Average Quality", f"{filtered_data['quality'].mean():.2f}")
col3.metric("Median Alcohol", f"{filtered_data['alcohol'].median():.1f}%")
col4.metric("High Quality (≥6)", f"{(filtered_data['quality'] >= 6).mean() * 100:.1f}%")

# Train model
if st.button("Train Model", type="primary"):
    with st.spinner('Training model...'):
        model, scaler, X_train, X_test, y_train, y_test, y_pred, report, feature_importances, shap_values = train_model(filtered_data)
        if model:
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            st.session_state['report'] = report
            st.session_state['feature_importances'] = feature_importances
            st.session_state['shap_values'] = shap_values
            st.success("Model trained successfully!")

# Tabs for analysis, visualizations, and recommendations
if len(filtered_data) >= 50:
    tabs = st.tabs(["📊 Market Overview", "🔮 Quality Prediction", "📉 Model Performance", "💡 Advanced Insights", "🍷 Recommendations"])

    with tabs[0]:
        st.markdown('<div class="sub-header">📊 Market Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Quality Distribution")
        chart_type = st.radio("Select Chart Type", ["Histogram", "Box Plot", "KDE"], horizontal=True)
        if chart_type == "Histogram":
            fig = px.histogram(filtered_data, x="quality", nbins=6, title="Quality Distribution")
            fig.update_layout(xaxis_title="Quality Score", yaxis_title="Count", height=500)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Box Plot":
            fig = px.box(filtered_data, y="quality", x="alcohol", title="Quality by Alcohol Content")
            fig.update_layout(xaxis_title="Alcohol (%)", yaxis_title="Quality Score", height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            import seaborn as sns
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.kdeplot(filtered_data['quality'], ax=ax, fill=True, color="#2563EB")
            ax.set_title("Quality Density Distribution")
            ax.set_xlabel("Quality Score")
            ax.set_ylabel("Density")
            st.pyplot(fig)
            plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sub-header">Feature Analysis</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            feature = st.selectbox("Select Feature for Analysis", ['alcohol', 'volatile_acidity', 'sulphates', 'citric_acid'])
            fig = px.box(filtered_data, x="quality", y=feature, title=f"{feature.replace('_', ' ').title()} by Quality")
            fig.update_layout(xaxis_title="Quality Score", yaxis_title=feature.replace('_', ' ').title(), height=500)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Key Feature Insights**")
            high_quality = filtered_data[filtered_data['quality'] >= 6]
            low_quality = filtered_data[filtered_data['quality'] <= 4]
            st.markdown(f"• Avg {feature.replace('_', ' ').title()} (High Quality): {high_quality[feature].mean():.2f}")
            st.markdown(f"• Avg {feature.replace('_', ' ').title()} (Low Quality): {low_quality[feature].mean():.2f}")
            st.markdown(f"• Range: {filtered_data[feature].min():.2f} - {filtered_data[feature].max():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="sub-header">🔮 Quality Prediction</div>', unsafe_allow_html=True)
        if 'model' in st.session_state:
            st.markdown(f'<div class="card"><p>Using Random Forest Classifier with Accuracy: {st.session_state["report"]["accuracy"]*100:.1f}%</p></div>', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Predict Wine Quality")
                with st.form("wine_quality_form"):
                    fixed_acidity = st.number_input("Fixed Acidity (g/L)", min_value=0.0, max_value=20.0, value=float(filtered_data['fixed_acidity'].median()), step=0.1)
                    volatile_acidity = st.number_input("Volatile Acidity (g/L)", min_value=0.0, max_value=2.0, value=float(filtered_data['volatile_acidity'].median()), step=0.01)
                    citric_acid = st.number_input("Citric Acid (g/L)", min_value=0.0, max_value=1.0, value=float(filtered_data['citric_acid'].median()), step=0.01)
                    residual_sugar = st.number_input("Residual Sugar (g/L)", min_value=0.0, max_value=20.0, value=float(filtered_data['residual_sugar'].median()), step=0.1)
                    chlorides = st.number_input("Chlorides (g/L)", min_value=0.0, max_value=0.5, value=float(filtered_data['chlorides'].median()), step=0.001)
                    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide (mg/L)", min_value=0.0, max_value=100.0, value=float(filtered_data['free_sulfur_dioxide'].median()), step=1.0)
                    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide (mg/L)", min_value=0.0, max_value=300.0, value=float(filtered_data['total_sulfur_dioxide'].median()), step=1.0)
                    density = st.number_input("Density (g/cm³)", min_value=0.9, max_value=1.1, value=float(filtered_data['density'].median()), step=0.0001)
                    pH = st.number_input("pH", min_value=2.0, max_value=4.5, value=float(filtered_data['pH'].median()), step=0.01)
                    sulphates = st.number_input("Sulphates (g/L)", min_value=0.0, max_value=2.0, value=float(filtered_data['sulphates'].median()), step=0.01)
                    alcohol = st.number_input("Alcohol (%)", min_value=0.0, max_value=20.0, value=float(filtered_data['alcohol'].median()), step=0.1)
                    submit_button = st.form_submit_button("Predict Quality")
                if submit_button and 'model' in st.session_state:
                    sample = pd.DataFrame({
                        'fixed_acidity': [fixed_acidity], 'volatile_acidity': [volatile_acidity], 'citric_acid': [citric_acid],
                        'residual_sugar': [residual_sugar], 'chlorides': [chlorides], 'free_sulfur_dioxide': [free_sulfur_dioxide],
                        'total_sulfur_dioxide': [total_sulfur_dioxide], 'density': [density], 'pH': [pH],
                        'sulphates': [sulphates], 'alcohol': [alcohol],
                        'alcohol_to_acidity': [alcohol / fixed_acidity if fixed_acidity > 0 else 0],
                        'sulfur_ratio': [free_sulfur_dioxide / total_sulfur_dioxide if total_sulfur_dioxide > 0 else 0]
                    })
                    model = st.session_state['model']
                    scaler = st.session_state['scaler']
                    prediction = model.predict(scaler.transform(sample))[0]
                    st.success(f"Predicted Quality: {prediction}")
                    comparables = filtered_data[filtered_data['quality'] == prediction]
                    if not comparables.empty:
                        st.markdown("##### Similar Wines")
                        for _, wine in comparables.head(3).iterrows():
                            st.markdown(f"• Alcohol: {wine['alcohol']:.1f}%, Sulphates: {wine['sulphates']:.2f} g/L, Volatile Acidity: {wine['volatile_acidity']:.2f} g/L")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Quality Drivers")
                if 'feature_importances' in st.session_state:
                    fig = px.bar(st.session_state['feature_importances'].head(10), x='Importance', y='Feature', orientation='h', title="Top Factors Affecting Wine Quality")
                    fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
                    st.plotly_chart(fig, use_container_width=True)
                if 'shap_values' in st.session_state:
                    st.markdown("#### SHAP Analysis")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(st.session_state['shap_values'], features=st.session_state['scaler'].transform(st.session_state['X_test']), feature_names=st.session_state['X_test'].columns, plot_type="bar", show=False)
                    st.pyplot(fig)
                    plt.close(fig)
                st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="sub-header">📉 Model Performance</div>', unsafe_allow_html=True)
        if 'report' in st.session_state:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{st.session_state['report']['accuracy']*100:.2f}%")
            with col2:
                st.metric("Macro Precision", f"{st.session_state['report']['macro avg']['precision']*100:.2f}%")
            with col3:
                st.metric("Macro Recall", f"{st.session_state['report']['macro avg']['recall']*100:.2f}%")
            with col4:
                st.metric("Macro F1-Score", f"{st.session_state['report']['macro avg']['f1-score']*100:.2f}%")
            st.markdown("#### Classification Report")
            report_df = pd.DataFrame(st.session_state['report']).T
            report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
            report_df = report_df.apply(lambda x: x.round(2) if x.dtype == "float64" else x)
            st.dataframe(report_df, hide_index=False)
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
            fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"), title="Confusion Matrix")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Prediction Distribution")
            pred_dist = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': st.session_state['y_pred']})
            fig = px.histogram(pred_dist, x=['Actual', 'Predicted'], barmode='overlay', title="Actual vs Predicted Quality Distribution")
            fig.update_layout(xaxis_title="Quality Score", yaxis_title="Count", height=500)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.markdown('<div class="sub-header">💡 Advanced Insights</div>', unsafe_allow_html=True)
        st.markdown("#### Raw Data")
        st.dataframe(filtered_data, hide_index=True, height=400)
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        csv = convert_df(filtered_data)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="wine_quality_data.csv",
            mime="text/csv",
        )
        st.markdown("#### Data Summary")
        summary = filtered_data.describe().T
        summary['count'] = summary['count'].astype(int)
        for col in ['mean', '50%', 'min', 'max']:
            if col in summary.columns:
                summary[col] = summary[col].apply(lambda x: f"{x:.2f}")
        st.dataframe(summary, height=300)
        st.markdown("#### Correlation Matrix")
        corr = filtered_data.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Feature Correlation Matrix")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.markdown('<div class="sub-header">🍷 Recommendations</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### High-Quality Wine Recommendations")
        quality_threshold = st.slider("Minimum Quality for Recommendations", 3, 8, 6)
        if 'model' in st.session_state:
            X_scaled = st.session_state['scaler'].transform(filtered_data.drop('quality', axis=1))
            predictions = st.session_state['model'].predict(X_scaled)
            filtered_data['predicted_quality'] = predictions
            recommended_wines = filtered_data[filtered_data['predicted_quality'] >= quality_threshold]
            if not recommended_wines.empty:
                st.markdown(f"**Found {len(recommended_wines)} wines with predicted quality ≥ {quality_threshold}**")
                st.dataframe(recommended_wines[['alcohol', 'sulphates', 'volatile_acidity', 'citric_acid', 'predicted_quality']].head(10), hide_index=True)
            else:
                st.warning("No wines meet the quality threshold.")
        else:
            st.warning("Please train the model to generate recommendations.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<p class="error-message">Insufficient data (less than 50 wines). Please adjust filters.</p>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p style="color: #5c5c5c;">Powered by xAI | Built with Streamlit | Data updated as of July 5, 2025</p>
</div>
""", unsafe_allow_html=True)
