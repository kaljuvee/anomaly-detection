import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pycaret.datasets import get_data
import io

# Configure page
st.set_page_config(
    page_title="Data Explorer - Anomaly Detection",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load PyCaret sample anomaly dataset"""
    try:
        data = get_data('anomaly')
        return data
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None

def analyze_data(df):
    """Perform basic data analysis"""
    analysis = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'statistics': df.describe()
    }
    return analysis

def create_correlation_heatmap(df):
    """Create correlation heatmap for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Feature Correlation Matrix",
                       color_continuous_scale="RdBu_r")
        fig.update_layout(height=600)
        return fig
    return None

def create_distribution_plots(df):
    """Create distribution plots for all numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return None
    
    # Calculate subplot dimensions
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=numeric_cols,
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols + 1
        col_pos = i % n_cols + 1
        
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        height=300 * n_rows,
        title_text="Feature Distributions",
        title_x=0.5
    )
    
    return fig

def main():
    st.title("📊 Data Explorer")
    st.markdown("Upload your dataset or explore the sample data to understand its structure and characteristics.")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    
    # Data source selection
    st.markdown("## 📁 Data Source")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="upload-section">
            <h4>📤 Upload Your Data</h4>
            <p>Upload a CSV file to analyze your own dataset</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with numerical features for anomaly detection"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.session_state.data_source = "uploaded"
                st.success(f"✅ Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with col2:
        st.markdown("""
        <div class="upload-section">
            <h4>🎯 Use Sample Data</h4>
            <p>Explore with PyCaret's built-in anomaly dataset</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Load Sample Dataset", type="primary"):
            with st.spinner("Loading sample data..."):
                sample_data = load_sample_data()
                if sample_data is not None:
                    st.session_state.data = sample_data
                    st.session_state.data_source = "sample"
                    st.success(f"✅ Successfully loaded sample dataset with {sample_data.shape[0]} rows and {sample_data.shape[1]} columns")
    
    # Data analysis section
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.markdown("---")
        st.markdown("## 🔍 Data Analysis")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>📏 Rows</h4>
                <h2>{:,}</h2>
            </div>
            """.format(df.shape[0]), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>📊 Columns</h4>
                <h2>{}</h2>
            </div>
            """.format(df.shape[1]), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>🔢 Numeric</h4>
                <h2>{}</h2>
            </div>
            """.format(len(df.select_dtypes(include=[np.number]).columns)), unsafe_allow_html=True)
        
        with col4:
            missing_count = df.isnull().sum().sum()
            st.markdown("""
            <div class="metric-card">
                <h4>❓ Missing</h4>
                <h2>{}</h2>
            </div>
            """.format(missing_count), unsafe_allow_html=True)
        
        # Data preview
        st.markdown("### 👀 Data Preview")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.dataframe(df.head(20), use_container_width=True)
        
        with col2:
            st.markdown("**Column Information:**")
            for col in df.columns:
                dtype = str(df[col].dtype)
                missing = df[col].isnull().sum()
                st.text(f"{col}: {dtype}")
                if missing > 0:
                    st.text(f"  Missing: {missing}")
        
        # Statistical summary
        st.markdown("### 📈 Statistical Summary")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        else:
            st.warning("No numeric columns found for statistical analysis.")
        
        # Visualizations
        st.markdown("---")
        st.markdown("## 📊 Data Visualizations")
        
        # Distribution plots
        if len(numeric_cols) > 0:
            st.markdown("### 📊 Feature Distributions")
            dist_fig = create_distribution_plots(df)
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)
            
            # Correlation heatmap
            if len(numeric_cols) > 1:
                st.markdown("### 🔥 Correlation Heatmap")
                corr_fig = create_correlation_heatmap(df)
                if corr_fig:
                    st.plotly_chart(corr_fig, use_container_width=True)
            
            # Box plots for outlier detection
            st.markdown("### 📦 Box Plots (Outlier Detection)")
            
            selected_features = st.multiselect(
                "Select features to visualize:",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if selected_features:
                # Normalize data for better visualization
                df_normalized = df[selected_features].copy()
                for col in selected_features:
                    df_normalized[col] = (df_normalized[col] - df_normalized[col].mean()) / df_normalized[col].std()
                
                fig = px.box(df_normalized.melt(), 
                           x='variable', 
                           y='value',
                           title="Box Plots (Normalized Values)",
                           labels={'variable': 'Features', 'value': 'Normalized Values'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        # Data quality assessment
        st.markdown("---")
        st.markdown("## ✅ Data Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Readiness for Anomaly Detection")
            
            checks = []
            
            # Check for numeric data
            if len(numeric_cols) > 0:
                checks.append("✅ Contains numeric features")
            else:
                checks.append("❌ No numeric features found")
            
            # Check for missing values
            if df.isnull().sum().sum() == 0:
                checks.append("✅ No missing values")
            else:
                checks.append(f"⚠️ {df.isnull().sum().sum()} missing values found")
            
            # Check data size
            if df.shape[0] >= 100:
                checks.append("✅ Sufficient data size")
            else:
                checks.append("⚠️ Small dataset (< 100 rows)")
            
            # Check for constant features
            constant_features = [col for col in numeric_cols if df[col].nunique() <= 1]
            if len(constant_features) == 0:
                checks.append("✅ No constant features")
            else:
                checks.append(f"⚠️ {len(constant_features)} constant features")
            
            for check in checks:
                st.markdown(check)
        
        with col2:
            st.markdown("### 💡 Recommendations")
            
            recommendations = []
            
            if df.isnull().sum().sum() > 0:
                recommendations.append("Consider handling missing values before training")
            
            if len(constant_features) > 0:
                recommendations.append("Remove constant features to improve model performance")
            
            if df.shape[0] < 1000:
                recommendations.append("Consider collecting more data for better anomaly detection")
            
            if len(numeric_cols) < 2:
                recommendations.append("Anomaly detection works better with multiple features")
            
            if not recommendations:
                recommendations.append("✅ Data looks good for anomaly detection!")
            
            for rec in recommendations:
                st.markdown(f"• {rec}")
        
        # Export options
        st.markdown("---")
        st.markdown("## 📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download processed data
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name="anomaly_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download data summary
            analysis = analyze_data(df)
            summary_text = f"""
Data Summary Report
==================

Dataset Shape: {analysis['shape'][0]} rows × {analysis['shape'][1]} columns
Numeric Features: {len(analysis['numeric_columns'])}
Categorical Features: {len(analysis['categorical_columns'])}
Missing Values: {sum(analysis['missing_values'].values())}

Columns:
{chr(10).join([f"- {col}: {dtype}" for col, dtype in analysis['dtypes'].items()])}

Statistical Summary:
{analysis['statistics'].to_string()}
            """
            
            st.download_button(
                label="📊 Download Summary Report",
                data=summary_text,
                file_name="data_summary.txt",
                mime="text/plain"
            )
    
    else:
        st.info("👆 Please upload a dataset or load the sample data to begin exploration.")

if __name__ == "__main__":
    main()

