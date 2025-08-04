import streamlit as st
import pandas as pd
import plotly.express as px
from pycaret.datasets import get_data

# Configure page
st.set_page_config(
    page_title="Anomaly Detection Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🔍 Anomaly Detection Demo</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    Welcome to the **PyCaret Anomaly Detection Interactive Demo**! This application demonstrates 
    various anomaly detection algorithms using PyCaret's powerful machine learning library.
    
    Anomaly detection is used to identify rare items, events, or observations that raise suspicions 
    by differing significantly from the majority of the data.
    """)
    
    # Feature overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Key Features</h3>
            <ul>
                <li><strong>12 Algorithms</strong> - Compare different anomaly detection methods</li>
                <li><strong>Interactive Visualizations</strong> - Explore data and results</li>
                <li><strong>Real-time Training</strong> - Train models with live progress</li>
                <li><strong>Model Comparison</strong> - Side-by-side algorithm analysis</li>
                <li><strong>Export Results</strong> - Download predictions and models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔬 Available Algorithms</h3>
            <ul>
                <li><strong>Isolation Forest</strong> - Tree-based anomaly detection</li>
                <li><strong>Local Outlier Factor</strong> - Density-based detection</li>
                <li><strong>One-Class SVM</strong> - Support vector machines</li>
                <li><strong>PCA</strong> - Principal component analysis</li>
                <li><strong>K-Nearest Neighbors</strong> - Distance-based detection</li>
                <li><strong>And 7 more algorithms!</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start section
    st.markdown("---")
    st.markdown("## 🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>1️⃣ Load Data</h4>
            <p>Upload your dataset or use our sample data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>2️⃣ Choose Algorithm</h4>
            <p>Select from 12 different anomaly detection methods</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>3️⃣ Analyze Results</h4>
            <p>Visualize anomalies and export findings</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample data preview
    st.markdown("---")
    st.markdown("## 📊 Sample Dataset Preview")
    
    try:
        # Load sample data
        with st.spinner("Loading sample dataset..."):
            data = get_data('anomaly')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Dataset Overview:**")
            st.dataframe(data.head(10), use_container_width=True)
        
        with col2:
            st.markdown("**Dataset Statistics:**")
            st.metric("Total Rows", f"{data.shape[0]:,}")
            st.metric("Features", data.shape[1])
            st.metric("Data Type", "Numerical")
            st.metric("Missing Values", data.isnull().sum().sum())
        
        # Quick visualization
        st.markdown("**Feature Distribution:**")
        fig = px.histogram(data.melt(), x='value', color='variable', 
                          title="Distribution of All Features",
                          labels={'value': 'Feature Value', 'variable': 'Feature'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        st.info("Please check your internet connection and try again.")
    
    # Navigation guide
    st.markdown("---")
    st.markdown("## 🧭 Navigation Guide")
    
    st.markdown("""
    Use the sidebar to navigate through different sections:
    
    - **🏠 Home** - Overview and introduction (current page)
    - **📊 Data Explorer** - Upload and explore your data
    - **🤖 Model Training** - Train anomaly detection models
    - **📈 Visualizations** - Interactive plots and analysis
    - **⚖️ Model Comparison** - Compare multiple algorithms
    - **📥 Export Results** - Download predictions and models
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built with ❤️ using <strong>Streamlit</strong> and <strong>PyCaret</strong></p>
        <p>Explore the power of automated machine learning for anomaly detection!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

