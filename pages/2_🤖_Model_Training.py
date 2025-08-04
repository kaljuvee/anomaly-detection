import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pycaret.datasets import get_data
import pycaret.anomaly as anomaly
import time
import pickle
import io

# Configure page
st.set_page_config(
    page_title="Model Training - Anomaly Detection",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .algorithm-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .training-progress {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .model-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Algorithm information
ALGORITHMS = {
    'iforest': {
        'name': 'Isolation Forest',
        'description': 'Tree-based algorithm that isolates anomalies by randomly selecting features and split values',
        'best_for': 'Large datasets, high-dimensional data',
        'pros': 'Fast, scalable, handles mixed data types',
        'cons': 'May struggle with very sparse data'
    },
    'lof': {
        'name': 'Local Outlier Factor',
        'description': 'Density-based algorithm that compares local density of a point with its neighbors',
        'best_for': 'Datasets with varying densities',
        'pros': 'Good for local anomalies, interpretable',
        'cons': 'Sensitive to parameter choice, slower on large datasets'
    },
    'svm': {
        'name': 'One-Class SVM',
        'description': 'Support Vector Machine that learns a decision boundary around normal data',
        'best_for': 'High-dimensional data, complex boundaries',
        'pros': 'Effective in high dimensions, memory efficient',
        'cons': 'Sensitive to feature scaling, parameter tuning required'
    },
    'pca': {
        'name': 'Principal Component Analysis',
        'description': 'Linear dimensionality reduction technique that identifies anomalies in reconstruction error',
        'best_for': 'Linear relationships, dimensionality reduction',
        'pros': 'Fast, interpretable, good for linear data',
        'cons': 'Assumes linear relationships, may miss complex patterns'
    },
    'knn': {
        'name': 'K-Nearest Neighbors',
        'description': 'Distance-based algorithm that identifies anomalies based on distance to k nearest neighbors',
        'best_for': 'Small to medium datasets, clear distance metrics',
        'pros': 'Simple, intuitive, no assumptions about data distribution',
        'cons': 'Computationally expensive, sensitive to curse of dimensionality'
    },
    'cluster': {
        'name': 'Clustering-Based Local Outlier',
        'description': 'Uses clustering to identify points that are far from cluster centers',
        'best_for': 'Data with natural clusters',
        'pros': 'Good for clustered data, interpretable',
        'cons': 'Requires good clustering, sensitive to cluster number'
    },
    'histogram': {
        'name': 'Histogram-based Outlier Detection',
        'description': 'Constructs histograms for each feature and identifies low-frequency regions',
        'best_for': 'Categorical or discrete features',
        'pros': 'Fast, simple, good for categorical data',
        'cons': 'May not work well with continuous features'
    },
    'abod': {
        'name': 'Angle-based Outlier Detection',
        'description': 'Considers the variance in angles between data points to identify outliers',
        'best_for': 'High-dimensional data',
        'pros': 'Effective in high dimensions, robust to noise',
        'cons': 'Computationally expensive, complex interpretation'
    },
    'cof': {
        'name': 'Connectivity-based Outlier Factor',
        'description': 'Uses connectivity patterns to identify outliers in sparse regions',
        'best_for': 'Sparse datasets, connectivity patterns',
        'pros': 'Good for sparse data, considers connectivity',
        'cons': 'Complex parameter tuning, slower computation'
    },
    'mcd': {
        'name': 'Minimum Covariance Determinant',
        'description': 'Robust estimator that identifies outliers based on Mahalanobis distance',
        'best_for': 'Gaussian-distributed data, multivariate outliers',
        'pros': 'Robust to outliers, statistical foundation',
        'cons': 'Assumes Gaussian distribution, sensitive to high dimensions'
    },
    'sod': {
        'name': 'Subspace Outlier Detection',
        'description': 'Identifies outliers in relevant subspaces of the feature space',
        'best_for': 'High-dimensional data, subspace analysis',
        'pros': 'Good for high dimensions, finds relevant subspaces',
        'cons': 'Complex interpretation, parameter sensitive'
    },
    'sos': {
        'name': 'Stochastic Outlier Selection',
        'description': 'Uses stochastic approach to compute outlier probabilities',
        'best_for': 'Probabilistic outlier detection',
        'pros': 'Provides probability scores, robust',
        'cons': 'Stochastic results, complex parameters'
    }
}

def load_data():
    """Load data from session state or sample data"""
    if 'data' in st.session_state and st.session_state.data is not None:
        return st.session_state.data
    else:
        # Load sample data if no data in session
        try:
            return get_data('anomaly')
        except:
            return None

def setup_pycaret_environment(data, session_id=123):
    """Setup PyCaret environment"""
    try:
        exp = anomaly.setup(data, session_id=session_id, verbose=False)
        return exp
    except Exception as e:
        st.error(f"Error setting up PyCaret environment: {str(e)}")
        return None

def train_model(algorithm, contamination=0.05):
    """Train anomaly detection model"""
    try:
        model = anomaly.create_model(algorithm, contamination=contamination, verbose=False)
        return model
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def assign_anomaly_labels(model):
    """Assign anomaly labels to the dataset"""
    try:
        results = anomaly.assign_model(model)
        return results
    except Exception as e:
        st.error(f"Error assigning labels: {str(e)}")
        return None

def main():
    st.title("🤖 Model Training")
    st.markdown("Select and train anomaly detection algorithms on your dataset.")
    
    # Load data
    data = load_data()
    
    if data is None:
        st.warning("⚠️ No data available. Please go to the Data Explorer page to load a dataset first.")
        st.stop()
    
    # Initialize session state for models
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'pycaret_setup' not in st.session_state:
        st.session_state.pycaret_setup = None
    
    # Data overview
    st.markdown("## 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{data.shape[0]:,}")
    with col2:
        st.metric("Features", data.shape[1])
    with col3:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        st.metric("Numeric Features", len(numeric_cols))
    with col4:
        st.metric("Missing Values", data.isnull().sum().sum())
    
    # Algorithm selection
    st.markdown("---")
    st.markdown("## 🎯 Algorithm Selection")
    
    # Algorithm information display
    with st.expander("📚 Algorithm Information", expanded=False):
        selected_algo_info = st.selectbox(
            "Select algorithm to learn more:",
            list(ALGORITHMS.keys()),
            format_func=lambda x: ALGORITHMS[x]['name']
        )
        
        algo_info = ALGORITHMS[selected_algo_info]
        st.markdown(f"""
        <div class="algorithm-card">
            <h4>{algo_info['name']}</h4>
            <p><strong>Description:</strong> {algo_info['description']}</p>
            <p><strong>Best for:</strong> {algo_info['best_for']}</p>
            <p><strong>Pros:</strong> {algo_info['pros']}</p>
            <p><strong>Cons:</strong> {algo_info['cons']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Training configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ⚙️ Training Configuration")
        
        selected_algorithm = st.selectbox(
            "Choose Algorithm:",
            list(ALGORITHMS.keys()),
            format_func=lambda x: ALGORITHMS[x]['name'],
            help="Select the anomaly detection algorithm to train"
        )
        
        contamination = st.slider(
            "Contamination Rate:",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Expected proportion of anomalies in the dataset"
        )
        
        session_id = st.number_input(
            "Random Seed:",
            min_value=1,
            max_value=9999,
            value=123,
            help="Random seed for reproducible results"
        )
    
    with col2:
        st.markdown("### 📋 Training Summary")
        st.markdown(f"""
        <div class="model-info">
            <p><strong>Algorithm:</strong> {ALGORITHMS[selected_algorithm]['name']}</p>
            <p><strong>Contamination:</strong> {contamination:.2%}</p>
            <p><strong>Data Size:</strong> {data.shape[0]:,} × {data.shape[1]}</p>
            <p><strong>Random Seed:</strong> {session_id}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Training button and process
    st.markdown("---")
    st.markdown("## 🚀 Model Training")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🎯 Train Model", type="primary", use_container_width=True):
            
            # Setup PyCaret environment if not already done
            if st.session_state.pycaret_setup is None:
                with st.spinner("Setting up PyCaret environment..."):
                    st.session_state.pycaret_setup = setup_pycaret_environment(data, session_id)
                
                if st.session_state.pycaret_setup is None:
                    st.error("Failed to setup PyCaret environment")
                    st.stop()
            
            # Training progress
            progress_container = st.container()
            
            with progress_container:
                st.markdown("""
                <div class="training-progress">
                    <h4>🔄 Training in Progress...</h4>
                    <p>Please wait while the model is being trained.</p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate training progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("Initializing algorithm...")
                    elif i < 70:
                        status_text.text("Training model...")
                    else:
                        status_text.text("Finalizing results...")
                    time.sleep(0.02)
                
                # Train the model
                model = train_model(selected_algorithm, contamination)
                
                if model is not None:
                    # Store the trained model
                    model_key = f"{selected_algorithm}_{contamination}_{session_id}"
                    st.session_state.trained_models[model_key] = {
                        'model': model,
                        'algorithm': selected_algorithm,
                        'contamination': contamination,
                        'session_id': session_id,
                        'timestamp': time.time()
                    }
                    
                    progress_container.empty()
                    st.success(f"✅ Model trained successfully! ({ALGORITHMS[selected_algorithm]['name']})")
                    
                    # Get predictions
                    with st.spinner("Generating predictions..."):
                        results = assign_anomaly_labels(model)
                        
                        if results is not None:
                            st.session_state.trained_models[model_key]['results'] = results
                            
                            # Display basic results
                            anomaly_count = results['Anomaly'].sum()
                            anomaly_percentage = (anomaly_count / len(results)) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Anomalies", f"{anomaly_count:,}")
                            with col2:
                                st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
                            with col3:
                                st.metric("Normal Points", f"{len(results) - anomaly_count:,}")
                else:
                    progress_container.empty()
                    st.error("❌ Model training failed. Please try again.")
    
    with col2:
        if st.button("🔄 Reset Environment", use_container_width=True):
            st.session_state.pycaret_setup = None
            st.session_state.trained_models = {}
            st.success("Environment reset successfully!")
            st.rerun()
    
    # Display trained models
    if st.session_state.trained_models:
        st.markdown("---")
        st.markdown("## 📋 Trained Models")
        
        models_data = []
        for key, model_info in st.session_state.trained_models.items():
            models_data.append({
                'Algorithm': ALGORITHMS[model_info['algorithm']]['name'],
                'Contamination': f"{model_info['contamination']:.2%}",
                'Session ID': model_info['session_id'],
                'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_info['timestamp'])),
                'Key': key
            })
        
        models_df = pd.DataFrame(models_data)
        
        # Display models table
        st.dataframe(models_df[['Algorithm', 'Contamination', 'Session ID', 'Timestamp']], 
                    use_container_width=True)
        
        # Model selection for detailed view
        selected_model_key = st.selectbox(
            "Select model for detailed view:",
            list(st.session_state.trained_models.keys()),
            format_func=lambda x: f"{ALGORITHMS[st.session_state.trained_models[x]['algorithm']]['name']} (Contamination: {st.session_state.trained_models[x]['contamination']:.2%})"
        )
        
        if selected_model_key:
            model_info = st.session_state.trained_models[selected_model_key]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔍 Model Details")
                st.markdown(f"""
                <div class="model-info">
                    <p><strong>Algorithm:</strong> {ALGORITHMS[model_info['algorithm']]['name']}</p>
                    <p><strong>Contamination Rate:</strong> {model_info['contamination']:.2%}</p>
                    <p><strong>Random Seed:</strong> {model_info['session_id']}</p>
                    <p><strong>Training Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_info['timestamp']))}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if 'results' in model_info:
                    results = model_info['results']
                    anomaly_count = results['Anomaly'].sum()
                    anomaly_percentage = (anomaly_count / len(results)) * 100
                    
                    st.markdown("### 📊 Results Summary")
                    st.metric("Anomalies Detected", f"{anomaly_count:,}")
                    st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
                    st.metric("Normal Points", f"{len(results) - anomaly_count:,}")
            
            # Results preview
            if 'results' in model_info:
                st.markdown("### 👀 Results Preview")
                results = model_info['results']
                
                # Show anomalies first
                anomalies = results[results['Anomaly'] == 1].head(10)
                normal = results[results['Anomaly'] == 0].head(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Detected Anomalies (Top 10):**")
                    st.dataframe(anomalies, use_container_width=True)
                
                with col2:
                    st.markdown("**Normal Points (Sample):**")
                    st.dataframe(normal, use_container_width=True)
                
                # Download options
                st.markdown("### 📥 Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download results
                    csv_buffer = io.StringIO()
                    results.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📄 Download Results (CSV)",
                        data=csv_data,
                        file_name=f"anomaly_results_{model_info['algorithm']}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Download model
                    model_buffer = io.BytesIO()
                    pickle.dump(model_info['model'], model_buffer)
                    model_data = model_buffer.getvalue()
                    
                    st.download_button(
                        label="🤖 Download Model (PKL)",
                        data=model_data,
                        file_name=f"anomaly_model_{model_info['algorithm']}.pkl",
                        mime="application/octet-stream"
                    )
                
                with col3:
                    # Delete model
                    if st.button("🗑️ Delete Model", key=f"delete_{selected_model_key}"):
                        del st.session_state.trained_models[selected_model_key]
                        st.success("Model deleted successfully!")
                        st.rerun()
    
    # Quick training section
    st.markdown("---")
    st.markdown("## ⚡ Quick Training")
    st.markdown("Train multiple algorithms quickly with default settings:")
    
    quick_algorithms = st.multiselect(
        "Select algorithms for quick training:",
        ['iforest', 'lof', 'svm', 'pca', 'knn'],
        default=['iforest', 'lof'],
        format_func=lambda x: ALGORITHMS[x]['name']
    )
    
    if st.button("🚀 Quick Train Selected Models") and quick_algorithms:
        
        # Setup environment if needed
        if st.session_state.pycaret_setup is None:
            with st.spinner("Setting up PyCaret environment..."):
                st.session_state.pycaret_setup = setup_pycaret_environment(data, 123)
        
        if st.session_state.pycaret_setup is not None:
            progress_bar = st.progress(0)
            
            for i, algo in enumerate(quick_algorithms):
                progress_bar.progress((i + 1) / len(quick_algorithms))
                
                with st.spinner(f"Training {ALGORITHMS[algo]['name']}..."):
                    model = train_model(algo, 0.05)
                    
                    if model is not None:
                        model_key = f"{algo}_0.05_123_quick"
                        st.session_state.trained_models[model_key] = {
                            'model': model,
                            'algorithm': algo,
                            'contamination': 0.05,
                            'session_id': 123,
                            'timestamp': time.time()
                        }
                        
                        # Get predictions
                        results = assign_anomaly_labels(model)
                        if results is not None:
                            st.session_state.trained_models[model_key]['results'] = results
            
            st.success(f"✅ Successfully trained {len(quick_algorithms)} models!")
            st.rerun()

if __name__ == "__main__":
    main()

