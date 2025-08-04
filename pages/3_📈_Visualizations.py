import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pycaret.anomaly as anomaly
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="Visualizations - Anomaly Detection",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .viz-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .metric-container {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_anomaly_scatter_plot(results, x_col, y_col, title="Anomaly Detection Results"):
    """Create scatter plot showing anomalies vs normal points"""
    fig = px.scatter(
        results, 
        x=x_col, 
        y=y_col,
        color='Anomaly',
        color_discrete_map={0: 'blue', 1: 'red'},
        title=title,
        labels={'Anomaly': 'Point Type'},
        hover_data=['Anomaly_Score']
    )
    
    # Update traces for better legend
    fig.for_each_trace(
        lambda trace: trace.update(name="Normal" if trace.name == "0" else "Anomaly")
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(title="Point Type")
    )
    
    return fig

def create_anomaly_score_distribution(results, title="Anomaly Score Distribution"):
    """Create histogram of anomaly scores"""
    fig = go.Figure()
    
    # Normal points
    normal_scores = results[results['Anomaly'] == 0]['Anomaly_Score']
    fig.add_trace(go.Histogram(
        x=normal_scores,
        name='Normal',
        opacity=0.7,
        nbinsx=30,
        marker_color='blue'
    ))
    
    # Anomaly points
    anomaly_scores = results[results['Anomaly'] == 1]['Anomaly_Score']
    fig.add_trace(go.Histogram(
        x=anomaly_scores,
        name='Anomaly',
        opacity=0.7,
        nbinsx=30,
        marker_color='red'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Anomaly Score',
        yaxis_title='Count',
        barmode='overlay',
        height=400
    )
    
    return fig

def create_pca_visualization(results, n_components=2):
    """Create PCA visualization of anomalies"""
    # Get numeric columns (exclude Anomaly and Anomaly_Score)
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['Anomaly', 'Anomaly_Score']]
    
    if len(feature_cols) < 2:
        return None
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(results[feature_cols])
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['Anomaly'] = results['Anomaly'].values
    pca_df['Anomaly_Score'] = results['Anomaly_Score'].values
    
    if n_components == 2:
        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Anomaly',
            color_discrete_map={0: 'blue', 1: 'red'},
            title=f'PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})',
            hover_data=['Anomaly_Score']
        )
    else:
        fig = px.scatter_3d(
            pca_df,
            x='PC1',
            y='PC2',
            z='PC3',
            color='Anomaly',
            color_discrete_map={0: 'blue', 1: 'red'},
            title=f'3D PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})',
            hover_data=['Anomaly_Score']
        )
    
    # Update traces for better legend
    fig.for_each_trace(
        lambda trace: trace.update(name="Normal" if trace.name == "0" else "Anomaly")
    )
    
    fig.update_layout(height=600)
    
    return fig, pca.explained_variance_ratio_

def create_feature_importance_plot(results, top_n=10):
    """Create feature importance plot based on anomaly scores correlation"""
    # Get numeric columns (exclude Anomaly and Anomaly_Score)
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['Anomaly', 'Anomaly_Score']]
    
    if len(feature_cols) == 0:
        return None
    
    # Calculate correlation with anomaly scores
    correlations = []
    for col in feature_cols:
        corr = abs(results[col].corr(results['Anomaly_Score']))
        correlations.append({'Feature': col, 'Importance': corr})
    
    importance_df = pd.DataFrame(correlations).sort_values('Importance', ascending=True).tail(top_n)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Feature Importance (Correlation with Anomaly Score)',
        labels={'Importance': 'Absolute Correlation with Anomaly Score'}
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_anomaly_heatmap(results):
    """Create heatmap showing anomaly patterns across features"""
    # Get numeric columns (exclude Anomaly and Anomaly_Score)
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['Anomaly', 'Anomaly_Score']]
    
    if len(feature_cols) == 0:
        return None
    
    # Separate anomalies and normal points
    anomalies = results[results['Anomaly'] == 1][feature_cols]
    normal = results[results['Anomaly'] == 0][feature_cols]
    
    if len(anomalies) == 0:
        return None
    
    # Calculate mean values for each group
    anomaly_means = anomalies.mean()
    normal_means = normal.mean()
    
    # Calculate difference (normalized)
    diff = (anomaly_means - normal_means) / normal.std()
    
    # Create heatmap data
    heatmap_data = diff.values.reshape(1, -1)
    
    fig = px.imshow(
        heatmap_data,
        x=feature_cols,
        y=['Anomaly vs Normal'],
        color_continuous_scale='RdBu_r',
        title='Feature Deviation in Anomalies (Normalized Difference from Normal)',
        aspect='auto'
    )
    
    fig.update_layout(height=200)
    
    return fig

def create_anomaly_timeline(results):
    """Create timeline view if there's an index that can represent time"""
    if len(results) < 10:
        return None
    
    # Use index as pseudo-time
    timeline_data = results.copy()
    timeline_data['Index'] = range(len(timeline_data))
    
    fig = px.scatter(
        timeline_data,
        x='Index',
        y='Anomaly_Score',
        color='Anomaly',
        color_discrete_map={0: 'blue', 1: 'red'},
        title='Anomaly Detection Timeline',
        labels={'Index': 'Data Point Index', 'Anomaly_Score': 'Anomaly Score'}
    )
    
    # Add threshold line
    threshold = timeline_data['Anomaly_Score'].quantile(0.95)
    fig.add_hline(y=threshold, line_dash="dash", line_color="orange", 
                  annotation_text="95th Percentile")
    
    fig.for_each_trace(
        lambda trace: trace.update(name="Normal" if trace.name == "0" else "Anomaly")
    )
    
    fig.update_layout(height=400)
    
    return fig

def main():
    st.title("📈 Visualizations")
    st.markdown("Explore your anomaly detection results through interactive visualizations.")
    
    # Check if models are available
    if 'trained_models' not in st.session_state or not st.session_state.trained_models:
        st.warning("⚠️ No trained models found. Please go to the Model Training page to train a model first.")
        st.stop()
    
    # Model selection
    st.markdown("## 🎯 Select Model for Visualization")
    
    model_options = {}
    for key, model_info in st.session_state.trained_models.items():
        if 'results' in model_info:
            algorithm_name = model_info['algorithm']
            contamination = model_info['contamination']
            display_name = f"{algorithm_name.upper()} (Contamination: {contamination:.2%})"
            model_options[display_name] = key
    
    if not model_options:
        st.warning("⚠️ No models with results found. Please train a model and generate predictions first.")
        st.stop()
    
    selected_model_display = st.selectbox(
        "Choose a trained model:",
        list(model_options.keys())
    )
    
    selected_model_key = model_options[selected_model_display]
    model_info = st.session_state.trained_models[selected_model_key]
    results = model_info['results']
    
    # Results overview
    st.markdown("---")
    st.markdown("## 📊 Results Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    anomaly_count = results['Anomaly'].sum()
    normal_count = len(results) - anomaly_count
    anomaly_rate = (anomaly_count / len(results)) * 100
    avg_anomaly_score = results['Anomaly_Score'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{anomaly_count:,}</h3>
            <p>Anomalies Detected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{normal_count:,}</h3>
            <p>Normal Points</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{anomaly_rate:.2f}%</h3>
            <p>Anomaly Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{avg_anomaly_score:.3f}</h3>
            <p>Avg Anomaly Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization options
    st.markdown("---")
    st.markdown("## 🎨 Visualization Options")
    
    viz_tabs = st.tabs([
        "📊 Score Distribution", 
        "🔍 Scatter Plots", 
        "🧮 PCA Analysis", 
        "📈 Feature Analysis",
        "🌡️ Heatmaps",
        "⏱️ Timeline View"
    ])
    
    # Tab 1: Score Distribution
    with viz_tabs[0]:
        st.markdown("### 📊 Anomaly Score Distribution")
        st.markdown("This histogram shows the distribution of anomaly scores for normal and anomalous points.")
        
        score_fig = create_anomaly_score_distribution(results)
        st.plotly_chart(score_fig, use_container_width=True)
        
        # Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Normal Points Statistics:**")
            normal_scores = results[results['Anomaly'] == 0]['Anomaly_Score']
            st.write(f"Count: {len(normal_scores):,}")
            st.write(f"Mean: {normal_scores.mean():.4f}")
            st.write(f"Std: {normal_scores.std():.4f}")
            st.write(f"Min: {normal_scores.min():.4f}")
            st.write(f"Max: {normal_scores.max():.4f}")
        
        with col2:
            st.markdown("**Anomaly Points Statistics:**")
            anomaly_scores = results[results['Anomaly'] == 1]['Anomaly_Score']
            if len(anomaly_scores) > 0:
                st.write(f"Count: {len(anomaly_scores):,}")
                st.write(f"Mean: {anomaly_scores.mean():.4f}")
                st.write(f"Std: {anomaly_scores.std():.4f}")
                st.write(f"Min: {anomaly_scores.min():.4f}")
                st.write(f"Max: {anomaly_scores.max():.4f}")
            else:
                st.write("No anomalies detected")
    
    # Tab 2: Scatter Plots
    with viz_tabs[1]:
        st.markdown("### 🔍 Feature Scatter Plots")
        st.markdown("Explore relationships between features and identify anomaly patterns.")
        
        # Get numeric columns for plotting
        numeric_cols = results.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['Anomaly', 'Anomaly_Score']]
        
        if len(feature_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("X-axis feature:", feature_cols, key="scatter_x")
            
            with col2:
                y_axis = st.selectbox("Y-axis feature:", 
                                    [col for col in feature_cols if col != x_axis], 
                                    key="scatter_y")
            
            if x_axis and y_axis:
                scatter_fig = create_anomaly_scatter_plot(results, x_axis, y_axis, 
                                                        f"Anomalies: {x_axis} vs {y_axis}")
                st.plotly_chart(scatter_fig, use_container_width=True)
                
                # Correlation analysis
                correlation = results[x_axis].corr(results[y_axis])
                st.info(f"Correlation between {x_axis} and {y_axis}: {correlation:.3f}")
        else:
            st.warning("Need at least 2 numeric features for scatter plots.")
    
    # Tab 3: PCA Analysis
    with viz_tabs[2]:
        st.markdown("### 🧮 Principal Component Analysis")
        st.markdown("Dimensionality reduction to visualize high-dimensional anomaly patterns.")
        
        numeric_cols = results.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['Anomaly', 'Anomaly_Score']]
        
        if len(feature_cols) >= 2:
            pca_type = st.radio("PCA Visualization Type:", ["2D", "3D"], horizontal=True)
            
            n_components = 2 if pca_type == "2D" else 3
            
            with st.spinner("Computing PCA..."):
                pca_result = create_pca_visualization(results, n_components)
                
                if pca_result:
                    pca_fig, explained_variance = pca_result
                    st.plotly_chart(pca_fig, use_container_width=True)
                    
                    # Explained variance
                    st.markdown("**Explained Variance by Component:**")
                    for i, var in enumerate(explained_variance):
                        st.write(f"PC{i+1}: {var:.2%}")
                    st.write(f"Total: {explained_variance.sum():.2%}")
                else:
                    st.error("Could not compute PCA visualization.")
        else:
            st.warning("Need at least 2 numeric features for PCA analysis.")
    
    # Tab 4: Feature Analysis
    with viz_tabs[3]:
        st.markdown("### 📈 Feature Importance Analysis")
        st.markdown("Understand which features contribute most to anomaly detection.")
        
        numeric_cols = results.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['Anomaly', 'Anomaly_Score']]
        
        if len(feature_cols) > 0:
            importance_fig = create_feature_importance_plot(results)
            if importance_fig:
                st.plotly_chart(importance_fig, use_container_width=True)
            
            # Feature statistics comparison
            st.markdown("### 📊 Feature Statistics: Anomalies vs Normal")
            
            anomalies = results[results['Anomaly'] == 1]
            normal = results[results['Anomaly'] == 0]
            
            if len(anomalies) > 0 and len(normal) > 0:
                comparison_data = []
                
                for col in feature_cols[:10]:  # Limit to top 10 features
                    comparison_data.append({
                        'Feature': col,
                        'Normal_Mean': normal[col].mean(),
                        'Anomaly_Mean': anomalies[col].mean(),
                        'Normal_Std': normal[col].std(),
                        'Anomaly_Std': anomalies[col].std(),
                        'Difference': anomalies[col].mean() - normal[col].mean()
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
        else:
            st.warning("No numeric features available for analysis.")
    
    # Tab 5: Heatmaps
    with viz_tabs[4]:
        st.markdown("### 🌡️ Feature Deviation Heatmap")
        st.markdown("Visualize how anomalous points deviate from normal patterns across features.")
        
        heatmap_fig = create_anomaly_heatmap(results)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - Red areas indicate features where anomalies have higher values than normal
            - Blue areas indicate features where anomalies have lower values than normal
            - White areas indicate little difference between anomalies and normal points
            """)
        else:
            st.warning("Could not generate heatmap. Need numeric features and detected anomalies.")
    
    # Tab 6: Timeline View
    with viz_tabs[5]:
        st.markdown("### ⏱️ Anomaly Timeline")
        st.markdown("View anomalies in the order they appear in your dataset.")
        
        timeline_fig = create_anomaly_timeline(results)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Anomaly clusters analysis
            if anomaly_count > 0:
                st.markdown("### 🎯 Anomaly Distribution Analysis")
                
                anomaly_indices = results[results['Anomaly'] == 1].index.tolist()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("First Anomaly Index", min(anomaly_indices))
                    st.metric("Last Anomaly Index", max(anomaly_indices))
                
                with col2:
                    # Calculate clustering
                    if len(anomaly_indices) > 1:
                        gaps = [anomaly_indices[i+1] - anomaly_indices[i] for i in range(len(anomaly_indices)-1)]
                        avg_gap = np.mean(gaps)
                        st.metric("Average Gap Between Anomalies", f"{avg_gap:.1f}")
                        st.metric("Max Gap Between Anomalies", max(gaps))
        else:
            st.warning("Could not generate timeline view.")
    
    # Export visualizations
    st.markdown("---")
    st.markdown("## 📥 Export Visualizations")
    
    st.markdown("""
    **Note:** To save individual plots, use the camera icon in the top-right corner of each plot.
    You can save plots as PNG, HTML, or PDF formats.
    """)
    
    # Summary report
    if st.button("📊 Generate Summary Report"):
        summary_report = f"""
Anomaly Detection Visualization Summary
=====================================

Model: {model_info['algorithm'].upper()}
Contamination Rate: {model_info['contamination']:.2%}
Dataset Size: {len(results):,} points

Results:
- Anomalies Detected: {anomaly_count:,} ({anomaly_rate:.2f}%)
- Normal Points: {normal_count:,}
- Average Anomaly Score: {avg_anomaly_score:.4f}

Feature Analysis:
- Total Features: {len(feature_cols)}
- Numeric Features: {len(feature_cols)}

Score Statistics:
Normal Points:
  - Mean Score: {results[results['Anomaly'] == 0]['Anomaly_Score'].mean():.4f}
  - Std Score: {results[results['Anomaly'] == 0]['Anomaly_Score'].std():.4f}

Anomaly Points:
  - Mean Score: {results[results['Anomaly'] == 1]['Anomaly_Score'].mean():.4f if anomaly_count > 0 else 'N/A'}
  - Std Score: {results[results['Anomaly'] == 1]['Anomaly_Score'].std():.4f if anomaly_count > 0 else 'N/A'}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_report,
            file_name=f"anomaly_visualization_summary_{model_info['algorithm']}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()

