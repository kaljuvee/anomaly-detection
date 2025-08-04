import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from sklearn.metrics import silhouette_score
from scipy import stats

# Configure page
st.set_page_config(
    page_title="Model Comparison - Anomaly Detection",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .comparison-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #6f42c1;
    }
    .metric-comparison {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .winner-card {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def calculate_model_metrics(results):
    """Calculate various metrics for model evaluation"""
    metrics = {}
    
    # Basic counts
    total_points = len(results)
    anomaly_count = results['Anomaly'].sum()
    normal_count = total_points - anomaly_count
    anomaly_rate = anomaly_count / total_points
    
    metrics['total_points'] = total_points
    metrics['anomaly_count'] = anomaly_count
    metrics['normal_count'] = normal_count
    metrics['anomaly_rate'] = anomaly_rate
    
    # Score statistics
    metrics['mean_anomaly_score'] = results['Anomaly_Score'].mean()
    metrics['std_anomaly_score'] = results['Anomaly_Score'].std()
    metrics['min_anomaly_score'] = results['Anomaly_Score'].min()
    metrics['max_anomaly_score'] = results['Anomaly_Score'].max()
    
    # Separation metrics
    if anomaly_count > 0 and normal_count > 0:
        normal_scores = results[results['Anomaly'] == 0]['Anomaly_Score']
        anomaly_scores = results[results['Anomaly'] == 1]['Anomaly_Score']
        
        metrics['normal_mean_score'] = normal_scores.mean()
        metrics['anomaly_mean_score'] = anomaly_scores.mean()
        metrics['score_separation'] = anomaly_scores.mean() - normal_scores.mean()
        
        # Statistical test for separation
        try:
            t_stat, p_value = stats.ttest_ind(anomaly_scores, normal_scores)
            metrics['separation_p_value'] = p_value
            metrics['separation_significant'] = p_value < 0.05
        except:
            metrics['separation_p_value'] = None
            metrics['separation_significant'] = False
    else:
        metrics['normal_mean_score'] = None
        metrics['anomaly_mean_score'] = None
        metrics['score_separation'] = None
        metrics['separation_p_value'] = None
        metrics['separation_significant'] = False
    
    # Silhouette score (if we have both classes)
    if anomaly_count > 0 and normal_count > 0:
        try:
            # Use numeric features for silhouette calculation
            numeric_cols = results.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in ['Anomaly', 'Anomaly_Score']]
            
            if len(feature_cols) >= 2:
                silhouette = silhouette_score(results[feature_cols], results['Anomaly'])
                metrics['silhouette_score'] = silhouette
            else:
                metrics['silhouette_score'] = None
        except:
            metrics['silhouette_score'] = None
    else:
        metrics['silhouette_score'] = None
    
    return metrics

def create_comparison_table(model_metrics):
    """Create comparison table for multiple models"""
    comparison_data = []
    
    for model_name, metrics in model_metrics.items():
        comparison_data.append({
            'Model': model_name,
            'Anomalies': f"{metrics['anomaly_count']:,}",
            'Anomaly Rate': f"{metrics['anomaly_rate']:.2%}",
            'Mean Score': f"{metrics['mean_anomaly_score']:.4f}",
            'Score Std': f"{metrics['std_anomaly_score']:.4f}",
            'Score Separation': f"{metrics['score_separation']:.4f}" if metrics['score_separation'] is not None else "N/A",
            'Silhouette Score': f"{metrics['silhouette_score']:.4f}" if metrics['silhouette_score'] is not None else "N/A",
            'Significant Separation': "✅" if metrics['separation_significant'] else "❌"
        })
    
    return pd.DataFrame(comparison_data)

def create_score_comparison_plot(model_results):
    """Create box plot comparing anomaly scores across models"""
    plot_data = []
    
    for model_name, results in model_results.items():
        for _, row in results.iterrows():
            plot_data.append({
                'Model': model_name,
                'Anomaly_Score': row['Anomaly_Score'],
                'Point_Type': 'Anomaly' if row['Anomaly'] == 1 else 'Normal'
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    fig = px.box(
        plot_df,
        x='Model',
        y='Anomaly_Score',
        color='Point_Type',
        title='Anomaly Score Distribution Comparison',
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'}
    )
    
    fig.update_layout(height=500)
    return fig

def create_anomaly_rate_comparison(model_metrics):
    """Create bar chart comparing anomaly rates"""
    models = list(model_metrics.keys())
    rates = [model_metrics[model]['anomaly_rate'] * 100 for model in models]
    
    fig = px.bar(
        x=models,
        y=rates,
        title='Anomaly Detection Rate Comparison',
        labels={'x': 'Model', 'y': 'Anomaly Rate (%)'},
        color=rates,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_score_separation_plot(model_metrics):
    """Create plot showing score separation between normal and anomaly points"""
    models = []
    separations = []
    
    for model_name, metrics in model_metrics.items():
        if metrics['score_separation'] is not None:
            models.append(model_name)
            separations.append(metrics['score_separation'])
    
    if not models:
        return None
    
    fig = px.bar(
        x=models,
        y=separations,
        title='Score Separation Between Normal and Anomaly Points',
        labels={'x': 'Model', 'y': 'Score Separation'},
        color=separations,
        color_continuous_scale='RdYlBu_r'
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_performance_radar_chart(model_metrics):
    """Create radar chart for model performance comparison"""
    if len(model_metrics) < 2:
        return None
    
    # Normalize metrics for radar chart
    metrics_to_plot = ['anomaly_rate', 'score_separation', 'silhouette_score']
    
    # Filter models that have all required metrics
    valid_models = {}
    for model_name, metrics in model_metrics.items():
        if all(metrics.get(metric) is not None for metric in metrics_to_plot):
            valid_models[model_name] = metrics
    
    if len(valid_models) < 2:
        return None
    
    fig = go.Figure()
    
    # Normalize values to 0-1 scale
    all_values = {metric: [model_metrics[model][metric] for model in valid_models.keys()] 
                  for metric in metrics_to_plot}
    
    normalized_values = {}
    for metric in metrics_to_plot:
        values = all_values[metric]
        min_val, max_val = min(values), max(values)
        if max_val > min_val:
            normalized_values[metric] = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized_values[metric] = [0.5] * len(values)
    
    # Add traces for each model
    for i, (model_name, metrics) in enumerate(valid_models.items()):
        values = [normalized_values[metric][i] for metric in metrics_to_plot]
        values.append(values[0])  # Close the radar chart
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_to_plot + [metrics_to_plot[0]],
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Comparison (Normalized)",
        height=500
    )
    
    return fig

def main():
    st.title("⚖️ Model Comparison")
    st.markdown("Compare the performance of different anomaly detection algorithms side by side.")
    
    # Check if models are available
    if 'trained_models' not in st.session_state or not st.session_state.trained_models:
        st.warning("⚠️ No trained models found. Please go to the Model Training page to train models first.")
        st.stop()
    
    # Get models with results
    available_models = {}
    for key, model_info in st.session_state.trained_models.items():
        if 'results' in model_info:
            algorithm_name = model_info['algorithm']
            contamination = model_info['contamination']
            display_name = f"{algorithm_name.upper()} ({contamination:.2%})"
            available_models[display_name] = key
    
    if len(available_models) < 1:
        st.warning("⚠️ No models with results found. Please train models and generate predictions first.")
        st.stop()
    
    # Model selection for comparison
    st.markdown("## 🎯 Select Models for Comparison")
    
    selected_models = st.multiselect(
        "Choose models to compare:",
        list(available_models.keys()),
        default=list(available_models.keys())[:min(3, len(available_models))],
        help="Select 2 or more models to compare their performance"
    )
    
    if len(selected_models) < 1:
        st.info("Please select at least one model to analyze.")
        st.stop()
    
    # Get model results and calculate metrics
    model_results = {}
    model_metrics = {}
    
    for display_name in selected_models:
        model_key = available_models[display_name]
        model_info = st.session_state.trained_models[model_key]
        results = model_info['results']
        
        model_results[display_name] = results
        model_metrics[display_name] = calculate_model_metrics(results)
    
    # Overview section
    st.markdown("---")
    st.markdown("## 📊 Comparison Overview")
    
    if len(selected_models) == 1:
        # Single model analysis
        model_name = selected_models[0]
        metrics = model_metrics[model_name]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-comparison">
                <h3>{metrics['anomaly_count']:,}</h3>
                <p>Anomalies Detected</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-comparison">
                <h3>{metrics['anomaly_rate']:.2%}</h3>
                <p>Anomaly Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-comparison">
                <h3>{metrics['mean_anomaly_score']:.4f}</h3>
                <p>Mean Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            separation = metrics['score_separation']
            st.markdown(f"""
            <div class="metric-comparison">
                <h3>{separation:.4f if separation is not None else 'N/A'}</h3>
                <p>Score Separation</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Multi-model comparison
        st.markdown("### 📋 Comparison Table")
        comparison_df = create_comparison_table(model_metrics)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Highlight best performing models
        st.markdown("### 🏆 Performance Highlights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Best separation
            best_separation_model = None
            best_separation_value = -float('inf')
            
            for model_name, metrics in model_metrics.items():
                if metrics['score_separation'] is not None and metrics['score_separation'] > best_separation_value:
                    best_separation_value = metrics['score_separation']
                    best_separation_model = model_name
            
            if best_separation_model:
                st.markdown(f"""
                <div class="winner-card">
                    <h4>🎯 Best Score Separation</h4>
                    <p><strong>{best_separation_model}</strong></p>
                    <p>Separation: {best_separation_value:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Best silhouette score
            best_silhouette_model = None
            best_silhouette_value = -float('inf')
            
            for model_name, metrics in model_metrics.items():
                if metrics['silhouette_score'] is not None and metrics['silhouette_score'] > best_silhouette_value:
                    best_silhouette_value = metrics['silhouette_score']
                    best_silhouette_model = model_name
            
            if best_silhouette_model:
                st.markdown(f"""
                <div class="winner-card">
                    <h4>📊 Best Silhouette Score</h4>
                    <p><strong>{best_silhouette_model}</strong></p>
                    <p>Score: {best_silhouette_value:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            # Most consistent (lowest std)
            most_consistent_model = None
            lowest_std = float('inf')
            
            for model_name, metrics in model_metrics.items():
                if metrics['std_anomaly_score'] < lowest_std:
                    lowest_std = metrics['std_anomaly_score']
                    most_consistent_model = model_name
            
            if most_consistent_model:
                st.markdown(f"""
                <div class="winner-card">
                    <h4>🎯 Most Consistent</h4>
                    <p><strong>{most_consistent_model}</strong></p>
                    <p>Std: {lowest_std:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Visualization comparisons
    st.markdown("---")
    st.markdown("## 📈 Visual Comparisons")
    
    viz_tabs = st.tabs([
        "📊 Score Distributions",
        "📈 Anomaly Rates", 
        "🎯 Score Separation",
        "🕸️ Performance Radar",
        "📋 Detailed Analysis"
    ])
    
    # Tab 1: Score Distributions
    with viz_tabs[0]:
        st.markdown("### 📊 Anomaly Score Distribution Comparison")
        
        if len(selected_models) > 1:
            score_comparison_fig = create_score_comparison_plot(model_results)
            st.plotly_chart(score_comparison_fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - Compare the distribution of anomaly scores across different models
            - Models with better separation show distinct distributions for normal vs anomaly points
            - Look for models where anomaly scores are consistently higher for detected anomalies
            """)
        else:
            # Single model score distribution
            model_name = selected_models[0]
            results = model_results[model_name]
            
            fig = px.histogram(
                results,
                x='Anomaly_Score',
                color='Anomaly',
                color_discrete_map={0: 'blue', 1: 'red'},
                title=f'Score Distribution - {model_name}',
                barmode='overlay',
                opacity=0.7
            )
            
            fig.for_each_trace(
                lambda trace: trace.update(name="Normal" if trace.name == "0" else "Anomaly")
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Anomaly Rates
    with viz_tabs[1]:
        st.markdown("### 📈 Anomaly Detection Rate Comparison")
        
        if len(selected_models) > 1:
            rate_fig = create_anomaly_rate_comparison(model_metrics)
            st.plotly_chart(rate_fig, use_container_width=True)
            
            # Rate analysis
            rates = [model_metrics[model]['anomaly_rate'] * 100 for model in selected_models]
            avg_rate = np.mean(rates)
            std_rate = np.std(rates)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Rate", f"{avg_rate:.2f}%")
            with col2:
                st.metric("Rate Std Dev", f"{std_rate:.2f}%")
            with col3:
                st.metric("Rate Range", f"{max(rates) - min(rates):.2f}%")
        else:
            model_name = selected_models[0]
            rate = model_metrics[model_name]['anomaly_rate'] * 100
            st.metric(f"Anomaly Rate - {model_name}", f"{rate:.2f}%")
    
    # Tab 3: Score Separation
    with viz_tabs[2]:
        st.markdown("### 🎯 Score Separation Analysis")
        
        if len(selected_models) > 1:
            separation_fig = create_score_separation_plot(model_metrics)
            if separation_fig:
                st.plotly_chart(separation_fig, use_container_width=True)
                
                st.markdown("""
                **Score Separation** measures how well the model distinguishes between normal and anomalous points:
                - Higher values indicate better separation
                - Positive values mean anomalies have higher scores than normal points
                - Negative values suggest the model may be confused
                """)
            else:
                st.warning("No models have sufficient data for separation analysis.")
        else:
            model_name = selected_models[0]
            separation = model_metrics[model_name]['score_separation']
            if separation is not None:
                st.metric(f"Score Separation - {model_name}", f"{separation:.4f}")
                
                if separation > 0:
                    st.success("✅ Good separation: Anomalies have higher scores than normal points")
                else:
                    st.warning("⚠️ Poor separation: Model may need tuning")
            else:
                st.info("Separation analysis requires both normal and anomaly points.")
    
    # Tab 4: Performance Radar
    with viz_tabs[3]:
        st.markdown("### 🕸️ Multi-dimensional Performance Comparison")
        
        if len(selected_models) > 1:
            radar_fig = create_performance_radar_chart(model_metrics)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
                
                st.markdown("""
                **Radar Chart Interpretation:**
                - Each axis represents a different performance metric (normalized 0-1)
                - Larger areas indicate better overall performance
                - Compare the shape and size of different models
                """)
            else:
                st.warning("Radar chart requires multiple models with complete metrics.")
        else:
            st.info("Radar chart requires multiple models for comparison.")
    
    # Tab 5: Detailed Analysis
    with viz_tabs[4]:
        st.markdown("### 📋 Detailed Statistical Analysis")
        
        for model_name in selected_models:
            metrics = model_metrics[model_name]
            results = model_results[model_name]
            
            with st.expander(f"📊 {model_name} - Detailed Analysis"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Basic Statistics:**")
                    st.write(f"Total Points: {metrics['total_points']:,}")
                    st.write(f"Anomalies: {metrics['anomaly_count']:,}")
                    st.write(f"Normal Points: {metrics['normal_count']:,}")
                    st.write(f"Anomaly Rate: {metrics['anomaly_rate']:.2%}")
                    
                    st.markdown("**Score Statistics:**")
                    st.write(f"Mean Score: {metrics['mean_anomaly_score']:.4f}")
                    st.write(f"Score Std: {metrics['std_anomaly_score']:.4f}")
                    st.write(f"Min Score: {metrics['min_anomaly_score']:.4f}")
                    st.write(f"Max Score: {metrics['max_anomaly_score']:.4f}")
                
                with col2:
                    st.markdown("**Advanced Metrics:**")
                    
                    if metrics['score_separation'] is not None:
                        st.write(f"Score Separation: {metrics['score_separation']:.4f}")
                        st.write(f"Normal Mean Score: {metrics['normal_mean_score']:.4f}")
                        st.write(f"Anomaly Mean Score: {metrics['anomaly_mean_score']:.4f}")
                        
                        if metrics['separation_p_value'] is not None:
                            st.write(f"Separation P-value: {metrics['separation_p_value']:.6f}")
                            st.write(f"Statistically Significant: {'Yes' if metrics['separation_significant'] else 'No'}")
                    
                    if metrics['silhouette_score'] is not None:
                        st.write(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
                        
                        if metrics['silhouette_score'] > 0.5:
                            st.success("✅ Good clustering quality")
                        elif metrics['silhouette_score'] > 0.25:
                            st.warning("⚠️ Moderate clustering quality")
                        else:
                            st.error("❌ Poor clustering quality")
                
                # Feature-level analysis
                numeric_cols = results.select_dtypes(include=[np.number]).columns
                feature_cols = [col for col in numeric_cols if col not in ['Anomaly', 'Anomaly_Score']]
                
                if len(feature_cols) > 0 and metrics['anomaly_count'] > 0:
                    st.markdown("**Feature Impact Analysis:**")
                    
                    feature_impact = []
                    normal_data = results[results['Anomaly'] == 0]
                    anomaly_data = results[results['Anomaly'] == 1]
                    
                    for col in feature_cols[:10]:  # Limit to top 10
                        if len(anomaly_data) > 0:
                            normal_mean = normal_data[col].mean()
                            anomaly_mean = anomaly_data[col].mean()
                            impact = abs(anomaly_mean - normal_mean) / normal_data[col].std()
                            
                            feature_impact.append({
                                'Feature': col,
                                'Impact Score': impact,
                                'Normal Mean': normal_mean,
                                'Anomaly Mean': anomaly_mean
                            })
                    
                    if feature_impact:
                        impact_df = pd.DataFrame(feature_impact).sort_values('Impact Score', ascending=False)
                        st.dataframe(impact_df, use_container_width=True)
    
    # Export comparison results
    st.markdown("---")
    st.markdown("## 📥 Export Comparison Results")
    
    if st.button("📊 Generate Comparison Report"):
        
        report_content = f"""
Anomaly Detection Model Comparison Report
========================================

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
Models Compared: {len(selected_models)}

SUMMARY
-------
"""
        
        for model_name in selected_models:
            metrics = model_metrics[model_name]
            report_content += f"""
{model_name}:
  - Anomalies Detected: {metrics['anomaly_count']:,} ({metrics['anomaly_rate']:.2%})
  - Mean Anomaly Score: {metrics['mean_anomaly_score']:.4f}
  - Score Std Dev: {metrics['std_anomaly_score']:.4f}
  - Score Separation: {metrics['score_separation']:.4f if metrics['score_separation'] is not None else 'N/A'}
  - Silhouette Score: {metrics['silhouette_score']:.4f if metrics['silhouette_score'] is not None else 'N/A'}
  - Significant Separation: {'Yes' if metrics['separation_significant'] else 'No'}
"""
        
        if len(selected_models) > 1:
            # Add rankings
            report_content += "\nRANKINGS\n--------\n"
            
            # Best separation
            valid_separations = {name: metrics['score_separation'] 
                               for name, metrics in model_metrics.items() 
                               if metrics['score_separation'] is not None}
            if valid_separations:
                best_sep = max(valid_separations, key=valid_separations.get)
                report_content += f"Best Score Separation: {best_sep} ({valid_separations[best_sep]:.4f})\n"
            
            # Best silhouette
            valid_silhouettes = {name: metrics['silhouette_score'] 
                               for name, metrics in model_metrics.items() 
                               if metrics['silhouette_score'] is not None}
            if valid_silhouettes:
                best_sil = max(valid_silhouettes, key=valid_silhouettes.get)
                report_content += f"Best Silhouette Score: {best_sil} ({valid_silhouettes[best_sil]:.4f})\n"
        
        report_content += f"""

RECOMMENDATIONS
---------------
Based on the analysis:

1. For best anomaly separation, consider: {max(model_metrics.keys(), key=lambda x: model_metrics[x]['score_separation'] or -float('inf'))}
2. For most consistent results, consider: {min(model_metrics.keys(), key=lambda x: model_metrics[x]['std_anomaly_score'])}
3. Always validate results with domain expertise
4. Consider ensemble methods for improved performance

Note: This analysis is based on unsupervised metrics. 
Ground truth validation is recommended when available.
"""
        
        st.download_button(
            label="📄 Download Comparison Report",
            data=report_content,
            file_name=f"model_comparison_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()

