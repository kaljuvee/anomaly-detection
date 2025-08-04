import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import zipfile
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Export Results - Anomaly Detection",
    page_icon="📥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .export-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
    }
    .download-section {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .format-info {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def create_comprehensive_report(model_info, results, include_visualizations=False):
    """Create a comprehensive analysis report"""
    
    algorithm = model_info['algorithm']
    contamination = model_info['contamination']
    session_id = model_info['session_id']
    timestamp = model_info['timestamp']
    
    # Basic statistics
    total_points = len(results)
    anomaly_count = results['Anomaly'].sum()
    normal_count = total_points - anomaly_count
    anomaly_rate = anomaly_count / total_points
    
    # Score statistics
    mean_score = results['Anomaly_Score'].mean()
    std_score = results['Anomaly_Score'].std()
    min_score = results['Anomaly_Score'].min()
    max_score = results['Anomaly_Score'].max()
    
    # Separation analysis
    if anomaly_count > 0 and normal_count > 0:
        normal_scores = results[results['Anomaly'] == 0]['Anomaly_Score']
        anomaly_scores = results[results['Anomaly'] == 1]['Anomaly_Score']
        score_separation = anomaly_scores.mean() - normal_scores.mean()
    else:
        score_separation = None
    
    # Feature analysis
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['Anomaly', 'Anomaly_Score']]
    
    report = f"""
ANOMALY DETECTION ANALYSIS REPORT
=================================

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report ID: {int(time.time())}

EXECUTIVE SUMMARY
-----------------
This report presents the results of anomaly detection analysis using the {algorithm.upper()} algorithm.
The analysis identified {anomaly_count:,} anomalous data points out of {total_points:,} total observations,
representing an anomaly rate of {anomaly_rate:.2%}.

MODEL CONFIGURATION
-------------------
Algorithm: {algorithm.upper()}
Contamination Rate: {contamination:.2%}
Random Seed: {session_id}
Training Date: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
----------------
Total Data Points: {total_points:,}
Features: {len(feature_cols)}
Feature Names: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}

RESULTS SUMMARY
---------------
Anomalies Detected: {anomaly_count:,}
Normal Points: {normal_count:,}
Anomaly Rate: {anomaly_rate:.2%}

ANOMALY SCORE ANALYSIS
----------------------
Mean Anomaly Score: {mean_score:.6f}
Standard Deviation: {std_score:.6f}
Minimum Score: {min_score:.6f}
Maximum Score: {max_score:.6f}
Score Range: {max_score - min_score:.6f}

"""

    if score_separation is not None:
        report += f"""
SEPARATION ANALYSIS
-------------------
Normal Points Mean Score: {normal_scores.mean():.6f}
Anomaly Points Mean Score: {anomaly_scores.mean():.6f}
Score Separation: {score_separation:.6f}
Separation Quality: {'Good' if score_separation > 0 else 'Poor'}

"""

    # Feature analysis
    if len(feature_cols) > 0 and anomaly_count > 0:
        report += "FEATURE IMPACT ANALYSIS\n"
        report += "-" * 23 + "\n"
        
        normal_data = results[results['Anomaly'] == 0]
        anomaly_data = results[results['Anomaly'] == 1]
        
        feature_impacts = []
        for col in feature_cols:
            normal_mean = normal_data[col].mean()
            anomaly_mean = anomaly_data[col].mean()
            difference = anomaly_mean - normal_mean
            relative_diff = (difference / normal_mean * 100) if normal_mean != 0 else 0
            
            feature_impacts.append({
                'feature': col,
                'normal_mean': normal_mean,
                'anomaly_mean': anomaly_mean,
                'difference': difference,
                'relative_diff': relative_diff
            })
        
        # Sort by absolute relative difference
        feature_impacts.sort(key=lambda x: abs(x['relative_diff']), reverse=True)
        
        for impact in feature_impacts[:10]:  # Top 10 features
            report += f"""
{impact['feature']}:
  Normal Mean: {impact['normal_mean']:.4f}
  Anomaly Mean: {impact['anomaly_mean']:.4f}
  Difference: {impact['difference']:.4f}
  Relative Change: {impact['relative_diff']:.2f}%
"""

    # Top anomalies
    if anomaly_count > 0:
        report += "\nTOP ANOMALIES (Highest Scores)\n"
        report += "-" * 30 + "\n"
        
        top_anomalies = results[results['Anomaly'] == 1].nlargest(10, 'Anomaly_Score')
        
        for idx, (_, row) in enumerate(top_anomalies.iterrows(), 1):
            report += f"\nAnomaly #{idx}:\n"
            report += f"  Index: {row.name}\n"
            report += f"  Anomaly Score: {row['Anomaly_Score']:.6f}\n"
            
            # Show feature values for top features
            for col in feature_cols[:5]:  # Top 5 features
                report += f"  {col}: {row[col]:.4f}\n"

    # Recommendations
    report += f"""

RECOMMENDATIONS
---------------
1. VALIDATION: Review the {anomaly_count} detected anomalies manually to validate results
2. THRESHOLD: Consider adjusting contamination rate if anomaly rate seems too high/low
3. FEATURES: Focus investigation on features showing highest impact differences
4. MONITORING: Implement ongoing monitoring for similar anomaly patterns
5. ACTION: Develop specific response procedures for confirmed anomalies

TECHNICAL NOTES
---------------
- This analysis uses unsupervised anomaly detection
- Results should be validated with domain expertise
- Consider ensemble methods for improved robustness
- Regular model retraining may be necessary as data patterns evolve

ALGORITHM DETAILS: {algorithm.upper()}
"""

    # Add algorithm-specific notes
    algorithm_notes = {
        'iforest': 'Isolation Forest works by isolating anomalies through random feature splits. Effective for large datasets.',
        'lof': 'Local Outlier Factor identifies anomalies based on local density. Good for varying density patterns.',
        'svm': 'One-Class SVM learns a boundary around normal data. Effective in high-dimensional spaces.',
        'pca': 'PCA-based detection uses reconstruction error. Assumes linear relationships in data.',
        'knn': 'K-Nearest Neighbors uses distance to neighbors. Simple but can be computationally expensive.',
        'cluster': 'Clustering-based detection identifies points far from cluster centers.',
        'histogram': 'Histogram-based method identifies low-frequency regions. Good for categorical features.',
        'abod': 'Angle-based detection considers variance in angles. Effective in high dimensions.',
        'cof': 'Connectivity-based method uses connectivity patterns. Good for sparse data.',
        'mcd': 'Minimum Covariance Determinant uses robust statistical estimation.',
        'sod': 'Subspace Outlier Detection finds relevant subspaces for anomaly detection.',
        'sos': 'Stochastic Outlier Selection provides probabilistic anomaly scores.'
    }
    
    if algorithm in algorithm_notes:
        report += f"\n{algorithm_notes[algorithm]}\n"

    report += f"""

DISCLAIMER
----------
This automated analysis is provided for informational purposes only.
Results should be interpreted by qualified personnel familiar with the data domain.
The accuracy of anomaly detection depends on data quality and algorithm suitability.

Report generated by Anomaly Detection Demo Application
Contact: [Your Contact Information]
Version: 1.0
"""

    return report

def create_model_metadata(model_info):
    """Create metadata file for the model"""
    metadata = {
        'model_info': {
            'algorithm': model_info['algorithm'],
            'contamination': model_info['contamination'],
            'session_id': model_info['session_id'],
            'training_timestamp': model_info['timestamp'],
            'training_date': datetime.fromtimestamp(model_info['timestamp']).isoformat()
        },
        'export_info': {
            'export_timestamp': time.time(),
            'export_date': datetime.now().isoformat(),
            'version': '1.0'
        }
    }
    
    if 'results' in model_info:
        results = model_info['results']
        metadata['results_summary'] = {
            'total_points': len(results),
            'anomalies_detected': int(results['Anomaly'].sum()),
            'anomaly_rate': float(results['Anomaly'].mean()),
            'mean_anomaly_score': float(results['Anomaly_Score'].mean()),
            'features': results.select_dtypes(include=[np.number]).columns.tolist()
        }
    
    return metadata

def create_export_package(model_key, model_info, include_items):
    """Create a comprehensive export package"""
    
    # Create in-memory zip file
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        
        # Add model file
        if include_items.get('model', False):
            model_buffer = io.BytesIO()
            pickle.dump(model_info['model'], model_buffer)
            zip_file.writestr(f"model_{model_info['algorithm']}.pkl", model_buffer.getvalue())
        
        # Add results CSV
        if include_items.get('results', False) and 'results' in model_info:
            results_csv = model_info['results'].to_csv(index=False)
            zip_file.writestr(f"results_{model_info['algorithm']}.csv", results_csv)
        
        # Add comprehensive report
        if include_items.get('report', False):
            report = create_comprehensive_report(model_info, model_info['results'])
            zip_file.writestr(f"analysis_report_{model_info['algorithm']}.txt", report)
        
        # Add metadata
        if include_items.get('metadata', False):
            metadata = create_model_metadata(model_info)
            metadata_json = json.dumps(metadata, indent=2)
            zip_file.writestr(f"metadata_{model_info['algorithm']}.json", metadata_json)
        
        # Add anomalies only CSV
        if include_items.get('anomalies_only', False) and 'results' in model_info:
            anomalies = model_info['results'][model_info['results']['Anomaly'] == 1]
            anomalies_csv = anomalies.to_csv(index=False)
            zip_file.writestr(f"anomalies_only_{model_info['algorithm']}.csv", anomalies_csv)
        
        # Add summary statistics
        if include_items.get('statistics', False) and 'results' in model_info:
            results = model_info['results']
            numeric_cols = results.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in ['Anomaly', 'Anomaly_Score']]
            
            if len(feature_cols) > 0:
                stats_summary = results[feature_cols].describe()
                stats_csv = stats_summary.to_csv()
                zip_file.writestr(f"statistics_{model_info['algorithm']}.csv", stats_csv)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def main():
    st.title("📥 Export Results")
    st.markdown("Download your anomaly detection models, results, and comprehensive analysis reports.")
    
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
            display_name = f"{algorithm_name.upper()} (Contamination: {contamination:.2%})"
            available_models[display_name] = key
    
    if not available_models:
        st.warning("⚠️ No models with results found. Please train models and generate predictions first.")
        st.stop()
    
    # Model selection
    st.markdown("## 🎯 Select Model for Export")
    
    selected_model_display = st.selectbox(
        "Choose a model to export:",
        list(available_models.keys())
    )
    
    selected_model_key = available_models[selected_model_display]
    model_info = st.session_state.trained_models[selected_model_key]
    results = model_info['results']
    
    # Model overview
    st.markdown("---")
    st.markdown("## 📊 Model Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    anomaly_count = results['Anomaly'].sum()
    normal_count = len(results) - anomaly_count
    anomaly_rate = (anomaly_count / len(results)) * 100
    
    with col1:
        st.metric("Algorithm", model_info['algorithm'].upper())
    with col2:
        st.metric("Anomalies", f"{anomaly_count:,}")
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    with col4:
        st.metric("Total Points", f"{len(results):,}")
    
    # Export options
    st.markdown("---")
    st.markdown("## 📦 Export Options")
    
    export_tabs = st.tabs([
        "📄 Individual Files",
        "📦 Complete Package", 
        "📊 Custom Reports",
        "🔄 Batch Export"
    ])
    
    # Tab 1: Individual Files
    with export_tabs[0]:
        st.markdown("### 📄 Download Individual Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="export-card">
                <h4>🤖 Model Files</h4>
                <p>Download trained models for deployment or further analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model pickle file
            model_buffer = io.BytesIO()
            pickle.dump(model_info['model'], model_buffer)
            model_data = model_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Model (PKL)",
                data=model_data,
                file_name=f"anomaly_model_{model_info['algorithm']}_{int(time.time())}.pkl",
                mime="application/octet-stream",
                help="Trained model in pickle format for Python deployment"
            )
            
            # Model metadata
            metadata = create_model_metadata(model_info)
            metadata_json = json.dumps(metadata, indent=2)
            
            st.download_button(
                label="📋 Download Metadata (JSON)",
                data=metadata_json,
                file_name=f"model_metadata_{model_info['algorithm']}.json",
                mime="application/json",
                help="Model configuration and training information"
            )
        
        with col2:
            st.markdown("""
            <div class="export-card">
                <h4>📊 Results & Data</h4>
                <p>Download predictions and analysis results</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Complete results
            results_csv = results.to_csv(index=False)
            st.download_button(
                label="📄 Download All Results (CSV)",
                data=results_csv,
                file_name=f"anomaly_results_{model_info['algorithm']}.csv",
                mime="text/csv",
                help="Complete dataset with anomaly predictions and scores"
            )
            
            # Anomalies only
            anomalies_only = results[results['Anomaly'] == 1]
            if len(anomalies_only) > 0:
                anomalies_csv = anomalies_only.to_csv(index=False)
                st.download_button(
                    label="🚨 Download Anomalies Only (CSV)",
                    data=anomalies_csv,
                    file_name=f"anomalies_only_{model_info['algorithm']}.csv",
                    mime="text/csv",
                    help="Only the detected anomalous data points"
                )
            else:
                st.info("No anomalies detected to export")
        
        # Reports section
        st.markdown("### 📋 Analysis Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Comprehensive report
            comprehensive_report = create_comprehensive_report(model_info, results)
            st.download_button(
                label="📊 Download Comprehensive Report (TXT)",
                data=comprehensive_report,
                file_name=f"comprehensive_report_{model_info['algorithm']}.txt",
                mime="text/plain",
                help="Detailed analysis report with insights and recommendations"
            )
        
        with col2:
            # Summary statistics
            numeric_cols = results.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in ['Anomaly', 'Anomaly_Score']]
            
            if len(feature_cols) > 0:
                stats_summary = results[feature_cols].describe()
                stats_csv = stats_summary.to_csv()
                
                st.download_button(
                    label="📈 Download Statistics Summary (CSV)",
                    data=stats_csv,
                    file_name=f"statistics_summary_{model_info['algorithm']}.csv",
                    mime="text/csv",
                    help="Statistical summary of all features"
                )
    
    # Tab 2: Complete Package
    with export_tabs[1]:
        st.markdown("### 📦 Complete Export Package")
        st.markdown("Download everything in a single ZIP file for easy sharing and archiving.")
        
        # Package options
        st.markdown("**Select items to include in the package:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_model = st.checkbox("🤖 Trained Model", value=True, help="Include the trained model file")
            include_results = st.checkbox("📊 Complete Results", value=True, help="Include all predictions and scores")
            include_report = st.checkbox("📋 Analysis Report", value=True, help="Include comprehensive analysis report")
        
        with col2:
            include_metadata = st.checkbox("📋 Model Metadata", value=True, help="Include model configuration and info")
            include_anomalies = st.checkbox("🚨 Anomalies Only", value=True, help="Include separate file with only anomalies")
            include_statistics = st.checkbox("📈 Statistics Summary", value=True, help="Include statistical summary")
        
        # Package preview
        include_items = {
            'model': include_model,
            'results': include_results,
            'report': include_report,
            'metadata': include_metadata,
            'anomalies_only': include_anomalies,
            'statistics': include_statistics
        }
        
        selected_items = [item for item, selected in include_items.items() if selected]
        
        if selected_items:
            st.markdown(f"**Package will include:** {', '.join(selected_items)}")
            
            if st.button("📦 Create Export Package", type="primary"):
                with st.spinner("Creating export package..."):
                    package_data = create_export_package(selected_model_key, model_info, include_items)
                
                st.download_button(
                    label="📥 Download Complete Package (ZIP)",
                    data=package_data,
                    file_name=f"anomaly_detection_package_{model_info['algorithm']}_{int(time.time())}.zip",
                    mime="application/zip",
                    help="Complete package with all selected items"
                )
                
                st.success("✅ Export package created successfully!")
        else:
            st.warning("Please select at least one item to include in the package.")
    
    # Tab 3: Custom Reports
    with export_tabs[2]:
        st.markdown("### 📊 Custom Analysis Reports")
        st.markdown("Generate customized reports with specific focus areas.")
        
        # Report customization options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Report Sections:**")
            include_executive_summary = st.checkbox("📋 Executive Summary", value=True)
            include_technical_details = st.checkbox("🔧 Technical Details", value=True)
            include_feature_analysis = st.checkbox("📊 Feature Analysis", value=True)
            include_top_anomalies = st.checkbox("🚨 Top Anomalies", value=True)
            include_recommendations = st.checkbox("💡 Recommendations", value=True)
        
        with col2:
            st.markdown("**Report Format:**")
            report_format = st.radio("Choose format:", ["Text Report", "JSON Data", "CSV Summary"])
            
            if include_top_anomalies:
                top_n_anomalies = st.slider("Number of top anomalies to include:", 5, 50, 10)
            else:
                top_n_anomalies = 10
        
        # Generate custom report
        if st.button("📊 Generate Custom Report"):
            
            if report_format == "Text Report":
                # Generate custom text report
                custom_report = f"""
CUSTOM ANOMALY DETECTION REPORT
===============================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {model_info['algorithm'].upper()}
"""
                
                if include_executive_summary:
                    custom_report += f"""

EXECUTIVE SUMMARY
-----------------
Analysis of {len(results):,} data points using {model_info['algorithm'].upper()} algorithm.
Detected {anomaly_count:,} anomalies ({anomaly_rate:.2f}% of total data).
"""
                
                if include_technical_details:
                    custom_report += f"""

TECHNICAL CONFIGURATION
-----------------------
Algorithm: {model_info['algorithm'].upper()}
Contamination Rate: {model_info['contamination']:.2%}
Random Seed: {model_info['session_id']}
Training Date: {datetime.fromtimestamp(model_info['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
"""
                
                if include_feature_analysis and len(results.select_dtypes(include=[np.number]).columns) > 2:
                    custom_report += "\nFEATURE ANALYSIS\n" + "-" * 16 + "\n"
                    
                    numeric_cols = results.select_dtypes(include=[np.number]).columns
                    feature_cols = [col for col in numeric_cols if col not in ['Anomaly', 'Anomaly_Score']]
                    
                    if anomaly_count > 0:
                        normal_data = results[results['Anomaly'] == 0]
                        anomaly_data = results[results['Anomaly'] == 1]
                        
                        for col in feature_cols[:5]:  # Top 5 features
                            normal_mean = normal_data[col].mean()
                            anomaly_mean = anomaly_data[col].mean()
                            difference = anomaly_mean - normal_mean
                            
                            custom_report += f"""
{col}:
  Normal Mean: {normal_mean:.4f}
  Anomaly Mean: {anomaly_mean:.4f}
  Difference: {difference:.4f}
"""
                
                if include_top_anomalies and anomaly_count > 0:
                    custom_report += f"\nTOP {min(top_n_anomalies, anomaly_count)} ANOMALIES\n"
                    custom_report += "-" * (len(f"TOP {min(top_n_anomalies, anomaly_count)} ANOMALIES")) + "\n"
                    
                    top_anomalies = results[results['Anomaly'] == 1].nlargest(top_n_anomalies, 'Anomaly_Score')
                    
                    for idx, (_, row) in enumerate(top_anomalies.iterrows(), 1):
                        custom_report += f"\n{idx}. Index {row.name}: Score {row['Anomaly_Score']:.6f}\n"
                
                if include_recommendations:
                    custom_report += f"""

RECOMMENDATIONS
---------------
1. Review the {anomaly_count} detected anomalies for validation
2. Consider domain expertise for anomaly interpretation
3. Monitor similar patterns in future data
4. Adjust contamination rate if needed ({model_info['contamination']:.2%} currently)
"""
                
                st.download_button(
                    label="📄 Download Custom Report",
                    data=custom_report,
                    file_name=f"custom_report_{model_info['algorithm']}_{int(time.time())}.txt",
                    mime="text/plain"
                )
            
            elif report_format == "JSON Data":
                # Generate JSON report
                json_report = {
                    "report_metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "model_algorithm": model_info['algorithm'],
                        "contamination_rate": model_info['contamination']
                    },
                    "summary": {
                        "total_points": len(results),
                        "anomalies_detected": int(anomaly_count),
                        "anomaly_rate": float(anomaly_rate / 100),
                        "normal_points": int(normal_count)
                    }
                }
                
                if include_top_anomalies and anomaly_count > 0:
                    top_anomalies = results[results['Anomaly'] == 1].nlargest(top_n_anomalies, 'Anomaly_Score')
                    json_report["top_anomalies"] = []
                    
                    for _, row in top_anomalies.iterrows():
                        anomaly_data = {
                            "index": int(row.name),
                            "anomaly_score": float(row['Anomaly_Score'])
                        }
                        # Add feature values
                        numeric_cols = results.select_dtypes(include=[np.number]).columns
                        feature_cols = [col for col in numeric_cols if col not in ['Anomaly', 'Anomaly_Score']]
                        for col in feature_cols:
                            anomaly_data[col] = float(row[col])
                        
                        json_report["top_anomalies"].append(anomaly_data)
                
                json_data = json.dumps(json_report, indent=2)
                
                st.download_button(
                    label="📄 Download JSON Report",
                    data=json_data,
                    file_name=f"custom_report_{model_info['algorithm']}_{int(time.time())}.json",
                    mime="application/json"
                )
            
            else:  # CSV Summary
                # Generate CSV summary
                summary_data = {
                    'Metric': [
                        'Total Points',
                        'Anomalies Detected',
                        'Normal Points',
                        'Anomaly Rate (%)',
                        'Mean Anomaly Score',
                        'Std Anomaly Score',
                        'Min Anomaly Score',
                        'Max Anomaly Score'
                    ],
                    'Value': [
                        len(results),
                        anomaly_count,
                        normal_count,
                        anomaly_rate,
                        results['Anomaly_Score'].mean(),
                        results['Anomaly_Score'].std(),
                        results['Anomaly_Score'].min(),
                        results['Anomaly_Score'].max()
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_csv = summary_df.to_csv(index=False)
                
                st.download_button(
                    label="📄 Download CSV Summary",
                    data=summary_csv,
                    file_name=f"custom_summary_{model_info['algorithm']}_{int(time.time())}.csv",
                    mime="text/csv"
                )
    
    # Tab 4: Batch Export
    with export_tabs[3]:
        st.markdown("### 🔄 Batch Export All Models")
        st.markdown("Export results from multiple models at once.")
        
        if len(available_models) > 1:
            
            # Select models for batch export
            selected_models = st.multiselect(
                "Select models for batch export:",
                list(available_models.keys()),
                default=list(available_models.keys())
            )
            
            if selected_models:
                # Batch export options
                batch_format = st.radio(
                    "Batch export format:",
                    ["Individual ZIP files", "Combined comparison report", "Merged results CSV"]
                )
                
                if st.button("🚀 Start Batch Export"):
                    
                    if batch_format == "Individual ZIP files":
                        # Create individual packages for each model
                        st.info("Creating individual export packages...")
                        
                        for model_display in selected_models:
                            model_key = available_models[model_display]
                            model_info = st.session_state.trained_models[model_key]
                            
                            include_items = {
                                'model': True,
                                'results': True,
                                'report': True,
                                'metadata': True,
                                'anomalies_only': True,
                                'statistics': True
                            }
                            
                            package_data = create_export_package(model_key, model_info, include_items)
                            
                            st.download_button(
                                label=f"📥 Download {model_display} Package",
                                data=package_data,
                                file_name=f"package_{model_info['algorithm']}_{int(time.time())}.zip",
                                mime="application/zip",
                                key=f"batch_{model_key}"
                            )
                    
                    elif batch_format == "Combined comparison report":
                        # Create comparison report
                        comparison_report = "MULTI-MODEL COMPARISON REPORT\n"
                        comparison_report += "=" * 32 + "\n\n"
                        comparison_report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        comparison_report += f"Models Compared: {len(selected_models)}\n\n"
                        
                        for model_display in selected_models:
                            model_key = available_models[model_display]
                            model_info = st.session_state.trained_models[model_key]
                            results = model_info['results']
                            
                            anomaly_count = results['Anomaly'].sum()
                            anomaly_rate = (anomaly_count / len(results)) * 100
                            
                            comparison_report += f"""
{model_display}:
  Algorithm: {model_info['algorithm'].upper()}
  Contamination: {model_info['contamination']:.2%}
  Total Points: {len(results):,}
  Anomalies: {anomaly_count:,} ({anomaly_rate:.2f}%)
  Mean Score: {results['Anomaly_Score'].mean():.6f}
  Score Std: {results['Anomaly_Score'].std():.6f}

"""
                        
                        st.download_button(
                            label="📄 Download Comparison Report",
                            data=comparison_report,
                            file_name=f"multi_model_comparison_{int(time.time())}.txt",
                            mime="text/plain"
                        )
                    
                    else:  # Merged results CSV
                        # Combine all results into one CSV
                        merged_results = []
                        
                        for model_display in selected_models:
                            model_key = available_models[model_display]
                            model_info = st.session_state.trained_models[model_key]
                            results = model_info['results'].copy()
                            
                            # Add model identifier columns
                            results['Model_Algorithm'] = model_info['algorithm']
                            results['Model_Contamination'] = model_info['contamination']
                            results['Model_Display_Name'] = model_display
                            
                            merged_results.append(results)
                        
                        combined_df = pd.concat(merged_results, ignore_index=True)
                        combined_csv = combined_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📄 Download Merged Results",
                            data=combined_csv,
                            file_name=f"merged_results_{int(time.time())}.csv",
                            mime="text/csv"
                        )
            else:
                st.info("Please select models for batch export.")
        else:
            st.info("Only one model available. Use individual export options above.")
    
    # Export format information
    st.markdown("---")
    st.markdown("## ℹ️ Export Format Information")
    
    with st.expander("📋 File Format Details"):
        st.markdown("""
        **PKL (Pickle) Files:**
        - Python-specific binary format for trained models
        - Can be loaded using `pickle.load()` in Python
        - Suitable for deployment in Python environments
        
        **CSV Files:**
        - Comma-separated values, readable by most data tools
        - Contains all data points with anomaly predictions and scores
        - Compatible with Excel, R, Python pandas, etc.
        
        **JSON Files:**
        - Human-readable structured data format
        - Contains metadata and configuration information
        - Easy to parse in most programming languages
        
        **TXT Files:**
        - Plain text reports with analysis and insights
        - Human-readable format for sharing findings
        - Includes recommendations and interpretations
        
        **ZIP Files:**
        - Compressed archives containing multiple files
        - Convenient for sharing complete analysis packages
        - Preserves file organization and reduces size
        """)

if __name__ == "__main__":
    main()

