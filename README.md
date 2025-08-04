# 🔍 Anomaly Detection Interactive Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.3+-orange.svg)](https://pycaret.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive web application that demonstrates various anomaly detection algorithms using PyCaret's powerful machine learning library. This application converts the comprehensive PyCaret anomaly detection tutorial into a user-friendly, interactive Streamlit interface.

## 🌟 Overview

Anomaly detection is a critical technique used to identify rare items, events, or observations that raise suspicions by differing significantly from the majority of the data. This application provides an intuitive interface to explore, train, and compare 12 different anomaly detection algorithms without requiring extensive programming knowledge.

### Key Features

- **🎯 12 Anomaly Detection Algorithms**: Compare different methods including Isolation Forest, Local Outlier Factor, One-Class SVM, PCA, and more
- **📊 Interactive Data Exploration**: Upload your own datasets or explore with built-in sample data
- **🤖 Real-time Model Training**: Train models with live progress indicators and immediate results
- **📈 Rich Visualizations**: Interactive plots using Plotly for comprehensive data analysis
- **⚖️ Model Comparison**: Side-by-side algorithm performance analysis
- **📥 Export Capabilities**: Download trained models, predictions, and comprehensive reports
- **🎨 Professional UI/UX**: Clean, intuitive interface built with Streamlit

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended for larger datasets

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kaljuvee/anomaly-detection.git
   cd anomaly-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run Home.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📋 Application Structure

The application is organized following Streamlit best practices with a multi-page architecture:

```
anomaly-detection/
├── Home.py                           # Main entry point
├── pages/
│   ├── 1_📊_Data_Explorer.py        # Data upload and exploration
│   ├── 2_🤖_Model_Training.py       # Algorithm training interface
│   ├── 3_📈_Visualizations.py       # Interactive plots and analysis
│   ├── 4_⚖️_Model_Comparison.py     # Multi-model comparison
│   └── 5_📥_Export_Results.py       # Results export and reporting
├── requirements.txt                  # Python dependencies
├── Tutorial_Anomaly_Detection.ipynb # Original PyCaret notebook
└── README.md                        # This documentation
```

## 🔬 Supported Algorithms

The application supports 12 state-of-the-art anomaly detection algorithms:

| Algorithm | Description | Best Use Case |
|-----------|-------------|---------------|
| **Isolation Forest** | Tree-based algorithm that isolates anomalies through random feature splits | Large datasets, high-dimensional data |
| **Local Outlier Factor (LOF)** | Density-based algorithm comparing local density with neighbors | Datasets with varying densities |
| **One-Class SVM** | Support Vector Machine learning decision boundary around normal data | High-dimensional data, complex boundaries |
| **Principal Component Analysis (PCA)** | Linear dimensionality reduction identifying reconstruction errors | Linear relationships, dimensionality reduction |
| **K-Nearest Neighbors (KNN)** | Distance-based algorithm using k nearest neighbors | Small to medium datasets, clear distance metrics |
| **Clustering-Based Local Outlier (CBLOF)** | Uses clustering to identify points far from cluster centers | Data with natural clusters |
| **Histogram-based Outlier Detection (HBOS)** | Constructs histograms and identifies low-frequency regions | Categorical or discrete features |
| **Angle-based Outlier Detection (ABOD)** | Considers variance in angles between data points | High-dimensional data |
| **Connectivity-based Outlier Factor (COF)** | Uses connectivity patterns in sparse regions | Sparse datasets, connectivity patterns |
| **Minimum Covariance Determinant (MCD)** | Robust estimator using Mahalanobis distance | Gaussian-distributed data, multivariate outliers |
| **Subspace Outlier Detection (SOD)** | Identifies outliers in relevant subspaces | High-dimensional data, subspace analysis |
| **Stochastic Outlier Selection (SOS)** | Stochastic approach computing outlier probabilities | Probabilistic outlier detection |

## 📖 User Guide

### 1. Data Explorer

The Data Explorer page allows you to upload and analyze your dataset:

- **Upload CSV Files**: Drag and drop or browse for CSV files up to 200MB
- **Sample Dataset**: Use PyCaret's built-in anomaly detection dataset (1000 rows × 10 features)
- **Data Analysis**: Automatic statistical analysis and data quality assessment
- **Visualizations**: Distribution plots, correlation heatmaps, and box plots for outlier detection

### 2. Model Training

Train anomaly detection models with customizable parameters:

- **Algorithm Selection**: Choose from 12 available algorithms
- **Parameter Tuning**: Adjust contamination rate (expected anomaly proportion)
- **Training Progress**: Real-time progress indicators and status updates
- **Results Preview**: Immediate display of detected anomalies and scores

### 3. Visualizations

Explore your results through interactive visualizations:

- **Score Distributions**: Histogram comparisons of normal vs anomalous points
- **Scatter Plots**: Feature relationships with anomaly highlighting
- **PCA Analysis**: Dimensionality reduction for high-dimensional data visualization
- **Feature Importance**: Identify which features contribute most to anomaly detection
- **Timeline Views**: Temporal patterns in anomaly detection

### 4. Model Comparison

Compare multiple algorithms side-by-side:

- **Performance Metrics**: Score separation, silhouette scores, and statistical significance
- **Visual Comparisons**: Box plots, radar charts, and performance rankings
- **Best Model Selection**: Automated recommendations based on multiple criteria

### 5. Export Results

Download comprehensive results and reports:

- **Individual Files**: Models (PKL), results (CSV), metadata (JSON)
- **Complete Packages**: ZIP files with all components
- **Custom Reports**: Tailored analysis reports in multiple formats
- **Batch Export**: Export multiple models simultaneously

## 🛠️ Technical Details

### Dependencies

The application relies on several key Python libraries:

- **Streamlit (≥1.47.0)**: Web application framework
- **PyCaret (≥3.3.0)**: AutoML library for anomaly detection
- **Plotly (≥5.24.0)**: Interactive visualization library
- **Pandas (≥2.1.0)**: Data manipulation and analysis
- **NumPy (≥1.26.0)**: Numerical computing
- **Scikit-learn (≥1.4.0)**: Machine learning algorithms
- **SciPy (≥1.11.0)**: Scientific computing

### Performance Considerations

- **Memory Usage**: Approximately 2-4GB RAM for typical datasets (1000-10000 rows)
- **Processing Time**: Model training typically takes 10-60 seconds depending on algorithm and data size
- **Scalability**: Tested with datasets up to 100,000 rows and 50 features

### Data Requirements

- **Format**: CSV files with numerical features
- **Size**: Recommended maximum 200MB file size
- **Features**: At least 2 numerical columns for meaningful analysis
- **Missing Values**: Automatically handled through imputation

## 🎯 Use Cases

This application is suitable for various anomaly detection scenarios:

### Business Applications
- **Fraud Detection**: Identify suspicious transactions or activities
- **Quality Control**: Detect defective products in manufacturing
- **Network Security**: Identify unusual network traffic patterns
- **Customer Behavior**: Detect unusual purchasing patterns

### Research and Education
- **Algorithm Comparison**: Compare different anomaly detection approaches
- **Educational Tool**: Learn about anomaly detection concepts interactively
- **Prototype Development**: Rapid prototyping of anomaly detection solutions
- **Data Exploration**: Understand data patterns and outliers

### Data Science Workflows
- **Exploratory Data Analysis**: Initial data quality assessment
- **Feature Engineering**: Identify important features for anomaly detection
- **Model Selection**: Choose the best algorithm for specific datasets
- **Results Validation**: Comprehensive analysis and reporting

## 🔧 Configuration

### Environment Variables

The application can be configured through environment variables:

```bash
# Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# PyCaret configuration
PYCARET_SETUP_SILENT=True
PYCARET_SETUP_VERBOSE=False
```

### Customization Options

- **Contamination Rate**: Adjust expected anomaly proportion (0.01-0.5)
- **Random Seed**: Set for reproducible results
- **Visualization Themes**: Plotly themes can be customized
- **Export Formats**: Multiple output formats available

## 📊 Example Workflow

Here's a typical workflow using the application:

1. **Data Upload**: Load your CSV dataset or use the sample data
2. **Data Exploration**: Review statistics, distributions, and data quality
3. **Algorithm Selection**: Choose appropriate algorithm based on data characteristics
4. **Model Training**: Train the model with optimal parameters
5. **Results Analysis**: Examine detected anomalies and scores
6. **Visualization**: Create plots to understand patterns
7. **Model Comparison**: Compare with other algorithms if needed
8. **Export**: Download results and reports for further analysis

## 🤝 Contributing

We welcome contributions to improve the application! Here's how you can help:

### Development Setup

1. Fork the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install black flake8 pytest
   ```

### Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting
- Add docstrings for all functions
- Include type hints where appropriate

### Testing

Run tests before submitting pull requests:
```bash
pytest tests/
streamlit run Home.py  # Manual testing
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyCaret Team**: For the excellent AutoML library and tutorial notebook
- **Streamlit Team**: For the amazing web application framework
- **PyOD Contributors**: For the underlying anomaly detection algorithms
- **Plotly Team**: For the interactive visualization capabilities

## 📞 Support

If you encounter any issues or have questions:

1. **Check the Documentation**: Review this README and inline help text
2. **Search Issues**: Look through existing GitHub issues
3. **Create New Issue**: Submit detailed bug reports or feature requests
4. **Community Support**: Join discussions in the repository

## 🔮 Future Enhancements

Planned improvements for future versions:

- **Deep Learning Algorithms**: Integration of neural network-based anomaly detection
- **Real-time Processing**: Streaming data anomaly detection capabilities
- **Advanced Visualizations**: 3D plots and interactive dashboards
- **API Integration**: REST API for programmatic access
- **Cloud Deployment**: One-click deployment to cloud platforms
- **Multi-language Support**: Internationalization for global users

---

**Built with ❤️ using Streamlit and PyCaret**

*Explore the power of automated machine learning for anomaly detection!*

