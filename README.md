# ML Algorithm Accuracy Testing Framework

A comprehensive, production-ready Python framework for automated machine learning model evaluation with extensive preprocessing, multiple algorithms, and detailed performance analysis.

## 📁 Project Structure

```
Accuracy_testing/
├── accuracy_testing.py     # Main ML testing framework
├── ml_accuracy_tester.py   # Production-ready class-based version
├── config.py              # Configuration file for file paths
├── b.bat                  # Windows batch file for easy execution
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```

## 📋 File Details

### `accuracy_testing.py` - Main Framework
The core ML testing script that provides:
- **Automatic Problem Detection**: Detects classification vs regression tasks
- **Data Preprocessing**: Handles missing values, categorical encoding, feature scaling
- **Multiple Algorithms**: Tests 8+ classification and 8+ regression models
- **Performance Metrics**: Comprehensive evaluation with accuracy, precision, recall, F1-score, ROC-AUC for classification; MAE, MSE, RMSE, R² for regression
- **Auto-encoding**: Automatically converts categorical target variables to numeric

### `ml_accuracy_tester.py` - Production Version
Class-based implementation with:
- Object-oriented design for better code organization
- Enhanced error handling and logging
- Type hints for better code clarity
- Comprehensive documentation

### `config.py` - Configuration
Simple configuration file to set:
```python
FILE_PATH = "your_data.csv"     # Path to your CSV file
TARGET_COLUMN = None            # Target column name or None for last column
```

### `b.bat` - Windows Batch File
User-friendly batch script that:
- Opens file dialog popup for CSV selection
- Prompts for target column name
- Automatically installs required packages
- Runs the ML testing framework
- Shows only accuracy results and best performing model

### `requirements.txt` - Dependencies
Lists all required Python packages:
- numpy, pandas, scikit-learn
- matplotlib, seaborn (for visualizations)
- tabulate (for formatted tables)
- xgboost (optional, for XGBoost models)

## 🚀 Quick Start

### Method 1: Using Batch File (Recommended for Windows)
1. Double-click `b.bat`
2. Select your CSV file in the popup dialog
3. Enter target column name (or press Enter for last column)
4. View results automatically

### Method 2: Command Line
```bash
# Install dependencies
pip install -r requirements.txt

# Run with your data
python accuracy_testing.py --file your_data.csv --target target_column

# Run with sample data
python accuracy_testing.py
```

### Method 3: Configuration File
1. Edit `config.py` with your file path
2. Run: `python accuracy_testing.py`

## 🤖 Supported Algorithms

### Classification Models
- **Logistic Regression**: Linear probabilistic classifier
- **Decision Tree**: Tree-based classifier with feature importance
- **Random Forest**: Ensemble of decision trees
- **SVM**: Support Vector Machine with RBF kernel
- **Naive Bayes**: Gaussian Naive Bayes classifier
- **K-Nearest Neighbors**: Distance-based classifier
- **Neural Network**: Multi-layer Perceptron classifier
- **Gradient Boosting**: Gradient boosting classifier
- **XGBoost**: Extreme gradient boosting (if installed)

### Regression Models
- **Linear Regression**: Basic linear regression
- **Ridge Regression**: L2 regularized linear regression
- **Lasso Regression**: L1 regularized linear regression
- **Support Vector Regression**: SVM for regression tasks
- **Decision Tree Regressor**: Tree-based regressor
- **Random Forest Regressor**: Ensemble regression
- **Gradient Boosting Regressor**: Gradient boosting for regression
- **Neural Network Regressor**: MLP for regression
- **XGBoost Regressor**: XGBoost for regression (if installed)

## 📊 Features

### Automatic Data Processing
- **Missing Value Handling**: Mean imputation for numeric, mode for categorical
- **Categorical Encoding**: Automatic label encoding for string variables
- **Feature Scaling**: StandardScaler applied to algorithms that need it
- **Target Auto-encoding**: Converts categorical targets to numeric automatically

### Smart Model Selection
- **Scaled Models**: Logistic Regression, SVM, KNN, Neural Networks
- **Unscaled Models**: Tree-based algorithms, Naive Bayes
- **Problem Type Detection**: Automatically detects classification vs regression

### Performance Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: MAE, MSE, RMSE, R² Score
- **Best Model Identification**: Automatically highlights top performer

### Visualizations (in full version)
- **Confusion Matrices**: For classification models
- **Feature Importance**: For tree-based models
- **Performance Comparison**: Tabulated results

## 💡 Usage Examples

### Example 1: Classification Dataset
```python
# Your CSV with features and a categorical target
# File: customer_data.csv
# Columns: age, income, education, purchased (target)

# Run via batch file or command line
# Output: Accuracy scores for all models + best performer
```

### Example 2: Regression Dataset
```python
# Your CSV with features and a numeric target
# File: house_prices.csv  
# Columns: bedrooms, bathrooms, sqft, price (target)

# Automatically detects regression task
# Output: R² scores for all models + best performer
```

### Example 3: Mixed Data Types
```python
# CSV with numeric and categorical features
# Automatic preprocessing handles:
# - Missing values → Imputed
# - Categorical features → Label encoded
# - String targets → Auto-encoded to numeric
```

## 🔧 Configuration Options

### File Path Configuration
Edit `config.py`:
```python
FILE_PATH = "C:/path/to/your/data.csv"
TARGET_COLUMN = "target_column_name"  # or None for last column
```

### Command Line Arguments
```bash
python accuracy_testing.py --file data.csv --target column_name
```

### Batch File Usage
- Automatic file dialog popup
- User-friendly prompts
- No technical knowledge required

## 📈 Output Format

### Batch File Output (Simplified)
```
CLASSIFICATION RESULTS - ACCURACY:
========================================
Logistic Regression : 0.847
Decision Tree       : 0.823
Random Forest       : 0.891
SVM                 : 0.856
Naive Bayes        : 0.834
KNN                : 0.829
Neural Network     : 0.863
Gradient Boosting  : 0.878

🏆 BEST MODEL: Random Forest (Accuracy: 0.891)
```

### Full Output (Command Line)
- Detailed metrics table with all performance measures
- Confusion matrices for classification
- Feature importance plots for tree models
- Comprehensive analysis

## 🛠️ Technical Requirements

### System Requirements
- **Operating System**: Windows (for batch file), Linux/Mac (command line)
- **Python**: 3.7 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 100MB for dependencies

### Python Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tabulate>=0.8.0
xgboost>=1.5.0 (optional)
```

## 🚨 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Solution: Install missing packages
pip install -r requirements.txt
```

**2. File Not Found**
- Check file path in `config.py`
- Ensure CSV file exists
- Use absolute file paths

**3. Convergence Warning**
- Neural network max iterations increased to 2000
- Warning is informational, results still valid

**4. Categorical Target Error**
- Framework auto-encodes string targets
- No manual preprocessing needed

### Performance Tips
- **Large datasets**: Use sampling for faster testing
- **Many features**: Consider feature selection
- **Slow execution**: Reduce model complexity or use fewer algorithms

## 📝 Data Format Requirements

### CSV File Structure
```csv
feature1,feature2,feature3,target
1.2,category_A,100,yes
2.1,category_B,150,no
1.8,category_A,120,yes
```

### Supported Data Types
- **Numeric**: Integers, floats
- **Categorical**: Strings, categories
- **Missing Values**: NaN, empty cells (auto-handled)
- **Target Variable**: Any column (numeric or categorical)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For issues, questions, or feature requests:
1. Check troubleshooting section above
2. Review code documentation
3. Create an issue in the repository
4. Contact the development team

---

**Happy Machine Learning! 🤖📊**