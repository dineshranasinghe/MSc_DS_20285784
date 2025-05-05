
# Retail Sales Forecasting with Traffic Data

This repository contains the code and necessary data files for a machine learning project focused on **Retail Sales Forecasting** using traffic data. The project aims to predict retail sales based on historical data of website traffic, holidays, weekends, and other features. The model is built using XGBoost and includes exploratory data analysis (EDA), data preprocessing, outlier detection, and model training.

## Project Structure

The project is organized into the following directories:

- **Code**: This folder contains all the Python scripts for data preprocessing, analysis, model training, and evaluation.
- **Required_Data**: This folder contains the necessary data files for training the model, including traffic and sales data.

## Installation

### Prerequisites

- Python 3.x
- Jupyter Notebook (or any other preferred environment for running notebooks)
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost
  - plotly

You can install the required dependencies using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost plotly
```

Alternatively, you can install the dependencies from a `requirements.txt` file (if available) by running:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Exploratory Data Analysis (EDA)

The initial analysis and visualizations of the data are provided in the `EDA` notebook within the **Code** folder. This will help you understand the structure of the data and guide you through the preprocessing steps.

### Step 2: Data Preprocessing

Data preprocessing is performed in the `Data_Cleaning` notebook. This includes handling missing values, converting data types, encoding categorical features, and performing outlier detection using different techniques such as Z-score, IQR, and Isolation Forest.

### Step 3: Feature Engineering and Model Training

The `Model_Training` notebook in the **Code** folder is where the model is trained using various machine learning algorithms. The model is evaluated on predictive accuracy, with a focus on using XGBoost for sales forecasting.

### Step 4: Model Evaluation

After training the model, it's important to assess its performance. Metrics such as R-squared (R²) and mean squared error (MSE) are used to evaluate the model’s predictive accuracy.

## Files

### Code Folder:
- **Data_Cleaning.ipynb**: Contains the data cleaning and preprocessing code.
- **Exploratory_Data_Analysis.ipynb**: Contains the exploratory data analysis code, including visualizations.
- **Model_Training.ipynb**: Contains the code for training and evaluating the XGBoost model.

### Required_Data Folder:
- **sales_data.csv**: Sales data for retail transactions.
- **traffic_data.csv**: Website traffic data used for prediction.

## Results

The final model demonstrates that **traffic volume** is a key predictor for retail sales, showing a positive correlation. The **XGBoost model** outperforms other algorithms, providing actionable insights for business decision-making.

### Key Findings:
- Traffic volume is the strongest predictor of sales.
- Holidays have minimal impact, while weekends show higher sales activity.
- Time of day influences sales, with peaks during afternoon and evening hours.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was inspired by research on retail sales forecasting and machine learning.
- Thanks to the creators of the libraries and datasets used in this project.
