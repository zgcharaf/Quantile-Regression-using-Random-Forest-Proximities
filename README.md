
# README: Quantile Regression Comparison Using Random Forest (QRF vs RF-GAP)

## Overview
This project demonstrates the application of two different methods for quantile regression using Random Forests: Quantile Regression Forests (QRF) and a custom method referred to as Random Forests with Gradient-based Adjustment for Prediction (RF-GAP). The comparison focuses on the Mean Squared Error (MSE) of the predictions at different quantiles (5th, 50th, and 95th) using the California Housing dataset.

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Code Overview](#code-overview)
4. [Running the Code](#running-the-code)
5. [Results](#results)
6. [Custom Functions](#custom-functions)
7. [Further Improvements](#further-improvements)

## Installation

To run the code, you need Python 3.x installed along with the following packages:
- `numpy`
- `scikit-learn`
- `optuna`
- `matplotlib`
- `joblib`

You can install the dependencies using pip:
```bash
pip install numpy scikit-learn optuna matplotlib joblib
```

## Dataset

The code uses the California Housing dataset from the `scikit-learn` library. This dataset contains information on housing prices in California based on features like median income, house age, and others.

## Code Overview

### 1. Data Preparation
- The California Housing dataset is fetched using `fetch_california_housing()`.
- The features are standardized using `StandardScaler` to ensure the model performs optimally.
- A random subset of 5000 data points is selected for training and testing to reduce computation time.
- The data is split into training and testing sets.

### 2. Hyperparameter Optimization
- `Optuna` is used to tune the hyperparameters of the Random Forest model.
- The hyperparameters optimized include the number of estimators, the maximum depth of the trees, and the minimum weight fraction of leaf nodes.
- Cross-validation is used to evaluate the performance of each hyperparameter set.

### 3. Model Training
- The Random Forest model is trained using the best hyperparameters obtained from `Optuna`.
- Predictions are made for the specified quantiles (5th, 50th, and 95th) using both QRF and RF-GAP methods.

### 4. Evaluation
- The Mean Squared Error (MSE) is calculated for each quantile and each method (QRF vs RF-GAP).
- The results are printed and visualized using scatter plots.

### 5. Visualization
- Scatter plots compare the predicted quantile values against the true values for both methods.
- A line representing the perfect prediction (y=x) is added for reference.

## Running the Code

1. Clone the repository or download the code.
2. Ensure all dependencies are installed (see Installation section).
3. Run the script in a Python environment:

```bash
python quantile_regression_comparison.py
```

4. The code will output MSE for each quantile and display scatter plots comparing the two methods.

## Results

- The MSE for each quantile and method is printed to the console.
- Scatter plots are generated to visualize the performance of RF-GAP and QRF for each quantile.

## Custom Functions

### 1. `rf_gap_quantile_regression_parallel(X_train, y_train, X_test, quantiles, rf)`
   - Implements RF-GAP quantile regression.
   - Uses parallel processing for faster computation.

### 2. `qrf_predict(rf, X_test, quantiles)`
   - Implements standard Quantile Regression Forest (QRF) prediction.

These functions are located in the `utils.py` file (which should be created or added separately).

## Further Improvements

- **Feature Engineering:** More advanced feature engineering could improve model performance.
- **Optimization:** Increase the number of trials in Optuna for better hyperparameter tuning.
- **Parallelization:** The entire process can be parallelized to reduce computation time further.
- **Extended Analysis:** Experiment with other datasets and models for more comprehensive analysis.

## Conclusion

This project provides a hands-on implementation of quantile regression using Random Forests. By comparing the QRF and RF-GAP methods, it offers insights into their performance for predicting different quantiles in a dataset.
