import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *
# Load the California Housing dataset
data = fetch_california_housing()

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.data)

# Choose 5000 random data points
np.random.seed(42)  # For reproducibility
indices = np.random.choice(X_scaled.shape[0], 5000, replace=False)
X_subset = X_scaled[indices]
y_subset = data.target[indices]

# Split the subset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 10, 30)
    min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5)

    # Create the model with suggested hyperparameters
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
         random_state=42,
        oob_score=True,
    )

    # Evaluate the model using cross-validation
    score = cross_val_score(rf, X_train, y_train, cv=3, scoring="neg_mean_squared_error").mean()

    return score

# Create the Optuna study and optimize
#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=50)

# Get the best hyperparameters
#best_params = study.best_params
#print("Best hyperparameters:", best_params)

# Fit Random Forest model
rf = RandomForestRegressor(n_estimators=187, max_depth = 22,min_weight_fraction_leaf=0.0012196356011150902, random_state=42, oob_score=True)
rf.fit(X_train, y_train)

quantiles = [5, 50, 95]
predictions_rf_gap = rf_gap_quantile_regression_parallel(X_train, y_train, X_test, quantiles, rf)
predictions_qrf = qrf_predict(rf, X_test, quantiles)
# Calculate MAE for each quantile for RF-GAP and QRF
mae_rf_gap = [mean_squared_error(y_test, predictions_rf_gap[:, i]) for i in range(len(quantiles))]
mae_qrf = [mean_squared_error(y_test, predictions_qrf[:, i]) for i in range(len(quantiles))]

# Output the comparison
for i, q in enumerate(quantiles):
    print(f"Quantile {q}% - MSE RF-GAP: {mae_rf_gap[i]:.4f}, MSE QRF: {mae_qrf[i]:.4f}")

# Visualize the results
for i, q in enumerate(quantiles):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, predictions_rf_gap[:, i], alpha=0.3, label='RF-GAP Predictions')
    plt.scatter(y_test, predictions_qrf[:, i], alpha=0.3, label='QRF Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel(f'Predicted {q}% Quantile')
    plt.title(f'Comparison of RF-GAP vs. QRF for {q}% Quantile')
    plt.legend()
    plt.show()
