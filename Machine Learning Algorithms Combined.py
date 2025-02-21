import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import uniform, randint
import math
import matplotlib.pyplot as plt

# -----------------------------
# 1. Data Loading and Preprocessing
# -----------------------------
file_path = 'Data prepared.xlsx'
df = pd.read_excel(file_path, sheet_name='Data')

# Define explanatory variables and target
explanatory_vars = [
    'CapStatus', 'Had Substantive Repair', 'ExternalWallMaterial',
    'RoofMaterial', 'YearBuilt', 'NoOfLevels', 'FoundationTypeInd',
    'Total Building Paid Incl GST', 'Damage Label'
]
X = df[explanatory_vars]
y = df['Recovery time']

# Save the original 'Damage Label' for later output
damage_labels = df['Damage Label']

# Process binary variables directly
binary_vars = ['CapStatus', 'Had Substantive Repair']
X_binary = X[binary_vars]

# List of multi-class categorical variables to be one-hot encoded
multiclass_vars = [
    'ExternalWallMaterial', 'RoofMaterial', 'FoundationTypeInd', 'Damage Label'
]
X_multiclass_encoded = pd.get_dummies(X[multiclass_vars])

# Extract numerical variables
numerical_vars = ['YearBuilt', 'NoOfLevels', 'Total Building Paid Incl GST']
X_numerical = X[numerical_vars]

# Concatenate all processed parts
X_processed = pd.concat([X_binary, X_multiclass_encoded, X_numerical], axis=1)

# Split into training and testing sets (80:20)
X_train, X_test, y_train, y_test, damage_labels_train, damage_labels_test = train_test_split(
    X_processed, y, damage_labels, test_size=0.2, random_state=42)

# Number of predictors (used in AIC computation)
p = X_train.shape[1]
n_test = len(y_test)

# Function to compute AIC (using mse computed on test data)
def compute_aic(mse, n, p):
    return n * np.log(mse) + 2 * (p + 1)

# Function to compute Mean Bias Deviation (MBD)
def compute_mbd(y_true, y_pred):
    return np.mean(y_pred - y_true)

# -----------------------------
# 2. Model Training and Evaluation for Each Model
# -----------------------------

# Dictionary to store results for each model
results_dict = {}

### 2.1 Gradient Boosting Regressor (GBM)
print("Training Gradient Boosting Regressor...")
gbm = GradientBoostingRegressor(random_state=42)
param_grid_gbm = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
grid_search_gbm = GridSearchCV(estimator=gbm, param_grid=param_grid_gbm, cv=5, scoring='neg_mean_squared_error')
grid_search_gbm.fit(X_train, y_train)

best_params_gbm = grid_search_gbm.best_params_
best_cv_score_gbm = -grid_search_gbm.best_score_

best_gbm = grid_search_gbm.best_estimator_
best_gbm.fit(X_train, y_train)
y_pred_gbm = best_gbm.predict(X_test)

mse_gbm = mean_squared_error(y_test, y_pred_gbm)
mae_gbm = mean_absolute_error(y_test, y_pred_gbm)
mbd_gbm = compute_mbd(y_test, y_pred_gbm)
aic_gbm = compute_aic(mse_gbm, n_test, p)

results_gbm = pd.DataFrame({
    "Observed Recovery Time": y_test,
    "Predicted Recovery Time": y_pred_gbm,
    "Damage Label": damage_labels_test,
    "MSE": [mse_gbm] * n_test,
    "MAE": [mae_gbm] * n_test,
    "Best CV MSE": [best_cv_score_gbm] * n_test,
    "Best Params": [str(best_params_gbm)] * n_test,
    "MBD": [mbd_gbm] * n_test,
    "AIC": [aic_gbm] * n_test
})
results_dict["GBM"] = results_gbm

print("Gradient Boosting Results:")
print("Mean Squared Error:", mse_gbm)
print("Mean Absolute Error:", mae_gbm)

### 2.2 Random Forest Regressor
print("\nTraining Random Forest Regressor...")
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500]
}
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)

best_params_rf = grid_search_rf.best_params_
best_cv_score_rf = -grid_search_rf.best_score_

best_rf = grid_search_rf.best_estimator_
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mbd_rf = compute_mbd(y_test, y_pred_rf)
aic_rf = compute_aic(mse_rf, n_test, p)

results_rf = pd.DataFrame({
    "Observed Recovery Time": y_test,
    "Predicted Recovery Time": y_pred_rf,
    "Damage Label": damage_labels_test,
    "MSE": [mse_rf] * n_test,
    "MAE": [mae_rf] * n_test,
    "Best CV MSE": [best_cv_score_rf] * n_test,
    "Best Params": [str(best_params_rf)] * n_test,
    "MBD": [mbd_rf] * n_test,
    "AIC": [aic_rf] * n_test
})
results_dict["RandomForest"] = results_rf

print("Random Forest Results:")
print("Mean Squared Error:", mse_rf)
print("Mean Absolute Error:", mae_rf)

### 2.3 XGBoost Regressor
print("\nTraining XGBoost Regressor...")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
param_dist_xgb = {
    'n_estimators': randint(50, 300),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.8, 0.2),
    'colsample_bytree': uniform(0.8, 0.2)
}
random_search_xgb = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist_xgb,
                                       n_iter=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search_xgb.fit(X_train, y_train)

best_params_xgb = random_search_xgb.best_params_
best_cv_score_xgb = -random_search_xgb.best_score_

best_xgb = random_search_xgb.best_estimator_
best_xgb.fit(X_train, y_train)
y_pred_xgb = best_xgb.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mbd_xgb = compute_mbd(y_test, y_pred_xgb)
aic_xgb = compute_aic(mse_xgb, n_test, p)

results_xgb = pd.DataFrame({
    "Observed Recovery Time": y_test,
    "Predicted Recovery Time": y_pred_xgb,
    "Damage Label": damage_labels_test,
    "MSE": [mse_xgb] * n_test,
    "MAE": [mae_xgb] * n_test,
    "Best CV MSE": [best_cv_score_xgb] * n_test,
    "Best Params": [str(best_params_xgb)] * n_test,
    "MBD": [mbd_xgb] * n_test,
    "AIC": [aic_xgb] * n_test
})
results_dict["XGBoost"] = results_xgb

print("XGBoost Results:")
print("Mean Squared Error:", mse_xgb)
print("Mean Absolute Error:", mae_xgb)

### 2.4 Decision Tree Regressor
print("\nTraining Decision Tree Regressor...")
dt = DecisionTreeRegressor(random_state=42)
param_grid_dt = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}
grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, cv=5, scoring='neg_mean_squared_error')
grid_search_dt.fit(X_train, y_train)

best_params_dt = grid_search_dt.best_params_
best_cv_score_dt = -grid_search_dt.best_score_

best_dt = grid_search_dt.best_estimator_
best_dt.fit(X_train, y_train)
y_pred_dt = best_dt.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mbd_dt = compute_mbd(y_test, y_pred_dt)
aic_dt = compute_aic(mse_dt, n_test, p)

results_dt = pd.DataFrame({
    "Observed Recovery Time": y_test,
    "Predicted Recovery Time": y_pred_dt,
    "Damage Label": damage_labels_test,
    "MSE": [mse_dt] * n_test,
    "MAE": [mae_dt] * n_test,
    "Best CV MSE": [best_cv_score_dt] * n_test,
    "Best Params": [str(best_params_dt)] * n_test,
    "MBD": [mbd_dt] * n_test,
    "AIC": [aic_dt] * n_test
})
results_dict["DecisionTree"] = results_dt

print("Decision Tree Results:")
print("Mean Squared Error:", mse_dt)
print("Mean Absolute Error:", mae_dt)

# -----------------------------
# 3. Save All Results to a Single Excel File with Separate Sheets
# -----------------------------
output_file_path = 'Integrated_Model_Results.xlsx'
with pd.ExcelWriter(output_file_path) as writer:
    for model_name, df_results in results_dict.items():
        df_results.to_excel(writer, sheet_name=model_name, index=False)

print("\nAll model results have been saved to:", output_file_path)

# -----------------------------
# 4. Plotting Comparison of Observed vs. Predicted Recovery Times
# -----------------------------
# Load all sheets from the saved Excel file into a dictionary of DataFrames
sheets_dict = pd.read_excel(output_file_path, sheet_name=None)

# Since the observed recovery times are the same across sheets,
# we use the observed data from the first sheet (e.g., "GBM")
observed_data = sheets_dict["GBM"]['Observed Recovery Time']

# Dictionary to store predicted recovery times from each model
predicted_data = {}
for model_name, df in sheets_dict.items():
    predicted_data[model_name] = df['Predicted Recovery Time']

# Set Times New Roman font for all text in plots
plt.rcParams['font.family'] = 'Times New Roman'

# Create the histogram plot
plt.figure(figsize=(12, 8))
bins = 30
alpha = 0.7

# Plot histogram for observed recovery times
plt.hist(observed_data, bins=bins, alpha=alpha, label='Observed Recovery Time',
         color='blue', edgecolor='black')

# Define distinct colors for each model
colors = {'GBM': 'red', 'RandomForest': 'green', 'XGBoost': 'purple', 'DecisionTree': 'orange'}

# Plot histograms for predicted recovery times for each model
for model_name, data in predicted_data.items():
    plt.hist(data, bins=bins, alpha=alpha, label=f'{model_name} - Predicted',
             color=colors.get(model_name, 'gray'), edgecolor='black')

# Add labels and grid
plt.xlabel('Recovery Time (days)', fontsize=24)
plt.ylabel('Frequency', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, alpha=0.5)

# Place legend outside the plot area on the upper left side
plt.legend(fontsize=18, bbox_to_anchor=(0.55, 1), loc='upper left')

# Adjust layout to accommodate the legend
plt.subplots_adjust(left=0.08, right=0.8, top=0.95, bottom=0.1)

# Save the figure with high dpi
plt.savefig('Figure_7.jpg', format='jpeg', dpi=800)
plt.show()
