import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Upload your autompg.csv file to Google Colab first
df = pd.read_csv('auto-mpg.csv')

print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check unique values in horsepower (likely contains '?' for missing values)
print(f"\nUnique horsepower values: {df['horsepower'].unique()}")

# Data Preprocessing
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# 1. Handle missing values in horsepower (replace '?' with NaN)
df['horsepower'] = df['horsepower'].replace('?', np.nan)

# Convert horsepower to numeric
df['horsepower'] = pd.to_numeric(df['horsepower'])

print(f"Missing values after conversion:")
print(df.isnull().sum())

# 2. Handle missing values - we'll use median imputation for horsepower
df['horsepower'].fillna(df['horsepower'].median(), inplace=True)

print(f"Missing values after imputation:")
print(df.isnull().sum())

# 3. Feature Engineering
# Convert model year to actual year (assuming 70 means 1970)
df['model_year_full'] = df['model year'] + 1900

# Create age of car feature (assuming current year is 2024 for reference)
df['car_age'] = 2024 - df['model_year_full']

# Create power-to-weight ratio
df['power_to_weight'] = df['horsepower'] / df['weight']

# Create displacement per cylinder
df['displacement_per_cylinder'] = df['displacement'] / df['cylinders']

print("\nNew features created:")
print("- model_year_full: Full year (1970-1982)")
print("- car_age: Age of the car")
print("- power_to_weight: Power to weight ratio")
print("- displacement_per_cylinder: Displacement per cylinder")

# 4. Encode categorical variables
# Origin can be treated as categorical (1=USA, 2=Europe, 3=Japan)
df['origin_usa'] = (df['origin'] == 1).astype(int)
df['origin_europe'] = (df['origin'] == 2).astype(int)
df['origin_japan'] = (df['origin'] == 3).astype(int)

# 5. Select features for modeling
# Exclude car name as it's too specific and has too many unique values
feature_columns = [
    'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
    'model year', 'model_year_full', 'car_age', 'power_to_weight',
    'displacement_per_cylinder', 'origin_usa', 'origin_europe', 'origin_japan'
]

# Target variable
target = 'mpg'

# Create feature matrix and target vector
X = df[feature_columns]
y = df[target]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"\nSelected features: {feature_columns}")

# 6. Check for correlations
print("\nCorrelation with target variable (MPG):")
correlations = df[feature_columns + [target]].corr()[target].sort_values(ascending=False)
print(correlations)

# 7. Split the data
print("\n" + "="*50)
print("TRAIN-TEST SPLIT")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# 8. Feature Scaling
print("\n" + "="*50)
print("FEATURE SCALING")
print("="*50)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features have been standardized (mean=0, std=1)")
print(f"Training set mean: {X_train_scaled.mean(axis=0).round(4)}")
print(f"Training set std: {X_train_scaled.std(axis=0).round(4)}")

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)

# 9. Model Training and Evaluation
print("\n" + "="*50)
print("MODEL TRAINING AND EVALUATION")
print("="*50)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Lasso Regression': Lasso(alpha=0.1, random_state=42)
}

# Store results
results = {}

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * len(name))

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Store results
    results[name] = {
        'model': model,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'y_test_pred': y_test_pred
    }

    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training RMSE: {np.sqrt(train_mse):.4f}")
    print(f"Test RMSE: {np.sqrt(test_mse):.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

# 10. Feature Importance Analysis
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# For Linear and Ridge Regression - coefficients
for name in ['Linear Regression', 'Ridge Regression']:
    print(f"\n{name} Coefficients:")
    model = results[name]['model']
    coef_df = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    print(coef_df)

# For Lasso - show non-zero coefficients
print(f"\nLasso Regression - Non-zero Coefficients:")
lasso_model = results['Lasso Regression']['model']
lasso_coef = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': lasso_model.coef_
})
non_zero_coef = lasso_coef[lasso_coef['Coefficient'] != 0].sort_values('Coefficient', key=abs, ascending=False)
print(non_zero_coef)
print(f"Number of features selected by Lasso: {len(non_zero_coef)}")

# 11. Data Visualization
print("\n" + "="*50)
print("CREATING VISUALIZATIONS")
print("="*50)

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create subplots for visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Auto MPG Dataset Analysis and Model Performance', fontsize=16)

# 1. Distribution of target variable
axes[0, 0].hist(df['mpg'], bins=20, alpha=0.7)
axes[0, 0].set_title('Distribution of MPG')
axes[0, 0].set_xlabel('Miles Per Gallon')
axes[0, 0].set_ylabel('Frequency')

# 2. Correlation heatmap of top features
top_features = correlations.abs().sort_values(ascending=False).head(8).index.tolist()
corr_matrix = df[top_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
axes[0, 1].set_title('Feature Correlation Heatmap')

# 3. MPG vs Weight (strongest negative correlation)
axes[0, 2].scatter(df['weight'], df['mpg'], alpha=0.6)
axes[0, 2].set_title('MPG vs Weight')
axes[0, 2].set_xlabel('Weight')
axes[0, 2].set_ylabel('MPG')

# 4. Model Performance Comparison
model_names = list(results.keys())
test_r2_scores = [results[name]['test_r2'] for name in model_names]
test_rmse_scores = [np.sqrt(results[name]['test_mse']) for name in model_names]

x_pos = np.arange(len(model_names))
axes[1, 0].bar(x_pos, test_r2_scores)
axes[1, 0].set_title('Model Performance (R² Score)')
axes[1, 0].set_xlabel('Models')
axes[1, 0].set_ylabel('R² Score')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(model_names, rotation=45)

# 5. Actual vs Predicted for best model
best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
best_predictions = results[best_model_name]['y_test_pred']
axes[1, 1].scatter(y_test, best_predictions, alpha=0.6)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_title(f'Actual vs Predicted ({best_model_name})')
axes[1, 1].set_xlabel('Actual MPG')
axes[1, 1].set_ylabel('Predicted MPG')

# 6. Residuals plot for best model
residuals = y_test - best_predictions
axes[1, 2].scatter(best_predictions, residuals, alpha=0.6)
axes[1, 2].axhline(y=0, color='r', linestyle='--')
axes[1, 2].set_title(f'Residuals Plot ({best_model_name})')
axes[1, 2].set_xlabel('Predicted MPG')
axes[1, 2].set_ylabel('Residuals')

plt.tight_layout()
plt.show()

# 12. Summary
print("\n" + "="*50)
print("PREPROCESSING SUMMARY")
print("="*50)
print(f"• Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"• Missing values handled: {6} missing horsepower values imputed with median")
print(f"• New features created: 4 (model_year_full, car_age, power_to_weight, displacement_per_cylinder)")
print(f"• Categorical encoding: Origin variable one-hot encoded")
print(f"• Feature scaling: StandardScaler applied to all features")
print(f"• Final feature set: {len(feature_columns)} features")
print(f"• Train-test split: 80%-20%")

print(f"\nBest performing model: {best_model_name}")
print(f"Test R² Score: {results[best_model_name]['test_r2']:.4f}")
print(f"Test RMSE: {np.sqrt(results[best_model_name]['test_mse']):.4f}")

print("\nThe data is now ready for machine learning!")
print("\nKey preprocessing steps completed:")
print("✓ Missing value imputation")
print("✓ Feature engineering")
print("✓ Categorical encoding")
print("✓ Feature scaling")
print("✓ Train-test split")
print("✓ Model training and evaluation")