import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*60)
print("ADVANCED AUTO MPG MACHINE LEARNING PIPELINE")
print("="*60)

# Load the dataset
df = pd.read_csv('auto-mpg.csv')

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")

# Advanced Data Preprocessing
print("\n" + "="*50)
print("ADVANCED DATA PREPROCESSING")
print("="*50)

# 1. Handle missing values in horsepower
print("Step 1: Handling missing values...")
df['horsepower'] = df['horsepower'].replace('?', np.nan)
df['horsepower'] = pd.to_numeric(df['horsepower'])


missing_hp_count = df['horsepower'].isnull().sum()
print(f"Missing horsepower values: {missing_hp_count}")

# Use median imputation for horsepower
df['horsepower'].fillna(df['horsepower'].median(), inplace=True)
print(f"Imputed with median: {df['horsepower'].median():.1f}")

# 2. Advanced Feature Engineering
print("\nStep 2: Advanced feature engineering...")

# Basic engineered features
df['model_year_full'] = df['model year'] + 1900
df['car_age'] = 2024 - df['model_year_full']
df['power_to_weight'] = df['horsepower'] / df['weight']
df['displacement_per_cylinder'] = df['displacement'] / df['cylinders']

# Advanced engineered features
df['weight_per_cylinder'] = df['weight'] / df['cylinders']
df['horsepower_per_cylinder'] = df['horsepower'] / df['cylinders']
df['efficiency_ratio'] = df['horsepower'] / df['displacement']
df['power_density'] = df['horsepower'] / (df['displacement'] * df['cylinders'])
df['acceleration_per_weight'] = df['acceleration'] / (df['weight'] / 1000)  # normalized by weight in tons

# Logarithmic transformations for skewed features
df['log_weight'] = np.log(df['weight'])
df['log_displacement'] = np.log(df['displacement'])
df['log_horsepower'] = np.log(df['horsepower'])

# Polynomial features for key relationships
df['weight_squared'] = df['weight'] ** 2
df['horsepower_squared'] = df['horsepower'] ** 2
df['displacement_squared'] = df['displacement'] ** 2

# Interaction features
df['weight_horsepower'] = df['weight'] * df['horsepower']
df['displacement_horsepower'] = df['displacement'] * df['horsepower']
df['cylinders_displacement'] = df['cylinders'] * df['displacement']

# 3. Enhanced categorical encoding
print("Step 3: Enhanced categorical encoding...")

# One-hot encoding for origin
df['origin_usa'] = (df['origin'] == 1).astype(int)
df['origin_europe'] = (df['origin'] == 2).astype(int)
df['origin_japan'] = (df['origin'] == 3).astype(int)

# Decade-based encoding for model year
df['decade_70s'] = ((df['model year'] >= 70) & (df['model year'] < 80)).astype(int)
df['decade_80s'] = (df['model year'] >= 80).astype(int)

# Cylinder category encoding
df['cyl_4'] = (df['cylinders'] == 4).astype(int)
df['cyl_6'] = (df['cylinders'] == 6).astype(int)
df['cyl_8'] = (df['cylinders'] == 8).astype(int)

print(f"Total engineered features: {len([col for col in df.columns if col not in ['mpg', 'car name']])}")

# 4. Feature Selection
print("\nStep 4: Feature selection based on correlation analysis...")

# Select all numeric features except target and car name
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('mpg')  # Remove target variable

# Calculate correlation with target
correlations = df[numeric_features + ['mpg']].corr()['mpg'].abs().sort_values(ascending=False)
print("\nTop 15 features by correlation with MPG:")
print(correlations.head(15))

# Select features with correlation > 0.1 and remove highly correlated features among themselves
selected_features = []
correlation_threshold = 0.1
multicollinearity_threshold = 0.95

for feature in correlations.index:
    if feature != 'mpg' and correlations[feature] > correlation_threshold:
        # Check for multicollinearity with already selected features
        if not selected_features:
            selected_features.append(feature)
        else:
            feature_corr_with_selected = df[selected_features + [feature]].corr()[feature].abs()
            max_corr = feature_corr_with_selected.drop(feature).max()
            if max_corr < multicollinearity_threshold:
                selected_features.append(feature)

print(f"\nSelected {len(selected_features)} features after correlation and multicollinearity filtering")
print("Selected features:", selected_features)

# Prepare feature matrix and target
X = df[selected_features]
y = df['mpg']

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# 5. Advanced Train-Test Split with Stratification
print("\n" + "="*50)
print("ADVANCED TRAIN-TEST SPLIT")
print("="*50)

# Create MPG bins for stratified sampling
y_binned = pd.cut(y, bins=5, labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_binned
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training target distribution: {y_train.describe()}")

# 6. Advanced Scaling and Pipeline Setup
print("\n" + "="*50)
print("MODEL PIPELINE SETUP")
print("="*50)

# Create cross-validation strategy
cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)

# Define comprehensive parameter grids for hyperparameter tuning
param_grids = {
    'Linear Regression': {},

    'Ridge Regression': {
        'regressor__alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    },

    'Lasso Regression': {
        'regressor__alpha': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    },

    'ElasticNet': {
        'regressor__alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
        'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
}

# Initialize models
base_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Lasso Regression': Lasso(random_state=42, max_iter=2000),
    'ElasticNet': ElasticNet(random_state=42, max_iter=2000)
}

# 7. Comprehensive Model Training and Hyperparameter Tuning
print("\n" + "="*50)
print("HYPERPARAMETER TUNING AND MODEL TRAINING")
print("="*50)

best_models = {}
cv_results = {}

for name, model in base_models.items():
    print(f"\nTraining {name}...")

    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])

    # Perform hyperparameter tuning if parameters exist
    if param_grids[name]:
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=cv_strategy,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best CV R² score: {grid_search.best_score_:.4f}")

    else:
        # For Linear Regression (no hyperparameters)
        pipeline.fit(X_train, y_train)
        best_models[name] = pipeline

        # Get CV scores manually
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, scoring='r2')
        print(f"CV R² score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 8. Comprehensive Model Evaluation
print("\n" + "="*50)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*50)

results = {}
feature_importance = {}

for name, model in best_models.items():
    print(f"\n{name} Evaluation:")
    print("-" * (len(name) + 12))

    # Cross-validation scores
    cv_r2_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='r2')
    cv_mse_scores = -cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='neg_mean_squared_error')
    cv_mae_scores = -cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='neg_mean_absolute_error')

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Store results
    results[name] = {
        'model': model,
        'cv_r2_mean': cv_r2_scores.mean(),
        'cv_r2_std': cv_r2_scores.std(),
        'cv_mse_mean': cv_mse_scores.mean(),
        'cv_mse_std': cv_mse_scores.std(),
        'cv_mae_mean': cv_mae_scores.mean(),
        'cv_mae_std': cv_mae_scores.std(),
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'y_test_pred': y_test_pred,
        'overfitting': train_r2 - test_r2
    }

    # Print results
    print(f"Cross-Validation Results:")
    print(f"  R² Score: {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
    print(f"  MSE: {cv_mse_scores.mean():.4f} ± {cv_mse_scores.std():.4f}")
    print(f"  RMSE: {np.sqrt(cv_mse_scores.mean()):.4f}")
    print(f"  MAE: {cv_mae_scores.mean():.4f} ± {cv_mae_scores.std():.4f}")

    print(f"\nTraining Set Performance:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  MSE: {train_mse:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE: {train_mae:.4f}")

    print(f"\nTest Set Performance:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  MSE: {test_mse:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")

    print(f"\nModel Characteristics:")
    print(f"  Overfitting (Train R² - Test R²): {train_r2 - test_r2:.4f}")

    # Feature importance analysis
    if hasattr(model.named_steps['regressor'], 'coef_'):
        coef = model.named_steps['regressor'].coef_
        feature_imp = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': coef,
            'Abs_Coefficient': np.abs(coef)
        }).sort_values('Abs_Coefficient', ascending=False)

        feature_importance[name] = feature_imp

        print(f"\nTop 10 Most Important Features:")
        print(feature_imp.head(10)[['Feature', 'Coefficient']].to_string(index=False))

        # For Lasso, show feature selection
        if 'Lasso' in name or 'ElasticNet' in name:
            non_zero_features = (coef != 0).sum()
            print(f"\nFeature Selection: {non_zero_features}/{len(selected_features)} features selected")

# 9. Model Comparison and Ranking
print("\n" + "="*50)
print("MODEL COMPARISON AND RANKING")
print("="*50)

# Create comparison DataFrame
comparison_data = []
for name, result in results.items():
    comparison_data.append({
        'Model': name,
        'CV_R2_Mean': result['cv_r2_mean'],
        'CV_R2_Std': result['cv_r2_std'],
        'Test_R2': result['test_r2'],
        'Test_RMSE': result['test_rmse'],
        'Test_MAE': result['test_mae'],
        'Overfitting': result['overfitting']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Test_R2', ascending=False)

print("Model Performance Comparison (Ranked by Test R²):")
print("=" * 80)
print(comparison_df.to_string(index=False, float_format='%.4f'))

# Identify best model
best_model_name = comparison_df.iloc[0]['Model']
best_model_result = results[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test R² Score: {best_model_result['test_r2']:.4f}")
print(f"   Test RMSE: {best_model_result['test_rmse']:.4f}")
print(f"   Cross-Validation R²: {best_model_result['cv_r2_mean']:.4f} ± {best_model_result['cv_r2_std']:.4f}")


# 10. Advanced Visualizations
print("\n" + "="*50)
print("CREATING ADVANCED VISUALIZATIONS")
print("="*50)

# Set up plotting style
plt.style.use('seaborn-v0_8')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

# 1. Model Performance Comparison
ax1 = fig.add_subplot(gs[0, :2])
x_pos = np.arange(len(comparison_df))
bars = ax1.bar(x_pos, comparison_df['Test_R2'], color=colors[:len(comparison_df)])
ax1.set_title('Model Performance Comparison (Test R² Score)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Models')
ax1.set_ylabel('R² Score')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(comparison_df['Model'], rotation=45)
ax1.set_ylim(0, 1)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. Cross-Validation Reliability
ax2 = fig.add_subplot(gs[0, 2:])
cv_means = comparison_df['CV_R2_Mean'].values
cv_stds = comparison_df['CV_R2_Std'].values
ax2.errorbar(x_pos, cv_means, yerr=cv_stds, fmt='o', capsize=5, capthick=2, linewidth=2)
ax2.set_xlabel('Models')
ax2.set_ylabel('CV R² Score')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(comparison_df['Model'], rotation=45)
ax2.grid(True, alpha=0.3)

# 3. Actual vs Predicted for Best Model
ax3 = fig.add_subplot(gs[1, :2])
best_predictions = best_model_result['y_test_pred']
ax3.scatter(y_test, best_predictions, alpha=0.7, color=colors[0], s=60)
min_val, max_val = min(y_test.min(), best_predictions.min()), max(y_test.max(), best_predictions.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax3.set_title(f'Actual vs Predicted - {best_model_name}', fontsize=14, fontweight='bold')























