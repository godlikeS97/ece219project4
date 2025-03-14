#!/usr/bin/env python3
"""
Script to compare different regression models for predicting tweet retweet counts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_data(csv_path):
    """
    Load the features dataset from CSV.
    
    Args:
        csv_path (str): Path to the features CSV file.
        
    Returns:
        pandas.DataFrame: Loaded features dataframe.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features.")
    return df

def preprocess_data(df):
    """
    Preprocess the data for regression modeling.
    
    Args:
        df (pandas.DataFrame): Raw features dataframe.
        
    Returns:
        tuple: X (features), y (target), column_names
    """
    print("Preprocessing data...")
    
    # Drop rows with NaN values
    df_clean = df.dropna()
    print(f"Dropped {len(df) - len(df_clean)} rows with missing values.")
    
    # Separate features and target
    X = df_clean.drop('retweet_count', axis=1)
    y = df_clean['retweet_count']
    
    # Create a binary target for logistic regression (0 for no retweets, 1 for at least one retweet)
    y_binary = (y > 0).astype(int)
    
    # Save column names for feature importance analysis
    column_names = X.columns.tolist()
    
    print(f"Target value distribution:")
    print(f"  - Mean: {y.mean():.2f}")
    print(f"  - Median: {y.median():.2f}")
    print(f"  - Min: {y.min():.2f}")
    print(f"  - Max: {y.max():.2f}")
    print(f"  - Tweets with retweets: {y_binary.sum()} ({y_binary.mean()*100:.2f}%)")
    
    return X, y, y_binary, column_names

def train_test_data_split(X, y, y_binary):
    """
    Split data into training and testing sets.
    
    Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Regression target.
        y_binary (pandas.Series): Binary classification target.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, y_binary_train, y_binary_test, scaler
    """
    print("Splitting data into training and testing sets...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _, _, y_binary_train, y_binary_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, y_binary_train, y_binary_test, scaler

def evaluate_regression_model(model_name, y_true, y_pred):
    """
    Evaluate regression model performance.
    
    Args:
        model_name (str): Name of the model.
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        
    Returns:
        dict: Performance metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    print(f"Performance of {model_name}:")
    print(f"  - MSE: {mse:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - R²: {r2:.4f}")
    
    return metrics

def evaluate_classification_model(model_name, y_true, y_pred):
    """
    Evaluate classification model performance.
    
    Args:
        model_name (str): Name of the model.
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        
    Returns:
        dict: Performance metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy
    }
    
    print(f"Performance of {model_name}:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Classification Report:")
    print(classification_report(y_true, y_pred))
    
    return metrics

def train_linear_models(X_train, X_test, y_train, y_test, column_names):
    """
    Train and evaluate linear regression models.
    
    Args:
        X_train (array-like): Training features.
        X_test (array-like): Testing features.
        y_train (array-like): Training target.
        y_test (array-like): Testing target.
        column_names (list): Feature names.
        
    Returns:
        list: Performance metrics for each model.
        dict: Trained models.
    """
    print("\n=== Linear Regression Models ===")
    metrics_list = []
    models = {}
    
    # Simple Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    metrics_lr = evaluate_regression_model("Linear Regression", y_test, y_pred_lr)
    metrics_list.append(metrics_lr)
    models['Linear Regression'] = lr
    
    # Print top coefficients
    coef_df = pd.DataFrame({'Feature': column_names, 'Coefficient': lr.coef_})
    coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)
    print("\nTop 10 features by coefficient magnitude (Linear Regression):")
    print(coef_df.head(10))
    
    # Lasso Regression with cross-validation for alpha selection
    lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
    lasso_cv = GridSearchCV(Lasso(max_iter=10000, random_state=42), lasso_params, cv=5, scoring='neg_mean_squared_error')
    lasso_cv.fit(X_train, y_train)
    best_alpha_lasso = lasso_cv.best_params_['alpha']
    print(f"\nBest Lasso alpha: {best_alpha_lasso}")
    
    lasso = Lasso(alpha=best_alpha_lasso, max_iter=10000, random_state=42)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    metrics_lasso = evaluate_regression_model(f"Lasso Regression (alpha={best_alpha_lasso})", y_test, y_pred_lasso)
    metrics_list.append(metrics_lasso)
    models['Lasso Regression'] = lasso
    
    # Print top non-zero coefficients
    coef_df = pd.DataFrame({'Feature': column_names, 'Coefficient': lasso.coef_})
    coef_df = coef_df.loc[coef_df.Coefficient != 0]
    coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)
    print("\nTop non-zero features by coefficient magnitude (Lasso Regression):")
    print(coef_df.head(10))
    
    # Ridge Regression with cross-validation for alpha selection
    ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge_cv = GridSearchCV(Ridge(random_state=42), ridge_params, cv=5, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_train, y_train)
    best_alpha_ridge = ridge_cv.best_params_['alpha']
    print(f"\nBest Ridge alpha: {best_alpha_ridge}")
    
    ridge = Ridge(alpha=best_alpha_ridge, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    metrics_ridge = evaluate_regression_model(f"Ridge Regression (alpha={best_alpha_ridge})", y_test, y_pred_ridge)
    metrics_list.append(metrics_ridge)
    models['Ridge Regression'] = ridge
    
    # Print top coefficients
    coef_df = pd.DataFrame({'Feature': column_names, 'Coefficient': ridge.coef_})
    coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)
    print("\nTop 10 features by coefficient magnitude (Ridge Regression):")
    print(coef_df.head(10))
    
    return metrics_list, models

def train_logistic_regression(X_train, X_test, y_binary_train, y_binary_test, column_names):
    """
    Train and evaluate logistic regression model.
    
    Args:
        X_train (array-like): Training features.
        X_test (array-like): Testing features.
        y_binary_train (array-like): Training binary target.
        y_binary_test (array-like): Testing binary target.
        column_names (list): Feature names.
        
    Returns:
        dict: Performance metrics.
        object: Trained model.
    """
    print("\n=== Logistic Regression Model ===")
    
    # Logistic Regression with regularization
    log_reg_params = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
    log_reg_cv = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), log_reg_params, cv=5, scoring='accuracy')
    log_reg_cv.fit(X_train, y_binary_train)
    best_C = log_reg_cv.best_params_['C']
    print(f"Best Logistic Regression C: {best_C}")
    
    log_reg = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_binary_train)
    y_pred_log_reg = log_reg.predict(X_test)
    metrics_log_reg = evaluate_classification_model("Logistic Regression", y_binary_test, y_pred_log_reg)
    
    # Print top coefficients
    coef_df = pd.DataFrame({'Feature': column_names, 'Coefficient': log_reg.coef_[0]})
    coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)
    print("\nTop 10 features by coefficient magnitude (Logistic Regression):")
    print(coef_df.head(10))
    
    return metrics_log_reg, log_reg

def train_xgboost(X_train, X_test, y_train, y_test, column_names):
    """
    Train and evaluate XGBoost model.
    
    Args:
        X_train (array-like): Training features.
        X_test (array-like): Testing features.
        y_train (array-like): Training target.
        y_test (array-like): Testing target.
        column_names (list): Feature names.
        
    Returns:
        dict: Performance metrics.
        object: Trained model.
    """
    print("\n=== XGBoost Model ===")
    
    # XGBoost with cross-validation for hyperparameter selection
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
    
    xgb_reg = xgb.XGBRegressor(random_state=42)
    xgb_cv = GridSearchCV(xgb_reg, xgb_params, cv=3, scoring='neg_mean_squared_error', verbose=0)
    xgb_cv.fit(X_train, y_train)
    
    best_params = xgb_cv.best_params_
    print(f"Best XGBoost parameters: {best_params}")
    
    # Train with best parameters
    xgb_best = xgb.XGBRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        random_state=42
    )
    xgb_best.fit(X_train, y_train)
    y_pred_xgb = xgb_best.predict(X_test)
    metrics_xgb = evaluate_regression_model("XGBoost", y_test, y_pred_xgb)
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': column_names,
        'Importance': xgb_best.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    print("\nTop 10 features by importance (XGBoost):")
    print(feature_importance.head(10))
    
    return metrics_xgb, xgb_best

def train_neural_network(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a 3-layer neural network.
    
    Args:
        X_train (array-like): Training features.
        X_test (array-like): Testing features.
        y_train (array-like): Training target.
        y_test (array-like): Testing target.
        
    Returns:
        dict: Performance metrics.
        object: Trained model.
    """
    print("\n=== Neural Network Model (3 layers) ===")
    
    # Create a sequential model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    print("Training neural network...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate the model
    y_pred_nn = model.predict(X_test).flatten()
    metrics_nn = evaluate_regression_model("Neural Network (3 layers)", y_test, y_pred_nn)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Neural Network Training History')
    plt.legend()
    plt.savefig('neural_network_training.png')
    plt.close()
    
    print(f"Neural network training took {len(history.history['loss'])} epochs")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    return metrics_nn, model

def plot_feature_importance(models, column_names):
    """
    Plot feature importance for different models.
    
    Args:
        models (dict): Trained models.
        column_names (list): Feature names.
    """
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    # Linear Regression Coefficients
    lr_coef = pd.DataFrame({'Feature': column_names, 'Coefficient': models['Linear Regression'].coef_})
    lr_coef = lr_coef.reindex(lr_coef.Coefficient.abs().sort_values(ascending=False).index)
    top_lr_features = lr_coef.head(10)
    axes[0].barh(top_lr_features['Feature'], top_lr_features['Coefficient'])
    axes[0].set_title('Linear Regression: Top 10 Feature Coefficients')
    axes[0].set_xlabel('Coefficient Value')
    
    # Lasso Coefficients (non-zero only)
    lasso_coef = pd.DataFrame({'Feature': column_names, 'Coefficient': models['Lasso Regression'].coef_})
    lasso_coef = lasso_coef.loc[lasso_coef.Coefficient != 0]
    lasso_coef = lasso_coef.reindex(lasso_coef.Coefficient.abs().sort_values(ascending=False).index)
    top_lasso_features = lasso_coef.head(10)
    axes[1].barh(top_lasso_features['Feature'], top_lasso_features['Coefficient'])
    axes[1].set_title('Lasso Regression: Top Non-zero Feature Coefficients')
    axes[1].set_xlabel('Coefficient Value')
    
    # XGBoost Feature Importance
    xgb_importance = pd.DataFrame({
        'Feature': column_names,
        'Importance': models['XGBoost'].feature_importances_
    })
    xgb_importance = xgb_importance.sort_values('Importance', ascending=False)
    top_xgb_features = xgb_importance.head(10)
    axes[2].barh(top_xgb_features['Feature'], top_xgb_features['Importance'])
    axes[2].set_title('XGBoost: Top 10 Feature Importance')
    axes[2].set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png')
    plt.close()

def plot_model_comparison(regression_metrics):
    """
    Plot performance comparison of regression models.
    
    Args:
        regression_metrics (list): Performance metrics for each model.
    """
    # Create DataFrame for plotting
    metrics_df = pd.DataFrame(regression_metrics)
    
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot RMSE
    axes[0].bar(metrics_df['Model'], metrics_df['RMSE'])
    axes[0].set_title('RMSE Comparison (lower is better)')
    axes[0].set_ylabel('RMSE')
    axes[0].set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
    
    # Plot R² Score
    axes[1].bar(metrics_df['Model'], metrics_df['R²'])
    axes[1].set_title('R² Score Comparison (higher is better)')
    axes[1].set_ylabel('R²')
    axes[1].set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def plot_predictions(models, X_test, y_test):
    """
    Plot actual vs predicted values for each model.
    
    Args:
        models (dict): Dictionary of trained models.
        X_test (array-like): Test features.
        y_test (array-like): Test targets.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    models_to_plot = [
        'Linear Regression',
        'Lasso Regression',
        'Ridge Regression',
        'XGBoost'
    ]
    
    for i, model_name in enumerate(models_to_plot):
        model = models[model_name]
        y_pred = model.predict(X_test)
        
        # Select a random subset for clearer visualization (max 1000 points)
        if len(y_test) > 1000:
            indices = np.random.choice(len(y_test), 1000, replace=False)
            y_true_sample = y_test.iloc[indices] if hasattr(y_test, 'iloc') else y_test[indices]
            y_pred_sample = y_pred[indices]
        else:
            y_true_sample = y_test
            y_pred_sample = y_pred
        
        # Create scatter plot
        axes[i].scatter(y_true_sample, y_pred_sample, alpha=0.5)
        
        # Add perfect prediction line
        max_val = max(y_true_sample.max(), y_pred_sample.max())
        axes[i].plot([0, max_val], [0, max_val], 'r--')
        
        axes[i].set_title(f'{model_name}: Actual vs Predicted')
        axes[i].set_xlabel('Actual Values')
        axes[i].set_ylabel('Predicted Values')
        
        # Add R² to the plot
        r2 = r2_score(y_test, y_pred)
        axes[i].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[i].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('predictions_comparison.png')
    plt.close()

def main():
    """
    Main function to run the regression model comparison.
    """
    # Load the data
    csv_path = 'data/superbowl_features.csv'
    df = load_data(csv_path)
    
    # Preprocess the data
    X, y, y_binary, column_names = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test, y_binary_train, y_binary_test, scaler = train_test_data_split(X, y, y_binary)
    
    # Train and evaluate linear models
    linear_metrics, linear_models = train_linear_models(X_train, X_test, y_train, y_test, column_names)
    
    # Train and evaluate logistic regression
    logistic_metrics, logistic_model = train_logistic_regression(X_train, X_test, y_binary_train, y_binary_test, column_names)
    
    # Train and evaluate XGBoost
    xgb_metrics, xgb_model = train_xgboost(X_train, X_test, y_train, y_test, column_names)
    linear_models['XGBoost'] = xgb_model
    
    # Train and evaluate neural network
    nn_metrics, nn_model = train_neural_network(X_train, X_test, y_train.values, y_test.values)
    
    # Combine all regression metrics
    all_regression_metrics = linear_metrics + [xgb_metrics, nn_metrics]
    
    # Plot model comparison
    plot_model_comparison(all_regression_metrics)
    
    # Plot feature importance
    plot_feature_importance(linear_models, column_names)
    
    # Plot predictions
    plot_predictions(linear_models, X_test, y_test)
    
    # Print summary of best model
    print("\n=== Model Comparison Summary ===")
    metrics_df = pd.DataFrame(all_regression_metrics)
    best_rmse_model = metrics_df.loc[metrics_df['RMSE'].idxmin()]
    best_r2_model = metrics_df.loc[metrics_df['R²'].idxmax()]
    
    print(f"Best model by RMSE: {best_rmse_model['Model']} (RMSE = {best_rmse_model['RMSE']:.4f})")
    print(f"Best model by R²: {best_r2_model['Model']} (R² = {best_r2_model['R²']:.4f})")
    
    print("\nLogistic Regression performance for binary classification (has retweets or not):")
    print(f"Accuracy: {logistic_metrics['Accuracy']:.4f}")
    
    print("\nThe model comparison results have been saved as:")
    print("- model_comparison.png (RMSE and R² comparison)")
    print("- feature_importance_comparison.png (top features by model)")
    print("- predictions_comparison.png (actual vs predicted values)")
    print("- neural_network_training.png (neural network training history)")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 