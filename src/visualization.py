import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

def plot_analysis(df, y_test, results, model, X_test):
    print("Generating plots...")
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    # 1. Correlation Heatmap (numeric only)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation")
    plt.savefig("plots/correlation.png")
    plt.close()
    
    # 2. Actual vs Predicted
    # Use XGBoost or Random Forest predictions usually
    y_pred = list(results.values())[-1] # Take the last one (XGBoost)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted")
    plt.savefig("plots/actual_vs_predicted.png")
    plt.close()
    
    # 3. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X_test.columns
        ids = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[ids], y=[features[i] for i in ids], palette="viridis")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png")
        plt.close()
