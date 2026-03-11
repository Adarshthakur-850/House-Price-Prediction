from src.data_loader import load_data
from src.feature_engineering import feature_engineering
from src.model import train_models
from src.visualization import plot_analysis
import os

def main():
    print("Starting House Price Prediction Pipeline...")
    
    # 1. Load Data
    df = load_data()
    
    # 2. Feature Engineering
    df_processed = feature_engineering(df)
    
    # 3. Training
    best_model, X_test, y_test, results = train_models(df_processed)
    
    # 4. Visualization
    plot_analysis(df, y_test, results, best_model, X_test)
    
    print("Pipeline completed.")

if __name__ == "__main__":
    main()
