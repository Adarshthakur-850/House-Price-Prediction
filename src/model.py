from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import numpy as np

def train_models(df):
    print("Training models...")
    
    target = 'price'
    X = df.drop(columns=[target, 'year_built']) # Drop original year_built as we use age
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    best_rmse = float('inf')
    best_model = None
    results = {}
    
    print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'R2':<10}")
    print("-" * 50)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name:<20} {mae:.2f}     {rmse:.2f}     {r2:.4f}")
        
        results[name] = y_pred
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            
    # Save Model
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(best_model, "models/price_model.pkl")
    print("Best model saved.")
    
    return best_model, X_test, y_test, results
