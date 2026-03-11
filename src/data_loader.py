import pandas as pd
import numpy as np
import random
import os

def load_data(filepath="data/housing_data.csv"):
    if not os.path.exists(filepath):
        print("Generating synthetic housing data...")
        generate_synthetic_data(filepath)
    return pd.read_csv(filepath)

def generate_synthetic_data(filepath):
    np.random.seed(42)
    random.seed(42)
    
    n_samples = 1500
    locations = ['Downtown', 'Suburb', 'Countryside', 'City Center', 'Uptown']
    
    data = []
    
    for _ in range(n_samples):
        area = random.randint(800, 4000)
        bedrooms = random.randint(1, 6)
        bathrooms = random.randint(1, 4)
        floors = random.randint(1, 3)
        year_built = random.randint(1950, 2023)
        garage_size = random.randint(0, 3)
        location = random.choice(locations)
        
        # Price Calculation Logic (Synthetic Ground Truth)
        base_price_per_sqft = 150
        
        loc_multiplier = {
            'Downtown': 1.5,
            'City Center': 1.4,
            'Uptown': 1.3,
            'Suburb': 1.0,
            'Countryside': 0.8
        }
        
        price = (area * base_price_per_sqft * loc_multiplier[location])
        price += (bedrooms * 10000) + (bathrooms * 15000) + (garage_size * 5000)
        
        # Age depreciation
        age = 2024 - year_built
        price -= (age * 500)
        
        # Noise
        noise = np.random.normal(0, 20000)
        price += noise
        
        price = max(50000, price)
        
        data.append({
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'floors': floors,
            'year_built': year_built,
            'garage_size': garage_size,
            'location': location,
            'price': round(price, 2)
        })
        
    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data generated at {filepath}")
