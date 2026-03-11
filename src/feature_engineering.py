import pandas as pd
from sklearn.preprocessing import LabelEncoder

def feature_engineering(df):
    print("Feature Engineering...")
    
    # 1. House Age
    current_year = 2024
    df['house_age'] = current_year - df['year_built']
    
    # 2. Total Rooms
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # 3. Location Encoding (One-Hot)
    df = pd.get_dummies(df, columns=['location'], drop_first=True)
    
    # 4. Interaction Feature: Area per Room
    df['area_per_room'] = df['area'] / df['total_rooms']
    
    return df
