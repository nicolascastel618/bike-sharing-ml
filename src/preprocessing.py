import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath='../data/raw/hour.csv'):
    """
    Loads raw data, removes data leakage, and creates targets.
    """
    df = pd.read_csv(filepath)
    
    # 1. Remove Data Leakage and Redundancy
    # 'casual' and 'registered' sum up to 'cnt'. If we keep them, the model cheats.
    # 'atemp' is highly correlated with 'temp' (multicollinearity identified in EDA).
    # 'dteday' and 'instant' are redundant identifiers.
    df_clean = df.drop(columns=['casual', 'registered', 'atemp', 'dteday', 'instant'])
    
    # 2. Create Classification Target (is_peak_demand)
    # Define peak demand as the top 25% of hourly rentals
    threshold = df_clean['cnt'].quantile(0.75)
    df_clean['is_peak_demand'] = (df_clean['cnt'] >= threshold).astype(int)
    
    # 3. Separate Features (X) and Targets (y_reg, y_clf)
    X = df_clean.drop(columns=['cnt', 'is_peak_demand'])
    y_reg = df_clean['cnt']
    y_clf = df_clean['is_peak_demand']
    
    return X, y_reg, y_clf

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets, and applies standard scaling.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale numerical features (Crucial for regularized models and K-Means)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to Pandas DataFrames to retain column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Quick test to ensure the script works when run directly
    X, y_reg, y_clf = load_and_preprocess()
    print(f"Features shape: {X.shape}")
    print(f"Regression target shape: {y_reg.shape}")
    print(f"Classification target shape: {y_clf.shape}")