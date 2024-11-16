import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
import time
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def load_and_prepare_data(file_paths, seasons_to_use=2):
    dfs = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Extract season from file name
            season = re.search(r'(\d{4}-\d{4})', file_path).group(1)
            df['season'] = season
            dfs.append(df)
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not dfs:
        raise ValueError("No valid CSV files found.")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert relevant columns to numeric type
    numeric_columns = ['gp', 'min', 'pts', 'ast', 'reb', 'stl', 'blk', 'tov', 'fgm', 'fga', 'tpm', 'tpa', 'ftm', 'fta', 'pdk']
    combined_df[numeric_columns] = combined_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Create a unique player identifier
    combined_df['player_id'] = combined_df['first_name'] + ' ' + combined_df['last_name']
    
    # Convert season to a sortable format (use the first year of the season)
    combined_df['season_year'] = combined_df['season'].apply(lambda x: int(x.split('-')[0]))
    
    # Sort the dataframe by player_id and season_year
    combined_df = combined_df.sort_values(['player_id', 'season_year'])
    
    # Compute basic per-game and per-minute stats
    per_game_stats = ['pts', 'ast', 'reb', 'stl', 'blk', 'tov', 'min']
    for stat in per_game_stats:
        combined_df[f'{stat}_per_game'] = combined_df[stat] / combined_df['gp']
    
    per_minute_stats = ['pts', 'ast', 'reb', 'stl', 'blk']
    for stat in per_minute_stats:
        combined_df[f'{stat}_per_min'] = combined_df[stat] / combined_df['min']
    
    # Compute advanced stats
    combined_df['efficiency'] = (combined_df['pts'] + combined_df['reb'] + combined_df['ast'] + combined_df['stl'] + combined_df['blk'] - 
                                 (combined_df['fga'] - combined_df['fgm']) - (combined_df['fta'] - combined_df['ftm']) - combined_df['tov']) / combined_df['gp']
    combined_df['true_shooting'] = combined_df['pts'] / (2 * (combined_df['fga'] + 0.44 * combined_df['fta']))
    combined_df['effective_fg'] = (combined_df['fgm'] + 0.5 * combined_df['tpm']) / combined_df['fga']
    combined_df['usage_rate'] = 100 * ((combined_df['fga'] + 0.44 * combined_df['fta'] + combined_df['tov']) * 
                                       (combined_df['min'] / 5)) / (combined_df['min'] * (combined_df['fga'] + 0.44 * combined_df['fta'] + combined_df['tov']))
    combined_df['assist_ratio'] = combined_df['ast'] / (combined_df['fga'] + 0.44 * combined_df['fta'] + combined_df['ast'] + combined_df['tov'])
    combined_df['turnover_ratio'] = combined_df['tov'] / (combined_df['fga'] + 0.44 * combined_df['fta'] + combined_df['tov'])
    
    # Compute year-over-year changes
    features = ['gp', 'min', 'pts', 'ast', 'reb', 'stl', 'blk', 'tov', 'fgm', 'fga', 'tpm', 'tpa', 'ftm', 'fta',
                'pts_per_game', 'ast_per_game', 'reb_per_game', 'stl_per_game', 'blk_per_game', 'tov_per_game', 'min_per_game',
                'pts_per_min', 'ast_per_min', 'reb_per_min', 'stl_per_min', 'blk_per_min',
                'efficiency', 'true_shooting', 'effective_fg', 'usage_rate', 'assist_ratio', 'turnover_ratio']
    
    for feature in features:
        combined_df[f'{feature}_change'] = combined_df.groupby('player_id')[feature].diff()
    
    # Compute rolling averages
    for feature in features:
        combined_df[f'{feature}_rolling_2'] = combined_df.groupby('player_id')[feature].rolling(window=2, min_periods=1).mean().reset_index(0, drop=True)
        combined_df[f'{feature}_rolling_3'] = combined_df.groupby('player_id')[feature].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    
    # Create feature matrix
    all_features = features + [f'{f}_change' for f in features] + [f'{f}_rolling_2' for f in features] + [f'{f}_rolling_3' for f in features]
    feature_matrices = []
    pdks = []

    for _, player_data in combined_df.groupby('player_id'):
        if len(player_data) >= seasons_to_use + 1:
            for i in range(len(player_data) - seasons_to_use):
                feature_matrix = player_data.iloc[i:i+seasons_to_use][all_features].values.flatten()
                feature_matrices.append(feature_matrix)
                pdks.append(player_data.iloc[i+seasons_to_use]['pdk'])

    X = pd.DataFrame(feature_matrices)
    y = pd.Series(pdks)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Number of features created: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")

    return X_scaled, y, combined_df, scaler

def train_model(X, y, model_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Perform cross-validation
    start_time = time.time()
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    end_time = time.time()
    
    print(f"Cross-validation completed in {end_time - start_time:.2f} seconds")
    print(f"Cross-validation R-squared scores: {cv_scores}")
    print(f"Mean cross-validation R-squared: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train the final model
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Final model training completed in {end_time - start_time:.2f} seconds")
    
    # Evaluate on test set
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    test_r2 = r2_score(y_test, model.predict(X_test))
    
    print(f"Test set Mean Squared Error: {test_mse:.4f}")
    print(f"Test set R-squared Score: {test_r2:.4f}")
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved in {model_path}")
    
    return model

def predict_next_season(model, historical_data, scaler, seasons_to_use=2):
    features = ['gp', 'min', 'pts', 'ast', 'reb', 'stl', 'blk', 'tov', 'fgm', 'fga', 'tpm', 'tpa', 'ftm', 'fta',
                'pts_per_game', 'ast_per_game', 'reb_per_game', 'stl_per_game', 'blk_per_game', 'tov_per_game', 'min_per_game',
                'pts_per_min', 'ast_per_min', 'reb_per_min', 'stl_per_min', 'blk_per_min',
                'efficiency', 'true_shooting', 'effective_fg', 'usage_rate', 'assist_ratio', 'turnover_ratio']
    
    change_features = [f'{feature}_change' for feature in features]
    rolling_features = [f'{feature}_rolling_2' for feature in features] + [f'{feature}_rolling_3' for feature in features]
    all_features = features + change_features + rolling_features

    feature_matrices = []
    player_info = []

    for player_id, player_data in historical_data.groupby('player_id'):
        player_data = player_data.sort_values('season_year', ascending=False)
        
        if len(player_data) >= seasons_to_use:
            # Use the most recent seasons_to_use seasons
            feature_matrix = player_data.head(seasons_to_use)[all_features].values.flatten()
        else:
            # For players with fewer seasons, repeat the most recent season data
            most_recent_season = player_data.iloc[0][all_features].values
            feature_matrix = np.tile(most_recent_season, seasons_to_use)[:len(all_features) * seasons_to_use]
        
        feature_matrices.append(feature_matrix)
        player_info.append(player_data.iloc[0][['first_name', 'last_name', 'position', 'team_name']])

    X_last = pd.DataFrame(feature_matrices)
    X_last_scaled = scaler.transform(X_last)

    # Make predictions
    predictions = model.predict(X_last_scaled)

    # Create a result DataFrame
    result_df = pd.DataFrame(player_info)
    result_df['predicted_pdk'] = predictions
    result_df['seasons_of_data'] = historical_data.groupby('player_id').size().reindex(result_df.index).values

    # Sort by predicted PDK in descending order
    result_df = result_df.sort_values('predicted_pdk', ascending=False)

    return result_df

if __name__ == "__main__":
    data_dir = "/app/data"
    model_dir = "/app/model"
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the data directory.")
    else:
        file_paths = [os.path.join(data_dir, f) for f in csv_files]
        print("CSV files found:")
        for path in file_paths:
            print(path)
        
        try:
            X, y, historical_data, scaler = load_and_prepare_data(file_paths, seasons_to_use=2)
            model = train_model(X, y, model_path=os.path.join(model_dir, 'model.joblib'))
            
            predictions_next_season = predict_next_season(model, historical_data, scaler, seasons_to_use=2)
            
            if predictions_next_season.empty:
                print("No predictions were made. Check if there are players with data for at least 2 seasons.")
            else:
                predictions_next_season.to_csv("/app/model/predictions_next_season.csv", index=False)
                print("Predictions for next season saved to predictions_next_season.csv")
                
                print("\nTop 20 predicted players for next season:")
                print(predictions_next_season.head(20).to_string(index=False))
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            # Print additional debugging information
            print(f"Number of seasons in historical data: {historical_data['season_year'].nunique()}")
            print(f"Seasons available: {sorted(historical_data['season_year'].unique())}")
            print(f"Total number of players: {historical_data['player_id'].nunique()}")
            print(f"Number of players with at least 2 seasons:")
            print(historical_data.groupby('player_id').filter(lambda x: len(x) >= 2)['player_id'].nunique())
            print("\nSample of players with their season counts:")
            print(historical_data.groupby('player_id').size().sort_values(ascending=False).head(10))
            
            # Check for LeBron James specifically
            lebron_data = historical_data[historical_data['player_id'] == 'LeBron James']
            print("\nLeBron James data:")
            print(lebron_data[['season_year', 'team_name']])