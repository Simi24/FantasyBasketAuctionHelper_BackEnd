import pandas as pd
import numpy as np
from pulp import *
import os
import json

def load_predictions(predictions_path):
    return pd.read_csv(predictions_path)

def load_nba_players(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data['data']

def filter_active_players(predictions_df, nba_players):
    active_players = set()
    for player in nba_players:
        active_players.add((player['first_name'], player['last_name']))
    
    return predictions_df[predictions_df.apply(lambda row: (row['first_name'], row['last_name']) in active_players, axis=1)]

def assign_costs(predictions_df):
    predictions_df = predictions_df.copy()
    predictions_df['cost'] = predictions_df['predicted_pdk'].rank(pct=True) * 35 + 5
    return predictions_df

def optimize_squad(players_df, total_budget, num_centers=2, num_guards=4, num_forwards=4, excluded_squads=[], max_players_per_team=2):
    prob = LpProblem("Fantasy_Basketball_Team_Selection", LpMaximize)
    player_vars = LpVariable.dicts("Players", players_df.index, cat='Binary')

    # Objective function
    prob += lpSum([player_vars[i] * players_df.loc[i, 'predicted_pdk'] for i in players_df.index])

    # Total budget constraint
    prob += lpSum([player_vars[i] * players_df.loc[i, 'cost'] for i in players_df.index]) <= total_budget

    # Position constraints
    prob += lpSum([player_vars[i] for i in players_df[players_df['position'] == 'C'].index]) == num_centers
    prob += lpSum([player_vars[i] for i in players_df[players_df['position'] == 'G'].index]) == num_guards
    prob += lpSum([player_vars[i] for i in players_df[players_df['position'] == 'F'].index]) == num_forwards

    # Exclude previously found squads
    for squad in excluded_squads:
        prob += lpSum([player_vars[i] for i in squad if i in player_vars]) <= len(squad) - 1

    # Cost balancing constraints
    avg_player_cost = total_budget / (num_centers + num_guards + num_forwards)
    max_cost = players_df['cost'].max()
    min_cost = players_df['cost'].min()

    # Ensure at least one player is above average cost
    prob += lpSum([player_vars[i] for i in players_df[players_df['cost'] > avg_player_cost].index]) >= 1

    # Ensure at least one player is below average cost
    prob += lpSum([player_vars[i] for i in players_df[players_df['cost'] < avg_player_cost].index]) >= 1

    # Limit the number of very expensive players
    prob += lpSum([player_vars[i] for i in players_df[players_df['cost'] > (max_cost * 0.8)].index]) <= 3

    # Limit the number of very cheap players
    prob += lpSum([player_vars[i] for i in players_df[players_df['cost'] < (min_cost * 1.2)].index]) <= 3

    # Limit players from the same team
    for team in players_df['team_name'].unique():
        prob += lpSum([player_vars[i] for i in players_df[players_df['team_name'] == team].index]) <= max_players_per_team

    # Solve the problem
    prob.solve()

    if LpStatus[prob.status] != 'Optimal':
        return None, []

    selected_players = [i for i in players_df.index if player_vars[i].value() > 0.5]
    return players_df.loc[selected_players], selected_players

def generate_multiple_squads(players_df, total_budget, num_squads=50, max_appearances=5):
    squads = []
    excluded_squads = []
    player_appearances = {player: 0 for player in players_df.index}

    for i in range(num_squads):
        # Exclude players that have reached max appearances
        players_df_filtered = players_df[players_df.index.map(lambda x: player_appearances[x] < max_appearances)]
        
        optimized_squad, selected_players = optimize_squad(players_df_filtered, total_budget, excluded_squads=excluded_squads)
        if optimized_squad is None:
            print(f"Could not find more unique squads after {i} iterations.")
            break
        
        squads.append(optimized_squad)
        excluded_squads.append(selected_players)
        
        # Update player appearances
        for player in selected_players:
            player_appearances[player] += 1

    return squads

def save_squads_to_csv(squads, output_path):
    all_squads = pd.DataFrame()
    for i, squad in enumerate(squads, 1):
        squad = squad.copy()  # Create a copy to avoid SettingWithCopyWarning
        squad['squad_number'] = i
        squad['total_cost'] = squad['cost'].sum()
        squad['total_predicted_pdk'] = squad['predicted_pdk'].sum()
        all_squads = pd.concat([all_squads, squad], ignore_index=True)
        
        # Add a blank row after each squad
        blank_row = pd.DataFrame([[None] * len(squad.columns)], columns=squad.columns)
        all_squads = pd.concat([all_squads, blank_row], ignore_index=True)
    
    all_squads.to_csv(output_path, index=False)
    print(f"All squads saved to {output_path}")

def main():
    predictions_path = '/app/model/predictions_next_season.csv'
    nba_players_path = '/app/data/nba_players_2024-2025.json'
    
    predictions_df = load_predictions(predictions_path)
    nba_players = load_nba_players(nba_players_path)
    
    # Filter out players not in the NBA for the next season
    active_predictions_df = filter_active_players(predictions_df, nba_players)
    
    players_df = assign_costs(active_predictions_df)
    
    total_budget = 150
    num_squads = 50  # Number of squads to generate

    squads = generate_multiple_squads(players_df, total_budget, num_squads)

    for i, squad in enumerate(squads, 1):
        print(f"\nOptimized Squad {i}:")
        print(squad)
        print(f"Total Cost: {squad['cost'].sum():.2f}")
        print(f"Total Predicted PDK: {squad['predicted_pdk'].sum():.2f}")

        for position in ['C', 'G', 'F']:
            print(f"\n{position} players:")
            print(squad[squad['position'] == position])

    print("\nPlayer pool statistics:")
    print(players_df['cost'].describe())
    print("\nPlayers by position:")
    print(players_df['position'].value_counts())

    # Save all squads to a CSV file
    output_path = '/app/model/optimized_squads.csv'
    save_squads_to_csv(squads, output_path)

if __name__ == "__main__":
    main()
