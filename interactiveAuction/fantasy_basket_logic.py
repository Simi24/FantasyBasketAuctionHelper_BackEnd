import pandas as pd
import numpy as np
from pulp import *
import json
import logging

def load_predictions(predictions_path):
    return pd.read_csv(predictions_path)

def load_nba_players(csv_path):
    df = pd.read_csv(csv_path)
    return df[['Nome', 'Cognome']]

def filter_active_players(predictions_df, nba_players):
    active_players = set(zip(
        nba_players['Nome'].str.strip().str.lower(),
        nba_players['Cognome'].str.strip().str.lower()
    ))
    
    return predictions_df[predictions_df.apply(lambda row: 
        (row['first_name'].strip().lower(), row['last_name'].strip().lower()) in active_players, axis=1)]

def assign_costs(predictions_df):
    predictions_df = predictions_df.copy()
    predictions_df['cost'] = predictions_df['predicted_pdk'].rank(pct=True) * 35 + 5
    return predictions_df

def assign_dynamic_costs(predictions_df, total_budget, min_cost=5, max_cost_ratio=0.25):
    predictions_df = predictions_df.copy()
    
    # Normalize predicted pdk to range [0, 1]
    min_pdk = predictions_df['predicted_pdk'].min()
    max_pdk = predictions_df['predicted_pdk'].max()
    predictions_df['normalized_pdk'] = (predictions_df['predicted_pdk'] - min_pdk) / (max_pdk - min_pdk)
    
    # Calculate maximum allowed cost per player (initial step)
    max_cost = total_budget * max_cost_ratio
    
    # Apply logarithmic function to calculate base costs
    predictions_df['cost'] = min_cost + (max_cost - min_cost) * np.log1p(predictions_df['normalized_pdk']) / np.log1p(1)
    
    # Round costs to two decimal places
    predictions_df['cost'] = predictions_df['cost'].round(2)

    logging.info(predictions_df['cost'])
    
    return predictions_df

def update_costs(available_players, remaining_budget):
    updated_players = assign_dynamic_costs(available_players, remaining_budget)
    
    return updated_players

def generate_multiple_squads(available_players, my_team, remaining_budget, num_squads=3):
    squads = []
    already_selected_players = [] 

    for i in range(num_squads):
        # Filter out already selected players
        available_players_for_iteration = available_players[~available_players.index.isin(already_selected_players)]
        
        optimized_squad, selected_players = optimize_squad(available_players_for_iteration, my_team, remaining_budget)
        if optimized_squad is None:
            print(f"Could not find a valid squad in iteration {i+1}.")
            break
        
        squads.append(optimized_squad)
        already_selected_players.extend(selected_players)

    return squads


def optimize_squad(available_players, my_team, remaining_budget, max_players_per_team=2):
    prob = LpProblem("Fantasy_Basketball_Team_Selection", LpMaximize)

    logging.info(my_team)
    logging.info(remaining_budget)
    
    # Reset index before concatenation
    my_team = my_team.reset_index(drop=True)
    available_players = available_players.reset_index(drop=True)
    
    all_players = pd.concat([my_team, available_players], ignore_index=True)
    player_vars = LpVariable.dicts("Players", range(len(all_players)), cat='Binary')

    my_team_indices = range(len(my_team))
    available_players_indices = range(len(my_team), len(all_players))

    # Ensure all players from my_team are always selected
    for i in my_team_indices:
        prob += player_vars[i] == 1

    # Objective function: maximize predicted_pdk for available players
    prob += lpSum([player_vars[i] * all_players.loc[i, 'predicted_pdk'] for i in available_players_indices])

    # Total number of players should be 10
    prob += lpSum([player_vars[i] for i in range(len(all_players))]) == 10
    
    # Budget constraint for available players
    prob += lpSum([player_vars[i] * all_players.loc[i, 'cost'] for i in available_players_indices]) <= remaining_budget

    # Position constraints
    prob += lpSum([player_vars[i] for i in all_players[all_players['position'] == 'C'].index]) == 2
    prob += lpSum([player_vars[i] for i in all_players[all_players['position'] == 'G'].index]) == 4
    prob += lpSum([player_vars[i] for i in all_players[all_players['position'] == 'F'].index]) == 4

    #max_budget_per_player = remaining_budget * 0.20
    #for i in available_players_indices:
        #prob += all_players.loc[i, 'cost'] * player_vars[i] <= max_budget_per_player

    # Limit the number of players per team, but only for available players
    for team in available_players['team_name'].unique():
        prob += lpSum([player_vars[i] for i in available_players_indices if all_players.loc[i, 'team_name'] == team]) <= max_players_per_team

    prob.solve()

    if LpStatus[prob.status] != 'Optimal':
        return None, []

    selected_players = [i for i in range(len(all_players)) if player_vars[i].value() > 0.5]
    return all_players.loc[selected_players], selected_players

def print_squad(squad, squad_number):
    print(f"\nOptimized Squad {squad_number}:")
    print(squad)
    print(f"Total Cost: {squad['cost'].sum():.2f}")
    print(f"Total Predicted PDK: {squad['predicted_pdk'].sum():.2f}")

    for position in ['C', 'G', 'F']:
        print(f"\n{position} players:")
        print(squad[squad['position'] == position])