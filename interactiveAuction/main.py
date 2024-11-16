from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import numpy as np
import pandas as pd
import math
from fantasy_basket_logic import (
    load_predictions, load_nba_players, filter_active_players, 
    update_costs, generate_multiple_squads, assign_dynamic_costs
)
from player import Player

app = Flask(__name__)
CORS(app)

# Global variables to store state
available_players = None
my_team = None
opponent_teams = None
total_players = 0
opponent_budget = 0
total_budget = 0
remaining_budget = 150

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/initialize', methods=['POST'])
def initialize():
    global available_players, my_team, opponent_teams, remaining_budget, total_budget, total_players, opponent_budget

    data = request.json
    opponent_names = data.get('opponent_names', ['Opponent'])
    my_team = Player(name="My Team", remaining_budget=remaining_budget, players=[])

    remaining_budget = float(data.get('budget', 150))

    opponent_teams = {name: Player(name=name, remaining_budget=remaining_budget, players=[]) for name in opponent_names}
    opponent_budget = remaining_budget * len(opponent_names)
    total_budget = opponent_budget + remaining_budget
    
    if len(opponent_names) < 2:
        return jsonify({"error": "At least 2 opponents are required to start an auction."}), 400
    
    predictions_path = '/app/model/predictions_next_season.csv'
    nba_players_path = '/app/data/players_2024-25.csv'
    
    try:
        predictions_df = load_predictions(predictions_path)
        nba_players = load_nba_players(nba_players_path)
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        return jsonify({"error": str(e)}), 500

    active_predictions_df = filter_active_players(predictions_df, nba_players)
    total_players = len(active_predictions_df)
    available_players = update_costs(active_predictions_df, remaining_budget, total_players=total_players, opponent_budgets=[team.remaining_budget for team in opponent_teams.values()], total_budget=total_budget, players_bought=0)
    my_team = pd.DataFrame(columns=available_players.columns)

    logging.info(f"Auction initialized with opponents: {', '.join(opponent_names)}")
    return jsonify({"message": f"Auction initialized successfully with opponents: {', '.join(opponent_names)}"}), 200

@app.route('/finish', methods=['POST'])
def reset():
    global available_players, my_team, opponent_teams, remaining_budget
    
    available_players = None
    my_team = None
    opponent_teams = None
    remaining_budget = 150

    logging.info("Auction reset successfully")
    return jsonify({"message": "Auction reset successfully"}), 200

@app.route('/buy', methods=['POST'])
def buy_player():
    global available_players, my_team, remaining_budget
    
    data = request.json
    player_name = data.get('player_name')
    cost = float(data.get('cost', 0))

    player = available_players[
        (available_players['first_name'] + ' ' + available_players['last_name']).str.lower() == player_name.lower()
    ]

    if player.empty:
        logging.warning(f"Player {player_name} not found or already taken")
        return jsonify({"error": f"Player {player_name} not found or already taken."}), 400
    elif cost > remaining_budget:
        logging.warning(f"Not enough budget to buy {player_name}")
        return jsonify({"error": f"Not enough budget to buy {player_name}."}), 400
    else:
        my_team = pd.concat([my_team, player], ignore_index=True)
        available_players = available_players.drop(player.index)
        available_players = update_costs(available_players, remaining_budget, total_players=total_players, opponent_budgets=[team.remaining_budget for team in opponent_teams.values()], total_budget=total_budget, players_bought=total_players - len(available_players))
        remaining_budget -= cost
        logging.info(f"Added {player_name} to your team. Remaining budget: {remaining_budget}")
        return jsonify({
            "message": f"Added {player_name} to your team.",
            "remaining_budget": remaining_budget,
            "predicted_pdk": player['predicted_pdk'].values[0],
            "role": player['position'].values[0]
        }), 200

@app.route('/opponent', methods=['POST'])
def opponent_pick():
    global available_players, opponent_teams, opponent_budget, total_players

    data = request.json
    player_name = data.get('player_name')
    opponent_name = data.get('opponent_name')
    cost = float(data.get('cost', 0))

    if opponent_name not in opponent_teams:
        logging.warning(f"Invalid opponent name: {opponent_name}")
        return jsonify({"error": f"Invalid opponent name. Must be one of: {', '.join(opponent_teams.keys())}"}), 400

    player = available_players[
        (available_players['first_name'] + ' ' + available_players['last_name']).str.lower() == player_name.lower()
    ]

    if player.empty:
        logging.warning(f"Player {player_name} not found or already taken")
        return jsonify({"error": f"Player {player_name} not found or already taken."}), 400
    elif cost > opponent_teams[opponent_name].remaining_budget:
        logging.warning(f"Not enough budget for {opponent_name} to buy {player_name}")
        return jsonify({"error": f"Not enough budget for {opponent_name} to buy {player_name}."}), 400
    else:
        opponent_teams[opponent_name].players.append(player.iloc[0].to_dict())
        opponent_teams[opponent_name].remaining_budget -= cost
        available_players = available_players.drop(player.index)
        
        players_bought = total_players - len(available_players)
        available_players = update_costs(available_players, remaining_budget, total_players, 
                                         [team.remaining_budget for team in opponent_teams.values()], 
                                         total_budget, players_bought)
        
        logging.info(f"Removed {player_name} from available players. Picked by opponent {opponent_name}")
        return jsonify({
            "message": f"Removed {player_name} from available players. Picked by opponent {opponent_name}",
            "opponent_remaining_budget": opponent_teams[opponent_name].remaining_budget,
            "predicted_pdk": player['predicted_pdk'].values[0],
            "role": player['position'].values[0]
        }), 200

@app.route('/generate', methods=['GET'])
def generate_squads():
    global available_players, my_team, remaining_budget
    
    num_squads = int(request.args.get('num_squads', 3))
    squads = generate_multiple_squads(available_players, my_team, remaining_budget, num_squads)
    
    squad_data = []
    for i, squad in enumerate(squads, 1):
        players_data = squad.to_dict(orient='records')
        for player in players_data:
            for key, value in player.items():
                if pd.isna(value):
                    player[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    player[key] = float(value)

        squad_data.append({
            "squad_number": i,
            "players": players_data,
            "total_cost": float(squad['cost'].sum()),
            "total_predicted_pdk": float(squad['predicted_pdk'].sum())
        })
    
    logging.info(f"Generated {num_squads} optimized squads")
    response = make_response(jsonify(squad_data))
    response.headers['Content-Type'] = 'application/json'
    return response, 200

@app.route('/team', methods=['GET'])
def get_team():
    logging.info("Retrieving current team information")
    return jsonify({
        "my_team": my_team.to_dict(orient='records'),
        "remaining_budget": remaining_budget,
        "opponent_teams": opponent_teams
    }), 200

@app.route('/available', methods=['GET'])
def get_available_players():
    logging.info("Retrieving available players")
    return jsonify((available_players['first_name'] + ' ' + available_players['last_name'] + ' ' + available_players['cost'].astype(str)).str.lower().tolist()), 200

def update_costs(available_players, remaining_budget, total_players, opponent_budgets, total_budget, players_bought):
    scarcity_factor = calculate_scarcity_factor(available_players, total_players)
    quality_factor = calculate_quality_factor(available_players)
    budget_factor = calculate_budget_factor(remaining_budget, opponent_budgets, total_budget)
    auction_progress_factor = calculate_auction_progress_factor(players_bought, total_players)
    
    logging.info(f"Scarcity factor: {scarcity_factor}")
    logging.info(f"Quality factor: {quality_factor}")
    logging.info(f"Budget factor: {budget_factor}")
    logging.info(f"Auction progress factor: {auction_progress_factor}")

    total_factor = calculate_total_factor(scarcity_factor, quality_factor, budget_factor, auction_progress_factor)
    logging.info(f"Total factor: {total_factor}")

    updated_players = assign_dynamic_costs(available_players, remaining_budget)
    updated_players['cost'] *= total_factor
    updated_players['cost'] = updated_players['cost'].round(2)
    
    return updated_players

def calculate_scarcity_factor(available_players, total_players, top_players_threshold=0.30):
    remaining_percentage = len(available_players) / total_players
    top_players_remaining = available_players[available_players['predicted_pdk'] > available_players['predicted_pdk'].quantile(1 - top_players_threshold)]
    top_players_percentage = len(top_players_remaining) / len(available_players)
    
    scarcity_factor = 1 + 0.5 * (1 - remaining_percentage) + 0.2 * (1 - top_players_percentage)
    return scarcity_factor

def calculate_quality_factor(available_players, threshold=0.30):
    top_players = available_players[available_players['predicted_pdk'] > available_players['predicted_pdk'].quantile(1 - threshold)]
    quality_factor = 1 + 0.2 * (1 - len(top_players) / len(available_players))
    return quality_factor

def calculate_budget_factor(my_remaining_budget, opponent_budgets, total_budget):
    total_remaining_budget = my_remaining_budget + sum(opponent_budgets)
    budget_factor = 1 + 0.1 * ((total_remaining_budget / total_budget) - 1)
    return max(budget_factor, 1)

def calculate_auction_progress_factor(players_bought, total_players):
    progress = players_bought / total_players
    return 1 + 0.5 * progress

def calculate_total_factor(scarcity_factor, quality_factor, budget_factor, auction_progress_factor):
    combined_factor = (scarcity_factor * quality_factor * budget_factor * auction_progress_factor) - 1
    total_factor = 1 + 0.5 * combined_factor
    return min(max(total_factor, 1.0), 1.5)


if __name__ == "__main__":
    logging.info("Starting Fantasy Basketball Auction API")
    app.run(host='0.0.0.0', port=5000)