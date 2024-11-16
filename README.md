# FantasyBasketAuctionHelper_BackEnd 🏀

Backend service for FantasyBasketAuctionHelper, providing ML-powered predictions and team optimization for fantasy basketball auctions. This service uses XGBoost for player performance predictions and linear optimization algorithms to generate optimal team compositions.

## 📚 Project Structure

```
FantasyBasketAuctionHelper_BackEnd/
├── data/                    # NBA players predictions data
├── interactiveAuction/     # Flask server implementation
├── model/                  # Trained XGBoost model
├── optimizeSquad/         # Team optimization algorithms
├── trainModel/           # Model training scripts
└── docker-compose files  # Different deployment configurations
```

## 🛠️ Technology Stack

- **Python**: Core programming language
- **Flask**: Web server framework
- **XGBoost**: Machine learning model for player performance predictions
- **Linear Optimization**: Team composition optimization algorithms
- **Docker**: Application containerization

## 🚀 Getting Started

### Prerequisites
- Docker
- Docker Compose

### Running the Server

To start the interactive auction server:
```bash
docker compose -f docker-compose.interactive.yml up
```

## 📦 Components

### 1. Interactive Auction Server (`interactiveAuction/`)
- Flask-based RESTful API server
- Endpoints for:
  - Player predictions retrieval
  - Team optimization requests
  - Auction state management
- Real-time team suggestions based on current auction state

### 2. Team Optimization (`optimizeSquad/`)
- Implements linear optimization algorithms
- Generates optimal team compositions considering:
  - Budget constraints
  - Player availability
  - Predicted fantasy points
- Capable of generating up to 50 optimized team suggestions

### 3. Model Training (`trainModel/`)
- Scripts for training the XGBoost model
- Uses historical NBA data from the last 6 seasons
- Features engineering and model optimization

### 4. Data Directory (`data/`)
- Contains predictions for all NBA players
- Updated prediction datasets
- Historical performance data

## 🔄 Docker Compose Configurations

The project includes multiple Docker Compose configurations for different use cases:

- `docker-compose.interactive.yml`: For running the interactive auction server
- `docker-compose.squad.yml`: For batch generation of optimized teams
- `docker-compose.train.yml`: For model training tasks

[Prima parte del README rimane invariata fino alla sezione API Endpoints]

## 📡 API Endpoints

The interactive server exposes the following REST endpoints:

### Auction Management
- **Initialize Auction**
  ```
  POST /initialize
  ```
  Initializes the auction with the specified budget and opponents.
  ```json
  {
    "budget": 150,
    "opponent_names": ["Opponent1", "Opponent2", ...]
  }
  ```

- **Finish Auction**
  ```
  POST /finish
  ```
  Resets the auction state.

### Player Operations
- **Buy Player**
  ```
  POST /buy
  ```
  Records a player purchase for your team.
  ```json
  {
    "player_name": "Player Name",
    "cost": 50
  }
  ```

- **Record Opponent Pick**
  ```
  POST /opponent
  ```
  Records a player purchase by an opponent.
  ```json
  {
    "player_name": "Player Name",
    "opponent_name": "Opponent Name",
    "cost": 50
  }
  ```

### Data Retrieval
- **Get Available Players**
  ```
  GET /available
  ```
  Returns a list of all available players with their costs.

- **Get Team Information**
  ```
  GET /team
  ```
  Returns current team composition, remaining budget, and opponent teams' status.

### Team Generation
- **Generate Optimized Squads**
  ```
  GET /generate?num_squads=3
  ```
  Generates optimized squad suggestions based on available players and budget.
  - Query Parameters:
    - `num_squads`: Number of squad suggestions to generate (default: 3)

### Response Formats

#### Successful Buy Response
```json
{
    "message": "Added [Player Name] to your team.",
    "remaining_budget": 100.0,
    "predicted_pdk": 25.5,
    "role": "G"
}
```

#### Generated Squads Response
```json
[
    {
        "squad_number": 1,
        "players": [...],
        "total_cost": 150.0,
        "total_predicted_pdk": 100.5
    },
    ...
]
```

## 🎯 Model Performance

The XGBoost model is trained on 6 seasons of NBA data to predict fantasy points. The model considers various factors including:
- Historical player performance
- Playing time
- Team dynamics

## 🤝 Contributing

Interested in contributing? Great! Please:
1. Fork the repository
2. Create a new branch for your modifications
3. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

This backend is designed to work with [FantasyBasketAuctionHelper](https://github.com/Simi24/FantasyBasketAuctionHelper) frontend application.