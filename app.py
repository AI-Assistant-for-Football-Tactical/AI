import sys
import os
import json
import requests
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

# Add the API directories to sys.path so we can import their modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'apis', 'pre_match'))
sys.path.append(os.path.join(current_dir, 'apis', 'post_match'))

# Import the specific functions from the respective modules
from pre_match import (
    get_team_lnm,
    get_match_stats,
    get_players_stats,
    get_players_scores,
    get_player_real_position_multimatch,
    analyze_opponent_comprehensive_multimatch,
    formation_suggestions,
    get_training_player_stats,
    get_training_recommendations,
    get_best_starting_lineup_from_recommendations,
    get_season_tournament_ids,
    get_tacticale
)

from post_match import generate_post_match_report

app = Flask(__name__)

# Common API base used in both functions
api_base = 'https://football-backend-app.victoriouswater-69fff737.swedencentral.azurecontainerapps.io/'

@app.route('/')
def index():
    return jsonify({
        "status": "success",
        "message": "Football Analysis APIs (Pre-Match & Post-Match) are running successfully!"
    }), 200


@app.route('/post_match', methods=['POST'])
def post_match():
    data = request.get_json()

    # Validate input
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    team_id = data.get('team_id')
    event_id = data.get('event_id')

    if team_id is None or event_id is None:
        return jsonify({
            "error": "team_id and event_id are required"
        }), 400

    try:
        # Call the core analytical function defined in post_match.py
        result = generate_post_match_report(api_base=api_base, team_id=int(team_id), event_id=int(event_id))
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/pre_match', methods=['POST'])
def pre_match():
    data = request.get_json()

    # Validate input
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    team_id = data.get('team_id')
    num_of_matches = data.get('num_matches')
    opponent_id = data.get('opponent_id')

    if team_id is None or num_of_matches is None or opponent_id is None:
        return jsonify({
            "error": "team_id, num_matches, and opponent_id are required"
        }), 400

    try:
        # body of the function to process the pre-match data and generate result
        # not parallel part
        matches_ids = get_team_lnm(api_base, team_id, num_of_matches)

        # first parallel part 
        matches_stats = get_match_stats(api_base, matches_ids)

        # second parallel part
        players_stats = get_players_stats(api_base, matches_ids)
        players_score = get_players_scores(api_base, players_stats, matches_ids.get('target_team_name'), matches_ids)

        # third parallel part
        players_real_positions = get_player_real_position_multimatch(api_base, matches_ids)

        # fourth parallel part
        opponent_matches_ids = get_team_lnm(api_base, opponent_id, num_of_matches)
        opponent_analysis = analyze_opponent_comprehensive_multimatch(api_base, opponent_matches_ids)

        # fifth parallel part
        formation_suggestion = formation_suggestions(api_base, opponent_id)

        # sixth parallel part
        players_training_stats = get_training_player_stats(api_base, matches_ids)
        training_recommendations = get_training_recommendations(api_base, players_training_stats)

        # not parallel part
        team_selection_output = get_best_starting_lineup_from_recommendations(
            players_stats_scored=players_score, 
            recommended_formations=formation_suggestion,
            real_pos_df=players_real_positions 
        )

        get_season_tournament_ids(api_base, list(players_score['player_id']))

        tacticle = get_tacticale(api_base, team_selection_output, opponent_id)
        if isinstance(tacticle, str):
            tacticle = json.loads(tacticle)
        
        # getting the date for the next match
        response_data = requests.get(api_base + f'teams/{team_id}/events/next/0').json()
        event = response_data["events"][0]  # or any event you want
        ts = event["startTimestamp"]

        match_date = (
            datetime
            .fromtimestamp(ts, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

        training_plan_data = training_recommendations
        if isinstance(training_plan_data, str):
            training_plan_data = json.loads(training_plan_data)

        # constructing final json
        result = {
            "meta": {
                "teamId": team_id,
                "opponentId": opponent_id,
                "matchDate": match_date,
                "analysisTimestamp": datetime.utcnow().isoformat() + "Z"
            },
            "opponentAnalysis": opponent_analysis.get('opponentAnalysis'),
            "teamSelection": {
                "suggestedFormation": tacticle.get('suggestedFormation'),
                "strategyCode": tacticle.get('strategyCode'),
                "startingXI": team_selection_output.get('startingXI'),
                "substitutes": team_selection_output.get('substitutes')
            },
            "trainingPlan": training_plan_data.get('trainingPlan')
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Use the PORT environment variable if available (required for Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
