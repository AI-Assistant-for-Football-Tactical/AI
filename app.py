import sys
import os , re
import json
import requests
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor




load_dotenv()

# Add the API directories to sys.path so we can import their modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'apis', 'pre_match'))
sys.path.append(os.path.join(current_dir, 'apis', 'post_match'))

# Import the specific functions from the respective modules
from pre_match2 import (
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
    get_tacticale,
    cached_get,
    clear_api_cache
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

    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    team_id = data.get('team_id')
    num_of_matches = data.get('num_matches')
    opponent_id = data.get('opponent_id')

    if team_id is None or num_of_matches is None or opponent_id is None:
        return jsonify({
            "error": "team_id, num_of_matches, and opponent_id are required"
        }), 400
    
    try:

        # Clear API cache for fresh analysis run
        clear_api_cache()


        # not parallel part
        matches_ids = get_team_lnm(api_base , team_id, num_of_matches)


        # first parallel part 
        matches_stats = get_match_stats(api_base ,matches_ids )



        # second parallel part
        players_real_positions = get_player_real_position_multimatch(api_base, matches_ids)
        


        # third parallel part
        players_stats = get_players_stats(api_base, matches_ids)
        players_score = get_players_scores(api_base , players_stats , matches_ids.get('target_team_name') , players_real_positions)



        # fourth parallel part
        opponent_matches_ids = get_team_lnm(api_base , opponent_id, num_of_matches)
        opponent_analysis = analyze_opponent_comprehensive_multimatch(api_base, opponent_matches_ids)




        # fifth parallel part
        formation_suggestion = formation_suggestions(api_base , opponent_id)


        # sixth parallel part
        players_training_stats = get_training_player_stats(api_base, matches_ids)
        training_recommendations = get_training_recommendations(api_base, players_training_stats)
    

        # team selection logic
        team_selection_output = get_best_starting_lineup_from_recommendations(
                    players_stats_scored=players_score, 
                    recommended_formations=formation_suggestion,
                    real_pos_df=players_real_positions 
                )

        tacticle = get_tacticale(api_base , team_selection_output , opponent_id)
        
        if isinstance(tacticle, str):
            match = re.search(r'\{[\s\S]*\}', tacticle)
            if match:
                tacticle = json.loads(match.group())
            else:
                tacticle = {"suggestedFormation": team_selection_output.get('suggestedFormation'), "strategyCode": "BALANCED"}

        # getting the date for the next match
        try:
            next_match_resp = requests.get(api_base + f'teams/{team_id}/events/next/0')
            if next_match_resp.status_code == 200:
                next_data = next_match_resp.json()
                if next_data.get("events"):
                    event = next_data["events"][0]
                    ts = event["startTimestamp"]
                    match_date = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                else:
                    match_date = datetime.utcnow().isoformat() + "Z"
            else:
                match_date = datetime.utcnow().isoformat() + "Z"
        except:
            match_date = datetime.utcnow().isoformat() + "Z"


        if isinstance(training_recommendations, str):
            match = re.search(r'\{[\s\S]*\}', training_recommendations)
        if match:
            training_recommendations = json.loads(match.group())
        else:
            training_recommendations = {"trainingPlan": None }


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
            "trainingPlan": training_recommendations.get('trainingPlan')
            
        }

        return result, 200    
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Use the PORT environment variable if available (required for Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
