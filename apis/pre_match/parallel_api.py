from flask import Flask, request, jsonify
from pre_match import *
from datetime import datetime , timezone
from concurrent.futures import ThreadPoolExecutor
import json
import re



current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '.env')


load_dotenv(dotenv_path=r".env", override=True)



app = Flask(__name__)

@app.route('/')
def index():
    return "server is working!"

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

    # Step 1: Not parallel
    matches_ids = get_team_lnm(api_base, team_id, num_of_matches)

    # Step 2: Parallel execution
    with ThreadPoolExecutor(max_workers=6) as executor:
        
        # submit tasks
        future_match_stats = executor.submit(get_match_stats, api_base, matches_ids)
        future_real_positions = executor.submit(get_player_real_position_multimatch, api_base, matches_ids)
        future_players_stats = executor.submit(get_players_stats, api_base, matches_ids)
        future_opponent_matches = executor.submit(get_team_lnm, api_base, opponent_id, num_of_matches)
        future_formation = executor.submit(formation_suggestions, api_base, opponent_id)
        future_training_stats = executor.submit(get_training_player_stats, api_base, matches_ids)

        # get results
        matches_stats = future_match_stats.result()
        players_real_positions = future_real_positions.result()
        players_stats = future_players_stats.result()
        opponent_matches_ids = future_opponent_matches.result()
        formation_suggestion = future_formation.result()
        players_training_stats = future_training_stats.result()

    # Step 3: Dependent computations (can partially parallelize)
    with ThreadPoolExecutor(max_workers=3) as executor:
        
        future_players_score = executor.submit(
            get_players_scores,
            api_base,
            players_stats,
            matches_ids.get('target_team_name'),
            players_real_positions
        )

        future_opponent_analysis = executor.submit(
            analyze_opponent_comprehensive_multimatch,
            api_base,
            opponent_matches_ids
        )

        future_training_recommendations = executor.submit(
            get_training_recommendations,
            api_base,
            players_training_stats
        )

        players_score = future_players_score.result()
        opponent_analysis = future_opponent_analysis.result()
        training_recommendations = future_training_recommendations.result()

    # Step 4: Not parallel (depends on previous outputs)
    team_selection_output = get_best_starting_lineup_from_recommendations(
        players_stats_scored=players_score,
        recommended_formations=formation_suggestion,
        real_pos_df=players_real_positions
    )

    get_season_tournament_ids(api_base, list(players_score['player_id']))

    # Step 5: Tactical
    tacticle = get_tacticale(api_base, team_selection_output, opponent_id)

    if isinstance(tacticle, str):
        tacticle = re.search(r'\{[\s\S]*\}', tacticle)
        tacticle = json.loads(tacticle.group())

    # Step 6: Get match date
    data = requests.get(api_base + f'teams/{team_id}/events/next/0').json()
    event = data["events"][0]
    ts = event["startTimestamp"]

    match_date = (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    # Step 7: Final JSON
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
        "trainingPlan": json.loads(training_recommendations).get('trainingPlan')
    }

    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)