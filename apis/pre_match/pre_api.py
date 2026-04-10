from flask import Flask, request, jsonify
from pre_match import *
from datetime import datetime


app = Flask(__name__)

@app.route('/')
def index():
    return "server is working!"

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
            "error": "team_id, num_of_matches, and opponent_id are required"
        }), 400

    # body of the function to process the pre-match data and generate result
        # not parallel part
    matches_ids = get_team_lnm(api_base , team_id, num_of_matches)


    # first parallel part 
    matches_stats = get_match_stats(api_base ,matches_ids )


    # second parallel part
    players_stats = get_players_stats(api_base, matches_ids)
    players_score = get_players_scores(api_base , players_stats , matches_ids.get('target_team_name') , matches_ids)


    # third parallel part
    players_real_positions = get_player_real_position_multimatch(api_base, matches_ids)


    # fourth parallel part
    opponent_matches_ids = get_team_lnm(api_base , opponent_id, num_of_matches)
    opponent_analysis = analyze_opponent_comprehensive_multimatch(api_base, opponent_matches_ids)



    # fifth parallel part
    formation_suggestion = formation_suggestions(api_base , opponent_id)




    # sixth parallel part
    players_training_stats = get_training_player_stats(api_base, matches_ids)
    training_recommendations = get_training_recommendations(api_base, players_training_stats)




    # not parallel part

    team_selection_output = get_best_starting_lineup_from_recommendations(
                players_stats_scored=players_score, 
                recommended_formations=formation_suggestion,
                real_pos_df=players_real_positions 
            )



    get_season_tournament_ids( api_base, list(players_score['player_id']))



    tacticle = get_tacticale(api_base , team_selection_output , opponent_id)
    tacticle = json.loads(tacticle)

    result = {
        "meta": {
        "teamId": team_id,
        "opponentId": opponent_id,
        "matchDate": "2025-11-24T20:00:00Z",
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

    return result, 200


if __name__ == '__main__':
    app.run(debug=True)