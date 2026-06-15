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
sys.path.append(os.path.join(current_dir, 'apis', 'in_match'))

import pandas as pd

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

from in_match import (
    get_one_team_data,
    all_deltas,
    get_on_pitch_players,
    compute_fatigue_score,
    compute_positional_drift,
    compute_team_passing_influence,
    detect_defensive_gaps,
    assess_shot_quality,
    merge_engineered_features,
    compute_performance_deviations,
    prepare_urgency_features,
    rank_substitution_urgency,
    generate_substitution_recommendations,
    compute_zone_threat,
    analyze,
    build_match_intelligence_export,
    generate_analysis,
    generate_analysis_without_llm
)

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


@app.route('/in_match', methods=['POST'])
def in_match():
    """
    In-Match Analysis Pipeline:
    1. Rolling Snapshot & Delta Engine (all_deltas)
    2. Feature Engineering & Baseline Comparisons
    3. Performance Deviation (Z-Score Model)
    4. Substitution Urgency XGBoost
    5. Pitch Grid Zone Threat Regressor
    6. Formation Effectiveness Model
    7. Inference Pipeline & LLM Aggregation
    """
    try:
        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        # getting event data
        event_stats1 = data.get('statistics1')
        team_id = data.get('target_team_id')
        event_stats2 = data.get('statistics2')

        # getting lineups data
        lineups1 = data.get('lineups1')
        lineups2 = data.get('lineups2')

        # getting shotmaps data
        players_shotmaps1 = data.get('shotmap1')
        players_shotmaps2 = data.get('shotmap2')

        # getting heatmaps data
        players_heatmaps1 = data.get('heatmap1').get('home') + data.get('heatmap1').get('away')
        players_heatmaps2 = data.get('heatmap2').get('home') + data.get('heatmap2').get('away')

        # getting rating data
        breakdowns1 = data.get('ratingBreakdown1').get('home') + data.get('ratingBreakdown1').get('away')
        if len(breakdowns1) < 1:
            players_rating_breakdowns1 = None
        else:
            players_rating_breakdowns1 = []
            for player in breakdowns1:
                players_rating_breakdowns1.append(player.get('ratingBreakdown'))

        breakdowns2 = data.get('ratingBreakdown2').get('home') + data.get('ratingBreakdown2').get('away')
        players_rating_breakdowns2 = []
        for player in breakdowns2:
            players_rating_breakdowns2.append(player.get('ratingBreakdown'))

        # determine which team is ours
        home_heatmap_team_id = data.get('heatmap1').get('home')[0].get('teamId')
        is_home = True if team_id == home_heatmap_team_id else False

        # getting data ready for section 1
        result = get_one_team_data(
            event_stats1,
            event_stats2,
            lineups1,
            lineups2,
            players_shotmaps1,
            players_shotmaps2,
            players_heatmaps1,
            players_heatmaps2,
            players_rating_breakdowns1,
            players_rating_breakdowns2,
            team_id,
            is_home
        )

        players = result['lineups']
        events = result['event_stats']
        heatmaps = result['heatmaps']
        shotmaps = result['shotmaps']
        features = result['rates']

        # ===== STEP 1: Rolling Snapshot & Delta Engine =====
        player_delta_result, events_delta_result, heatmaps_delta_result, shotmaps_delta_result, features_delta_result = all_deltas(
            players, events, heatmaps, shotmaps, features
        )

        # ===== STEP 2: Feature Engineering & Baseline Comparisons =====
        on_pitch = get_on_pitch_players(player_delta_result, players)
        fatigue_features = compute_fatigue_score(on_pitch)
        drift_features = compute_positional_drift(heatmaps_delta_result)
        pass_features = compute_team_passing_influence(player_delta_result)

        gaps = detect_defensive_gaps(pd.DataFrame(), pd.DataFrame())
        shot_metrics = assess_shot_quality(shotmaps_delta_result)
        engineered_features = merge_engineered_features(fatigue_features, drift_features, pass_features)

        # ===== STEP 3: Performance Deviation (Z-Score Model) =====
        z_scores_df = compute_performance_deviations(on_pitch)

        # ===== STEP 4: Substitution Urgency =====
        urgency_input = prepare_urgency_features(engineered_features, z_scores_df)
        urgency_ranked = rank_substitution_urgency(urgency_input)
        recommendations, remaining_bench = generate_substitution_recommendations(player_delta_result, urgency_ranked)

        # ===== STEP 5: Pitch Grid Zone Threat Regressor =====
        zone_threats = compute_zone_threat(heatmaps, shotmaps_delta_result, features_delta_result)

        # ===== STEP 6: Formation Effectiveness Model =====
        formation_output = analyze(heatmaps, players, on_pitch['id'])

        # ===== Match Intelligence Export =====
        match_intelligence = build_match_intelligence_export(
            events_delta_result, urgency_ranked, player_delta_result, gaps, shot_metrics, recommendations
        )

        # ===== STEP 7: Inference Pipeline & LLM Aggregation =====
        try:
            response = generate_analysis(
                fatigue_features, z_scores_df, urgency_ranked, zone_threats,
                formation_output, engineered_features, match_intelligence, shot_metrics, events_delta_result
            )
        except Exception as e:
            print(f'the llm is down as : {e}')
            response = generate_analysis_without_llm(
                fatigue_features, z_scores_df, urgency_ranked, zone_threats,
                formation_output, engineered_features, match_intelligence, shot_metrics, events_delta_result
            )

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == '__main__':
    # Use the PORT environment variable if available (required for Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
