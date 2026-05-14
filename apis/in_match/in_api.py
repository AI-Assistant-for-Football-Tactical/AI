from flask import Flask, request, jsonify
from in_match import * 

app = Flask(__name__)

@app.route('/')
def index():
    return "In Match Analysis API is running!"


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
        is_home = True
        
        # getting data ready for section 1
        result = get_one_team_data( event_stats1 , event_stats2 , lineups1 , lineups2 , players_shotmaps1 , players_shotmaps2 , 
        players_heatmaps1 , players_heatmaps2 , players_rating_breakdowns1  , players_rating_breakdowns2 , is_home )
        
        players = result['lineups']
        events = result['event_stats']
        heatmaps = result['heatmaps']
        shotmaps = result['shotmaps']
        features = result['rates']
            
        #  ===== STEP 1: Rolling Snapshot & Delta Engine =====
        player_delta_result, events_delta_result, heatmaps_delta_result, shotmaps_delta_result, features_delta_result = all_deltas(players, events, heatmaps, shotmaps, features)
        
        #  ===== STEP 2: Feature Engineering & Baseline Comparisons =====
        
        #  ===== STEP 3: Performance Deviation (Z-Score Model) =====
        
        #  ===== STEP 4: Substitution Urgency  =====
        
        #  ===== STEP 5: Pitch Grid Zone Threat Regressor =====
        zone_threats = compute_zone_threat(heatmaps , shotmaps_delta_result , features_delta_result)
        #  ===== STEP 6: Formation Effectiveness Model =====
        formation_output = analyze(heatmaps, players , on_pitch['id'])
        #  ===== STEP 7: Inference Pipeline & LLM Aggregation =====
        try:
            response = generate_analysis(fatigue_features , z_scores_df , urgency_ranked , zone_threats ,formation_output , engineered_features , match_intelligence , shot_metrics , events_delta_result )
        except Exception as e :
            print(f'the llm is down as : {e}')
            response = generate_analysis_without_llm(fatigue_features , z_scores_df , urgency_ranked , zone_threats ,formation_output , engineered_features , match_intelligence , shot_metrics , events_delta_result )

        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == '__main__':
    # Running on port 5002 to avoid conflicts with pre_match and post_match if running simultaneously
    app.run(debug=True, port=5002)
