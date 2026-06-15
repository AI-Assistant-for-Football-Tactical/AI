from flask import Flask, request, jsonify
from in_match import * 
import pandas as pd

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
        
            
        # getting event data
        event_stats1 = data.get('statistics1')
        team_id =  data.get('target_team_id')
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
        if len(breakdowns1) < 1 :
            players_rating_breakdowns1 = None
        else :
            players_rating_breakdowns1 =[]
            for player in breakdowns1 :
                players_rating_breakdowns1.append(player.get('ratingBreakdown'))

        breakdowns2 = data.get('ratingBreakdown2').get('home') + data.get('ratingBreakdown2').get('away')

        players_rating_breakdowns2 = []
        for player in breakdowns2 :
            players_rating_breakdowns2.append(player.get('ratingBreakdown'))
            
        # determine which team is ours

        home_heatmap_team_id = data.get('heatmap1').get('home')[0].get('teamId')   
        is_home = True if team_id == home_heatmap_team_id else False


        # getting data ready for section 1
        result = get_one_team_data(
            event_stats1 ,
            event_stats2 ,
            lineups1 , 
            lineups2 , 
            players_shotmaps1 , 
            players_shotmaps2 , 
            players_heatmaps1 , 
            players_heatmaps2 , 
            players_rating_breakdowns1  ,
            players_rating_breakdowns2 , 
            team_id ,
            is_home 
        )

        players = result['lineups']
        events = result['event_stats']
        heatmaps = result['heatmaps']
        shotmaps = result['shotmaps']
        features = result['rates']
        
        #  ===== STEP 1: Rolling Snapshot & Delta Engine =====
        player_delta_result, events_delta_result, heatmaps_delta_result, shotmaps_delta_result, features_delta_result = all_deltas(players, events, heatmaps, shotmaps, features)
        
        #  ===== STEP 2: Feature Engineering & Baseline Comparisons =====
        on_pitch = get_on_pitch_players(player_delta_result, players)
        fatigue_features = compute_fatigue_score(on_pitch)
        drift_features = compute_positional_drift(heatmaps_delta_result)
        pass_features = compute_team_passing_influence(player_delta_result)
        
        gaps = detect_defensive_gaps(pd.DataFrame(), pd.DataFrame()) 
        shot_metrics = assess_shot_quality(shotmaps_delta_result)
        engineered_features = merge_engineered_features(fatigue_features, drift_features, pass_features)
        
        #  ===== STEP 3: Performance Deviation (Z-Score Model) =====
        z_scores_df = compute_performance_deviations(on_pitch)
        
        #  ===== STEP 4: Substitution Urgency  =====
        urgency_input = prepare_urgency_features(engineered_features, z_scores_df)
        urgency_ranked = rank_substitution_urgency(urgency_input)
        recommendations, remaining_bench = generate_substitution_recommendations(player_delta_result, urgency_ranked)
        
        #  ===== STEP 5: Pitch Grid Zone Threat Regressor =====
        zone_threats = compute_zone_threat(heatmaps , shotmaps_delta_result , features_delta_result)
        
        #  ===== STEP 6: Formation Effectiveness Model =====
        formation_output = analyze(heatmaps, players , on_pitch['id'])
        
        # ===== Match Intelligence Export =====
        match_intelligence = build_match_intelligence_export(events_delta_result, urgency_ranked, player_delta_result, gaps, shot_metrics, recommendations)

        #  ===== STEP 7: Inference Pipeline & LLM Aggregation =====
        try:
            response = generate_analysis(fatigue_features, z_scores_df, urgency_ranked, zone_threats, formation_output, engineered_features, match_intelligence, shot_metrics, events_delta_result)
        except Exception as e:
            print(f'the llm is down as : {e}')
            response = generate_analysis_without_llm(fatigue_features, z_scores_df, urgency_ranked, zone_threats, formation_output, engineered_features, match_intelligence, shot_metrics, events_delta_result)

        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == '__main__':
    # Running on port 5002 to avoid conflicts with pre_match and post_match if running simultaneously
    app.run(debug=True, port=5002)
