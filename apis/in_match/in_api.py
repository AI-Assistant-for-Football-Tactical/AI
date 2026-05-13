from flask import Flask, request, jsonify
from in_match import * 
import pandas as pd
import numpy as np
import ast


app = Flask(__name__)

@app.route('/')
def index():
    return "In Match Analysis API is running!"


# ============================================================================
# SECTION 1: ROLLING SNAPSHOT & DELTA ENGINE
# ============================================================================

def players_level_delta(players_df):
    """For each player, compute delta = snapshot_2 − snapshot_1 for every numeric stat column."""
    shared_cols = ["id", "name", "shirt_number", "team_id", "position"]
    cols_1 = {c.rsplit("_", 1)[0] for c in players_df.columns if c.endswith("_1")}
    cols_2 = {c.rsplit("_", 1)[0] for c in players_df.columns if c.endswith("_2")}
    stat_bases = sorted(cols_1 & cols_2)
    
    col_data = {}
    for base in stat_bases:
        c1, c2 = f"{base}_1", f"{base}_2"
        col_data[f"{base}_delta"] = (
            pd.to_numeric(players_df[c2], errors="coerce").values
            - pd.to_numeric(players_df[c1], errors="coerce").values
        )
    
    result = pd.concat(
        [players_df[shared_cols].reset_index(drop=True),
         pd.DataFrame(col_data)],
        axis=1,
    )
    return result


def team_level_delta(team_df):
    """Compute delta between the two snapshot rows of team-level event statistics."""
    if len(team_df) < 2:
        raise ValueError("event_stats must have at least 2 rows (snap1, snap2)")
    
    snap1 = pd.to_numeric(team_df.iloc[0], errors="coerce")
    snap2 = pd.to_numeric(team_df.iloc[1], errors="coerce")
    delta = snap2 - snap1
    
    delta_df = pd.DataFrame([delta.values], columns=[f"{c}_delta" for c in team_df.columns])
    return delta_df


def _parse_heatmap(raw):
    """Safely parse a heatmap column value into a list of {x, y} dicts."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def heatmap_delta(heatmaps):
    """For each player, compute the new heatmap points that appeared between snapshot 1 and 2."""
    rows = []
    for _, row in heatmaps.iterrows():
        pid = row["player_id"]
        h1 = _parse_heatmap(row.get("heatmap1", "[]"))
        h2 = _parse_heatmap(row.get("heatmap2", "[]"))
        
        n1, n2 = len(h1), len(h2)
        n_new = max(0, n2 - n1)
        delta_points = h2[-n_new:] if n_new > 0 else []
        
        def centroid(pts):
            if not pts:
                return np.nan, np.nan
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            return np.mean(xs), np.mean(ys)
        
        cx1, cy1 = centroid(h1)
        cx2, cy2 = centroid(h2)
        
        rows.append({
            "player_id": pid,
            "heatmap_delta": delta_points,
            "n_snap1": n1,
            "n_snap2": n2,
            "n_new_points": n_new,
            "centroid_snap1_x": cx1,
            "centroid_snap1_y": cy1,
            "centroid_snap2_x": cx2,
            "centroid_snap2_y": cy2,
            "centroid_delta_x": cx2 - cx1 if not (np.isnan(cx1) or np.isnan(cx2)) else 0,
            "centroid_delta_y": cy2 - cy1 if not (np.isnan(cy1) or np.isnan(cy2)) else 0,
        })
    
    return pd.DataFrame(rows)


def shotmaps_delta(shotmaps_df):
    """Identify shots that exist only in snapshot 2 (new shots in the window)."""
    df = shotmaps_df.copy()
    has_1 = df["xg_1"].notna()
    has_2 = df["xg_2"].notna()
    
    df["is_new_in_snap2"] = (~has_1) & has_2
    df["is_in_both"] = has_1 & has_2
    
    return df


def features_delta(features_df):
    """Compute deltas for aggregated event features."""
    cols_1 = {c.rsplit("_", 1)[0] for c in features_df.columns
              if c.endswith("_1") and c != "total_events"}
    cols_2 = {c.rsplit("_", 1)[0] for c in features_df.columns
              if c.endswith("_2")}
    bases = sorted(cols_1 & cols_2)
    
    data = features_df.copy()
    result = pd.DataFrame()
    result['total_events'] = data['total_events']
    for base in bases:
        c1, c2 = f"{base}_1", f"{base}_2"
        result[f"{base}_delta"] = (
            pd.to_numeric(data[c2], errors="coerce")
            - pd.to_numeric(data[c1], errors="coerce")
        )
    return result


def all_deltas(players_df, team_df, heatmaps_df, shotmaps_df, features_df):
    """Accepts dataframes and returns delta dataframes for all sections."""
    try:
        if players_df is not None and not players_df.empty:
            players_delta = players_level_delta(players_df)
        else:
            players_delta = None
            
        if team_df is not None and not team_df.empty:
            team_delta = team_level_delta(team_df)
        else:
            team_delta = None
            
        if heatmaps_df is not None and not heatmaps_df.empty:
            heatmaps_delta = heatmap_delta(heatmaps_df)
        else:
            heatmaps_delta = None
            
        if shotmaps_df is not None and not shotmaps_df.empty:
            shotmaps_delta_result = shotmaps_delta(shotmaps_df)
        else:
            shotmaps_delta_result = None
            
        if features_df is not None and not features_df.empty:
            features_delta_result = features_delta(features_df)
        else:
            features_delta_result = None
        
        return players_delta, team_delta, heatmaps_delta, shotmaps_delta_result, features_delta_result
    
    except Exception as e:
        import traceback
        print(f"Error inside all_deltas: {e}")
        print(traceback.format_exc())
        return None, None, None, None, None


# ============================================================================
# SECTION 2: FEATURE ENGINEERING & BASELINE COMPARISONS
# ============================================================================

def prepare_player_delta_data(player_delta_result, lineups_stats_df):
    """Prepare player delta data with all required features."""
    if player_delta_result is not None and not player_delta_result.empty:
        player_delta = player_delta_result.copy()
    else:
        player_delta = lineups_stats_df.copy()
        for col in player_delta.columns:
            if col.endswith('_2'):
                base = col[:-2]
                if base + '_1' in player_delta.columns:
                    player_delta[base + '_delta'] = player_delta[col].fillna(0) - player_delta[base + '_1'].fillna(0)
    
    # Merge with original lineups_stats_df to get snapshot columns
    player_delta = player_delta.merge(
        lineups_stats_df[['id', 'minutesPlayed_2', 'rating_1', 'rating_2', 'touches_2']],
        on='id', how='left'
    )
    
    return player_delta


def prepare_heatmaps_delta_data(heatmaps_delta_result):
    """Prepare heatmaps delta data."""
    if heatmaps_delta_result is not None and not heatmaps_delta_result.empty:
        return heatmaps_delta_result.copy()
    else:
        return pd.DataFrame()


def get_on_pitch_players(player_delta_df, lineups_stats_df):
    """Filter to players active in this window using touches_delta (not cumulative touches_2).
    
    touches_delta > 0  → player is on pitch and active in this window
    touches_delta == 0 → player was subbed OFF before this window (cumulative unchanged)
    touches_delta NaN  → bench player (never entered)
    """
    on_pitch = player_delta_df[
        player_delta_df['touches_delta'].notna() & 
        (player_delta_df['touches_delta'] > 0)
    ].copy()
    
    cols_to_merge = ['id', 'minutesPlayed_2', 'rating_1', 'rating_2']
    cols_to_use = [c for c in cols_to_merge if c in lineups_stats_df.columns]
    
    if cols_to_use:
        on_pitch = on_pitch.merge(
            lineups_stats_df[cols_to_use],
            left_on='id', right_on='id', how='left', suffixes=('', '_merged')
        )
    
    return on_pitch


def compute_fatigue_score(on_pitch_df):
    """
    Computes a 0-100 fatigue proxy based on:
    - Minutes played (0-20 points): longer minutes = more fatigue
    - Rating drop (0-25 points): performance decline indicates fatigue
    - Pass accuracy drop (0-20 points): lower accuracy in this window = fatigue
    - Ball carry efficiency (0-15 points): low distance/touches = fatigue
    - Duel win rate (0-20 points): losing more duels = fatigue
    """
    df = on_pitch_df.copy()
    
    for col in ['minutesPlayed_2', 'rating_1', 'rating_2', 'accuratePass_delta', 'totalPass_delta', 
                'totalBallCarriesDistance_delta', 'touches_delta', 'duelWon_delta', 'duelLost_delta']:
        if col not in df.columns: 
            df[col] = 0.0
    
    # Minutes component
    df['fatigue_minutes'] = (df['minutesPlayed_2'].fillna(0) / 90.0) * 20.0
    df['fatigue_minutes'] = df['fatigue_minutes'].clip(upper=20.0)
    
    # Rating drop component
    df['rating_drop'] = df['rating_1'].fillna(6.0) - df['rating_2'].fillna(6.0)
    # A 0.5 rating drop in a window = max 25 points (divide by 0.5, not 1.0)
    df['fatigue_rating'] = (df['rating_drop'].clip(lower=0) / 0.5) * 25.0
    df['fatigue_rating'] = df['fatigue_rating'].clip(upper=25.0)
    
    # Pass accuracy component
    df['pass_acc_window'] = df['accuratePass_delta'] / df['totalPass_delta'].replace(0, np.nan)
    df['fatigue_pass'] = 0.0
    df.loc[df['pass_acc_window'] < 0.85, 'fatigue_pass'] = 10.0
    df.loc[df['pass_acc_window'] < 0.70, 'fatigue_pass'] = 20.0
    
    # Ball carry efficiency component
    df['fatigue_carry'] = 0.0
    low_carry = (df['totalBallCarriesDistance_delta'] < 5.0) & (df['touches_delta'] > 3)
    df.loc[low_carry, 'fatigue_carry'] = 15.0
    
    # Duel win rate component
    df['total_duels_delta'] = df['duelWon_delta'] + df['duelLost_delta']
    df['duel_win_pct'] = df['duelWon_delta'] / df['total_duels_delta'].replace(0, np.nan)
    df['fatigue_duel'] = 0.0
    df.loc[df['duel_win_pct'] < 0.50, 'fatigue_duel'] = 10.0
    df.loc[df['duel_win_pct'] < 0.40, 'fatigue_duel'] = 20.0
    
    # Final fatigue score
    df['fatigue_score'] = df[['fatigue_minutes', 'fatigue_rating', 'fatigue_pass', 'fatigue_carry', 'fatigue_duel']].sum(axis=1)
    df['fatigue_score'] = df['fatigue_score'].clip(upper=100.0).fillna(0)
    
    return df


def compute_positional_drift(heatmaps_delta_df, grid_cols=6, grid_rows=4):
    """Analyzes spatial drift using centroid shifts from the heatmap delta."""
    if heatmaps_delta_df.empty: 
        return pd.DataFrame()
    
    df = heatmaps_delta_df.copy()
    zone_width = 100 / grid_cols
    zone_height = 100 / grid_rows
    
    df['expected_zone_x'] = (df['centroid_snap1_x'] // zone_width).clip(0, grid_cols-1)
    df['expected_zone_y'] = (df['centroid_snap1_y'] // zone_height).clip(0, grid_rows-1)
    df['current_zone_x'] = (df['centroid_snap2_x'] // zone_width).clip(0, grid_cols-1)
    df['current_zone_y'] = (df['centroid_snap2_y'] // zone_height).clip(0, grid_rows-1)
    
    df['drift_distance'] = np.sqrt(df['centroid_delta_x']**2 + df['centroid_delta_y']**2)
    df['is_drifting'] = df['drift_distance'] > 10.0
    
    return df[['player_id', 'drift_distance', 'expected_zone_x', 'expected_zone_y', 'current_zone_x', 'current_zone_y', 'is_drifting']]


def compute_team_passing_influence(player_delta_df):
    """Calculates individual passing influence from delta volume metrics."""
    if player_delta_df.empty: 
        return pd.DataFrame()
    
    df = player_delta_df.copy()
    for col in ['totalProgression_delta', 'keyPass_delta', 'totalPass_delta']:
        if col not in df.columns: 
            df[col] = 0.0
    
    prog_score = (df['totalProgression_delta'] / 150.0).clip(lower=0, upper=1.0) * 5
    key_score = (df['keyPass_delta'] / 3.0).clip(lower=0, upper=1.0) * 3
    vol_score = (df['totalPass_delta'] / 30.0).clip(lower=0, upper=1.0) * 2
    
    df['pass_influence_score'] = (prog_score + key_score + vol_score).fillna(0)
    return df[['id', 'pass_influence_score', 'totalProgression_delta', 'keyPass_delta']]


def detect_defensive_gaps(events_df1, events_df2, grid_cols=6, grid_rows=4):
    """
    Identify zones with 0 defensive actions in the latest window that previously had actions.
    """
    def get_zone_counts(df):
        if df is None or df.empty or 'x' not in df.columns or 'y' not in df.columns:
            return {}
            
        target_df = df
        if 'type' in df.columns:
            target_df = df[df['type'] == 'defensive']
            
        if target_df.empty:
            return {}
            
        target_df = target_df.copy()
        target_df['zone_x'] = (target_df['x'] // (100 / grid_cols)).clip(0, grid_cols-1)
        target_df['zone_y'] = (target_df['y'] // (100 / grid_rows)).clip(0, grid_rows-1)
        
        return target_df.groupby(['zone_x', 'zone_y']).size().to_dict()

    snap1_counts = get_zone_counts(events_df1)
    snap2_counts = get_zone_counts(events_df2)
    
    exposed_zones = []
    for zone, count1 in snap1_counts.items():
        count2 = snap2_counts.get(zone, 0)
        if count1 > 0 and count2 == 0:
            exposed_zones.append(zone)
            
    return {"defensive_zone_coverage_snap2": snap2_counts, "exposed_zones": exposed_zones}


def assess_shot_quality(shotmaps_delta_df):
    """
    Aggregates shot quality from new shots in the window.
    """
    if shotmaps_delta_df is None or shotmaps_delta_df.empty or 'is_new_in_snap2' not in shotmaps_delta_df.columns:
        return pd.DataFrame()
        
    new_shots = shotmaps_delta_df[shotmaps_delta_df['is_new_in_snap2'] == True].copy()
    
    if new_shots.empty:
        return pd.DataFrame({"total_xg_this_window": [0], "shot_quality_avg": [0], "shots_inside_box": [0]})
    
    total_xg = new_shots['xg_2'].sum() if 'xg_2' in new_shots.columns else 0
    avg_xg = new_shots['xg_2'].mean() if 'xg_2' in new_shots.columns else 0
    
    # Inside box approximation
    shots_inside_box = len(new_shots[new_shots['x_2'] < 18]) if 'x_2' in new_shots.columns else 0
    
    metrics = {
        "total_xg_this_window": [total_xg],
        "shot_quality_avg": [avg_xg],
        "shots_inside_box": [shots_inside_box]
    }
    
    return pd.DataFrame(metrics)


def merge_engineered_features(fatigue_df, drift_df, passing_df):
    """
    Combines per-player engineered features into a single DataFrame.
    """
    res = fatigue_df.copy()
    
    if drift_df is not None and not drift_df.empty:
        res = res.merge(drift_df, left_on='id', right_on='player_id', how='left')
        
    if passing_df is not None and not passing_df.empty:
        res = res.merge(passing_df, on='id', how='left')
        
    return res


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
        players_data = data.get('players_data')
        events_data = data.get('events_data')
        heatmaps_data = data.get('heatmaps_data')
        shotmaps_data = data.get('shotmaps_data')
        features_data = data.get('features_data')
        lineups_stats_data = data.get('lineups_stats_data')
        
        # ===== STEP 1: Rolling Snapshot & Delta Engine =====
        player_delta_result, events_delta_result, heatmaps_delta_result, shotmaps_delta_result, features_delta_result = all_deltas(
            players_data, events_data, heatmaps_data, shotmaps_data, features_data
        )
        
        # Convert to DataFrames if not already
        if isinstance(lineups_stats_data, list):
            lineups_stats_df = pd.DataFrame(lineups_stats_data)
        else:
            lineups_stats_df = lineups_stats_data
        
        # ===== STEP 2: Feature Engineering & Baseline Comparisons =====
        
        # Prepare data
        player_delta = prepare_player_delta_data(player_delta_result, lineups_stats_df)
        heatmaps_delta = prepare_heatmaps_delta_data(heatmaps_delta_result)
        
        # Get on-pitch players
        on_pitch = get_on_pitch_players(player_delta, lineups_stats_df)
        
        # Compute fatigue scores
        fatigue_features = compute_fatigue_score(on_pitch)
        
        # Compute positional drift
        drift_features = compute_positional_drift(heatmaps_delta)
        
        # Compute passing influence
        pass_features = compute_team_passing_influence(player_delta)
        
        # Detect defensive gaps
        events_df1_data = data.get('events_df1_data', [])
        events_df2_data = data.get('events_df2_data', [])
        events_df1 = pd.DataFrame(events_df1_data) if events_df1_data else pd.DataFrame()
        events_df2 = pd.DataFrame(events_df2_data) if events_df2_data else pd.DataFrame()
        gaps = detect_defensive_gaps(events_df1, events_df2)

        # Assess shot quality
        shot_metrics = assess_shot_quality(shotmaps_delta_result)

        # Merge all engineered features
        engineered_features = merge_engineered_features(fatigue_features, drift_features, pass_features)
        
        # ===== PREPARE RESPONSE =====
        
        # Format gaps for response
        formatted_gaps = {
            "defensive_zone_coverage_snap2": [{"zone_x": k[0], "zone_y": k[1], "count": v} for k, v in gaps.get("defensive_zone_coverage_snap2", {}).items()],
            "exposed_zones": [{"zone_x": z[0], "zone_y": z[1]} for z in gaps.get("exposed_zones", [])]
        }
        
        response = {
            "success": True,
            "section_2": {
                "on_pitch_players": len(on_pitch),
                "fatigue_stats": {
                    "mean_fatigue": float(fatigue_features['fatigue_score'].mean()) if not fatigue_features.empty else 0,
                    "max_fatigue": float(fatigue_features['fatigue_score'].max()) if not fatigue_features.empty else 0,
                    "min_fatigue": float(fatigue_features['fatigue_score'].min()) if not fatigue_features.empty else 0
                },
                "top_fatigued_players": fatigue_features.nlargest(5, 'fatigue_score')[
                    ['id', 'name', 'fatigue_score', 'minutesPlayed_2', 'rating_drop']
                ].to_dict('records') if not fatigue_features.empty else [],
                "players_drifting": int(drift_features['is_drifting'].sum()) if not drift_features.empty else 0,
                "avg_pass_influence": float(pass_features['pass_influence_score'].mean()) if not pass_features.empty else 0,
                "defensive_gaps": formatted_gaps,
                "shot_quality": shot_metrics.to_dict('records')[0] if not shot_metrics.empty else {}
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == '__main__':
    # Running on port 5002 to avoid conflicts with pre_match and post_match if running simultaneously
    app.run(debug=True, port=5002)
