
api_base = r'https://football-backend-app.victoriouswater-69fff737.swedencentral.azurecontainerapps.io/'

import requests
import pandas as pd
import json
from pandas import DataFrame , Series
import numpy as np
import statistics
from google import genai

def get_team_lnm(api_base :str , team_id:int , num_matchs: int) -> dict :
  '''this function takes the base of the api without the endpoint , the team id and the number of last matchs wanted
  and returns a dictionary of the last matchs ids of the team with the home team and away team names'''
  try :
    response = requests.get(api_base +f'teams/{team_id}/events/last/0') # api response
  except:
    return 'the api is down'
  match_info = { # extracting match details
    match['id']: {
        'homeTeam': match['homeTeam']['name'],
        'awayTeam': match['awayTeam']['name']
    }
    for match in response.json()['events']
}
  match_info =  dict(reversed(list(match_info.items())[-num_matchs:])) # filtering the last n matches from the data and reverse them (the last match is the first)
  match_info['target_team_id'] =team_id  # adding the id of the team to the dict
  match_info['target_team_name'] = requests.get(api_base + f'teams/{team_id}').json( ).get('team').get('name') # adding the name of the team to the dict
  return match_info

def get_match_stats(api_base: str , matches_info: dict ) -> DataFrame :
    '''this function takes the base of the api without the endpoint , the match onfo dictionary  and returns the statistics of the matches as a dataframe'''
    all_matches_stats = [] # define a dict to contain all the information to convert it to a DataFrame later

    for match_id in matches_info.keys():
        if isinstance(match_id, int):
            # getting the correct key for the values based on whether the team is away or home
            if matches_info.get('target_team_name') == matches_info.get(match_id).get('awayTeam'):
                value_key = 'awayValue'
            else :
                value_key = 'homeValue'

            match_stats = {}
            home_team = matches_info[match_id]['homeTeam'] # just adding each match id , home team name and away team name to the dict
            away_team = matches_info[match_id]['awayTeam']
            match_stats['match_id'] = match_id
            match_stats['home_team'] = home_team
            match_stats['away_team'] = away_team
            match_stats['team_formation'] = requests.get(api_base + f'events/{match_id}/lineups').json().get('home' if matches_info.get('target_team_name') == home_team else 'away').get('formation')


            try : # handling the api if it's disconnected or failed
                response = requests.get(api_base + f'events/{match_id}/statistics').json()['statistics'] # getting the statistics of the match
            except :
                return 'api is down'

            for period in response: # loop over each period statistics
                if period.get('period') ==  'ALL': # get the general stats only
                    response = period['groups'] # filtering only the overall statistics

            for group_stat in response: # loop over each match group statistics
                for stat_item in group_stat['statisticsItems']: # loop over each statistic item (collection of stats under a specific category)
                    if stat_item.get('name') != None :
                        match_stats[stat_item.get('name')] = stat_item.get(value_key) # adding the feature and the value for it in the dict

            all_matches_stats.append(match_stats) # adding the whole match stats to the general list

    all_matches_stats = pd.DataFrame(all_matches_stats ).fillna(0) # convert the general list to data frame and fill none values with 0
    return all_matches_stats

def get_players_stats(api_base: str, matches_info: dict) -> DataFrame:
    '''Get player statistics for all players in the given matches'''
    all_players_stats = []

    for match_id in matches_info.keys():
        if isinstance(match_id, int):
            # Determine if target team is home or away
            is_home = matches_info.get('target_team_name') == matches_info.get(match_id).get('homeTeam')
            team_key = 'home' if is_home else 'away'

            try:
                # Get lineups endpoint
                response = requests.get(api_base + f'events/{match_id}/lineups')

                # Check if request was successful
                if response.status_code != 200:
                    print(f"Error fetching player stats for match {match_id}: HTTP {response.status_code}")
                    continue

                data = response.json()

                # Check if response is valid and has expected structure
                if not isinstance(data, dict):
                    print(f"Error fetching player stats for match {match_id}: Invalid response format")
                    continue

                # Get player statistics
                if team_key in data and isinstance(data[team_key], dict) and 'players' in data[team_key]:
                    players = data[team_key]['players']

                    for player in players:
                        if not isinstance(player, dict):
                            continue

                        player_stat = {
                            'match_id': match_id,
                            'player_id': player.get('player', {}).get('id') if isinstance(player.get('player'), dict) else None,
                            'player_name': player.get('player', {}).get('name') if isinstance(player.get('player'), dict) else None,
                            'position': player.get('position'),
                            'shirt_number': player.get('shirtNumber'),
                            'substitute': player.get('substitute', False),
                            'captain': player.get('captain', False)
                        }

                        # Add statistics if available - statistics is a DICT, not a list
                        if 'statistics' in player and isinstance(player['statistics'], dict):
                            # Iterate through all statistics in the dictionary
                            for stat_name, stat_value in player['statistics'].items():
                                # Handle nested dictionaries like ratingVersions and statisticsType
                                if isinstance(stat_value, dict):
                                    # For ratingVersions, extract original rating
                                    if stat_name == 'ratingVersions':
                                        player_stat['rating_original'] = stat_value.get('original', 0)
                                        player_stat['rating_alternative'] = stat_value.get('alternative', 0)
                                    # For statisticsType, extract the type
                                    elif stat_name == 'statisticsType':
                                        player_stat['statistics_type'] = stat_value.get('statisticsType', 'player')
                                    # For other nested dicts, skip or convert to string
                                    else:
                                        player_stat[stat_name] = str(stat_value)
                                else:
                                    player_stat[stat_name] = stat_value

                        all_players_stats.append(player_stat)
                else:
                    print(f"No player data found for match {match_id} (team: {team_key})")

            except Exception as e:
                print(f"Error fetching player stats for match {match_id}: {str(e)}")
                continue

    if not all_players_stats:
        print("Warning: No player statistics collected")
        return pd.DataFrame()

    players_df = pd.DataFrame(all_players_stats).fillna(0)

    # Drop the original ratingVersions column if it exists (since we extracted its values)
    if 'ratingVersions' in players_df.columns:
        players_df = players_df.drop('ratingVersions', axis=1)

    return players_df

def get_player_real_position_multimatch(api_base: str, matches_info: dict) -> pd.DataFrame:
    """Calculates the real average position of players based on their heatmap data across multiple matches."""

    target_team = matches_info.get('target_team_name')
    if not target_team:
        return 'Target team name not found in matches_info'

    # Data structure: {player_id: {'name': str, 'base_role': str, 'roles_played': set, 'right_points': int, 'center_points': int, 'left_points': int, 'match_count': int}}
    player_agg = {}

    match_ids = [k for k in matches_info.keys() if isinstance(k, int)]

    print(f"Analyzing {len(match_ids)} matches for '{target_team}'...")

    for match_id in match_ids:
        try:
            # 1. Determine side using matches_info (Reliable)
            match_meta = matches_info.get(match_id, {})
            home_team_name = match_meta.get('homeTeam')

            # Direct comparison with target_team from Section 1
            side = 'home' if home_team_name == target_team else 'away'

            # 2. Get lineups just for players
            lineups_response = requests.get(api_base + f'events/{match_id}/lineups', timeout=10)
            if lineups_response.status_code != 200:
                print(f"Skipping match {match_id}: Lineups not found")
                continue

            lineups = lineups_response.json()
            team_data = lineups.get(side, {})
            all_players = team_data.get('players', [])

            if not all_players:
                # Fallback check
                print(f"No players found for {side} in match {match_id}")
                continue

            for player_entry in all_players:
                player = player_entry['player']
                pid = str(player['id']) # Ensure ID is string for consistency
                pname = player['name']
                prole = player_entry.get('position', 'Unknown')

                # Fetch heatmap for this player in this match
                try:
                    heatmap_response = requests.get(api_base + f'events/{match_id}/player/{pid}/heatmap', timeout=5)
                    if heatmap_response.status_code == 200:
                        heatmap_data = heatmap_response.json().get('heatmap', [])
                        if heatmap_data:
                            # 1. Zone Thresholds (Widened Center: 25 to 75)
                            right_pts = sum(1 for pt in heatmap_data if pt['y'] < 25)
                            center_pts = sum(1 for pt in heatmap_data if 25 <= pt['y'] <= 75)
                            left_pts = sum(1 for pt in heatmap_data if pt['y'] > 75)

                            x_coords = [pt['x'] for pt in heatmap_data]
                            avg_x = sum(x_coords) / len(x_coords) if x_coords else 50

                            total_match_pts = right_pts + center_pts + left_pts

                            if total_match_pts > 0:
                                # Determine dominant zone for THIS MATCH
                                zones = {'Right': right_pts, 'Center': center_pts, 'Left': left_pts}
                                dominant_zone = max(zones, key=zones.get)

                                # Infer Specific Role for THIS MATCH
                                match_role = prole
                                if prole == 'G': match_role = 'G'
                                elif prole == 'D':
                                    if dominant_zone == 'Right': match_role = 'RB'
                                    elif dominant_zone == 'Left': match_role = 'LB'
                                    else: match_role = 'CB'
                                elif prole == 'M':
                                    if dominant_zone == 'Right': match_role = 'RM'
                                    elif dominant_zone == 'Left': match_role = 'LM'
                                    else:
                                        # Depth Inference
                                        if avg_x < 45: match_role = 'CDM'
                                        elif avg_x > 65: match_role = 'CAM'
                                        else: match_role = 'CM'
                                elif prole == 'F':
                                    if dominant_zone == 'Right': match_role = 'RW'
                                    elif dominant_zone == 'Left': match_role = 'LW'
                                    else: match_role = 'ST'

                                # Initialize aggregation dict
                                if pid not in player_agg:
                                    player_agg[pid] = {
                                        'name': pname,
                                        'base_role': prole,
                                        'roles_played': set(),
                                        'right_points': 0,
                                        'center_points': 0,
                                        'left_points': 0,
                                        'match_count': 0
                                    }

                                # Aggregate
                                player_agg[pid]['roles_played'].add(match_role)
                                player_agg[pid]['right_points'] += right_pts
                                player_agg[pid]['center_points'] += center_pts
                                player_agg[pid]['left_points'] += left_pts
                                player_agg[pid]['match_count'] += 1

                except Exception as e:
                    # Continue to next player if one fails
                    pass

        except Exception as e:
            print(f"Error processing match {match_id}: {e}")

    # Finalize positions
    real_positions = []

    for pid, data in player_agg.items():
        if data['match_count'] > 0:
            total_pts = data['right_points'] + data['center_points'] + data['left_points']
            if total_pts == 0:
                continue

            # Multi-Position Logic (combines roles with "or")
            specific_role = list(set(data['roles_played'])) if len(set(data['roles_played'])) > 1 else list(data['roles_played'])[0]

            # Calculate percentages for verification/display
            right_pct = (data['right_points'] / total_pts) * 100
            center_pct = (data['center_points'] / total_pts) * 100
            left_pct = (data['left_points'] / total_pts) * 100

            real_positions.append({
                'player_id': pid,
                'player_name': data['name'],
                'role': data['base_role'],
                'specific_role': specific_role,
                'matches_played': data['match_count'],
                'Right%': round(right_pct, 1),
                'Center%': round(center_pct, 1),
                'Left%': round(left_pct, 1)
            })

    return pd.DataFrame(real_positions)

def analyze_opponent_comprehensive_multimatch(api_base: str, matches_info: dict) -> dict:
    """
    Performs comprehensive opponent analysis based on multiple matches (last 5).
    Aggregates tactics, formations, and key threats with detailed position inference.
    """

    target_team = matches_info.get('target_team_name')
    if not target_team:
        return {'error': 'Target team name not found in matches_info'}

    print(f"Starting detailed multi-match analysis for '{target_team}'...")

    match_ids = [k for k in matches_info.keys() if isinstance(k, int)]

    # Aggregators
    total_possession = 0
    total_pass_accuracy = 0
    matches_with_stats = 0

    formation_counts = {}

    # Player Stats: {pid: {name, goals, assists, shots, rating_sum, rating_count, match_count, roles: set(), roles_played: set()}}
    player_stats_agg = {}

    for match_id in match_ids:
        try:
            # 1. Determine side
            match_meta = matches_info.get(match_id, {})
            home_team_name = match_meta.get('homeTeam')
            side = 'home' if home_team_name == target_team else 'away'

            # 2. Statistics
            try:
                stats_resp = requests.get(api_base + f'events/{match_id}/statistics', timeout=5)
                if stats_resp.status_code == 200:
                    st = stats_resp.json().get('statistics', [])
                    idx = 0 if side == 'home' else 1
                    if len(st) > idx:
                        team_stats_groups = st[idx].get('groups', [])
                        for group in team_stats_groups:
                            for item in group.get('statisticsItems', []):
                                name = item.get('name')
                                val_str = str(item.get('homeValue' if side == 'home' else 'awayValue', '0')).replace('%', '')
                                try: val = float(val_str)
                                except: val = 0.0

                                if name == 'Ball possession':
                                    total_possession += val
                                elif name == 'Accurate passes':
                                    total_pass_accuracy += val
                        matches_with_stats += 1
            except Exception as e:
                print(f"Stats error match {match_id}: {e}")

            # 3. Lineups & Heatmaps
            try:
                lineups_resp = requests.get(api_base + f'events/{match_id}/lineups', timeout=10)
                if lineups_resp.status_code == 200:
                    ln = lineups_resp.json()
                    side_data = ln.get(side, {})

                    # Formation
                    fmt = side_data.get('formation')
                    if fmt:
                        formation_counts[fmt] = formation_counts.get(fmt, 0) + 1

                    # Players
                    players = side_data.get('players', [])
                    for p in players:
                        pid = str(p['player']['id'])
                        pname = p['player']['name']
                        p_role = p.get('position', 'M')

                        # Init Aggregation Entry
                        if pid not in player_stats_agg:
                            player_stats_agg[pid] = {
                                'name': pname,
                                'goals': 0, 'assists': 0, 'shots': 0,
                                'rating_sum': 0, 'rating_count': 0, 'match_count': 0,
                                'roles': set(),
                                'roles_played': set()
                            }

                        # Stats
                        p_stats = p.get('statistics', {})
                        goals = int(p_stats.get('goals', 0))
                        assists = int(p_stats.get('goalAssist', 0))
                        shots = int(p_stats.get('totalShots', 0))
                        try: rating = float(p_stats.get('rating', 0))
                        except: rating = 0.0

                        player_stats_agg[pid]['goals'] += goals
                        player_stats_agg[pid]['assists'] += assists
                        player_stats_agg[pid]['shots'] += shots
                        player_stats_agg[pid]['match_count'] += 1
                        if rating > 0:
                            player_stats_agg[pid]['rating_sum'] += rating
                            player_stats_agg[pid]['rating_count'] += 1

                        # Store base role for fallback
                        player_stats_agg[pid]['roles'].add(p_role)

                        # Heatmap (Per-Match Logic)
                        match_role = p_role
                        try:
                            hm_resp = requests.get(api_base + f'events/{match_id}/player/{pid}/heatmap', timeout=5)
                            if hm_resp.status_code == 200:
                                hm = hm_resp.json().get('heatmap', [])
                                if hm:
                                    right_pts = sum(1 for pt in hm if pt['y'] < 25)
                                    center_pts = sum(1 for pt in hm if 25 <= pt['y'] <= 75)
                                    left_pts = sum(1 for pt in hm if pt['y'] > 75)

                                    x_coords = [pt['x'] for pt in hm]
                                    avg_x = sum(x_coords) / len(x_coords) if x_coords else 50

                                    total_match_pts = right_pts + center_pts + left_pts
                                    if total_match_pts > 0:
                                        zones = {'Right': right_pts, 'Center': center_pts, 'Left': left_pts}
                                        dominant_zone = max(zones, key=zones.get)

                                        if p_role == 'G': match_role = 'G'
                                        elif p_role == 'D':
                                            if dominant_zone == 'Right': match_role = 'RB'
                                            elif dominant_zone == 'Left': match_role = 'LB'
                                            else: match_role = 'CB'
                                        elif p_role == 'M':
                                            if dominant_zone == 'Right': match_role = 'RM'
                                            elif dominant_zone == 'Left': match_role = 'LM'
                                            else:
                                                if avg_x < 45: match_role = 'CDM'
                                                elif avg_x > 65: match_role = 'CAM'
                                                else: match_role = 'CM'
                                        elif p_role == 'F':
                                            if dominant_zone == 'Right': match_role = 'RW'
                                            elif dominant_zone == 'Left': match_role = 'LW'
                                            else: match_role = 'ST'
                        except:
                            pass

                        player_stats_agg[pid]['roles_played'].add(match_role)

            except Exception as e:
                print(f"Lineup error match {match_id}: {e}")

        except Exception as e:
            print(f"Match {match_id} error: {e}")

    # --- Synthesize Results ---

    # 1. Tactics
    avg_poss = total_possession / matches_with_stats if matches_with_stats > 0 else 50.0
    avg_pass = total_pass_accuracy / matches_with_stats if matches_with_stats > 0 else 0.0

    style_labels = []
    if avg_poss > 55: style_labels.append('POSSESSION_DOMINANT')
    elif avg_poss < 45: style_labels.append('COUNTER_ATTACK_FOCUSED')
    else: style_labels.append('BALANCED_PLAYSTYLE')

    if formation_counts:
        primary_formation = max(formation_counts, key=formation_counts.get)
    else:
        primary_formation = "Unknown"

    # 2. Key Threats & Positions
    scored_players = []
    for pid, data in player_stats_agg.items():
        if data['match_count'] == 0:
            continue

        avg_rating = data['rating_sum'] / data['rating_count'] if data['rating_count'] > 0 else 0

        # Position Inference from Per-Match Aggregated Heatmap
        if data['roles_played']:
            detailed_roles = sorted(list(data['roles_played']))
        else:
            detailed_roles = list(data['roles'])

        # Per-Match Averages for more accurate threat codes
        goals_per_game = data['goals'] / data['match_count']
        assists_per_game = data['assists'] / data['match_count']
        shots_per_game = data['shots'] / data['match_count']

        # Weighted Threat Score based on per-game rates
        threat_score = (avg_rating * 1.5) + (goals_per_game * 10.0) + (assists_per_game * 8.0) + (shots_per_game * 0.5)

        codes = []
        if goals_per_game >= 0.4: codes.append('CLINICAL_FINISHER')
        if assists_per_game >= 0.3: codes.append('DISTRIBUTOR')
        if shots_per_game >= 2.0: codes.append('HIGH_VOLUME_SHOOTER')

        scored_players.append({
            "playerId": pid,
            "name": data['name'],
            "position": detailed_roles,
            "threatScore": round(threat_score, 1),
            "threatCodes": codes,
            "stats": {
                "totalGoals": data['goals'],
                "totalAssists": data['assists'],
                "avgRating": round(avg_rating, 1)
            }
        })

    top_threats = sorted(scored_players, key=lambda x: x['threatScore'], reverse=True)

    # 3. Vulnerabilities
    vulns = []
    if avg_poss > 60: vulns.append('HIGH_DEFENSIVE_LINE')
    if avg_poss < 40: vulns.append('PASSIVE_MIDFIELD')

    report = {
        "opponentAnalysis": {
            "tacticalStyle": {
                "inferredFormation": primary_formation,
                "styleLabels": style_labels,
                "metrics": {
                    "avgPossession": round(avg_poss, 1),
                    "avgPassAccuracy": round(avg_pass, 1)
                }
            },
            "keyThreats": top_threats,
            "vulnerabilities": vulns
        }
    }

    return report

def analyze_opponent_comprehensive_multimatch(api_base: str, matches_info: dict) -> dict:
    """
    Performs comprehensive opponent analysis based on multiple matches (last 5).
    Aggregates tactics, formations, and key threats with detailed position inference.
    """

    target_team = matches_info.get('target_team_name')
    if not target_team:
        return {'error': 'Target team name not found in matches_info'}

    print(f"Starting detailed multi-match analysis for '{target_team}'...")

    match_ids = [k for k in matches_info.keys() if isinstance(k, int)]

    # Aggregators
    total_possession = 0
    total_pass_accuracy = 0
    matches_with_stats = 0

    formation_counts = {}

    # Player Stats: {pid: {name, goals, assists, shots, rating_sum, rating_count, match_count, roles: set(), roles_played: set()}}
    player_stats_agg = {}

    for match_id in match_ids:
        try:
            # 1. Determine side
            match_meta = matches_info.get(match_id, {})
            home_team_name = match_meta.get('homeTeam')
            side = 'home' if home_team_name == target_team else 'away'

            # 2. Statistics
            try:
                stats_resp = requests.get(api_base + f'events/{match_id}/statistics', timeout=5)
                if stats_resp.status_code == 200:
                    st = stats_resp.json().get('statistics', [])
                    idx = 0 if side == 'home' else 1
                    if len(st) > idx:
                        team_stats_groups = st[idx].get('groups', [])
                        for group in team_stats_groups:
                            for item in group.get('statisticsItems', []):
                                name = item.get('name')
                                val_str = str(item.get('homeValue' if side == 'home' else 'awayValue', '0')).replace('%', '')
                                try: val = float(val_str)
                                except: val = 0.0

                                if name == 'Ball possession':
                                    total_possession += val
                                elif name == 'Accurate passes':
                                    total_pass_accuracy += val
                        matches_with_stats += 1
            except Exception as e:
                print(f"Stats error match {match_id}: {e}")

            # 3. Lineups & Heatmaps
            try:
                lineups_resp = requests.get(api_base + f'events/{match_id}/lineups', timeout=10)
                if lineups_resp.status_code == 200:
                    ln = lineups_resp.json()
                    side_data = ln.get(side, {})

                    # Formation
                    fmt = side_data.get('formation')
                    if fmt:
                        formation_counts[fmt] = formation_counts.get(fmt, 0) + 1

                    # Players
                    players = side_data.get('players', [])
                    for p in players:
                        pid = str(p['player']['id'])
                        pname = p['player']['name']
                        p_role = p.get('position', 'M')

                        # Init Aggregation Entry
                        if pid not in player_stats_agg:
                            player_stats_agg[pid] = {
                                'name': pname,
                                'goals': 0, 'assists': 0, 'shots': 0,
                                'rating_sum': 0, 'rating_count': 0, 'match_count': 0,
                                'roles': set(),
                                'roles_played': set()
                            }

                        # Stats
                        p_stats = p.get('statistics', {})
                        goals = int(p_stats.get('goals', 0))
                        assists = int(p_stats.get('goalAssist', 0))
                        shots = int(p_stats.get('totalShots', 0))
                        try: rating = float(p_stats.get('rating', 0))
                        except: rating = 0.0

                        player_stats_agg[pid]['goals'] += goals
                        player_stats_agg[pid]['assists'] += assists
                        player_stats_agg[pid]['shots'] += shots
                        player_stats_agg[pid]['match_count'] += 1
                        if rating > 0:
                            player_stats_agg[pid]['rating_sum'] += rating
                            player_stats_agg[pid]['rating_count'] += 1

                        # Store base role for fallback
                        player_stats_agg[pid]['roles'].add(p_role)

                        # Heatmap (Per-Match Logic)
                        match_role = p_role
                        try:
                            hm_resp = requests.get(api_base + f'events/{match_id}/player/{pid}/heatmap', timeout=5)
                            if hm_resp.status_code == 200:
                                hm = hm_resp.json().get('heatmap', [])
                                if hm:
                                    right_pts = sum(1 for pt in hm if pt['y'] < 25)
                                    center_pts = sum(1 for pt in hm if 25 <= pt['y'] <= 75)
                                    left_pts = sum(1 for pt in hm if pt['y'] > 75)

                                    x_coords = [pt['x'] for pt in hm]
                                    avg_x = sum(x_coords) / len(x_coords) if x_coords else 50

                                    total_match_pts = right_pts + center_pts + left_pts
                                    if total_match_pts > 0:
                                        zones = {'Right': right_pts, 'Center': center_pts, 'Left': left_pts}
                                        dominant_zone = max(zones, key=zones.get)

                                        if p_role == 'G': match_role = 'G'
                                        elif p_role == 'D':
                                            if dominant_zone == 'Right': match_role = 'RB'
                                            elif dominant_zone == 'Left': match_role = 'LB'
                                            else: match_role = 'CB'
                                        elif p_role == 'M':
                                            if dominant_zone == 'Right': match_role = 'RM'
                                            elif dominant_zone == 'Left': match_role = 'LM'
                                            else:
                                                if avg_x < 45: match_role = 'CDM'
                                                elif avg_x > 65: match_role = 'CAM'
                                                else: match_role = 'CM'
                                        elif p_role == 'F':
                                            if dominant_zone == 'Right': match_role = 'RW'
                                            elif dominant_zone == 'Left': match_role = 'LW'
                                            else: match_role = 'ST'
                        except:
                            pass

                        player_stats_agg[pid]['roles_played'].add(match_role)

            except Exception as e:
                print(f"Lineup error match {match_id}: {e}")

        except Exception as e:
            print(f"Match {match_id} error: {e}")

    # --- Synthesize Results ---

    # 1. Tactics
    avg_poss = total_possession / matches_with_stats if matches_with_stats > 0 else 50.0
    avg_pass = total_pass_accuracy / matches_with_stats if matches_with_stats > 0 else 0.0

    style_labels = []
    if avg_poss > 55: style_labels.append('POSSESSION_DOMINANT')
    elif avg_poss < 45: style_labels.append('COUNTER_ATTACK_FOCUSED')
    else: style_labels.append('BALANCED_PLAYSTYLE')

    if formation_counts:
        primary_formation = max(formation_counts, key=formation_counts.get)
    else:
        primary_formation = "Unknown"

    # 2. Key Threats & Positions
    scored_players = []
    for pid, data in player_stats_agg.items():
        if data['match_count'] == 0:
            continue

        avg_rating = data['rating_sum'] / data['rating_count'] if data['rating_count'] > 0 else 0

        # Position Inference from Per-Match Aggregated Heatmap
        if data['roles_played']:
            detailed_roles = sorted(list(data['roles_played']))
        else:
            detailed_roles = list(data['roles'])

        # Per-Match Averages for more accurate threat codes
        goals_per_game = data['goals'] / data['match_count']
        assists_per_game = data['assists'] / data['match_count']
        shots_per_game = data['shots'] / data['match_count']

        # Weighted Threat Score based on per-game rates
        threat_score = (avg_rating * 1.5) + (goals_per_game * 10.0) + (assists_per_game * 8.0) + (shots_per_game * 0.5)

        codes = []
        if goals_per_game >= 0.4: codes.append('CLINICAL_FINISHER')
        if assists_per_game >= 0.3: codes.append('DISTRIBUTOR')
        if shots_per_game >= 2.0: codes.append('HIGH_VOLUME_SHOOTER')

        scored_players.append({
            "playerId": pid,
            "name": data['name'],
            "position": detailed_roles,
            "threatScore": round(threat_score, 1),
            "threatCodes": codes,
            "stats": {
                "totalGoals": data['goals'],
                "totalAssists": data['assists'],
                "avgRating": round(avg_rating, 1)
            }
        })

    top_threats = sorted(scored_players, key=lambda x: x['threatScore'], reverse=True)[:5]

    # 3. Vulnerabilities
    vulns = []
    if avg_poss > 60: vulns.append('HIGH_DEFENSIVE_LINE')
    if avg_poss < 40: vulns.append('PASSIVE_MIDFIELD')

    report = {
        "opponentAnalysis": {
            "tacticalStyle": {
                "inferredFormation": primary_formation,
                "styleLabels": style_labels,
                "metrics": {
                    "avgPossession": round(avg_poss, 1),
                    "avgPassAccuracy": round(avg_pass, 1)
                }
            },
            "keyThreats": top_threats,
            "vulnerabilities": vulns
        }
    }

    return report

# dectionary containing the importance of each feature for every position
scores = {

    # ---------------- GOALKEEPER ----------------
    "G": {
        "saves": 8.0,
        "savedShotsFromInsideTheBox": 10.0,
        "goalsPrevented": 18.0,
        "keeperSaveValue": 45.0,
        "goodHighClaim": 6.0,
        "totalKeeperSweeper": 3.0,
        "accurateKeeperSweeper": 5.0,
        "accurateLongBalls": 2.0,
        "passValueNormalized": 12.0,
        "errorLeadToAShot": -8.0,
        "errorLeadToAGoal": -30.0,
        "ownGoals": -40.0,
        "rating": 5.0,
        "minutesPlayed": 0.02
    },

    # ---------------- FULLBACKS ----------------
    "RB": {
        "accuratePass": 4.0,
        "accurateCross": 8.0,
        "totalCross": 3.0,
        "duelWon": 4.0,
        "wonTackle": 6.0,
        "interceptionWon": 6.0,
        "progressiveBallCarriesCount": 7.0,
        "totalProgression": 5.0,
        "keyPass": 6.0,
        "passValueNormalized": 14.0,
        "defensiveValueNormalized": 12.0,
        "possessionLostCtrl": -4.0,
        "rating": 4.0,
        "minutesPlayed": 0.02
    },

    "LB": {  # same logic as RB
        "accuratePass": 4.0,
        "accurateCross": 8.0,
        "totalCross": 3.0,
        "duelWon": 4.0,
        "wonTackle": 6.0,
        "interceptionWon": 6.0,
        "progressiveBallCarriesCount": 7.0,
        "totalProgression": 5.0,
        "keyPass": 6.0,
        "passValueNormalized": 14.0,
        "defensiveValueNormalized": 12.0,
        "possessionLostCtrl": -4.0,
        "rating": 4.0,
        "minutesPlayed": 0.02
    },

    # ---------------- CENTER BACK ----------------
    "CB": {
        "totalClearance": 7.0,
        "aerialWon": 8.0,
        "wonTackle": 7.0,
        "interceptionWon": 8.0,
        "outfielderBlock": 5.0,
        "duelWon": 6.0,
        "defensiveValueNormalized": 18.0,
        "passValueNormalized": 8.0,
        "errorLeadToAShot": -8.0,
        "errorLeadToAGoal": -25.0,
        "rating": 4.0,
        "minutesPlayed": 0.02
    },

    # ---------------- DEFENSIVE MIDFIELDER ----------------
    "CDM": {
        "ballRecovery": 8.0,
        "interceptionWon": 7.0,
        "wonTackle": 6.0,
        "accuratePass": 5.0,
        "accurateOwnHalfPasses": 4.0,
        "accurateOppositionHalfPasses": 4.0,
        "defensiveValueNormalized": 16.0,
        "passValueNormalized": 14.0,
        "possessionLostCtrl": -5.0,
        "rating": 4.0,
        "minutesPlayed": 0.02
    },

    # ---------------- CENTRAL MIDFIELDER ----------------
    "CM": {
        "accuratePass": 6.0,
        "totalPass": 2.0,
        "progressiveBallCarriesCount": 6.0,
        "totalProgression": 6.0,
        "keyPass": 6.0,
        "expectedAssists": 8.0,
        "passValueNormalized": 18.0,
        "dribbleValueNormalized": 10.0,
        "possessionLostCtrl": -4.0,
        "rating": 4.0,
        "minutesPlayed": 0.02
    },

    # ---------------- ATTACKING MIDFIELDER ----------------
    "CAM": {
        "keyPass": 10.0,
        "expectedAssists": 12.0,
        "bigChanceCreated": 14.0,
        "goalAssist": 18.0,
        "shotValueNormalized": 8.0,
        "dribbleValueNormalized": 12.0,
        "passValueNormalized": 16.0,
        "possessionLostCtrl": -4.0,
        "rating": 4.0,
        "minutesPlayed": 0.02
    },

    # ---------------- WIDE MIDFIELDERS ----------------
    "RM": {
        "accurateCross": 10.0,
        "totalCross": 4.0,
        "dribbleValueNormalized": 12.0,
        "progressiveBallCarriesCount": 8.0,
        "keyPass": 7.0,
        "expectedAssists": 6.0,
        "passValueNormalized": 14.0,
        "possessionLostCtrl": -4.0,
        "rating": 4.0,
        "minutesPlayed": 0.02
    },

    "LM": {  # mirror of RM
        "accurateCross": 10.0,
        "totalCross": 4.0,
        "dribbleValueNormalized": 12.0,
        "progressiveBallCarriesCount": 8.0,
        "keyPass": 7.0,
        "expectedAssists": 6.0,
        "passValueNormalized": 14.0,
        "possessionLostCtrl": -4.0,
        "rating": 4.0,
        "minutesPlayed": 0.02
    },

    # ---------------- WINGERS ----------------
    "RW": {
        "goals": 20.0,
        "expectedGoals": 12.0,
        "shotValueNormalized": 14.0,
        "dribbleValueNormalized": 16.0,
        "keyPass": 8.0,
        "expectedAssists": 8.0,
        "progressiveBallCarriesCount": 10.0,
        "bigChanceMissed": -8.0,
        "rating": 4.0,
        "minutesPlayed": 0.02
    },

    "LW": {  # mirror of RW
        "goals": 20.0,
        "expectedGoals": 12.0,
        "shotValueNormalized": 14.0,
        "dribbleValueNormalized": 16.0,
        "keyPass": 8.0,
        "expectedAssists": 8.0,
        "progressiveBallCarriesCount": 10.0,
        "bigChanceMissed": -8.0,
        "rating": 4.0,
        "minutesPlayed": 0.02
    },

    # ---------------- STRIKER ----------------
    "ST": {
        "goals": 30.0,
        "expectedGoals": 18.0,
        "expectedGoalsOnTarget": 12.0,
        "shotValueNormalized": 18.0,
        "bigChanceMissed": -10.0,
        "aerialWon": 6.0,
        "keyPass": 5.0,
        "rating": 4.0,
        "minutesPlayed": 0.02
    }
}

def apply_score_formula_singlevalue (row : Series ) -> Series :

    ''' this function is used to apply the score formulas on each row in the dataframe
    returns a score if the position is known else it'll return nan
    works only for single position values in the record'''

    # sets for all the possible positions and their commoun class
    attakers_pos = {"ST", "CF", "LW", "RW", "LF", "RF" , "F"}
    defenders_pos = {"CM", "CDM", "CAM", "LM", "RM", "AM", "DM" , "D"}
    midfielders_pos = {"CB", "LB", "RB", "LWB", "RWB" , "M"}
    goalkeepers_pos = {"GK" , "G"}

    # init value for score
    score = 0

    pos = row['specific_role'] # getting the position

    if str(pos).upper() in scores: # checking if the position is in positions

        pos = str(pos).upper() # converting the position to uppercase to match the keys in scores dictionary
        for feature in  scores[pos]:
            try :
                score += scores[pos][feature] * row[feature]
            except Exception as e :
                return f"some features is not found : {e}"

    elif type(pos) == list or type(pos) == set : # if the position is a list or a set (multiple positions)

        score =  apply_score_formula_multivalue(row)

    else:
        score = np.nan # if the position is unknown return nan

    return score

def apply_score_formula_multivalue (row : Series ) -> Series :

    ''' this function is used to apply the score formulas on each row in the dataframe
    returns a score if the position is known else it'll return nan
    works only for multi position values in the record (in list or set)'''

    try :
        scores = [] # list to hold the scores for each position

        for pos in row['specific_role']: # getting the score for each position
            row2 = row.copy() # creating a copy of the row to modify the position value
            row2['specific_role'] = pos
            scores.append( apply_score_formula_singlevalue (row2))
        return scores

    except Exception as e:
        return f"something wrong with the position values : {e}"

# some helper functions for nomalization and handling multi-value features
def numeric_equivalent_min(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.min(x)
    return x

def numeric_equivalent_max(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.max(x)
    return x

def normalize_value(x, min_s, max_s):

    if max_s == min_s:
        return 0

    if isinstance(x, (list, tuple, np.ndarray)):
        return [100 * (v - min_s) / (max_s - min_s) for v in x]

    return 100 * (x - min_s) / (max_s - min_s)

# def get_players_scores( api_base : str , players_stats : DataFrame , teams_info : DataFrame) -> DataFrame:

#     '''this function is used to take a dataframe of players stats and team name (each players should has stats for the last n matches)
#     and return the same dataframe of the players stats but also including a new column which is score column for each player
#     noting : this function works for the players who has single or multiple positions'''

#     try :

#         all_real_positions = get_player_real_position_multimatch(api_base , teams_info) # getting real positions
#         players_stats_filtered = players_stats[players_stats['player_name'].isin(all_real_positions['player_name'].unique())] # filtering our players from others

#         # getting the median of the statistics for the players
#         players_stats_filtered = players_stats_filtered.drop(columns=['match_id','position' , 'statistics_type']).groupby(by=['player_id' , 'player_name']).median().reset_index()

#         # concatinating the two dataframes
#         player_stats_with_scores = pd.concat([players_stats_filtered , all_real_positions.drop(columns='player_name')] , axis =1  )

#         # getting scores
#         player_stats_with_scores['score'] = player_stats_with_scores.apply(apply_score_formula_singlevalue , axis=1)

#         # getting the min and max values accross all positions
#         min_s = player_stats_with_scores['score'].apply(numeric_equivalent_min).min()
#         max_s = player_stats_with_scores['score'].apply(numeric_equivalent_max).max()

#         # normlizing the values
#         player_stats_with_scores["score"] = player_stats_with_scores["score"].apply(lambda x: normalize_value(x, min_s, max_s))

#         player_stats_with_scores = player_stats_with_scores.loc[:, ~player_stats_with_scores.columns.duplicated()]

#         return player_stats_with_scores

#     except Exception as e :
#         print(f'problem with getting real position of the players {e}')
#         return np.NaN


# player_stats_pos = get_players_scores(api_base , player_stats , teams_info)
# player_stats_pos.style.background_gradient(cmap='Greens').set_properties(**{'font-family': 'Segoe UI'})

def get_players_scores( api_base : str , players_stats : DataFrame , team_name : str) -> DataFrame:

    '''this function is used to take a dataframe of players stats and team name (each players should has stats for the last n matches)
    and return the same dataframe of the players stats but also including a new column which is score column for each player
    noting : this function works for the players who has single or multiple positions'''

    try :
        all_real_positions = get_player_real_position_multimatch(api_base , teams_info) # getting real positions
        players_stats_filtered = players_stats[players_stats['player_name'].isin(all_real_positions['player_name'].unique())] # filtering our players from others

        # getting the median of the statistics for the players
        players_stats_filtered = players_stats_filtered.drop(columns=['match_id','position' , 'statistics_type']).groupby(by=['player_id' , 'player_name']).median().reset_index()

        # concatinating the two dataframes
        player_stats_with_scores = pd.concat([players_stats_filtered , all_real_positions.drop(columns='player_name')] , axis =1  )

        player_stats_with_scores = player_stats_with_scores.loc[:, ~player_stats_with_scores.columns.duplicated()]

        # getting scores
        player_stats_with_scores['score'] = player_stats_with_scores.apply(apply_score_formula_singlevalue , axis=1)


        # separating the multiple positions for the same player into different rows
        player_stats_with_scores = player_stats_with_scores.explode(['specific_role', 'score']).reset_index(drop=True).drop_duplicates()

        # calculating the score inside each general position not accross all positions by setting the max value as 100 and the others is percentage from it
        player_stats_with_scores['score'] = player_stats_with_scores.groupby('role')['score'].transform(
            lambda x: np.where(
                x.max() != x.min(),
                (100 * x) / x.max(),
                x   # identical scores → assign 0
            )
        )



        # getting the names of the columns
        columns = list(player_stats_with_scores.columns)
        columns.remove('specific_role')
        columns.remove('score')

        return player_stats_with_scores.groupby(columns, as_index=False).agg({ # getting roles in a list intead of different rows
          'specific_role': lambda x: list(x) if len(list(x)) > 1 else x.iloc[0],
          'score': lambda x: list(x) if len(list(x)) > 1 else x.iloc[0]
      })

    except Exception as e :
        print(f'problem with getting real position of the players {e}')
        return np.NaN

def formation_suggestions(api_base: str , opponent_id: int) -> list:
    '''this function takes the base of the api without the endpoint , the opponent team id and returns the most suggested formation we should play with'''
    try :
        matches_info = get_team_lnm(api_base ,opponent_id , 5  ) # getting the last 5 matches info of the opponent team
    except Exception as e :
        return f'something went wrong with {e}'
    formations = [] # define a list to contain all the formations used by the opponent team in the last n matches
    for match_id in matches_info.keys(): # loop over each match id in the matches info dictionary
        if isinstance(match_id, int):
            # getting the correct key for the values based on whether the team is away or home
            if matches_info.get('target_team_name') == matches_info.get(match_id).get('awayTeam'):
                value_key = 'away'
            else :
                value_key = 'home'

            formations.append(requests.get(api_base + f'events/{match_id}/lineups').json().get(value_key).get('formation')) #getting the formation used by the opponent team in the match and adding it to the formations list

    opponent_most_used_formation = statistics.multimode(formations)[0]  # getting the most used formation by the opponent team in the last n matches

    counter_formations = { # formations to choose from
        "4-4-2": ["3-5-2", "4-3-3", "4-2-3-1"],
        "3-5-2": ["4-3-3", "4-4-2", "4-2-3-1"],
        "4-3-3": ["4-2-3-1", "5-4-1", "4-5-1", "3-5-2"],
        "4-2-3-1": ["4-3-3", "3-5-2"],
        "5-3-2": ["4-3-3", "4-2-3-1", "3-4-3"],
        "4-5-1": ["4-3-3", "3-5-2", "4-4-2"],
        "3-4-3": ["4-3-3", "4-2-3-1"],
        "4-3-2-1": ["3-5-2", "4-4-2"],
        "4-1-4-1": ["4-3-3", "4-2-3-1"],
        "4-2-2-2": ["4-2-3-1"]
    }
    suggested_formation = counter_formations[opponent_most_used_formation]

    return suggested_formation

def get_best_starting_lineup_from_recommendations(players_stats_scored: pd.DataFrame, recommended_formations: list, real_pos_df: pd.DataFrame = None) -> tuple:
    """
    Simulates building a lineup for EVERY formation.
    CRITICAL FIX: Bypasses corrupted positions in `players_stats_scored` by strictly remapping
    Real Positions directly from `real_pos_df` (Section 4) using `clean_id`.
    """
    try:
        if 'score' not in players_stats_scored.columns:
            return "Scores missing. Run Section 5 first.", None, None

        df = players_stats_scored.copy()

        # 1. PREPARE THE DATA & FIX SCRAMBLED POSITIONS
        df['score'] = df['score'].apply(lambda x: max(x) if isinstance(x, list) else x)

        # Get clean IDs first
        if 'player_id' in df.columns:
            df['clean_id'] = df['player_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        elif 'id' in df.columns:
            df['clean_id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        else:
            df['clean_id'] = 'N/A'

        # Get Names safely
        name_col = next((c for c in ['player_name', 'name', 'playerName'] if c in df.columns), None)
        df['Player Name'] = df[name_col] if name_col else 'Unknown'

        # =========================================================================
        # THE FIX: IGNORE CORRUPTED POSITIONS AND FETCH DIRECTLY FROM SECTION 4
        # =========================================================================
        if real_pos_df is not None and not real_pos_df.empty:
            temp_real = real_pos_df.copy()
            temp_real['clean_id'] = temp_real['player_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

            # Create a 100% accurate dictionary mapping from Section 4
            id_to_true_role = dict(zip(temp_real['clean_id'], temp_real['specific_role']))

            # Apply to df
            df['raw_true_position'] = df['clean_id'].map(id_to_true_role)

            # Format to list safely
            df['Real Position'] = df['raw_true_position'].apply(
                lambda x: [str(i).strip() for i in x] if isinstance(x, list)
                else [p.strip() for p in str(x).split('or')] if 'or' in str(x).lower()
                else [str(x).strip()] if pd.notna(x) else ['Unknown']
            )
            df.drop(columns=['raw_true_position'], inplace=True)
        else:
            print("WARNING: real_positions_df is missing! Positions cannot be verified and might be scrambled.")
            df['Real Position'] = [['Unknown']] * len(df)

        # Optional: update Name from real_positions just to be safe
        if real_pos_df is not None and not real_pos_df.empty:
            name_col_real = next((c for c in ['player_name', 'name', 'playerName'] if c in temp_real.columns), None)
            if name_col_real:
                id_to_name = dict(zip(temp_real['clean_id'], temp_real[name_col_real]))
                df['Player Name'] = df['clean_id'].map(id_to_name).fillna(df['Player Name'])

        df = df.sort_values(by='score', ascending=False).reset_index(drop=True)

        def matches_bucket(role_list, tags):
            for r in role_list:
                clean_r = r.upper()
                for tag in tags:
                    if tag in clean_r: return True
            return False

        # --- UNIVERSAL DYNAMIC BLUEPRINTS ---
        def get_formation_blueprint(formation_str):
            bp = {'Goalkeeper': {'max': 1, 'filled': 0, 'tags': ['GK'], 'players': []}}
            f = str(formation_str).strip()

            # --- 1. Defense Line ---
            if f.startswith('4-'):
                bp.update({
                    'Left Back': {'max': 1, 'filled': 0, 'tags': ['LB', 'LWB'], 'players': []},
                    'Right Back': {'max': 1, 'filled': 0, 'tags': ['RB', 'RWB'], 'players': []},
                    'Center Backs': {'max': 2, 'filled': 0, 'tags': ['CB'], 'players': []}
                })
            elif f.startswith('3-'):
                bp.update({
                    'Center Backs': {'max': 3, 'filled': 0, 'tags': ['CB', 'LB', 'RB'], 'players': []}
                })
            elif f.startswith('5-'):
                bp.update({
                    'Left Wing Back': {'max': 1, 'filled': 0, 'tags': ['LWB', 'LB', 'LM'], 'players': []},
                    'Right Wing Back': {'max': 1, 'filled': 0, 'tags': ['RWB', 'RB', 'RM'], 'players': []},
                    'Center Backs': {'max': 3, 'filled': 0, 'tags': ['CB'], 'players': []}
                })
            else:
                parts = [int(p) for p in f.split('-') if p.isdigit()]
                def_cnt = parts[0] if parts else 4
                bp['Defenders'] = {'max': def_cnt, 'filled': 0, 'tags': ['CB', 'LB', 'RB', 'LWB', 'RWB'], 'players': []}

            # --- 2. Midfield & Attack Lines ---
            if f in ["4-3-3", "4-1-2-3", "4-3-2-1"]:
                bp.update({
                    'Central Midfielders': {'max': 3, 'filled': 0, 'tags': ['CM', 'CDM', 'CAM'], 'players': []},
                    'Left Winger': {'max': 1, 'filled': 0, 'tags': ['LW', 'LM'], 'players': []},
                    'Right Winger': {'max': 1, 'filled': 0, 'tags': ['RW', 'RM'], 'players': []},
                    'Striker': {'max': 1, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}
                })
            elif f in ["4-2-3-1", "4-4-1-1"]:
                bp.update({
                    'Defensive Midfielders': {'max': 2, 'filled': 0, 'tags': ['CDM', 'CM'], 'players': []},
                    'Attacking Midfielder': {'max': 1, 'filled': 0, 'tags': ['CAM', 'CM'], 'players': []},
                    'Left Winger': {'max': 1, 'filled': 0, 'tags': ['LW', 'LM'], 'players': []},
                    'Right Winger': {'max': 1, 'filled': 0, 'tags': ['RW', 'RM'], 'players': []},
                    'Striker': {'max': 1, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}
                })
            elif f in ["4-4-2", "4-1-4-1", "4-1-3-2", "4-5-1"]:
                bp.update({
                    'Central Midfielders': {'max': 2, 'filled': 0, 'tags': ['CM', 'CDM', 'CAM'], 'players': []},
                    'Left Mid': {'max': 1, 'filled': 0, 'tags': ['LM', 'LW'], 'players': []},
                    'Right Mid': {'max': 1, 'filled': 0, 'tags': ['RM', 'RW'], 'players': []},
                    'Strikers': {'max': 2 if f in ["4-4-2", "4-1-3-2"] else 1, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}
                })
                if f == "4-1-4-1": bp['Defensive Midfielders'] = {'max': 1, 'filled': 0, 'tags': ['CDM'], 'players': []}
                if f == "4-5-1": bp['Central Midfielders']['max'] = 3
            elif f in ["3-5-2", "3-1-4-2"]:
                bp.update({
                    'Central Midfielders': {'max': 3, 'filled': 0, 'tags': ['CM', 'CDM', 'CAM'], 'players': []},
                    'Left Mid/Wing': {'max': 1, 'filled': 0, 'tags': ['LM', 'LWB', 'LW'], 'players': []},
                    'Right Mid/Wing': {'max': 1, 'filled': 0, 'tags': ['RM', 'RWB', 'RW'], 'players': []},
                    'Strikers': {'max': 2, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}
                })
            elif f in ["3-4-3", "3-4-2-1"]:
                bp.update({
                    'Central Midfielders': {'max': 2, 'filled': 0, 'tags': ['CM', 'CDM'], 'players': []},
                    'Left Mid/Wing': {'max': 1, 'filled': 0, 'tags': ['LM', 'LWB'], 'players': []},
                    'Right Mid/Wing': {'max': 1, 'filled': 0, 'tags': ['RM', 'RWB'], 'players': []},
                    'Left Forward': {'max': 1, 'filled': 0, 'tags': ['LW', 'CAM', 'ST'], 'players': []},
                    'Right Forward': {'max': 1, 'filled': 0, 'tags': ['RW', 'CAM', 'ST'], 'players': []},
                    'Striker': {'max': 1, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}
                })
            elif f in ["5-3-2", "5-4-1"]:
                bp.update({
                    'Central Midfielders': {'max': 3 if f=="5-3-2" else 2, 'filled': 0, 'tags': ['CM', 'CDM', 'CAM'], 'players': []},
                    'Strikers': {'max': 2 if f=="5-3-2" else 1, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}
                })
                if f == "5-4-1":
                    bp['Left Mid'] = {'max': 1, 'filled': 0, 'tags': ['LM', 'LW'], 'players': []}
                    bp['Right Mid'] = {'max': 1, 'filled': 0, 'tags': ['RM', 'RW'], 'players': []}
            else:
                parts = [int(p) for p in f.split('-') if p.isdigit()]
                if len(parts) >= 3:
                    mid_count = sum(parts[1:-1])
                    fwd_count = parts[-1]
                    bp['Midfielders'] = {'max': mid_count, 'filled': 0, 'tags': ['CM', 'CDM', 'CAM', 'LM', 'RM'], 'players': []}
                    bp['Forwards'] = {'max': fwd_count, 'filled': 0, 'tags': ['ST', 'LW', 'RW', 'CF', 'FW'], 'players': []}

            return bp

        best_total_score = -1
        best_formation_name = ""
        best_starting_xi = None
        best_bench = None

        # 2. SIMULATE EVERY RECOMMENDED FORMATION
        for target_formation in recommended_formations:
            formation_blueprint = get_formation_blueprint(str(target_formation))
            bench = []
            current_formation_score = 0

            # Fill the XI for this specific formation
            for _, row in df.iterrows():
                if row['Real Position'] == ['Unknown']: continue # Skip players without valid roles

                placed = False
                role_list = row['Real Position']

                # Check normal positions
                for bucket_name, bucket_info in formation_blueprint.items():
                    if bucket_info['filled'] < bucket_info['max'] and matches_bucket(role_list, bucket_info['tags']):
                        bucket_info['players'].append({
                            'Selected Slot': bucket_name,
                            'Player ID': row['clean_id'],
                            'Player Name': row['Player Name'],
                            'Real Position': role_list,
                            'Score': row['score']
                        })
                        bucket_info['filled'] += 1
                        current_formation_score += row['score']
                        placed = True
                        break

                if not placed:
                    bench.append({
                        'Player ID': row['clean_id'],
                        'Player Name': row['Player Name'],
                        'Real Position': role_list,
                        'Score': row['score']})

            # Force Fill missing slots from bench
            total_filled = sum(b['filled'] for b in formation_blueprint.values())
            while total_filled < 11 and bench:
                best_sub = bench.pop(0)
                for bucket_name, bucket_info in formation_blueprint.items():
                    if bucket_info['filled'] < bucket_info['max']:
                        bucket_info['players'].append({
                            'Selected Slot': f"{bucket_name} (Out of Position)",
                            'Player ID': best_sub['Player ID'],
                            'Player Name': best_sub['Player Name'],
                            'Real Position': best_sub['Real Position'],
                            'Score': best_sub['Score']
                        })
                        bucket_info['filled'] += 1
                        total_filled += 1
                        current_formation_score += (best_sub['Score'] * 0.8)
                        break

            if current_formation_score > best_total_score and total_filled >= 11:
                best_total_score = current_formation_score
                best_formation_name = target_formation

                lineup_list = []
                for b in formation_blueprint.values(): lineup_list.extend(b['players'])
                best_starting_xi = pd.DataFrame(lineup_list)

                bench_df = pd.DataFrame(bench).head(7)
                if not bench_df.empty: bench_df.insert(0, 'Selected Slot', 'Substitute')
                best_bench = bench_df

        return best_formation_name, best_starting_xi, best_bench

    except Exception as e:
        print(f"Algorithm Error: {e}")
        return None, None, None

def get_season_tournament_ids(api_base : str , players_ids : list) -> dict:
    '''this is a helper function that returns a dictionary that contains the season id and tournament id for each player
    input -> list of players id's
    output -> dictionary that contains each player id as the key then in it the season and tournament ids'''

    ids = dict()

    if players_ids == np.nan:
        return "players_ids list is missing"

    for player_id in players_ids:
        response = requests.get(f'{api_base}players/{player_id}/statistics/seasons').json() # fetch the data
        tournament_id = response.get('uniqueTournamentSeasons')[0].get('uniqueTournament').get('id') # getting the tournament id for the player
        # getting the last season id for the player also handles if the player has only one season
        season_id = response.get('uniqueTournamentSeasons')[0].get('seasons')[1].get('id') if len(response.get('uniqueTournamentSeasons')[0].get('seasons')) > 1 else response.get('uniqueTournamentSeasons')[0].get('seasons')[0].get('id')
        ids[player_id] = {'tournament_id': tournament_id, 'season_id': season_id} # adding data to the dictionary

    return ids

# #parallel version of the above function to speed up the process of fetching data for multiple players
# import requests
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from time import sleep

# def fetch_one(session: requests.Session, base_url: str, player_id: int, timeout=10):
#     url = f"{base_url.rstrip('/')}/players/{player_id}/statistics/seasons"
#     resp = session.get(url, timeout=timeout)
#     resp.raise_for_status()
#     return player_id, resp.json()

# def fetch_parallel_requests(base_url: str, player_ids: list[int], max_workers: int = 16) -> dict[int, dict]:
#     if not player_ids:
#         return {}

#     results: dict[int, dict] = {}
#     # Tip: If your API dislikes shared Sessions across threads,
#     # move Session() creation inside the worker function.
#     with requests.Session() as session:
#         session.headers.update({"Accept": "application/json"})

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = [executor.submit(fetch_one, session, base_url, pid) for pid in player_ids]
#             for f in as_completed(futures):
#                 try:
#                     pid, data = f.result()
#                     results[pid] = data
#                 except requests.HTTPError as e:
#                     code = e.response.status_code if e.response is not None else "unknown"
#                     results[pid] = {"error": f"http {code}"}
#                 except requests.RequestException as e:
#                     results[pid] = {"error": f"network {str(e)}"}
#                 except Exception as e:
#                     results[pid] = {"error": f"unexpected {str(e)}"}
#                 # Optional light pacing to respect rate limits
#                 # sleep(0.01)

#     return results
# get_season_tournament_ids( api_base, list(player_stats_pos['player_id']))

def get_tacticale_and_formation(api_base : str , recommended_players : DataFrame , opponent_id : int , suggested_formations : str) -> json :

    ''' this functions is used to get the tacticale style for the team and select the best formation to play with from the recommendations
    it works on --> players name , players id and recommended formations '''

    try :
        ids = get_season_tournament_ids( api_base, list(recommended_players['Player ID']) )

        general_statistics = [] # list to add the general statistics to it before merging
        for player_id in ids.keys() :
            response = requests.get(f"{api_base}players/{player_id}/unique-tournament/{ids[player_id]['tournament_id']}/season/{ids[player_id]['season_id']}/statistics/overall").json().get('statistics')
            response['id']= player_id
            general_statistics.append(response) # adding the general statistics to the list

        general_statistics = pd.DataFrame(general_statistics).fillna(0).drop(columns=['type' ,'statisticsType'])
        # general_statistics = recommended_players.merge(general_statistics )
        general_statistics = pd.concat([recommended_players[['Player ID' , 'Player Name' , 'Selected Slot' , 'Real Position', 'Score'] ] , general_statistics] , axis=1).drop(columns='Player ID')

        # getting opponent formation for the last match
        opponent_lastm_info = get_team_lnm(api_base , opponent_id , 1)
        opponent_formation = requests.get(api_base + f'events/{list(opponent_lastm_info.keys())[0]}/lineups').json().get('home' if opponent_lastm_info.get('target_team_name') == opponent_lastm_info[list(opponent_lastm_info.keys())[0]]['homeTeam'] else 'away').get('formation')

        # getting the prompt ready for the llm
        prompt = f"""
You are a professional football manager and tactical analyst. \
You are given a dataset containing the statistics and positions of the players selected for the next match. \

Player data: \
{general_statistics.to_json()} \

The suggested formation for our team to play with: \
{suggested_formations} \

The opponent played their last match using this formation: \
{opponent_formation} \

Your task:
1. Analyze the player statistics and positions.
2. Suggest a tactical strategy that best fits the squad and counters the opponent's formation. \

Return ONLY a valid JSON object in the following format:

{{ "suggestedFormation": "formation",
  "strategyCode": "strategy_name" }}

Example:
{{ "suggestedFormation": "4-3-3",
  "strategyCode": "COUNTER_ATTACK_DIRECT" }}
"""

        # llm call
        client = genai.Client(api_key="AIzaSyAaP3-wHjenRqi_vc0-4ved2wox1NRPqCY")


        response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        config={
            "system_instruction": "Only output the requierd output"
        },
        contents= prompt
    )
        # getting the result from the output of the llm
        result = response.text
        return result

    except Exception as e :
        return f'something went wrong with : {e}'

def get_training_player_stats(api_base: str, matches_info: dict) -> pd.DataFrame:
    '''This function gets player stats using the exact same logic as analyze_opponent_comprehensive_multimatch from Section 4.'''

    target_team = matches_info.get('target_team_name')
    if not target_team:
        return pd.DataFrame()

    match_ids = [k for k in matches_info.keys() if isinstance(k, int)]

    # Player Stats: {pid: {name, goals, assists, shots, rating_sum, rating_count, match_count, roles: set(), roles_played: set()}}
    player_stats_agg = {}

    for match_id in match_ids:
        try:
            # 1. Determine side
            match_meta = matches_info.get(match_id, {})
            home_team_name = match_meta.get('homeTeam')
            side = 'home' if home_team_name == target_team else 'away'

            # 3. Lineups & Heatmaps (Copied exactly from Section 4)
            try:
                lineups_resp = requests.get(api_base + f'events/{match_id}/lineups', timeout=10)
                if lineups_resp.status_code == 200:
                    ln = lineups_resp.json()
                    side_data = ln.get(side, {})

                    # Players
                    players = side_data.get('players', [])
                    for p in players:
                        pid = str(p['player']['id'])
                        pname = p['player']['name']
                        p_role = p.get('position', 'M')

                        # Init Aggregation Entry
                        if pid not in player_stats_agg:
                            player_stats_agg[pid] = {
                                'name': pname,
                                'goals': 0, 'assists': 0, 'shots': 0,
                                'rating_sum': 0, 'rating_count': 0, 'match_count': 0,
                                'roles': set(),
                                'roles_played': set()
                            }

                        # Stats
                        p_stats = p.get('statistics', {})
                        goals = int(p_stats.get('goals', 0))
                        assists = int(p_stats.get('goalAssist', 0))
                        shots = int(p_stats.get('totalShots', 0))
                        try: rating = float(p_stats.get('rating', 0))
                        except: rating = 0.0

                        player_stats_agg[pid]['goals'] += goals
                        player_stats_agg[pid]['assists'] += assists
                        player_stats_agg[pid]['shots'] += shots
                        player_stats_agg[pid]['match_count'] += 1
                        if rating > 0:
                            player_stats_agg[pid]['rating_sum'] += rating
                            player_stats_agg[pid]['rating_count'] += 1

                        # Store base role for fallback
                        player_stats_agg[pid]['roles'].add(p_role)

                        # Heatmap (Per-Match Logic)
                        match_role = p_role
                        try:
                            hm_resp = requests.get(api_base + f'events/{match_id}/player/{pid}/heatmap', timeout=5)
                            if hm_resp.status_code == 200:
                                hm = hm_resp.json().get('heatmap', [])
                                if hm:
                                    right_pts = sum(1 for pt in hm if pt['y'] < 25)
                                    center_pts = sum(1 for pt in hm if 25 <= pt['y'] <= 75)
                                    left_pts = sum(1 for pt in hm if pt['y'] > 75)

                                    x_coords = [pt['x'] for pt in hm]
                                    avg_x = sum(x_coords) / len(x_coords) if x_coords else 50

                                    total_match_pts = right_pts + center_pts + left_pts
                                    if total_match_pts > 0:
                                        zones = {'Right': right_pts, 'Center': center_pts, 'Left': left_pts}
                                        dominant_zone = max(zones, key=zones.get)

                                        if p_role == 'G': match_role = 'G'
                                        elif p_role == 'D':
                                            if dominant_zone == 'Right': match_role = 'RB'
                                            elif dominant_zone == 'Left': match_role = 'LB'
                                            else: match_role = 'CB'
                                        elif p_role == 'M':
                                            if dominant_zone == 'Right': match_role = 'RM'
                                            elif dominant_zone == 'Left': match_role = 'LM'
                                            else:
                                                if avg_x < 45: match_role = 'CDM'
                                                elif avg_x > 65: match_role = 'CAM'
                                                else: match_role = 'CM'
                                        elif p_role == 'F':
                                            if dominant_zone == 'Right': match_role = 'RW'
                                            elif dominant_zone == 'Left': match_role = 'LW'
                                            else: match_role = 'ST'
                        except:
                            pass

                        player_stats_agg[pid]['roles_played'].add(match_role)

            except Exception as e:
                print(f"Lineup error match {match_id}: {e}")

        except Exception as e:
            print(f"Match {match_id} error: {e}")

    # Now build the final dataframe replacing the JSON response with a dataframe of stats
    scored_players = []
    for pid, data in player_stats_agg.items():
        if data['match_count'] == 0:
            continue

        avg_rating = data['rating_sum'] / data['rating_count'] if data['rating_count'] > 0 else 0

        if data['roles_played']:
            detailed_roles = sorted(list(data['roles_played']))
        else:
            detailed_roles = list(data['roles'])

        goals_per_game = data['goals'] / data['match_count']
        assists_per_game = data['assists'] / data['match_count']
        shots_per_game = data['shots'] / data['match_count']

        threat_score = (avg_rating * 1.5) + (goals_per_game * 10.0) + (assists_per_game * 8.0) + (shots_per_game * 0.5)

        scored_players.append({
            "Player ID": pid,
            "Player Name": data['name'],
            "Positions": list(detailed_roles),
            "Matches Played": data['match_count'],
            "Mean Goals": round(goals_per_game, 2),
            "Mean Assists": round(assists_per_game, 2),
            "Mean Shots": round(shots_per_game, 2),
            "Average Rating": round(avg_rating, 1),
            "Threat Score": round(threat_score, 1)
        })

    return pd.DataFrame(scored_players)

def get_training_recommendations(api_base: str, average_stats_df: pd.DataFrame) -> json:
    '''This function sends the averaged player stats to an LLM to identify weaknesses and suggest training exercises.'''

    if average_stats_df.empty:
        return '{"error": "No player statistics provided"}'

    stats_json = average_stats_df.to_json(orient="records")

    prompt = f"""
You are a top-tier football coach and fitness trainer.
I am providing you with the average statistics of my team's players over their recent matches.

Player Average Match Statistics (JSON):
{stats_json}

Your task:
CRITICAL RULE: You MUST generate a recommendation object for EVERY SINGLE player provided in the input JSON. Do not omit any player. The output array MUST contain exactly the same number of players as the input array.

1. Analyze these numerical stats for every single player to identify their specific tactical or physical weaknesses (e.g., low average rating, low output in goals/assists for attackers, or fewer total shots).
2. For each identified weakness, recommend 1 to 2 targeted training drills or exercises.

Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json. The format MUST strongly adhere to the following structure.

Example Output format (One-shot Example):
[
  {{
    "playerId": "123456",
    "playerName": "Vinícius Júnior",
    "weakness": "Low average rating despite playing forward, suggesting poor final decision-making or conversion.",
    "recommendedExercises": ["1v1 offensive duels drill", "Finishing accuracy drills under pressure"]
  }},
  {{
    "playerId": "789101",
    "playerName": "Jude Bellingham",
    "weakness": "Low assist count for an attacking midfielder, implying lack of creative passes in the final third.",
    "recommendedExercises": ["Rondo drills with progressive passing", "Vision and spatial awareness scanning"]
  }}
]
"""

    try:
        client = genai.Client(api_key="AIzaSyAjd7dHUBIeyon7ODjg7MkeKMGEqdzhfoU")
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            config={
                "system_instruction": "You are a football tactician and trainer. Only output strictly valid JSON format without markdown blocks."
            },
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f'{{"error": "LLM API failed: {e}"}}'

