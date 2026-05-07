import requests
import pandas as pd
import json
import os
import statistics
import re
import numpy as np
from pandas import DataFrame, Series
from google import genai
from warnings import filterwarnings
from dotenv import load_dotenv

filterwarnings('ignore')
load_dotenv()

api_base = 'https://football-backend-app.victoriouswater-69fff737.swedencentral.azurecontainerapps.io/'
_api_cache: dict = {}
_DEFAULT_TIMEOUT = 15

def clear_api_cache():
    global _api_cache
    _api_cache.clear()

def cached_get(url: str, timeout: int=_DEFAULT_TIMEOUT):
    if url in _api_cache:
        return _api_cache[url]
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            try:
                resp.json()
            except ValueError:
                if '/heatmap' not in url:
                    print(f'[cached_get] ERROR: Invalid JSON received from {url}')
                return None
            _api_cache[url] = resp
        return resp
    except requests.exceptions.Timeout:
        if '/heatmap' not in url:
            print(f'[cached_get] TIMEOUT after {timeout}s: {url}')
        return None
    except requests.exceptions.RequestException as e:
        if '/heatmap' not in url:
            print(f'[cached_get] ERROR {e}: {url}')
        return None

def get_team_lnm(api_base: str, team_id: int, num_matchs: int) -> dict:
    response = cached_get(api_base + f'teams/{team_id}/events/last/0')
    if response is None or response.status_code != 200:
        return 'the api is down'
    match_info = {match['id']: {'homeTeam': match['homeTeam']['name'], 'awayTeam': match['awayTeam']['name']} for match in response.json()['events']}
    match_info = dict(reversed(list(match_info.items())[-num_matchs:]))
    match_info['target_team_id'] = team_id
    team_resp = cached_get(api_base + f'teams/{team_id}')
    if team_resp is None or team_resp.status_code != 200:
        return 'the api is down'
    match_info['target_team_name'] = team_resp.json().get('team').get('name')
    return match_info

def get_match_stats(api_base: str, matches_info: dict) -> DataFrame:
    all_matches_stats = []
    for match_id in matches_info.keys():
        if isinstance(match_id, int):
            if matches_info.get('target_team_name') == matches_info.get(match_id).get('awayTeam'):
                value_key = 'awayValue'
            else:
                value_key = 'homeValue'
            match_stats = {}
            home_team = matches_info[match_id]['homeTeam']
            away_team = matches_info[match_id]['awayTeam']
            match_stats['match_id'] = match_id
            match_stats['home_team'] = home_team
            match_stats['away_team'] = away_team
            lineups_resp = cached_get(api_base + f'events/{match_id}/lineups')
            if lineups_resp is not None and lineups_resp.status_code == 200:
                match_stats['team_formation'] = lineups_resp.json().get('home' if matches_info.get('target_team_name') == home_team else 'away', {}).get('formation')
            else:
                match_stats['team_formation'] = None
            try:
                stats_resp = cached_get(api_base + f'events/{match_id}/statistics')
                if stats_resp is None or stats_resp.status_code != 200:
                    print(f'Skipping match {match_id}: statistics endpoint failed')
                    continue
                response = stats_resp.json()['statistics']
            except:
                return 'api is down'
            for period in response:
                if period.get('period') == 'ALL':
                    response = period['groups']
            for group_stat in response:
                for stat_item in group_stat['statisticsItems']:
                    if stat_item.get('name') != None:
                        match_stats[stat_item.get('name')] = stat_item.get(value_key)
            all_matches_stats.append(match_stats)
    all_matches_stats = pd.DataFrame(all_matches_stats).fillna(0)
    return all_matches_stats

def get_players_stats(api_base: str, matches_info: dict) -> DataFrame:
    all_players_stats = []
    for match_id in matches_info.keys():
        if isinstance(match_id, int):
            is_home = matches_info.get('target_team_name') == matches_info.get(match_id).get('homeTeam')
            team_key = 'home' if is_home else 'away'
            try:
                response = cached_get(api_base + f'events/{match_id}/lineups')
                if response is None or response.status_code != 200:
                    print(f'Error fetching player stats for match {match_id}: HTTP {(response.status_code if response else "no response")}')
                    continue
                data = response.json()
                if not isinstance(data, dict):
                    print(f'Error fetching player stats for match {match_id}: Invalid response format')
                    continue
                if team_key in data and isinstance(data[team_key], dict) and ('players' in data[team_key]):
                    players = data[team_key]['players']
                    for player in players:
                        if not isinstance(player, dict):
                            continue
                        player_stat = {'match_id': match_id, 'player_id': player.get('player', {}).get('id') if isinstance(player.get('player'), dict) else None, 'player_name': player.get('player', {}).get('name') if isinstance(player.get('player'), dict) else None, 'position': player.get('position'), 'shirt_number': player.get('shirtNumber'), 'substitute': player.get('substitute', False), 'captain': player.get('captain', False)}
                        if 'statistics' in player and isinstance(player['statistics'], dict):
                            for stat_name, stat_value in player['statistics'].items():
                                if isinstance(stat_value, dict):
                                    if stat_name == 'ratingVersions':
                                        player_stat['rating_original'] = stat_value.get('original', 0)
                                        player_stat['rating_alternative'] = stat_value.get('alternative', 0)
                                    elif stat_name == 'statisticsType':
                                        player_stat['statistics_type'] = stat_value.get('statisticsType', 'player')
                                    else:
                                        player_stat[stat_name] = str(stat_value)
                                else:
                                    player_stat[stat_name] = stat_value
                        all_players_stats.append(player_stat)
                else:
                    print(f'No player data found for match {match_id} (team: {team_key})')
            except Exception as e:
                print(f'Error fetching player stats for match {match_id}: {str(e)}')
                continue
    if not all_players_stats:
        print('Warning: No player statistics collected')
        return pd.DataFrame()
    players_df = pd.DataFrame(all_players_stats).fillna(0)
    if 'ratingVersions' in players_df.columns:
        players_df = players_df.drop('ratingVersions', axis=1)
    return players_df

def get_player_real_position_multimatch(api_base: str, matches_info: dict) -> pd.DataFrame:
    target_team = matches_info.get('target_team_name')
    if not target_team:
        return 'Target team name not found in matches_info'
    player_agg = {}
    match_ids = [k for k in matches_info.keys() if isinstance(k, int)]
    for match_id in match_ids:
        try:
            match_meta = matches_info.get(match_id, {})
            home_team_name = match_meta.get('homeTeam')
            side = 'home' if home_team_name == target_team else 'away'
            lineups_response = cached_get(api_base + f'events/{match_id}/lineups')
            if lineups_response is None or lineups_response.status_code != 200:
                print(f'Skipping match {match_id}: Lineups not found')
                continue
            lineups = lineups_response.json()
            team_data = lineups.get(side, {})
            all_players = team_data.get('players', [])
            if not all_players:
                print(f'No players found for {side} in match {match_id}')
                continue
            for player_entry in all_players:
                player = player_entry['player']
                pid = str(player['id'])
                pname = player['name']
                prole = player_entry.get('position', 'Unknown')
                if pid not in player_agg:
                    player_agg[pid] = {'name': pname, 'base_role': prole, 'roles_played': set(), 'match_count': 0}
                player_agg[pid]['roles_played'].add(prole)
                player_agg[pid]['match_count'] += 1
        except Exception as e:
            print(f'Error processing match {match_id}: {e}')
    real_positions = []
    role_mapping = {'G': ['G'], 'D': ['CB', 'LB', 'RB', 'LWB', 'RWB'], 'M': ['CM', 'CDM', 'CAM', 'RM', 'LM'], 'F': ['ST', 'LW', 'RW', 'CF']}
    for pid, data in player_agg.items():
        if data['match_count'] > 0:
            base = data['base_role']
            mapped_roles = role_mapping.get(base, ['CM', 'CDM', 'CAM'])
            real_positions.append({'player_id': pid, 'player_name': data['name'], 'role': base, 'specific_role': mapped_roles, 'matches_played': data['match_count'], 'Right%': 0.0, 'Center%': 100.0, 'Left%': 0.0})
    return pd.DataFrame(real_positions)

def analyze_opponent_comprehensive_multimatch(api_base: str, matches_info: dict) -> dict:
    target_team = matches_info.get('target_team_name')
    if not target_team:
        return {'error': 'Target team name not found in matches_info'}
    match_ids = [k for k in matches_info.keys() if isinstance(k, int)]
    total_possession = 0
    total_pass_accuracy = 0
    matches_with_stats = 0
    formation_counts = {}
    player_stats_agg = {}
    for match_id in match_ids:
        try:
            match_meta = matches_info.get(match_id, {})
            home_team_name = match_meta.get('homeTeam')
            side = 'home' if home_team_name == target_team else 'away'
            try:
                stats_resp = cached_get(api_base + f'events/{match_id}/statistics')
                if stats_resp is not None and stats_resp.status_code == 200:
                    st = stats_resp.json().get('statistics', [])
                    idx = 0 if side == 'home' else 1
                    if len(st) > idx:
                        team_stats_groups = st[idx].get('groups', [])
                        for group in team_stats_groups:
                            for item in group.get('statisticsItems', []):
                                name = item.get('name')
                                val_str = str(item.get('homeValue' if side == 'home' else 'awayValue', '0')).replace('%', '')
                                try:
                                    val = float(val_str)
                                except:
                                    val = 0.0
                                if name == 'Ball possession':
                                    total_possession += val
                                elif name == 'Accurate passes':
                                    total_pass_accuracy += val
                        matches_with_stats += 1
            except Exception as e:
                print(f'Stats error match {match_id}: {e}')
            try:
                lineups_resp = cached_get(api_base + f'events/{match_id}/lineups')
                if lineups_resp is not None and lineups_resp.status_code == 200:
                    ln = lineups_resp.json()
                    side_data = ln.get(side, {})
                    fmt = side_data.get('formation')
                    if fmt:
                        formation_counts[fmt] = formation_counts.get(fmt, 0) + 1
                    players = side_data.get('players', [])
                    for p in players:
                        pid = str(p['player']['id'])
                        pname = p['player']['name']
                        p_role = p.get('position', 'M')
                        if pid not in player_stats_agg:
                            player_stats_agg[pid] = {'name': pname, 'goals': 0, 'assists': 0, 'shots': 0, 'rating_sum': 0, 'rating_count': 0, 'match_count': 0, 'roles': set(), 'roles_played': set()}
                        p_stats = p.get('statistics', {})
                        goals = int(p_stats.get('goals', 0))
                        assists = int(p_stats.get('goalAssist', 0))
                        shots = int(p_stats.get('totalShots', 0))
                        try:
                            rating = float(p_stats.get('rating', 0))
                        except:
                            rating = 0.0
                        player_stats_agg[pid]['goals'] += goals
                        player_stats_agg[pid]['assists'] += assists
                        player_stats_agg[pid]['shots'] += shots
                        player_stats_agg[pid]['match_count'] += 1
                        if rating > 0:
                            player_stats_agg[pid]['rating_sum'] += rating
                            player_stats_agg[pid]['rating_count'] += 1
                        player_stats_agg[pid]['roles'].add(p_role)
                        player_stats_agg[pid]['roles_played'].add(p_role)
            except Exception as e:
                print(f'Lineup error match {match_id}: {e}')
        except Exception as e:
            print(f'Match {match_id} error: {e}')
    avg_poss = total_possession / matches_with_stats if matches_with_stats > 0 else 50.0
    avg_pass = total_pass_accuracy / matches_with_stats if matches_with_stats > 0 else 0.0
    style_labels = []
    if avg_poss > 55:
        style_labels.append('POSSESSION_DOMINANT')
    elif avg_poss < 45:
        style_labels.append('COUNTER_ATTACK_FOCUSED')
    else:
        style_labels.append('BALANCED_PLAYSTYLE')
    if formation_counts:
        primary_formation = max(formation_counts, key=formation_counts.get)
    else:
        primary_formation = 'Unknown'
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
        threat_score = avg_rating * 1.5 + goals_per_game * 10.0 + assists_per_game * 8.0 + shots_per_game * 0.5
        codes = []
        if goals_per_game >= 0.4:
            codes.append('CLINICAL_FINISHER')
        if assists_per_game >= 0.3:
            codes.append('DISTRIBUTOR')
        if shots_per_game >= 2.0:
            codes.append('HIGH_VOLUME_SHOOTER')
        scored_players.append({'playerId': pid, 'name': data['name'], 'position': detailed_roles, 'threatScore': round(threat_score, 1), 'threatCodes': codes, 'stats': {'totalGoals': data['goals'], 'totalAssists': data['assists'], 'avgRating': round(avg_rating, 1)}})
    top_threats = sorted(scored_players, key=lambda x: x['threatScore'], reverse=True)[:5]
    vulns = []
    if avg_poss > 60:
        vulns.append('HIGH_DEFENSIVE_LINE')
    if avg_poss < 40:
        vulns.append('PASSIVE_MIDFIELD')
    report = {'opponentAnalysis': {'tacticalStyle': {'inferredFormation': primary_formation, 'styleLabels': style_labels, 'metrics': {'avgPossession': round(avg_poss, 1), 'avgPassAccuracy': round(avg_pass, 1)}}, 'keyThreats': top_threats, 'vulnerabilities': vulns}}
    return report

scores = {'G': {'saves': 8.0, 'savedShotsFromInsideTheBox': 10.0, 'goalsPrevented': 18.0, 'keeperSaveValue': 45.0, 'goodHighClaim': 6.0, 'totalKeeperSweeper': 3.0, 'accurateKeeperSweeper': 5.0, 'accurateLongBalls': 2.0, 'passValueNormalized': 12.0, 'errorLeadToAShot': -8.0, 'errorLeadToAGoal': -30.0, 'rating': 5.0, 'minutesPlayed': 0.02}, 'RB': {'accuratePass': 4.0, 'accurateCross': 8.0, 'totalCross': 3.0, 'duelWon': 4.0, 'wonTackle': 6.0, 'interceptionWon': 6.0, 'progressiveBallCarriesCount': 7.0, 'totalProgression': 5.0, 'keyPass': 6.0, 'passValueNormalized': 14.0, 'defensiveValueNormalized': 12.0, 'possessionLostCtrl': -4.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'LB': {'accuratePass': 4.0, 'accurateCross': 8.0, 'totalCross': 3.0, 'duelWon': 4.0, 'wonTackle': 6.0, 'interceptionWon': 6.0, 'progressiveBallCarriesCount': 7.0, 'totalProgression': 5.0, 'keyPass': 6.0, 'passValueNormalized': 14.0, 'defensiveValueNormalized': 12.0, 'possessionLostCtrl': -4.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'CB': {'totalClearance': 7.0, 'aerialWon': 8.0, 'wonTackle': 7.0, 'interceptionWon': 8.0, 'outfielderBlock': 5.0, 'duelWon': 6.0, 'defensiveValueNormalized': 18.0, 'passValueNormalized': 8.0, 'errorLeadToAShot': -8.0, 'errorLeadToAGoal': -25.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'CDM': {'ballRecovery': 8.0, 'interceptionWon': 7.0, 'wonTackle': 6.0, 'accuratePass': 5.0, 'accurateOwnHalfPasses': 4.0, 'accurateOppositionHalfPasses': 4.0, 'defensiveValueNormalized': 16.0, 'passValueNormalized': 14.0, 'possessionLostCtrl': -5.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'CM': {'accuratePass': 6.0, 'totalPass': 2.0, 'progressiveBallCarriesCount': 6.0, 'totalProgression': 6.0, 'keyPass': 6.0, 'expectedAssists': 8.0, 'passValueNormalized': 18.0, 'dribbleValueNormalized': 10.0, 'possessionLostCtrl': -4.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'CAM': {'keyPass': 10.0, 'expectedAssists': 12.0, 'bigChanceCreated': 14.0, 'goalAssist': 18.0, 'shotValueNormalized': 8.0, 'dribbleValueNormalized': 12.0, 'passValueNormalized': 16.0, 'possessionLostCtrl': -4.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'RM': {'accurateCross': 10.0, 'totalCross': 4.0, 'dribbleValueNormalized': 12.0, 'progressiveBallCarriesCount': 8.0, 'keyPass': 7.0, 'expectedAssists': 6.0, 'passValueNormalized': 14.0, 'possessionLostCtrl': -4.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'LM': {'accurateCross': 10.0, 'totalCross': 4.0, 'dribbleValueNormalized': 12.0, 'progressiveBallCarriesCount': 8.0, 'keyPass': 7.0, 'expectedAssists': 6.0, 'passValueNormalized': 14.0, 'possessionLostCtrl': -4.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'RW': {'goals': 20.0, 'expectedGoals': 12.0, 'shotValueNormalized': 14.0, 'dribbleValueNormalized': 16.0, 'keyPass': 8.0, 'expectedAssists': 8.0, 'progressiveBallCarriesCount': 10.0, 'bigChanceMissed': -8.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'LW': {'goals': 20.0, 'expectedGoals': 12.0, 'shotValueNormalized': 14.0, 'dribbleValueNormalized': 16.0, 'keyPass': 8.0, 'expectedAssists': 8.0, 'progressiveBallCarriesCount': 10.0, 'bigChanceMissed': -8.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'ST': {'goals': 30.0, 'expectedGoals': 18.0, 'expectedGoalsOnTarget': 12.0, 'shotValueNormalized': 18.0, 'bigChanceMissed': -10.0, 'aerialWon': 6.0, 'keyPass': 5.0, 'rating': 4.0, 'minutesPlayed': 0.02}}

def apply_score_formula_singlevalue(row: Series) -> Series:
    score = 0
    pos = row['specific_role']
    if str(pos).upper() in scores:
        pos = str(pos).upper()
        for feature in scores[pos]:
            try:
                score += scores[pos][feature] * row[feature]
            except Exception as e:
                return f'some features is not found : {e}'
    elif type(pos) == list or type(pos) == set:
        score = apply_score_formula_multivalue(row)
    else:
        score = np.nan
    return score

def apply_score_formula_multivalue(row: Series) -> Series:
    try:
        final_scores = []
        for pos in row['specific_role']:
            row2 = row.copy()
            row2['specific_role'] = pos
            final_scores.append(apply_score_formula_singlevalue(row2))
        return final_scores
    except Exception as e:
        return f'something wrong with the position values : {e}'

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

def normalize_score(x):
    max_val = x.max()
    if max_val == 0:
        return x * 0
    else:
        return 100 * x / max_val

def get_players_scores(api_base: str, players_stats: pd.DataFrame, team_name: str, real_positions: pd.DataFrame) -> pd.DataFrame:
    try:
        all_real_positions = real_positions.copy()
        players_stats_filtered = players_stats[players_stats['player_name'].isin(all_real_positions['player_name'].unique())].copy()
        players_stats_filtered = players_stats_filtered.drop(columns=['match_id', 'position', 'statistics_type']).groupby(by=['player_id', 'player_name']).median().reset_index()
        players_stats_filtered['player_id'] = players_stats_filtered['player_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        all_real_positions['player_id'] = all_real_positions['player_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        player_stats_with_scores = pd.merge(players_stats_filtered, all_real_positions.drop(columns='player_name'), on='player_id', how='inner')
        player_stats_with_scores = player_stats_with_scores.loc[:, ~player_stats_with_scores.columns.duplicated()]
        player_stats_with_scores['score'] = player_stats_with_scores.apply(apply_score_formula_singlevalue, axis=1)
        player_stats_with_scores = player_stats_with_scores.explode(['specific_role', 'score']).reset_index(drop=True).drop_duplicates()
        player_stats_with_scores['score'] = player_stats_with_scores.groupby('role')['score'].transform(normalize_score)
        columns = list(player_stats_with_scores.columns)
        columns.remove('specific_role')
        columns.remove('score')
        return player_stats_with_scores.groupby(columns, as_index=False).agg({'specific_role': lambda x: list(x) if len(list(x)) > 1 else x.iloc[0], 'score': lambda x: list(x) if len(list(x)) > 1 else x.iloc[0]})
    except Exception as e:
        print(f'problem with getting real position of the players {e}')
        return np.nan

def get_best_starting_lineup_from_recommendations(players_stats_scored: pd.DataFrame, recommended_formations: list, real_pos_df: pd.DataFrame=None) -> tuple:
    try:
        if 'score' not in players_stats_scored.columns:
            return ('Scores missing. Run Section 5 first.', None, None)
        df = players_stats_scored.copy()
        df['score'] = df['score'].apply(lambda x: max(x) if isinstance(x, list) else x)
        if 'player_id' in df.columns:
            df['clean_id'] = df['player_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        elif 'id' in df.columns:
            df['clean_id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        else:
            df['clean_id'] = 'N/A'
        name_col = next((c for c in ['player_name', 'name', 'playerName'] if c in df.columns), None)
        df['Player Name'] = df[name_col] if name_col else 'Unknown'
        if real_pos_df is not None and (not real_pos_df.empty):
            temp_real = real_pos_df.copy()
            temp_real['clean_id'] = temp_real['player_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            id_to_true_role = dict(zip(temp_real['clean_id'], temp_real['specific_role']))
            df['raw_true_position'] = df['clean_id'].map(id_to_true_role)
            df['Real Position'] = df['raw_true_position'].apply(lambda x: [str(i).strip() for i in x] if isinstance(x, list) else [p.strip() for p in str(x).split('or')] if 'or' in str(x).lower() else [str(x).strip()] if pd.notna(x) else ['Unknown'])
        else:
            df['Real Position'] = [['Unknown']] * len(df)
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df = df.sort_values(by='score', ascending=False).reset_index(drop=True)

        def matches_bucket(role_list, tags):
            for r in role_list:
                clean_r = r.upper()
                for tag in tags:
                    if tag in clean_r:
                        return True
            return False
        
        bucket_features = {'Goalkeeper': {'pos': 'GK', 'role': 'Shot Stopper'}, 'Left Back': {'pos': 'DL', 'role': 'Attacking Wingback'}, 'Left Wing Back': {'pos': 'WBL', 'role': 'Complete Wingback'}, 'Right Back': {'pos': 'DR', 'role': 'Attacking Fullback'}, 'Right Wing Back': {'pos': 'WBR', 'role': 'Complete Wingback'}, 'Center Backs': {'pos': 'DC', 'role': 'Ball Playing Defender'}, 'Defenders': {'pos': 'DC', 'role': 'No-Nonsense Defender'}, 'Defensive Midfielders': {'pos': 'DM', 'role': 'Defensive Anchor'}, 'Central Midfielders': {'pos': 'MC', 'role': 'Box-to-Box'}, 'Midfielders': {'pos': 'MC', 'role': 'Deep Lying Playmaker'}, 'Attacking Midfielder': {'pos': 'AM', 'role': 'Playmaker'}, 'Left Mid': {'pos': 'ML', 'role': 'Wide Midfielder'}, 'Right Mid': {'pos': 'MR', 'role': 'Wide Midfielder'}, 'Left Mid/Wing': {'pos': 'LW', 'role': 'Inverted Winger'}, 'Right Mid/Wing': {'pos': 'RW', 'role': 'Winger'}, 'Left Winger': {'pos': 'LW', 'role': 'Inside Forward'}, 'Right Winger': {'pos': 'RW', 'role': 'Inverted Winger'}, 'Left Forward': {'pos': 'AML', 'role': 'Shadow Striker'}, 'Right Forward': {'pos': 'AMR', 'role': 'Shadow Striker'}, 'Striker': {'pos': 'ST', 'role': 'Complete Forward'}, 'Strikers': {'pos': 'ST', 'role': 'Advanced Forward'}, 'Forwards': {'pos': 'ST', 'role': 'Poacher'}}

        def get_formation_blueprint(formation_str):
            bp = {'Goalkeeper': {'max': 1, 'filled': 0, 'tags': ['GK'], 'players': []}}
            f = str(formation_str).strip()
            if f.startswith('4-'):
                bp.update({'Left Back': {'max': 1, 'filled': 0, 'tags': ['LB', 'LWB'], 'players': []}, 'Right Back': {'max': 1, 'filled': 0, 'tags': ['RB', 'RWB'], 'players': []}, 'Center Backs': {'max': 2, 'filled': 0, 'tags': ['CB'], 'players': []}})
            elif f.startswith('3-'):
                bp.update({'Center Backs': {'max': 3, 'filled': 0, 'tags': ['CB', 'LB', 'RB'], 'players': []}})
            elif f.startswith('5-'):
                bp.update({'Left Wing Back': {'max': 1, 'filled': 0, 'tags': ['LWB', 'LB', 'LM'], 'players': []}, 'Right Wing Back': {'max': 1, 'filled': 0, 'tags': ['RWB', 'RB', 'RM'], 'players': []}, 'Center Backs': {'max': 3, 'filled': 0, 'tags': ['CB'], 'players': []}})
            else:
                bp['Defenders'] = {'max': 4, 'filled': 0, 'tags': ['CB', 'LB', 'RB'], 'players': []}
            if f in ['4-3-3', '4-1-2-3', '4-3-2-1']:
                bp.update({'Central Midfielders': {'max': 3, 'filled': 0, 'tags': ['CM', 'CDM', 'CAM'], 'players': []}, 'Left Winger': {'max': 1, 'filled': 0, 'tags': ['LW', 'LM'], 'players': []}, 'Right Winger': {'max': 1, 'filled': 0, 'tags': ['RW', 'RM'], 'players': []}, 'Striker': {'max': 1, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}})
            elif f in ['4-2-3-1', '4-4-1-1']:
                bp.update({'Defensive Midfielders': {'max': 2, 'filled': 0, 'tags': ['CDM', 'CM'], 'players': []}, 'Attacking Midfielder': {'max': 1, 'filled': 0, 'tags': ['CAM', 'CM'], 'players': []}, 'Left Winger': {'max': 1, 'filled': 0, 'tags': ['LW', 'LM'], 'players': []}, 'Right Winger': {'max': 1, 'filled': 0, 'tags': ['RW', 'RM'], 'players': []}, 'Striker': {'max': 1, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}})
            elif f in ['4-4-2', '4-1-4-1', '4-1-3-2', '4-5-1']:
                bp.update({'Central Midfielders': {'max': 2, 'filled': 0, 'tags': ['CM', 'CDM', 'CAM'], 'players': []}, 'Left Mid': {'max': 1, 'filled': 0, 'tags': ['LM', 'LW'], 'players': []}, 'Right Mid': {'max': 1, 'filled': 0, 'tags': ['RM', 'RW'], 'players': []}, 'Strikers': {'max': 2 if f in ['4-4-2', '4-1-3-2'] else 1, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}})
            elif f in ['3-5-2', '3-1-4-2']:
                bp.update({'Central Midfielders': {'max': 3, 'filled': 0, 'tags': ['CM', 'CDM', 'CAM'], 'players': []}, 'Left Mid/Wing': {'max': 1, 'filled': 0, 'tags': ['LM', 'LWB', 'LW'], 'players': []}, 'Right Mid/Wing': {'max': 1, 'filled': 0, 'tags': ['RM', 'RWB', 'RW'], 'players': []}, 'Strikers': {'max': 2, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}})
            else:
                bp['Midfielders'] = {'max': 3, 'filled': 0, 'tags': ['CM', 'CDM', 'CAM'], 'players': []}
                bp['Forwards'] = {'max': 3, 'filled': 0, 'tags': ['ST', 'LW', 'RW'], 'players': []}
            return bp
        
        best_total_score = -1
        best_formation_name = ''
        best_starting_xi = None
        best_bench = None
        
        for target_formation in recommended_formations:
            formation_blueprint = get_formation_blueprint(str(target_formation))
            bench = []
            current_formation_score = 0

            def build_player_dict(row, bucket_name, score_multiplier=1.0):
                final_score = row['score'] * score_multiplier
                form_val = round(final_score / 10, 1) if final_score <= 100 else 9.9
                fitness_val = min(100, int(75 + final_score / 4))
                pos_str = bucket_features.get(bucket_name, {}).get('pos', 'UNK')
                role_str = bucket_features.get(bucket_name, {}).get('role', 'Player')
                reasons = []
                if final_score > 90:
                    reasons.append('HIGH_RECENT_FORM')
                if 'Winger' in bucket_name or 'Back' in bucket_name:
                    reasons.append('PACE_MISMATCH_VS_OPPONENT')
                if 'Center Back' in bucket_name and final_score > 85:
                    reasons.append('AERIAL_DOMINANCE')
                if not reasons:
                    reasons.append('TACTICAL_FIT')
                return {'playerId': int(row['clean_id']) if str(row['clean_id']).isdigit() else row['clean_id'], 'name': row['Player Name'], 'position': pos_str, 'role': role_str, 'suitabilityScore': round(final_score, 1), 'selectionFactors': {'form': min(9.9, form_val), 'fitness': fitness_val, 'matchupAdvantage': min(98, fitness_val - 2)}, 'reasonCodes': reasons}
            
            for _, row in df.iterrows():
                if row['Real Position'] == ['Unknown']:
                    continue
                placed = False
                role_list = row['Real Position']
                for bucket_name, bucket_info in formation_blueprint.items():
                    if bucket_info['filled'] < bucket_info['max'] and matches_bucket(role_list, bucket_info['tags']):
                        bucket_info['players'].append(build_player_dict(row, bucket_name))
                        bucket_info['filled'] += 1
                        current_formation_score += row['score']
                        placed = True
                        break
                if not placed:
                    stashed_positions = [str(r).upper().strip() for r in role_list]
                    short_positions = [r if len(r) <= 3 else 'UNK' for r in stashed_positions]
                    bench.append({'playerId': int(row['clean_id']) if str(row['clean_id']).isdigit() else row['clean_id'], 'name': row['Player Name'], 'positions': short_positions, 'suitabilityScore': round(row['score'], 1), '_raw_row': row})
            
            total_filled = sum((b['filled'] for b in formation_blueprint.values()))
            while total_filled < 11 and bench:
                best_sub = bench.pop(0)
                for bucket_name, bucket_info in formation_blueprint.items():
                    if bucket_info['filled'] < bucket_info['max']:
                        raw_row = best_sub.pop('_raw_row')
                        oop_player = build_player_dict(raw_row, bucket_name, score_multiplier=0.8)
                        oop_player['reasonCodes'].append('OUT_OF_POSITION_COVERAGE')
                        bucket_info['players'].append(oop_player)
                        bucket_info['filled'] += 1
                        total_filled += 1
                        current_formation_score += raw_row['score'] * 0.8
                        break
            
            if current_formation_score > best_total_score and total_filled >= 11:
                best_total_score = current_formation_score
                best_formation_name = target_formation
                lineup_list = []
                for b in formation_blueprint.values():
                    lineup_list.extend(b['players'])
                best_starting_xi = lineup_list
                final_bench = []
                for i in range(min(7, len(bench))):
                    sub = bench[i].copy()
                    if '_raw_row' in sub:
                        del sub['_raw_row']
                    final_bench.append(sub)
                best_bench = final_bench
        
        team_selection_json = {'suggestedFormation': best_formation_name, 'startingXI': best_starting_xi, 'substitutes': best_bench}
        return team_selection_json
    except Exception as e:
        print(f'Algorithm Error: {e}')
        return None

def formation_suggestions(api_base: str, opponent_id: int) -> list:
    try:
        matches_info = get_team_lnm(api_base, opponent_id, 5)
    except Exception as e:
        return f'something went wrong with {e}'
    formations = []
    for match_id in matches_info.keys():
        if isinstance(match_id, int):
            if matches_info.get('target_team_name') == matches_info.get(match_id).get('awayTeam'):
                value_key = 'away'
            else:
                value_key = 'home'
            lineups_resp = cached_get(api_base + f'events/{match_id}/lineups')
            if lineups_resp is not None and lineups_resp.status_code == 200:
                formations.append(lineups_resp.json().get(value_key, {}).get('formation'))
    opponent_most_used_formation = statistics.multimode(formations)[0]
    counter_formations = {'4-4-2': ['3-5-2', '4-3-3', '4-2-3-1'], '3-5-2': ['4-3-3', '4-4-2', '4-2-3-1'], '4-3-3': ['4-2-3-1', '5-4-1', '4-5-1', '3-5-2'], '4-2-3-1': ['4-3-3', '3-5-2'], '5-3-2': ['4-3-3', '4-2-3-1', '3-4-3'], '4-5-1': ['4-3-3', '3-5-2', '4-4-2'], '3-4-3': ['4-3-3', '4-2-3-1'], '4-3-2-1': ['3-5-2', '4-4-2'], '4-1-4-1': ['4-3-3', '4-2-3-1'], '4-2-2-2': ['4-2-3-1']}
    suggested_formation = counter_formations[opponent_most_used_formation]
    return suggested_formation

def get_season_tournament_ids(api_base: str, players_ids: list) -> dict:
    ids = dict()
    if players_ids == np.nan:
        return 'players_ids list is missing'
    for player_id in players_ids:
        resp = cached_get(f'{api_base}players/{player_id}/statistics/seasons')
        if resp is None or resp.status_code != 200:
            print(f'Skipping player {player_id}: seasons endpoint failed')
            continue
        response = resp.json()
        tournament_id = response.get('uniqueTournamentSeasons')[0].get('uniqueTournament').get('id')
        season_id = response.get('uniqueTournamentSeasons')[0].get('seasons')[1].get('id') if len(response.get('uniqueTournamentSeasons')[0].get('seasons')) > 1 else response.get('uniqueTournamentSeasons')[0].get('seasons')[0].get('id')
        ids[player_id] = {'tournament_id': tournament_id, 'season_id': season_id}
    return ids

def get_tacticale(api_base: str, team_selection_output: json, opponent_id: int) -> json:
    try:
        api_key = os.environ.get('GEMINI_API_KEY_PRE_MATCH_1')
        client = genai.Client(api_key=api_key)
        recommended_players = pd.json_normalize(team_selection_output['startingXI'])
        recommended_players['suitabilityScore'] = pd.to_numeric(recommended_players['suitabilityScore'], errors='coerce').fillna(0.0)
        suggested_formations = team_selection_output['suggestedFormation']
        ids = get_season_tournament_ids(api_base, list(recommended_players['playerId']))
        general_statistics = []
        for player_id in ids.keys():
            stats_resp = cached_get(f'{api_base}players/{player_id}/unique-tournament/{ids[player_id]['tournament_id']}/season/{ids[player_id]['season_id']}/statistics/overall')
            if stats_resp is None or stats_resp.status_code != 200:
                print(f'Skipping player {player_id}: statistics endpoint failed')
                continue
            response = stats_resp.json().get('statistics')
            response['id'] = player_id
            general_statistics.append(response)
        general_statistics = pd.DataFrame(general_statistics).fillna(0).drop(columns=['type', 'statisticsType'])
        general_statistics = pd.concat([recommended_players[['playerId', 'name', 'role', 'position', 'suitabilityScore']], general_statistics], axis=1).drop(columns='playerId')
        opponent_lastm_info = get_team_lnm(api_base, opponent_id, 1)
        opp_lineups_resp = cached_get(api_base + f'events/{list(opponent_lastm_info.keys())[0]}/lineups')
        if opp_lineups_resp is not None and opp_lineups_resp.status_code == 200:
            opponent_formation = opp_lineups_resp.json().get('home' if opponent_lastm_info.get('target_team_name') == opponent_lastm_info[list(opponent_lastm_info.keys())[0]]['homeTeam'] else 'away', {}).get('formation')
        else:
            opponent_formation = 'Unknown'
        prompt = f"""\nYou are a professional football manager and tactical analyst. You are given a dataset containing the statistics and positions of the players selected for the next match. \nPlayer data: {general_statistics.to_json()} \nThe suggested formation for our team to play with: {suggested_formations} \nThe opponent played their last match using this formation: {opponent_formation} \nYour task:\n# 1. Analyze the player statistics and positions.\n# 2. Suggest a tactical strategy that best fits the squad and counters the opponent's formation. \n# Return ONLY a valid JSON object in the following format:\n\n# {{ "suggestedFormation": "formation",\n#   "strategyCode": "strategy_name" }}\n\n# Example:\n# {{ "suggestedFormation": "4-3-3",\n#   "strategyCode": "COUNTER_ATTACK_DIRECT" }}\n# """
        response = client.models.generate_content(model='gemini-2.5-flash-lite', config={'system_instruction': 'Only output the requierd output'}, contents=prompt)
        result = response.text
        return result
    except Exception as e:
        return ' { "suggestedFormation": null,\n        "strategyCode": null } '

def get_training_player_stats(api_base: str, matches_info: dict) -> pd.DataFrame:
    target_team = matches_info.get('target_team_name')
    if not target_team:
        return pd.DataFrame()
    match_ids = [k for k in matches_info.keys() if isinstance(k, int)]
    player_stats_agg = {}
    for match_id in match_ids:
        try:
            match_meta = matches_info.get(match_id, {})
            home_team_name = match_meta.get('homeTeam')
            side = 'home' if home_team_name == target_team else 'away'
            try:
                lineups_resp = cached_get(api_base + f'events/{match_id}/lineups')
                if lineups_resp is not None and lineups_resp.status_code == 200:
                    ln = lineups_resp.json()
                    side_data = ln.get(side, {})
                    players = side_data.get('players', [])
                    for p in players:
                        pid = str(p['player']['id'])
                        pname = p['player']['name']
                        p_role = p.get('position', 'M')
                        if pid not in player_stats_agg:
                            player_stats_agg[pid] = {'name': pname, 'goals': 0, 'assists': 0, 'shots': 0, 'rating_sum': 0, 'rating_count': 0, 'match_count': 0, 'roles': set(), 'roles_played': set()}
                        p_stats = p.get('statistics', {})
                        goals = int(p_stats.get('goals', 0))
                        assists = int(p_stats.get('goalAssist', 0))
                        shots = int(p_stats.get('totalShots', 0))
                        try:
                            rating = float(p_stats.get('rating', 0))
                        except:
                            rating = 0.0
                        player_stats_agg[pid]['goals'] += goals
                        player_stats_agg[pid]['assists'] += assists
                        player_stats_agg[pid]['shots'] += shots
                        player_stats_agg[pid]['match_count'] += 1
                        if rating > 0:
                            player_stats_agg[pid]['rating_sum'] += rating
                            player_stats_agg[pid]['rating_count'] += 1
                        player_stats_agg[pid]['roles'].add(p_role)
                        player_stats_agg[pid]['roles_played'].add(p_role)
            except Exception as e:
                print(f'Lineup error match {match_id}: {e}')
        except Exception as e:
            print(f'Match {match_id} error: {e}')
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
        threat_score = avg_rating * 1.5 + goals_per_game * 10.0 + assists_per_game * 8.0 + shots_per_game * 0.5
        scored_players.append({'Player ID': pid, 'Player Name': data['name'], 'Positions': list(detailed_roles), 'Matches Played': data['match_count'], 'Mean Goals': round(goals_per_game, 2), 'Mean Assists': round(assists_per_game, 2), 'Mean Shots': round(shots_per_game, 2), 'Average Rating': round(avg_rating, 1), 'Threat Score': round(threat_score, 1)})
    return pd.DataFrame(scored_players)

def get_training_recommendations(api_base: str, average_stats_df: pd.DataFrame) -> str:
    try:
        if average_stats_df.empty:
            return '{"error": "No player statistics provided"}'
        api_key = os.environ.get('GEMINI_API_KEY_PRE_MATCH_1')
        client = genai.Client(api_key=api_key)
        stats_data = average_stats_df.to_dict(orient='records')
        prompt = f'\n        Analyze the following player statistics and identify weaknesses.\n        Generate a strictly valid JSON training plan. DO NOT use markdown code blocks (e.g., no ```json).\n        The JSON MUST strictly follow this exact structure and key names:\n        {{\n          "trainingPlan": {{\n            "teamDrills": [\n              {{\n                "focusCode": "str (e.g., PRESS_RESISTANCE)",\n                "priority": "str (HIGH/MEDIUM/LOW)",\n                "linkedOpponentFeature": "str (e.g., HIGH_PRESS_INTENSITY)",\n                "targetedPositions": ["D", "M"] \n              }}\n            ],\n            "individualDrills": [\n              {{\n                "playerId": int,\n                "playerName": "str",\n                "drillCode": "str (e.g., 1V1_DEFENDING_WIDE)"\n              }}\n            ]\n          }}\n        }}\n\n        Player Statistics:\n        {json.dumps(stats_data)}\n        '
        response = client.models.generate_content(model='gemini-2.5-flash', config={'system_instruction': 'You are a professional football tactician. Output only strictly valid, unformatted JSON that perfectly matches the requested schema. No explanations.'}, contents=prompt)
        return response.text.replace('```json', '').replace('```', '').strip()
    except Exception as e:
        return f'{{"error": "LLM API failed: {e}"}}'
