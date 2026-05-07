import os
import time
import json
import statistics
import re
import requests
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from pandas import DataFrame, Series
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from google import genai
from warnings import filterwarnings
from dotenv import load_dotenv

filterwarnings('ignore')
load_dotenv()

api_base = 'https://football-backend-app.victoriouswater-69fff737.swedencentral.azurecontainerapps.io/'
_api_cache: dict = {}
_DEFAULT_TIMEOUT = 15

# ============================================================
# SESSION + RETRIES
# ============================================================
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=25,
    pool_maxsize=25
)
session.mount("http://", adapter)
session.mount("https://", adapter)

def clear_api_cache():
    global _api_cache
    _api_cache.clear()

def cached_get(url: str, timeout: int = _DEFAULT_TIMEOUT):
    if url in _api_cache:
        return _api_cache[url]
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 200:
            try:
                resp.json()
            except ValueError:
                if '/heatmap' not in url:
                    print(f'[cached_get] ERROR: Invalid JSON received from {url}')
                return None
            _api_cache[url] = resp
        return resp
    except Exception as e:
        if '/heatmap' not in url:
            print(f'[cached_get] ERROR {type(e).__name__}: {url}')
        return None

def safe_get_json(url: str, timeout: int = 30) -> Optional[Dict]:
    resp = cached_get(url, timeout=timeout)
    if resp and resp.status_code == 200:
        try:
            return resp.json()
        except:
            return None
    return None

# ─── Data Retrieval Functions ───────────────────────────────────────────────

def get_team_lnm(api_base: str, team_id: int, num_matchs: int) -> dict:
    data = safe_get_json(api_base + f'teams/{team_id}/events/last/0')
    if not data:
        return 'the api is down'
    match_info = {match['id']: {'homeTeam': match['homeTeam']['name'], 'awayTeam': match['awayTeam']['name']} for match in data['events']}
    match_info = dict(reversed(list(match_info.items())[-num_matchs:]))
    match_info['target_team_id'] = team_id
    team_data = safe_get_json(api_base + f'teams/{team_id}')
    if not team_data:
        return 'the api is down'
    match_info['target_team_name'] = team_data.get('team', {}).get('name')
    return match_info

def _process_match_stats(api_base: str, match_id: int, matches_info: dict) -> Optional[dict]:
    target_team = matches_info.get('target_team_name')
    home_team = matches_info[match_id]['homeTeam']
    away_team = matches_info[match_id]['awayTeam']
    value_key = 'awayValue' if target_team == away_team else 'homeValue'
    
    match_stats = {'match_id': match_id, 'home_team': home_team, 'away_team': away_team}
    
    lineup_json = safe_get_json(api_base + f'events/{match_id}/lineups')
    if lineup_json:
        side = 'home' if target_team == home_team else 'away'
        match_stats['team_formation'] = lineup_json.get(side, {}).get('formation')
    
    stats_json = safe_get_json(api_base + f'events/{match_id}/statistics')
    if stats_json:
        stats_resp = stats_json.get('statistics', [])
        for period in stats_resp:
            if period.get('period') == 'ALL':
                for group in period.get('groups', []):
                    for item in group.get('statisticsItems', []):
                        if item.get('name'):
                            match_stats[item['name']] = item.get(value_key)
                break
    return match_stats

def get_match_stats(api_base: str, matches_info: dict) -> DataFrame:
    match_ids = [m for m in matches_info.keys() if isinstance(m, int)]
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_process_match_stats, api_base, mid, matches_info) for mid in match_ids]
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)
    return pd.DataFrame(results).fillna(0)

def _process_player_stats_match(api_base: str, match_id: int, matches_info: dict) -> List[dict]:
    target_team = matches_info.get('target_team_name')
    home_team = matches_info[match_id]['homeTeam']
    is_home = (target_team == home_team)
    team_key = 'home' if is_home else 'away'
    
    players_data = []
    lineup_json = safe_get_json(api_base + f'events/{match_id}/lineups')
    if lineup_json and team_key in lineup_json:
        players = lineup_json[team_key].get('players', [])
        for p in players:
            player_meta = p.get('player', {})
            player_stat = {
                'match_id': match_id,
                'player_id': player_meta.get('id'),
                'player_name': player_meta.get('name'),
                'position': p.get('position'),
                'shirt_number': p.get('shirtNumber'),
                'substitute': p.get('substitute', False),
                'captain': p.get('captain', False)
            }
            stats = p.get('statistics', {})
            if isinstance(stats, dict):
                for k, v in stats.items():
                    if k == 'ratingVersions':
                        player_stat['rating_original'] = v.get('original', 0)
                        player_stat['rating_alternative'] = v.get('alternative', 0)
                    elif k == 'statisticsType':
                        player_stat['statistics_type'] = v.get('statisticsType', 'player')
                    elif isinstance(v, dict):
                        player_stat[k] = str(v)
                    else:
                        player_stat[k] = v
            players_data.append(player_stat)
    return players_data

def get_players_stats(api_base: str, matches_info: dict) -> DataFrame:
    match_ids = [m for m in matches_info.keys() if isinstance(m, int)]
    all_stats = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_process_player_stats_match, api_base, mid, matches_info) for mid in match_ids]
        for f in as_completed(futures):
            all_stats.extend(f.result())
    return pd.DataFrame(all_stats).fillna(0)

def get_player_real_position_multimatch(api_base: str, matches_info: dict) -> pd.DataFrame:
    target_team = matches_info.get('target_team_name')
    if not target_team: return pd.DataFrame()
    
    player_agg = {}
    match_ids = [m for m in matches_info.keys() if isinstance(m, int)]
    
    def process_match_pos(mid):
        meta = matches_info.get(mid, {})
        side = 'home' if meta.get('homeTeam') == target_team else 'away'
        ln = safe_get_json(api_base + f'events/{mid}/lineups')
        if not ln: return []
        players = ln.get(side, {}).get('players', [])
        return [(str(p['player']['id']), p['player']['name'], p.get('position', 'Unknown')) for p in players]

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(process_match_pos, match_ids)
        for match_players in results:
            for pid, name, pos in match_players:
                if pid not in player_agg:
                    player_agg[pid] = {'name': name, 'base_role': pos, 'roles_played': set(), 'match_count': 0}
                player_agg[pid]['roles_played'].add(pos)
                player_agg[pid]['match_count'] += 1
                
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
    if not target_team: return {'error': 'Target team name not found'}
    
    match_ids = [m for m in matches_info.keys() if isinstance(m, int)]
    
    # State
    total_possession = 0.0
    total_pass_accuracy = 0.0
    matches_with_stats = 0
    formation_counts = {}
    player_agg = {}
    lock = Lock()

    def process_match_comp(mid):
        nonlocal total_possession, total_pass_accuracy, matches_with_stats
        meta = matches_info.get(mid, {})
        side = 'home' if meta.get('homeTeam') == target_team else 'away'
        
        # Stats
        st_json = safe_get_json(api_base + f'events/{mid}/statistics')
        if st_json:
            st = st_json.get('statistics', [])
            idx = 0 if side == 'home' else 1
            if len(st) > idx:
                val_key = 'homeValue' if side == 'home' else 'awayValue'
                with lock:
                    matches_with_stats += 1
                    for group in st[idx].get('groups', []):
                        for item in group.get('statisticsItems', []):
                            name = item.get('name')
                            val = float(str(item.get(val_key, '0')).replace('%',''))
                            if name == 'Ball possession': total_possession += val
                            elif name == 'Accurate passes': total_pass_accuracy += val
        
        # Lineup
        ln_json = safe_get_json(api_base + f'events/{mid}/lineups')
        if ln_json:
            side_data = ln_json.get(side, {})
            fmt = side_data.get('formation')
            with lock:
                if fmt: formation_counts[fmt] = formation_counts.get(fmt, 0) + 1
            for p in side_data.get('players', []):
                pid = str(p['player']['id'])
                p_stats = p.get('statistics', {})
                with lock:
                    if pid not in player_agg:
                        player_agg[pid] = {'name': p['player']['name'], 'goals': 0, 'assists': 0, 'shots': 0, 'rating_sum': 0, 'rating_count': 0, 'match_count': 0, 'roles': set()}
                    agg = player_agg[pid]
                    agg['goals'] += int(p_stats.get('goals', 0))
                    agg['assists'] += int(p_stats.get('goalAssist', 0))
                    agg['shots'] += int(p_stats.get('totalShots', 0))
                    agg['match_count'] += 1
                    agg['roles'].add(p.get('position', 'M'))
                    try:
                        r = float(p_stats.get('rating', 0))
                        if r > 0:
                            agg['rating_sum'] += r
                            agg['rating_count'] += 1
                    except: pass

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(process_match_comp, match_ids))

    avg_poss = total_possession / matches_with_stats if matches_with_stats > 0 else 50.0
    avg_pass = total_pass_accuracy / matches_with_stats if matches_with_stats > 0 else 0.0
    
    style = ['POSSESSION_DOMINANT'] if avg_poss > 55 else ['COUNTER_ATTACK_FOCUSED'] if avg_poss < 45 else ['BALANCED_PLAYSTYLE']
    primary_fmt = max(formation_counts, key=formation_counts.get) if formation_counts else 'Unknown'
    
    scored_players = []
    for pid, data in player_agg.items():
        if data['match_count'] == 0: continue
        avg_rating = data['rating_sum'] / data['rating_count'] if data['rating_count'] > 0 else 0
        gpg = data['goals'] / data['match_count']
        apg = data['assists'] / data['match_count']
        spg = data['shots'] / data['match_count']
        threat = avg_rating * 1.5 + gpg * 10.0 + apg * 8.0 + spg * 0.5
        codes = []
        if gpg >= 0.4: codes.append('CLINICAL_FINISHER')
        if apg >= 0.3: codes.append('DISTRIBUTOR')
        if spg >= 2.0: codes.append('HIGH_VOLUME_SHOOTER')
        scored_players.append({
            'playerId': pid, 'name': data['name'], 'position': list(data['roles']),
            'threatScore': round(threat, 1), 'threatCodes': codes,
            'stats': {'totalGoals': data['goals'], 'totalAssists': data['assists'], 'avgRating': round(avg_rating, 1)}
        })
    
    top_threats = sorted(scored_players, key=lambda x: x['threatScore'], reverse=True)
    primary_formation = primary_fmt

    vulns = []

    if avg_poss > 60:
        vulns.append('HIGH_DEFENSIVE_LINE')

    elif avg_poss < 45:
        vulns.append('PASSIVE_MIDFIELD')

    if avg_pass < 75:
        vulns.append('INACCURATE_IN_BUILDUP')

    elif avg_pass > 90:
        vulns.append('SHORT_PASS_DEPENDENT')

    total_threat = sum(
        p['threatScore']
        for p in scored_players
    ) or 1

    top2_threat = sum(
        p['threatScore']
        for p in top_threats[:2]
    )

    if top2_threat / total_threat > 0.5:
        vulns.append('OVER_RELIANT_ON_KEY_PLAYERS')

    if primary_formation in ('4-4-2', '4-2-4'):
        vulns.append('VULNERABLE_THROUGH_MIDDLE')

    elif primary_formation in ('3-5-2', '3-4-3'):
        vulns.append('EXPOSED_WIDE_CHANNELS')

    if avg_poss > 55:
        vulns.append('EXPOSED_TO_COUNTER_ATTACKS')
    return {
        'opponentAnalysis': {
            'tacticalStyle': {'inferredFormation': primary_fmt, 'styleLabels': style, 'metrics': {'avgPossession': round(avg_poss, 1), 'avgPassAccuracy': round(avg_pass, 1)}},
            'keyThreats': sorted(scored_players, key=lambda x: x['threatScore'], reverse=True)[:5],
            'vulnerabilities': vulns
        }
    }

# ─── Scoring and Recommendation Logic ──────────────────────────────────────

scores = {'G': {'saves': 8.0, 'savedShotsFromInsideTheBox': 10.0, 'goalsPrevented': 18.0, 'keeperSaveValue': 45.0, 'goodHighClaim': 6.0, 'totalKeeperSweeper': 3.0, 'accurateKeeperSweeper': 5.0, 'accurateLongBalls': 2.0, 'passValueNormalized': 12.0, 'errorLeadToAShot': -8.0, 'errorLeadToAGoal': -30.0, 'rating': 5.0, 'minutesPlayed': 0.02}, 'RB': {'accuratePass': 4.0, 'accurateCross': 8.0, 'totalCross': 3.0, 'duelWon': 4.0, 'wonTackle': 6.0, 'interceptionWon': 6.0, 'progressiveBallCarriesCount': 7.0, 'totalProgression': 5.0, 'keyPass': 6.0, 'passValueNormalized': 14.0, 'defensiveValueNormalized': 12.0, 'possessionLostCtrl': -4.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'LB': {'accuratePass': 4.0, 'accurateCross': 8.0, 'totalCross': 3.0, 'duelWon': 4.0, 'wonTackle': 6.0, 'interceptionWon': 6.0, 'progressiveBallCarriesCount': 7.0, 'totalProgression': 5.0, 'keyPass': 6.0, 'passValueNormalized': 14.0, 'defensiveValueNormalized': 12.0, 'possessionLostCtrl': -4.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'CB': {'totalClearance': 7.0, 'aerialWon': 8.0, 'wonTackle': 7.0, 'interceptionWon': 8.0, 'outfielderBlock': 5.0, 'duelWon': 6.0, 'defensiveValueNormalized': 18.0, 'passValueNormalized': 8.0, 'errorLeadToAShot': -8.0, 'errorLeadToAGoal': -25.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'CDM': {'ballRecovery': 8.0, 'interceptionWon': 7.0, 'wonTackle': 6.0, 'accuratePass': 5.0, 'accurateOwnHalfPasses': 4.0, 'accurateOppositionHalfPasses': 4.0, 'defensiveValueNormalized': 16.0, 'passValueNormalized': 14.0, 'possessionLostCtrl': -5.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'CM': {'accuratePass': 6.0, 'totalPass': 2.0, 'progressiveBallCarriesCount': 6.0, 'totalProgression': 6.0, 'keyPass': 6.0, 'expectedAssists': 8.0, 'passValueNormalized': 18.0, 'dribbleValueNormalized': 10.0, 'possessionLostCtrl': -4.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'CAM': {'keyPass': 10.0, 'expectedAssists': 12.0, 'bigChanceCreated': 14.0, 'goalAssist': 18.0, 'shotValueNormalized': 8.0, 'dribbleValueNormalized': 12.0, 'passValueNormalized': 16.0, 'possessionLostCtrl': -4.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'RM': {'accurateCross': 10.0, 'totalCross': 4.0, 'dribbleValueNormalized': 12.0, 'progressiveBallCarriesCount': 8.0, 'keyPass': 7.0, 'expectedAssists': 6.0, 'passValueNormalized': 14.0, 'possessionLostCtrl': -4.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'LM': {'accurateCross': 10.0, 'totalCross': 4.0, 'dribbleValueNormalized': 12.0, 'progressiveBallCarriesCount': 8.0, 'keyPass': 7.0, 'expectedAssists': 6.0, 'passValueNormalized': 14.0, 'possessionLostCtrl': -4.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'RW': {'goals': 20.0, 'expectedGoals': 12.0, 'shotValueNormalized': 14.0, 'dribbleValueNormalized': 16.0, 'keyPass': 8.0, 'expectedAssists': 8.0, 'progressiveBallCarriesCount': 10.0, 'bigChanceMissed': -8.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'LW': {'goals': 20.0, 'expectedGoals': 12.0, 'shotValueNormalized': 14.0, 'dribbleValueNormalized': 16.0, 'keyPass': 8.0, 'expectedAssists': 8.0, 'progressiveBallCarriesCount': 10.0, 'bigChanceMissed': -8.0, 'rating': 4.0, 'minutesPlayed': 0.02}, 'ST': {'goals': 30.0, 'expectedGoals': 18.0, 'expectedGoalsOnTarget': 12.0, 'shotValueNormalized': 18.0, 'bigChanceMissed': -10.0, 'aerialWon': 6.0, 'keyPass': 5.0, 'rating': 4.0, 'minutesPlayed': 0.02}}

def apply_score_formula_singlevalue(row: Series) -> float:
    pos = str(row['specific_role']).upper()
    if pos not in scores: return np.nan
    score = 0
    for feat, weight in scores[pos].items():
        if feat in row: score += weight * row[feat]
    return score

def get_players_scores(api_base: str, players_stats: pd.DataFrame, team_name: str, real_positions: pd.DataFrame) -> pd.DataFrame:
    try:
        ps = players_stats.copy()
        rp = real_positions.copy()
        ps['player_id'] = ps['player_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        rp['player_id'] = rp['player_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        # Aggregate stats
        ps_agg = ps.drop(columns=['match_id', 'position', 'statistics_type']).groupby(['player_id', 'player_name']).median().reset_index()
        
        # Merge
        merged = pd.merge(ps_agg, rp.drop(columns='player_name'), on='player_id', how='inner')
        merged = merged.loc[:, ~merged.columns.duplicated()]
        
        # Score
        results = []
        for _, row in merged.iterrows():
            roles = row['specific_role']
            if not isinstance(roles, list): roles = [roles]
            for r in roles:
                r_row = row.copy()
                r_row['specific_role'] = r
                r_row['score'] = apply_score_formula_singlevalue(r_row)
                results.append(r_row)
        
        scored_df = pd.DataFrame(results)
        scored_df = scored_df.dropna(subset=['score'])
        
        # Normalize
        def normalize_score(x):
            mx = x.max()
            return (100 * x / mx) if mx > 0 else x * 0
            
        scored_df['score'] = scored_df.groupby('role')['score'].transform(normalize_score)
        
        # Group back
        cols = [c for c in scored_df.columns if c not in ['specific_role', 'score']]
        return scored_df.groupby(cols, as_index=False).agg({
            'specific_role': lambda x: list(x) if len(x) > 1 else x.iloc[0],
            'score': lambda x: list(x) if len(x) > 1 else x.iloc[0]
        })
    except Exception as e:
        print(f"Scoring error: {e}")
        return pd.DataFrame()

# def get_best_starting_lineup_from_recommendations(players_stats_scored: pd.DataFrame, recommended_formations: list, real_pos_df: pd.DataFrame=None) -> dict:
#     try:
#         if players_stats_scored.empty: return {}
#         df = players_stats_scored.copy()
#         df['score'] = df['score'].apply(lambda x: max(x) if isinstance(x, list) else x)
#         df['clean_id'] = df['player_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
#         if real_pos_df is not None:
#             id_map = dict(zip(real_pos_df['player_id'].astype(str).str.replace(r'\.0$', '', regex=True), real_pos_df['specific_role']))
#             df['Real Position'] = df['clean_id'].map(id_map).apply(lambda x: x if isinstance(x, list) else [x])
#         else:
#             df['Real Position'] = [['Unknown']] * len(df)
            
#         df = df.sort_values('score', ascending=False).reset_index(drop=True)
        
#         bucket_features = {'Goalkeeper': {'pos': 'GK', 'role': 'Shot Stopper'}, 'Left Back': {'pos': 'DL', 'role': 'Attacking Wingback'}, 'Left Wing Back': {'pos': 'WBL', 'role': 'Complete Wingback'}, 'Right Back': {'pos': 'DR', 'role': 'Attacking Fullback'}, 'Right Wing Back': {'pos': 'WBR', 'role': 'Complete Wingback'}, 'Center Backs': {'pos': 'DC', 'role': 'Ball Playing Defender'}, 'Defenders': {'pos': 'DC', 'role': 'No-Nonsense Defender'}, 'Defensive Midfielders': {'pos': 'DM', 'role': 'Defensive Anchor'}, 'Central Midfielders': {'pos': 'MC', 'role': 'Box-to-Box'}, 'Midfielders': {'pos': 'MC', 'role': 'Deep Lying Playmaker'}, 'Attacking Midfielder': {'pos': 'AM', 'role': 'Playmaker'}, 'Left Mid': {'pos': 'ML', 'role': 'Wide Midfielder'}, 'Right Mid': {'pos': 'MR', 'role': 'Wide Midfielder'}, 'Left Mid/Wing': {'pos': 'LW', 'role': 'Inverted Winger'}, 'Right Mid/Wing': {'pos': 'RW', 'role': 'Winger'}, 'Left Winger': {'pos': 'LW', 'role': 'Inside Forward'}, 'Right Winger': {'pos': 'RW', 'role': 'Inverted Winger'}, 'Left Forward': {'pos': 'AML', 'role': 'Shadow Striker'}, 'Right Forward': {'pos': 'AMR', 'role': 'Shadow Striker'}, 'Striker': {'pos': 'ST', 'role': 'Complete Forward'}, 'Strikers': {'pos': 'ST', 'role': 'Advanced Forward'}, 'Forwards': {'pos': 'ST', 'role': 'Poacher'}}

#         def get_blueprint(f):
#             bp = {'Goalkeeper': {'max': 1, 'tags': ['GK'], 'players': [], 'filled': 0}}
#             f = str(f).strip()
#             if f.startswith('4-'):
#                 bp.update({'Left Back': {'max': 1, 'tags': ['LB', 'LWB'], 'players': [], 'filled': 0}, 'Right Back': {'max': 1, 'tags': ['RB', 'RWB'], 'players': [], 'filled': 0}, 'Center Backs': {'max': 2, 'tags': ['CB'], 'players': [], 'filled': 0}})
#             elif f.startswith('3-'):
#                 bp.update({'Center Backs': {'max': 3, 'tags': ['CB', 'LB', 'RB'], 'players': [], 'filled': 0}})
#             elif f.startswith('5-'):
#                 bp.update({'Left Wing Back': {'max': 1, 'tags': ['LWB', 'LB', 'LM'], 'players': [], 'filled': 0}, 'Right Wing Back': {'max': 1, 'tags': ['RWB', 'RB', 'RM'], 'players': [], 'filled': 0}, 'Center Backs': {'max': 3, 'tags': ['CB'], 'players': [], 'filled': 0}})
            
#             if f in ['4-3-3', '4-1-2-3', '4-3-2-1']:
#                 bp.update({'Central Midfielders': {'max': 3, 'tags': ['CM', 'CDM', 'CAM'], 'players': [], 'filled': 0}, 'Left Winger': {'max': 1, 'tags': ['LW', 'LM'], 'players': [], 'filled': 0}, 'Right Winger': {'max': 1, 'tags': ['RW', 'RM'], 'players': [], 'filled': 0}, 'Striker': {'max': 1, 'tags': ['ST', 'CF', 'FW'], 'players': [], 'filled': 0}})
#             elif f in ['4-2-3-1', '4-4-1-1']:
#                 bp.update({'Defensive Midfielders': {'max': 2, 'tags': ['CDM', 'CM'], 'players': [], 'filled': 0}, 'Attacking Midfielder': {'max': 1, 'tags': ['CAM', 'CM'], 'players': [], 'filled': 0}, 'Left Winger': {'max': 1, 'tags': ['LW', 'LM'], 'players': [], 'filled': 0}, 'Right Winger': {'max': 1, 'tags': ['RW', 'RM'], 'players': [], 'filled': 0}, 'Striker': {'max': 1, 'tags': ['ST', 'CF', 'FW'], 'players': [], 'filled': 0}})
#             elif f in ['4-4-2', '4-5-1']:
#                 bp.update({'Central Midfielders': {'max': 2, 'tags': ['CM', 'CDM'], 'players': [], 'filled': 0}, 'Left Mid': {'max': 1, 'tags': ['LM', 'LW'], 'players': [], 'filled': 0}, 'Right Mid': {'max': 1, 'tags': ['RM', 'RW'], 'players': [], 'filled': 0}, 'Strikers': {'max': 2 if f == '4-4-2' else 1, 'tags': ['ST', 'CF'], 'players': [], 'filled': 0}})
#             return bp

#         best_score = -1
#         final_selection = {}

#         for form in recommended_formations:
#             bp = get_blueprint(form)
#             xi = []
#             bench = []
#             curr_score = 0
            
#             for _, row in df.iterrows():
#                 placed = False
#                 for b_name, b_info in bp.items():
#                     if b_info['filled'] < b_info['max']:
#                         if any(t in str(row['Real Position']).upper() for t in b_info['tags']):
#                             p_dict = {'playerId': row['clean_id'], 'name': row['player_name'], 'position': bucket_features.get(b_name, {}).get('pos', 'UNK'), 'role': bucket_features.get(b_name, {}).get('role', 'Player'), 'suitabilityScore': round(row['score'], 1), 'selectionFactors': {'form': round(row['score']/10, 1), 'fitness': 90}, 'reasonCodes': ['TACTICAL_FIT']}
#                             xi.append(p_dict)
#                             b_info['filled'] += 1
#                             curr_score += row['score']
#                             placed = True
#                             break
#                 if not placed:
#                     bench.append({'playerId': row['clean_id'], 'name': row['player_name'], 'positions': row['Real Position'], 'suitabilityScore': round(row['score'], 1)})
            
#             if curr_score > best_score and len(xi) >= 11:
#                 best_score = curr_score
#                 final_selection = {'suggestedFormation': form, 'startingXI': xi[:11], 'substitutes': bench[:7]}
        
#         return final_selection
#     except Exception as e:
#         print(f"Selection error: {e}")
#         return {}

def get_best_starting_lineup_from_recommendations(players_stats_scored: pd.DataFrame, recommended_formations: list, real_pos_df: pd.DataFrame = None) -> tuple:
    try:
        if 'score' not in players_stats_scored.columns:
            return "Scores missing. Run Section 5 first.", None, None
            
        df = players_stats_scored.copy()
        
        # 1. PREPARE THE DATA & FIX SCRAMBLED POSITIONS
        df['score'] = df['score'].apply(lambda x: max(x) if isinstance(x, list) else x)
        
        # Safe ID mapping
        if 'player_id' in df.columns:
            df['clean_id'] = df['player_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        elif 'id' in df.columns:
            df['clean_id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        else:
            df['clean_id'] = 'N/A'
            
        name_col = next((c for c in ['player_name', 'name', 'playerName'] if c in df.columns), None)
        df['Player Name'] = df[name_col] if name_col else 'Unknown'

        # STRICT ID TO POSITION MAPPING
        if real_pos_df is not None and not real_pos_df.empty:
            temp_real = real_pos_df.copy()
            temp_real['clean_id'] = temp_real['player_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            id_to_true_role = dict(zip(temp_real['clean_id'], temp_real['specific_role']))
            df['raw_true_position'] = df['clean_id'].map(id_to_true_role)
            df['Real Position'] = df['raw_true_position'].apply(
                lambda x: [str(i).strip() for i in x] if isinstance(x, list) 
                else [p.strip() for p in str(x).split('or')] if 'or' in str(x).lower()
                else [str(x).strip()] if pd.notna(x) else ['Unknown']
            )
        else:
            df['Real Position'] = [['Unknown']] * len(df)
            
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
        
        def matches_bucket(role_list, tags):
            for r in role_list:
                clean_r = r.upper()
                for tag in tags:
                    if tag in clean_r: return True
            return False

        # Dictionary to map 'Bucket Names' to 'JSON Style Position' and 'Tactical Role'
        bucket_features = {
            'Goalkeeper': {'pos': 'GK', 'role': 'Shot Stopper'},
            'Left Back': {'pos': 'DL', 'role': 'Attacking Wingback'},
            'Left Wing Back': {'pos': 'WBL', 'role': 'Complete Wingback'},
            'Right Back': {'pos': 'DR', 'role': 'Attacking Fullback'},
            'Right Wing Back': {'pos': 'WBR', 'role': 'Complete Wingback'},
            'Center Backs': {'pos': 'DC', 'role': 'Ball Playing Defender'},
            'Defenders': {'pos': 'DC', 'role': 'No-Nonsense Defender'},
            'Defensive Midfielders': {'pos': 'DM', 'role': 'Defensive Anchor'},
            'Central Midfielders': {'pos': 'MC', 'role': 'Box-to-Box'},
            'Midfielders': {'pos': 'MC', 'role': 'Deep Lying Playmaker'},
            'Attacking Midfielder': {'pos': 'AM', 'role': 'Playmaker'},
            'Left Mid': {'pos': 'ML', 'role': 'Wide Midfielder'},
            'Right Mid': {'pos': 'MR', 'role': 'Wide Midfielder'},
            'Left Mid/Wing': {'pos': 'LW', 'role': 'Inverted Winger'},
            'Right Mid/Wing': {'pos': 'RW', 'role': 'Winger'},
            'Left Winger': {'pos': 'LW', 'role': 'Inside Forward'},
            'Right Winger': {'pos': 'RW', 'role': 'Inverted Winger'},
            'Left Forward': {'pos': 'AML', 'role': 'Shadow Striker'},
            'Right Forward': {'pos': 'AMR', 'role': 'Shadow Striker'},
            'Striker': {'pos': 'ST', 'role': 'Complete Forward'},
            'Strikers': {'pos': 'ST', 'role': 'Advanced Forward'},
            'Forwards': {'pos': 'ST', 'role': 'Poacher'},
        }

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

            if f in ["4-3-3", "4-1-2-3", "4-3-2-1"]:
                bp.update({'Central Midfielders': {'max': 3, 'filled': 0, 'tags': ['CM', 'CDM', 'CAM'], 'players': []}, 'Left Winger': {'max': 1, 'filled': 0, 'tags': ['LW', 'LM'], 'players': []}, 'Right Winger': {'max': 1, 'filled': 0, 'tags': ['RW', 'RM'], 'players': []}, 'Striker': {'max': 1, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}})
            elif f in ["4-2-3-1", "4-4-1-1"]:
                bp.update({'Defensive Midfielders': {'max': 2, 'filled': 0, 'tags': ['CDM', 'CM'], 'players': []}, 'Attacking Midfielder': {'max': 1, 'filled': 0, 'tags': ['CAM', 'CM'], 'players': []}, 'Left Winger': {'max': 1, 'filled': 0, 'tags': ['LW', 'LM'], 'players': []}, 'Right Winger': {'max': 1, 'filled': 0, 'tags': ['RW', 'RM'], 'players': []}, 'Striker': {'max': 1, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}})
            elif f in ["4-4-2", "4-1-4-1", "4-1-3-2", "4-5-1"]:
                bp.update({'Central Midfielders': {'max': 2, 'filled': 0, 'tags': ['CM', 'CDM', 'CAM'], 'players': []}, 'Left Mid': {'max': 1, 'filled': 0, 'tags': ['LM', 'LW'], 'players': []}, 'Right Mid': {'max': 1, 'filled': 0, 'tags': ['RM', 'RW'], 'players': []}, 'Strikers': {'max': 2 if f in ["4-4-2", "4-1-3-2"] else 1, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}})
            elif f in ["3-5-2", "3-1-4-2"]:
                bp.update({'Central Midfielders': {'max': 3, 'filled': 0, 'tags': ['CM', 'CDM', 'CAM'], 'players': []}, 'Left Mid/Wing': {'max': 1, 'filled': 0, 'tags': ['LM', 'LWB', 'LW'], 'players': []}, 'Right Mid/Wing': {'max': 1, 'filled': 0, 'tags': ['RM', 'RWB', 'RW'], 'players': []}, 'Strikers': {'max': 2, 'filled': 0, 'tags': ['ST', 'CF', 'FW'], 'players': []}})
            else:
                bp['Midfielders'] = {'max': 3, 'filled': 0, 'tags': ['CM', 'CDM', 'CAM'], 'players': []}
                bp['Forwards'] = {'max': 3, 'filled': 0, 'tags': ['ST', 'LW', 'RW'], 'players': []}

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
            
            # Formatter for building the advanced JSON-ready dictionary
            def build_player_dict(row, bucket_name, score_multiplier=1.0):
                final_score = row['score'] * score_multiplier
                form_val = round((final_score / 10), 1) if final_score <= 100 else 9.9
                fitness_val = min(100, int(75 + (final_score / 4)))
                
                # Fetch mapped JSON roles
                pos_str = bucket_features.get(bucket_name, {}).get('pos', 'UNK')
                role_str = bucket_features.get(bucket_name, {}).get('role', 'Player')
                
                # Generate dynamic reason codes
                reasons = []
                if final_score > 90: reasons.append("HIGH_RECENT_FORM")
                if "Winger" in bucket_name or "Back" in bucket_name: reasons.append("PACE_MISMATCH_VS_OPPONENT")
                if "Center Back" in bucket_name and final_score > 85: reasons.append("AERIAL_DOMINANCE")
                if not reasons: reasons.append("TACTICAL_FIT")

                return {
                    'playerId': int(row['clean_id']) if str(row['clean_id']).isdigit() else row['clean_id'],
                    'name': row['Player Name'],
                    'position': pos_str,
                    'role': role_str,
                    'suitabilityScore': round(final_score, 1),
                    'selectionFactors': {
                        'form': min(9.9, form_val),
                        'fitness': fitness_val,
                        'matchupAdvantage': min(98, fitness_val - 2)
                    },
                    'reasonCodes': reasons
                }
            
            # Fill the XI for this specific formation
            for _, row in df.iterrows():
                if row['Real Position'] == ['Unknown']: continue
                
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
                    # Stash actual list of short roles for substitute parsing
                    stashed_positions = [str(r).upper().strip() for r in role_list]
                    short_positions = [r if len(r) <= 3 else 'UNK' for r in stashed_positions]
                    bench.append({
                        'playerId': int(row['clean_id']) if str(row['clean_id']).isdigit() else row['clean_id'],
                        'name': row['Player Name'],
                        'positions': short_positions,  # Bench expects a list "positions": ["ST", "LW"]
                        'suitabilityScore': round(row['score'], 1),
                        # Store raw row for OOP fallback
                        '_raw_row': row
                    })
                        
            # Force Fill missing slots from bench
            total_filled = sum(b['filled'] for b in formation_blueprint.values())
            while total_filled < 11 and bench:
                best_sub = bench.pop(0)
                for bucket_name, bucket_info in formation_blueprint.items():
                    if bucket_info['filled'] < bucket_info['max']:
                        raw_row = best_sub.pop('_raw_row')
                        oop_player = build_player_dict(raw_row, bucket_name, score_multiplier=0.8) # 20% penalty
                        oop_player['reasonCodes'].append("OUT_OF_POSITION_COVERAGE")
                        bucket_info['players'].append(oop_player)
                        
                        bucket_info['filled'] += 1
                        total_filled += 1
                        current_formation_score += (raw_row['score'] * 0.8)
                        break
                        
            if current_formation_score > best_total_score and total_filled >= 11:
                best_total_score = current_formation_score
                best_formation_name = target_formation
                
                lineup_list = []
                for b in formation_blueprint.values(): lineup_list.extend(b['players'])
                best_starting_xi = lineup_list # Now it's a list of dictionaries matching JSON!
                
                # Clean up bench format
                final_bench = []
                for i in range(min(7, len(bench))):
                    sub = bench[i].copy()
                    if '_raw_row' in sub: del sub['_raw_row']
                    final_bench.append(sub)
                
                best_bench = final_bench

        # Generate the final TeamSelection dictionary structure
        team_selection_json = {
            "suggestedFormation": best_formation_name,
            "startingXI": best_starting_xi,
            "substitutes": best_bench
        }

        return team_selection_json
        
    except Exception as e:
        print(f"Algorithm Error: {e}")
        return None

def formation_suggestions(api_base: str, opponent_id: int) -> list:
    info = get_team_lnm(api_base, opponent_id, 5)
    if isinstance(info, str): return []
    fmts = []
    for mid in [k for k in info.keys() if isinstance(k, int)]:
        ln = safe_get_json(api_base + f'events/{mid}/lineups')
        if ln:
            side = 'home' if info[mid]['homeTeam'] == info.get('target_team_name') else 'away'
            f = ln.get(side, {}).get('formation')
            if f: fmts.append(f)
    if not fmts: return ['4-3-3']
    mode = statistics.multimode(fmts)[0]
    counters = {'4-4-2': ['3-5-2', '4-3-3'], '3-5-2': ['4-3-3', '4-2-3-1'], '4-3-3': ['4-2-3-1', '5-4-1'], '4-2-3-1': ['4-3-3', '3-5-2']}
    return counters.get(mode, ['4-3-3', '4-2-3-1'])

# ─── Training and Tactical ───────────────────────────────────────────────────

def get_training_player_stats(api_base: str, matches_info: dict) -> pd.DataFrame:
    # Use the same aggregation logic as analyze_opponent
    res = analyze_opponent_comprehensive_multimatch(api_base, matches_info)
    threats = res.get('opponentAnalysis', {}).get('keyThreats', [])
    df_data = []
    for t in threats:
        df_data.append({
            'Player ID': t['playerId'], 'Player Name': t['name'], 'Positions': t['position'],
            'Matches Played': 5, 'Mean Goals': t['stats']['totalGoals']/5, 'Mean Assists': t['stats']['totalAssists']/5,
            'Average Rating': t['stats']['avgRating'], 'Threat Score': t['threatScore']
        })
    return pd.DataFrame(df_data)

def get_training_recommendations(api_base: str, df: pd.DataFrame) -> str:
    try:
        api_key = os.environ.get('GEMINI_API_KEY_PRE_MATCH_1')
        client = genai.Client(api_key=api_key)
        prompt = f'\n        Analyze the following player statistics and identify weaknesses.\n        Generate a strictly valid JSON training plan. DO NOT use markdown code blocks (e.g., no ```json).\n        The JSON MUST strictly follow this exact structure and key names:\n        {{\n          "trainingPlan": {{\n            "teamDrills": [\n              {{\n                "focusCode": "str (e.g., PRESS_RESISTANCE)",\n                "priority": "str (HIGH/MEDIUM/LOW)",\n                "linkedOpponentFeature": "str (e.g., HIGH_PRESS_INTENSITY)",\n                "targetedPositions": ["D", "M"] \n              }}\n            ],\n            "individualDrills": [\n              {{\n                "playerId": int,\n                "playerName": "str",\n                "drillCode": "str (e.g., 1V1_DEFENDING_WIDE)"\n              }}\n            ]\n          }}\n        }}\n\n        Player Statistics:\n        {df.to_json()}\n        '
        resp = client.models.generate_content(model='gemini-2.5-flash', config={'system_instruction': 'Output strictly valid JSON with keys: trainingPlan (teamDrills, individualDrills).'}, contents=prompt)
        return resp.text.strip()
    except: return '{"trainingPlan": {"teamDrills": [], "individualDrills": []}}'

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
            stats_resp = cached_get(f"{api_base}players/{player_id}/unique-tournament/{ids[player_id]['tournament_id']}/season/{ids[player_id]['season_id']}/statistics/overall")
            if stats_resp is None or stats_resp.status_code != 200:
                print(f'Skipping player {player_id}: statistics endpoint failed')
                continue
            response = stats_resp.json().get('statistics')
            response['id'] = player_id
            general_statistics.append(response)
        general_statistics = pd.DataFrame(general_statistics).fillna(0)
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
        print(e)
        return ' { "suggestedFormation": null,\n        "strategyCode": null } '


def get_season_tournament_ids(api_base: str, pids: list) -> dict:
    return {} # Implementation not critical for core flow if using stats directly
