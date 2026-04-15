import numpy as np, json, requests
import pandas as pd
from datetime import datetime
from google import genai

def get_team_lnm(api_base: str, team_id: int, num_matchs: int) -> dict:
    '''this function takes the base of the api without the endpoint, the team id and the number of last matchs wanted'''
    try:
        response = requests.get(api_base + f'teams/{team_id}/events/last/0')
    except:
        return 'the api is down'

    match_info = {
        match['id']: {
            'homeTeam': match['homeTeam']['name'],
            'awayTeam': match['awayTeam']['name']
        }
        for match in response.json().get('events', [])
    }
    match_info = dict(reversed(list(match_info.items())[-num_matchs:]))
    match_info['target_team_id'] = team_id

    try:
        match_info['target_team_name'] = requests.get(api_base + f'teams/{team_id}').json().get('team', {}).get('name')
    except:
        match_info['target_team_name'] = 'Unknown'
        
    return match_info

def get_fatigue(api_base: str, team_id: int) -> dict:
    players_details = [] 
    last_match_players_ids = []
    last_match_players_pos = []
    matches_info = get_team_lnm(api_base, team_id, 5) 
 
    for match_id in matches_info:
        if isinstance(match_id, int):
            is_home = matches_info.get('target_team_name') == matches_info.get(match_id).get('homeTeam')
            team_key = 'home' if is_home else 'away'
            
            try:
                response = requests.get(api_base + f'events/{match_id}/lineups')
                if response.status_code != 200: continue
                
                data = response.json()
                if team_key in data and isinstance(data[team_key], dict) and 'players' in data[team_key]:
                    players = data[team_key]['players']
                    
                    for player in players:
                        if not isinstance(player, dict): continue

                        if match_id == list(matches_info.keys())[0]:
                            last_match_players_ids.append(player.get('player').get('id'))
                            last_match_players_pos.append(player.get('position'))

                        players_details.append(
                            {
                                'player_id': player.get('player').get('id'), 
                                'name' :  player.get('player').get('name'),
                                'position' :  player.get('position') ,
                                'minutes_played' : player.get('statistics').get('minutesPlayed', 0) ,
                            }
                        )
            except Exception as e:
                pass
    
    players_df = pd.DataFrame(players_details).fillna(0)
    if players_df.empty:
        return {"players_analysis": []}
        
    players_df = players_df.groupby(by=['player_id', 'name', 'position']).agg({'minutes_played': 'sum'}).reset_index()
    min_s = players_df['minutes_played'].min()
    max_s = players_df['minutes_played'].max()
    players_df['fatigue_index'] = round(100 * (players_df['minutes_played'] - min_s) / (max_s - min_s) if max_s != min_s else 0)
    
    # ----------------------------------------------------
    # Applied condition for >180 minutes -> High Risk
    # ----------------------------------------------------
    players_df['injury_risk_level'] = players_df.apply(
        lambda row: 'High' if row['minutes_played'] >= 180 else ('Low' if row['fatigue_index'] < 60 else ('Moderate' if row['fatigue_index'] < 80 else 'High')), 
        axis=1
    )
    
    
    req = pd.DataFrame({'player_id': last_match_players_ids, 'position': last_match_players_pos})
    filtered = players_df.merge(req, on=['player_id', 'position'], how='inner')

    players_analysis = [
        {
            "player_id": str(row["player_id"]),
            "name": row["name"],
            "position": row["position"],
            "minutes_played": int(row["minutes_played"]),
            "fatigue_and_risk": {
                "fatigue_index": float(row["fatigue_index"]),
                "injury_risk_level": row["injury_risk_level"]
            }
        }
        for _, row in filtered.iterrows()
    ]

    return {"players_analysis": players_analysis}

def get_training_recommendations(api_base: str, team_id: int) -> str:
    try:
        matches_info = get_team_lnm(api_base, team_id, 1)
        match_id = None
        for k in matches_info.keys():
            if isinstance(k, int): 
                match_id = k
                break
                
        if not match_id:
            return '{"error": "No recent matches found"}'
            
        is_home = matches_info.get('target_team_name') == matches_info.get(match_id).get('homeTeam')
        team_key = 'home' if is_home else 'away'
        
        stats_data = []
        try:
            lineup_resp = requests.get(api_base + f'events/{match_id}/lineups').json()
            if team_key in lineup_resp and 'players' in lineup_resp[team_key]:
                for p in lineup_resp[team_key]['players']:
                    stats = p.get('statistics', {})
                    mins = stats.get('minutesPlayed', 0)
                    if mins > 0:
                        clean_stats = {k:v for k,v in stats.items() if v} 
                        stats_data.append({
                            "playerId": p.get('player', {}).get('id'),
                            "playerName": p.get('player', {}).get('name'),
                            "stats": clean_stats
                        })
        except Exception as e:
            pass

        if not stats_data:
            return '{"error": "No player statistics found for the last match"}'
        
        prompt = f"""
        Analyze the following player statistics from their SINGLE MOST RECENT MATCH and identify weaknesses.
        Generate a strictly valid JSON training plan. DO NOT use markdown code blocks (e.g., no ```json).
        The JSON MUST strictly follow this exact structure and key names:
        {{
          "trainingPlan": {{
            "teamDrills": [
              {{
                "focusCode": "str (e.g., PRESS_RESISTANCE)",
                "priority": "str (HIGH/MEDIUM/LOW)",
                "linkedOpponentFeature": "str (e.g., HIGH_PRESS_INTENSITY)",
                "targetedPositions": ["D", "M"] 
              }}
            ],
            "individualDrills": [
              {{
                "playerId": int,
                "playerName": "str",
                "drillCode": "str (e.g., 1V1_DEFENDING_WIDE)"
              }}
            ]
          }}
        }}

        Player Statistics:
        {json.dumps(stats_data)}
        """

        client = genai.Client(api_key="AIzaSyDETBLONMaxx0qBo86bcxbT0Xl35C97mKk")
        
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            config={
                "system_instruction": "You are a professional football tactician. Output only strictly valid, unformatted JSON that perfectly matches the requested schema. No explanations."
            },
            contents=prompt
        )
        return response.text.replace("```json", "").replace("```", "").strip()

    except Exception as e:
        return f'{{"error": "LLM API failed: {e}"}}'


# Combining Logic for the End API Result Match
def generate_post_match_report(api_base: str, team_id: int, event_id: int) -> dict:
    fatigue_data = get_fatigue(api_base, team_id)
    training_json_str = get_training_recommendations(api_base, team_id)
    
    try:
        training_data = json.loads(training_json_str)
        if "trainingPlan" in training_data:
            training_plan = training_data["trainingPlan"]
        else:
            training_plan = training_data
    except Exception as e:
        training_plan = {"error": "Could not parse training JSON"}

    current_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    match_context = {}
    try:
        event_resp = requests.get(api_base + f'events/{event_id}').json().get('event', {})
        is_home = event_resp.get('homeTeam', {}).get('id') == team_id
        opponent = event_resp.get('awayTeam') if is_home else event_resp.get('homeTeam')
        match_context['opponent_id'] = f"team_{opponent.get('id', 'N/A')}"
        home_score = event_resp.get('homeScore', {}).get('current', 0)
        away_score = event_resp.get('awayScore', {}).get('current', 0)
        if home_score == away_score:
            res = "Draw"
        elif (is_home and home_score > away_score) or (not is_home and away_score > home_score):
            res = "Win"
        else:
            res = "Loss"
        match_context['match_result'] = res
        
        lineup_resp = requests.get(api_base + f'events/{event_id}/lineups').json()
        tk = 'home' if is_home else 'away'
        match_context['team_formation'] = lineup_resp.get(tk, {}).get('formation', 'Unknown')
    except:
        match_context = {"opponent_id": "Unknown", "match_result": "Unknown", "team_formation": "Unknown"}

    final_output = {
      "event_id": str(event_id),
      "team_id": str(team_id),
      "analysis_timestamp": current_time,
      "match_context": match_context,
      "players_analysis": fatigue_data.get("players_analysis", []),
      "trainingPlan": training_plan
    }
    
    return final_output
