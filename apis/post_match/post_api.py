from flask import Flask, request, jsonify
from post_match import generate_post_match_report

app = Flask(__name__)
api_base = r'https://football-backend-app.victoriouswater-69fff737.swedencentral.azurecontainerapps.io/'

@app.route('/')
def index():
    return "Post Match Analysis API is running!"

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


if __name__ == '__main__':
    # Running on port 5001 to avoid conflicts with pre_match if running simultaneously
    app.run(debug=True, port=5001)
