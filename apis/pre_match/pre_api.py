from flask import Flask, request, jsonify
from pre_match import *
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
    num_matches = data.get('num_matches')
    opponent_id = data.get('opponent_id')

    if team_id is None or num_matches is None or opponent_id is None:
        return jsonify({
            "error": "team_id, num_matches, and opponent_id are required"
        }), 400

    # body of the function to process the pre-match data and generate result
    result = None

    return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True)