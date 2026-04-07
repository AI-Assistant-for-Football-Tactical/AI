from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "server is working!"

@app.route('/post_match', methods=['POST'])
def post_match():
    
    # Validate input
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid or missing JSON"}), 400

    team_id = data.get("team_id")
    num_matches = data.get("num_matches")

   
    if team_id is None or num_matches is None:
        return jsonify({"error": "team_id and num_matches are required"}), 400

    if not isinstance(num_matches, int) or num_matches <= 0:
        return jsonify({"error": "num_matches must be a positive integer"}), 400

    # body of the function to process the post-match data and generate result
    result = None

    return jsonify(result), 200



if __name__ == '__main__':
    app.run()
