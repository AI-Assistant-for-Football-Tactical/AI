from flask import Flask, request, jsonify
from in_match import * 



app = Flask(__name__)

@app.route('/')
def index():
    return "In Match Analysis API is running!"

@app.route('/in_match', methods=['POST'])
def in_match():
    
    # request's data handling part
    data = request.get_json()
    players_data = None 
    events_data = None
    heatmaps = None
    shotmaps = None
    features = None
    
    # first step
    all_deltas(players_data , events_data , heatmaps , shotmaps , features)
    
    # second step 
    
    # third step 
    
    # fourth step 
    
    # fifth step 
    
    # sixth step 
    
    # last step 
    try :
        return jsonify({"success": True}) , 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Running on port 5002 to avoid conflicts with pre_match and post_match if running simultaneously
    app.run(debug=True, port=5002)
