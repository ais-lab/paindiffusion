from flask import Flask, send_from_directory, request, jsonify
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    filename='experiment.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Ensure data directory exists
DATA_DIR = "experiment_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@app.route('/')
def serve_experiment():
    return send_from_directory('.', 'qualitative_experiment.html')

@app.route('/comparison_video/<path:filename>')
def serve_video(filename):
    return send_from_directory('comparison_video', filename)

@app.route('/logo/<path:filename>')
def serve_logo(filename):
    return send_from_directory('logo', filename)

@app.route('/save_data', methods=['POST'])
def save_data():
    try:
        data = request.get_json()
        
        # Generate filename based on timestamp and session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = data.get('session_id', 'unknown_session')
        filename = f"{timestamp}_{session_id}.json"
        
        # Save data to file
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logging.info(f"Saved data to {file_path}")
        return jsonify({"status": "success", "message": "Data saved successfully"})
        
    except Exception as e:
        logging.error(f"Error saving data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)