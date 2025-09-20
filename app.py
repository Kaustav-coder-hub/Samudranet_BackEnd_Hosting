
from flask import Flask, jsonify, request
from flask_cors import CORS
from image_classification import image_classification_bp
from otolith_classification import otolith_classification_bp
from edna_classification_clean import edna_classification_bp
import os

app = Flask(__name__)
# Configure CORS to allow requests from any origin
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True, 
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     expose_headers=["Content-Length", "X-Requested-With"])

# Debug route access
@app.before_request
def before_request():
    print(f"Request to: {request.path} [{request.method}]")

app.register_blueprint(image_classification_bp, url_prefix="/classify")
app.register_blueprint(otolith_classification_bp, url_prefix="/otolith")
app.register_blueprint(edna_classification_bp, url_prefix="/edna")

@app.route('/')
def home():
    return 'SamudraNet Flask backend is running.'

# Placeholder endpoints for temperature, pH, and salinity graphs for a given species
@app.route('/species/<species>/temperature')
def get_temperature_graph(species):
    # TODO: Implement actual graph generation
    return jsonify({'message': f'Temperature graph for {species}'}), 200

@app.route('/species/<species>/ph')
def get_ph_graph(species):
    # TODO: Implement actual graph generation
    return jsonify({'message': f'pH graph for {species}'}), 200

@app.route('/species/<species>/salinity')
def get_salinity_graph(species):
    # TODO: Implement actual graph generation
    return jsonify({'message': f'Salinity graph for {species}'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)