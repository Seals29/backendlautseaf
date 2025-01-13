from flask import Flask, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)
model = joblib.load('linear_model_chlor_a_micrograms_per_l.pkl')
@app.route('/')
def index():
    return 'Hello World!'


@app.route('/linear', methods=['POST'])
def linear_regression():
    try:
        # Get JSON data from the request
        data = request.get_json()
        print(data)
        
        # Extract feature values based on expected keys
        feature_keys = ['water_salinity', 'water_temp', 'oxygen_conc', 'oxygen_sat', 
                        'potential_density', 'phaeop', 'phosphate', 'silicate', 
                        'nitrate', 'nitrite', 'year_part','month_part']
        
        # Ensure all required features are present in the input
        if not all(key in data for key in feature_keys):
            return jsonify({'error': 'Missing one or more required features'}), 400

        # Create feature array
        features = np.array([data[key] for key in feature_keys]).reshape(1, -1)
        
        # Predict using the model
        prediction = model.predict(features)
        
        # Return prediction as JSON response
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)