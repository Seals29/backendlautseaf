from flask import Flask, request, jsonify
import joblib
import numpy as np
import xgboost as xgb
app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello World!'


@app.route('/xgboost/watersalinity', methods=['POST'])
def xgboost_model():
    try:
        # Get JSON data from the request
        data = request.get_json()
        print(data)

        feature_keys = ['chlor_a_micrograms_per_l', 'water_temp', 'o2_conc_milimeters_per_liter', 'o2_sat', 
                        'potential_density', 'phaeop_micrograms_per_l', 'phosphate_micromoles_per_l', 'silicate_micromoles_per_l', 
                        'nitrate_micromoles_per_l', 'nitrite_micromoles_per_l', 'year_part','month_part_cos', 'month_part_sin']
        
        if not all(key in data for key in feature_keys):
            return jsonify({'error': 'Missing one or more required features'}), 400

        features = np.array([data[key] for key in feature_keys]).reshape(1, -1)
        model = joblib.load('xgboost_model_water_salinity.pkl')
        prediction = model.predict(features)
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)



@app.route('/xgboost/chlorophyl', methods=['POST'])
def xgboost_model():
    try:
        # Get JSON data from the request
        data = request.get_json()
        print(data)

        feature_keys = ['water_salinity', 'water_temp', 'o2_conc_milimeters_per_liter', 'o2_sat', 
                        'potential_density', 'phaeop_micrograms_per_l', 'phosphate_micromoles_per_l', 'silicate_micromoles_per_l', 
                        'nitrate_micromoles_per_l', 'nitrite_micromoles_per_l', 'year_part','month_part_cos', 'month_part_sin']
        
        if not all(key in data for key in feature_keys):
            return jsonify({'error': 'Missing one or more required features'}), 400

        features = np.array([data[key] for key in feature_keys]).reshape(1, -1)
        model = joblib.load('./xgboost_model_chlor.pkl')
        prediction = model.predict(features)
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)



@app.route('/xgboost/phosphate', methods=['POST'])
def xgboost_model():
    try:
        # Get JSON data from the request
        data = request.get_json()
        print(data)

        feature_keys = ['chlor_a_micrograms_per_l', 'water_temp', 'o2_conc_milimeters_per_liter', 'o2_sat', 
                        'potential_density', 'phaeop_micrograms_per_l', 'water_salinity', 'silicate_micromoles_per_l', 
                        'nitrate_micromoles_per_l', 'nitrite_micromoles_per_l', 'year_part','month_part_cos', 'month_part_sin']
        
        if not all(key in data for key in feature_keys):
            return jsonify({'error': 'Missing one or more required features'}), 400

        features = np.array([data[key] for key in feature_keys]).reshape(1, -1)
        model = joblib.load('./xgboost_model_phosphate.pkl')
        prediction = model.predict(features)
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
