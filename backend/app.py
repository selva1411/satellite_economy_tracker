from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback
from model import predict_gdp_for_years, COUNTRY_CODES

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        country = request.form.get('country', 'India')
        years_raw = request.form.get('years', '2025,2026,2027')
        target_years = [int(y.strip()) for y in years_raw.split(',')]

        # Save uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, 'satellite.jpg')
        image_file.save(image_path)

        # Run prediction
        result = predict_gdp_for_years(country, image_path, target_years)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/countries', methods=['GET'])
def get_countries():
    return jsonify(list(COUNTRY_CODES.keys()))


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)