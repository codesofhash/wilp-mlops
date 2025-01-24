import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Set the tracking URI (assuming the MLflow server is running locally)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Load the model (replace "Gradient Boosting Regressor" and version 1 with your model)
model_name = "Gradient Boosting Regressor"
model_version = 1
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)


# Define a route for prediction using POST method (same as before)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'features' not in data:
            return jsonify({"error": "No features provided"}), 400

        features = data['features']
        prediction = model.predict([features])

        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Define a new route for testing via GET request
@app.route('/predict_get', methods=['GET'])
def predict_get():
    try:
        # Get the features as query parameters
        features = request.args.getlist('features', type=float)

        # Ensure features are provided
        if not features:
            return jsonify({"error": "No features provided"}), 400

        # Make prediction
        prediction = model.predict([features])

        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
