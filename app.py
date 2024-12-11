# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from mysklearn.myclassifiers import MyDecisionTreeClassifier

app = Flask(__name__)

# Load the trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
    # Verify the model has a tree
    if hasattr(model, 'tree') and model.tree is not None:
        print("Model tree verified")
    else:
        raise Exception("Model tree is None")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run train_model.py first to create a trained model")
    raise SystemExit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        if request.is_json:
            data = request.get_json()
            features = data['features']
        else:
            features = [request.form[f'feature_{i}'] for i in range(5)]
        
        print(f"Received features: {features}")
        
        # Verify features
        if not all(isinstance(f, str) for f in features):
            return jsonify({'error': 'All features must be strings'}), 400
            
        if not all(f in ['very_low', 'low', 'medium', 'high', 'very_high'] for f in features):
            return jsonify({'error': 'Invalid feature values'}), 400
        
        # Make prediction
        prediction = model.predict([features])
        print(f"Prediction result: {prediction}")
        
        if request.is_json:
            return jsonify({'prediction': prediction[0]})
        else:
            return render_template('index.html', prediction=prediction[0])
            
    except Exception as e:
        import traceback
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)