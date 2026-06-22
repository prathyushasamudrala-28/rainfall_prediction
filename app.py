from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained rainfall prediction model
# Ensure your model file is saved as 'rainfall_model.pkl'
model = pickle.load(open('best_model (1).pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            pressure = float(request.form['pressure'])
            wind_speed = float(request.form['wind_speed'])
            cloud_cover = float(request.form['cloud_cover'])
            precipitation = float(request.form['precipitation'])

            features = np.array([[temperature, humidity, pressure, wind_speed, cloud_cover, precipitation]])
            prediction = model.predict(features)

            # 🔧 Convert regression output to binary result using threshold
            if prediction[0] >= 0.5:
                result = "🌧 Rain Predicted"
            else:
                result = "☀ No Rain Predicted"

            return render_template('predict.html', prediction=result)

        except Exception as e:
            return render_template('predict.html', prediction=f"Error: {e}")

    return render_template('predict.html')


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
