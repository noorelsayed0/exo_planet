from flask import Flask,  request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("bestbinary_exoplanet_model.h5")
preprocessor = joblib.load("preprocessor.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        if not file:
            return jsonify({"error": "Please upload a file"}), 400

        # نفترض أن الملف CSV فيه عمود واحد (flux)
        data = pd.read_csv(file)
        X = preprocessor.transform(data)

        


        # ✅ prediction
        prediction = model.predict(X)
        # الموديل بيرجع احتمال الكلاس 1 بس (Exoplanet)
        p1 = float(prediction[0][0])     # Exoplanet
        p0 = 1.0 - p1                    # Not Exoplanet

        # نجهزهم في ليستة زي softmax
        probs = [p0, p1]

        # نختار الكلاس اللي عنده أعلى احتمال
        result = int(np.argmax(probs))  

        return jsonify({
            "predictions": [result],

            "probabilities": probs
            
              
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
