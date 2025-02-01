from flask import Flask, request, render_template_string
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = "random_forest_model.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Disease-to-specialist mapping
disease_specialist_map = {
    "Fungal infection": "Dermatologist",
    "Allergy": "Allergist/Immunologist",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Hepatologist",
    "Drug Reaction": "Dermatologist/Allergist",
    "Peptic ulcer disease": "Gastroenterologist",
    "AIDS": "Infectious Disease Specialist",
    "Diabetes": "Endocrinologist",
    "Gastroenteritis": "Gastroenterologist",
    "Bronchial Asthma": "Pulmonologist",
    "Hypertension": "Cardiologist",
    "Migraine": "Neurologist",
    "Cervical spondylosis": "Orthopedist/Neurologist",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "Jaundice": "Hepatologist",
    "Malaria": "Infectious Disease Specialist",
    "Chicken pox": "Dermatologist",
    "Dengue": "Infectious Disease Specialist",
    "Typhoid": "Infectious Disease Specialist",
    "hepatitis A": "Hepatologist",
    "Hepatitis B": "Hepatologist",
    "Hepatitis C": "Hepatologist",
    "Hepatitis D": "Hepatologist",
    "Hepatitis E": "Hepatologist",
    "Alcoholic hepatitis": "Hepatologist",
    "Tuberculosis": "Pulmonologist",
    "Common Cold": "General Practitioner",
    "Pneumonia": "Pulmonologist",
    "Dimorphic hemorrhoids(piles)": "Gastroenterologist",
    "Heart attack": "Cardiologist",
    "Varicose veins": "Vascular Surgeon",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist",
    "Osteoarthritis": "Orthopedist",
    "Arthritis": "Rheumatologist",
    "(vertigo) Paroxysmal Positional Vertigo": "Neurologist/ENT Specialist",
    "Acne": "Dermatologist",
    "Urinary tract infection": "Urologist",
    "Psoriasis": "Dermatologist",
    "Impetigo": "Dermatologist"
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Disease Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .form-container { max-width: 500px; margin: auto; }
        textarea { width: 100%; height: 100px; padding: 10px; font-size: 16px; }
        .output { margin-top: 20px; padding: 10px; background-color: #f9f9f9; border: 1px solid #ddd; }
        button { padding: 10px 20px; font-size: 16px; }
    </style>
</head>
<body>
    <div class="form-container">
        <h1>Disease Prediction System</h1>
        <form method="POST">
            <label for="symptoms">Enter symptoms (comma-separated):</label><br>
            <textarea id="symptoms" name="symptoms">{{ symptoms_input }}</textarea><br><br>
            <button type="submit">Predict Disease</button>
        </form>
        {% if result %}
        <div class="output">
            <h3>Prediction Result:</h3>
            <p><strong>Disease:</strong> {{ result.disease }}</p>
            <p><strong>Specialist:</strong> {{ result.specialist }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    symptoms_input = ""
    
    if request.method == "POST":
        # Get user input
        symptoms_input = request.form.get("symptoms", "")
        symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]

        # Handle up to 17 symptoms
        if len(symptoms) > 17:
            result = {
                "error": "Please enter at most 17 symptoms."
            }
        else:
            # Fill with zeros for missing symptoms
            input_features = np.zeros(17)
            for i, symptom in enumerate(symptoms):
                if i < 17:  # Only use up to 17 features
                    input_features[i] = 1

            # Predict the disease
            try:
                prediction = model.predict([input_features])[0]  # Predict using the model
                disease = prediction  # Get the predicted disease
                specialist = disease_specialist_map.get(disease, "General Physician")  # Get the specialist
                result = {
                    "disease": disease,
                    "specialist": specialist
                }
            except Exception as e:
                result = {
                    "error": f"Error in prediction: {str(e)}"
                }
    
    return render_template_string(HTML_TEMPLATE, result=result, symptoms_input=symptoms_input)

if __name__ == "__main__":
    app.run(debug=True)
