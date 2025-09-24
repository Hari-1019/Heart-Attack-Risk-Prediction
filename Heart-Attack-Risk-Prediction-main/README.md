# Heart Attack Risk Prediction

Predict your heart attack risk instantly with a modern web app powered by machine learning.

---

## 🚀 Quick Start

1. **Clone & Install**
   ```sh
   git clone https://github.com/dilitha21/Heart-Attack-Risk-Prediction.git
   cd Heart-Attack-Risk-Prediction
   pip install -r requirements.txt
   ```
2. **Train Models**
   - For Random Forest (API default):
     ```sh
     python train_model.py
     ```
   - For all models (Logistic Regression, Random Forest, XGBoost):
     ```sh
     python ML_CW.py
     ```
3. **Run the API**
   ```sh
   python api.py
   ```
4. **Open the Frontend**
   - Open `frontend/index.html` in your browser.
   - Ensure the API is running for predictions.

---

## 📝 Field Names (must match exactly)

    Age
    Gender
    Cholesterol
    Blood Pressure
    Heart Rate
    Smoking
    Obesity
    Diabetes
    Previous Heart Problems
    Medication Use
    Exercise Hours Per Week
    Stress Level

---

## 📦 Project Structure

    api.py           # Flask API backend
    train_model.py   # Model training script (Random Forest)
    ML_CW.py         # Train all models for comparison
    model/           # Saved models, scaler, encoders
    frontend/        # Web UI (HTML/CSS/JS)

---

## 🔗 Example API Request

```sh
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Gender": "Male",
    "Cholesterol": 240,
    "Blood Pressure": "140/90",
    "Heart Rate": 80,
    "Smoking": "Yes",
    "Obesity": "No",
    "Diabetes": "No",
    "Previous Heart Problems": "No",
    "Medication Use": "Yes",
    "Exercise Hours Per Week": 2,
    "Stress Level": 7
  }'
```

---

## ℹ️ Notes

- Field names must be capitalized and match exactly across frontend, backend, and model.
- To use a different model in the API, rename it to `rf_model.pkl` or update the API code.
- Retrain models if you update the dataset.
- Clear browser cache if you see old errors.

---

## License
MIT

