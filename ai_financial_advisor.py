import pandas as pd
import joblib
import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from flask_cors import CORS # Needed for local testing (frontend port != backend port)

# --- Configuration & Setup ---
app = Flask(__name__)
# Enable CORS to allow the HTML file (run locally or on a different port) to access this API
CORS(app) 

# IMPORTANT: Please ensure this path points to your Financial_data.csv
CSV_PATH = r"C:\Users\KIIT\Downloads\venv\Financial_data.csv"
MODEL_PATH = "models/savings_model.pkl"

if not os.path.exists("models"):
    os.makedirs("models")

# Load model globally when the application starts
SAVINGS_MODEL = None
try:
    if os.path.exists(MODEL_PATH):
        SAVINGS_MODEL = joblib.load(MODEL_PATH)
        print("✅ Savings model loaded successfully.")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}. Run 'python app.py train' first.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure the model was trained correctly.")


# ------------------------------
# TRAINING FUNCTION (Run via CLI: python app.py train)
# ------------------------------
def train_model():
    """Trains the Linear Regression model and saves it to a pickle file."""
    print("[INFO] Starting training...")
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV file not found at: {CSV_PATH}")
        return

    try:
        df = pd.read_csv(CSV_PATH)
        print("[INFO] Loaded CSV columns:", list(df.columns))

        expense_cols = [
            "Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport",
            "Eating_Out", "Entertainment", "Utilities", "Healthcare",
            "Education", "Miscellaneous"
        ]

        # Check for essential columns
        required_cols = ["Income", "Desired_Savings"] + expense_cols
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"[ERROR] CSV missing columns: {missing}")
            return

        # Prepare dataset: Create Total_Expenses feature
        df["Total_Expenses"] = df[expense_cols].sum(axis=1)

        X = df[["Income", "Total_Expenses"]]
        y = df["Desired_Savings"]

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_PATH)
        print("\n✅ MODEL TRAINED SUCCESSFULLY!")
        print(f"Saved to {MODEL_PATH}\n")

    except Exception as e:
        print(f"[CRITICAL ERROR] Training failed: {e}")


# ------------------------------
# API ENDPOINT for Budget Planning
# ------------------------------
@app.route("/api/plan", methods=["POST"])
def budget_plan_api():
    """Receives user data from the frontend and returns an AI-generated budget plan."""
    if SAVINGS_MODEL is None:
        return jsonify({"error": "AI model is not loaded. Please train the model first (python app.py train)."}), 503

    try:
        data = request.json
        # Extract and validate inputs from the frontend JSON
        income = float(data.get("income"))
        savings = float(data.get("savings"))
        goal = data.get("goal")
        target_amount = float(data.get("targetAmount"))
        time_frame = int(data.get("timeFrame"))

        if income <= 0 or target_amount <= 0 or time_frame <= 0:
            return jsonify({"error": "Income, Target Amount, and Time Frame must be positive values."}), 400

        # --- Budget Calculation ---
        amount_to_save = target_amount - savings
        
        if amount_to_save <= 0:
            monthly_required = 0.0
            plan_output = f"🎉 Goal Achieved! Your current savings of ₹{savings:,.2f} already meet or exceed your target of ₹{target_amount:,.2f} for '{goal}'."
            
            return jsonify({"plan": plan_output})
        
        monthly_required = amount_to_save / time_frame
        
        # --- ML Model Prediction ---
        # The model was trained on (Income, Total_Expenses) -> Desired_Savings.
        # Here, we treat the 'monthly_required' amount as the 'Total_Expenses' (i.e., the largest expense category)
        # to see what the model suggests given this required commitment.
        
        # Prepare data for prediction (must match the format used during training)
        ml_input = np.array([[income, monthly_required]])
        predicted_savings = SAVINGS_MODEL.predict(ml_input)[0]

        # Ensure savings is non-negative
        predicted_savings = max(0, predicted_savings)
        
        # --- AI Advisor Logic ---
        advice = ""
        if monthly_required > (income * 0.4):
            advice = "🚨 **High Commitment Alert:** Your required monthly saving of **₹{0:,.2f}** is over 40% of your income. This is aggressive. Consider increasing the time frame or finding additional income sources.".format(monthly_required)
        elif monthly_required > predicted_savings * 1.2:
             advice = "📈 **Stretch Goal:** The required monthly savings (₹{0:,.2f}) is significantly higher than the AI's typical recommendation (₹{1:,.2f}) for your income profile. Strict discipline is required.".format(monthly_required, predicted_savings)
        else:
            advice = "💰 **Feasible Goal:** Your required monthly savings aligns well with the AI's typical recommendation for your income level. You've got this!"

        # --- Plan Generation ---
        plan_output = f"""
        **🎯 Goal:** {goal}
        **🏦 Target Amount:** ₹{target_amount:,.2f}
        **🗓️ Time Frame:** {time_frame} months
        **💸 Current Savings:** ₹{savings:,.2f}

        ---

        **✅ Required Monthly Saving:** **₹{monthly_required:,.2f}**
        * *(Target amount remaining: ₹{amount_to_save:,.2f})*

        **💵 Remaining Monthly Budget for All Expenses:** ₹{(income - monthly_required):,.2f}
        * *Your total non-savings expenses (rent, food, etc.) must not exceed this amount.*

        **🧠 AI Recommendation:** {advice}

        **✨ Strategy Tip:** Try to allocate your remaining budget (₹{(income - monthly_required):,.2f}) using the 50/30/20 rule as a guideline:
        * **Needs (50%):** ₹{income * 0.5:,.2f}
        * **Wants (30%):** ₹{income * 0.3:,.2f}
        * **Savings/Buffer (20%):** ₹{income * 0.2:,.2f}
        """

        return jsonify({"plan": plan_output})

    except ValueError:
        return jsonify({"error": "Invalid input format. Please ensure all number fields contain numeric values."}), 400
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500


# ------------------------------
# MAIN Execution
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        train_model()
    else:
        # Flask server runs on http://127.0.0.1:5000/
        app.run(debug=True, host='127.0.0.1', port=5500)