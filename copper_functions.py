import pickle
import numpy as np
import os
from dotenv import load_dotenv
from google import genai as google_genai

load_dotenv()

def load_models():
    with open("reg_model.pkl", "rb") as f:
        reg_model = pickle.load(f)
    with open("class_model.pkl", "rb") as f:
        class_model = pickle.load(f)
    with open("le_item_type.pkl", "rb") as f:
        le = pickle.load(f)
    return reg_model, class_model, le

def predict_price(quantity_tons, customer, country, item_type,
                  application, thickness, width, product_ref):
    reg_model, _, le = load_models()
    item_type_enc = le.transform([item_type])[0]
    qty_log = np.log1p(quantity_tons)
    thick_log = np.log1p(thickness)
    features = np.array([[qty_log, customer, country, item_type_enc,
                          application, thick_log, width, product_ref]])
    price_log = reg_model.predict(features)[0]
    return round(np.expm1(price_log), 2)

def predict_status(quantity_tons, customer, country, item_type,
                   application, thickness, width, product_ref):
    _, class_model, le = load_models()
    item_type_enc = le.transform([item_type])[0]
    qty_log = np.log1p(quantity_tons)
    thick_log = np.log1p(thickness)
    features = np.array([[qty_log, customer, country, item_type_enc,
                          application, thick_log, width, product_ref]])
    prediction = class_model.predict(features)[0]
    return "WON" if prediction == 1 else "LOST"

def explain_prediction(mode, prediction, inputs):
    try:
        client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        if mode == "price":
            prompt = f"""A copper industry transaction has these details:
            Quantity: {inputs['quantity']} tons, Thickness: {inputs['thickness']}mm,
            Width: {inputs['width']}mm, Application: {inputs['application']}, Country code: {inputs['country']}
            The predicted selling price is ${prediction}.
            In 2-3 sentences, explain why this price makes sense based on these factors. Be specific and concise."""
        else:
            prompt = f"""A copper sales lead has these details:
            Quantity: {inputs['quantity']} tons, Thickness: {inputs['thickness']}mm,
            Width: {inputs['width']}mm, Application: {inputs['application']}, Country code: {inputs['country']}
            The model predicted this lead will be {prediction}.
            In 2-3 sentences, explain the likely reason based on these factors. Be specific and concise."""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI explanation unavailable: {e}"