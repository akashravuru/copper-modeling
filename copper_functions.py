import numpy as np
import pandas as pd
import os
import streamlit as st
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from google import genai as google_genai

load_dotenv()

@st.cache_resource
def load_models():
    # Load and clean data
    df = pd.read_excel("Copper_Set.xlsx")
    
    # Clean
    df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
    df['material_ref'] = df['material_ref'].apply(
        lambda x: None if str(x).startswith('00000') else x
    )
    df = df.dropna(subset=['selling_price'])
    df = df.drop(columns=['material_ref'])
    df = df.fillna(df.median(numeric_only=True))
    df = df.dropna(subset=['id', 'status'])
    df = df.drop(columns=['id', 'item_date', 'delivery date'])
    
    # Log transforms
    df['selling_price'] = np.log1p(df['selling_price'])
    df['quantity tons'] = np.log1p(df['quantity tons'])
    df['thickness'] = np.log1p(df['thickness'])
    df = df[df['selling_price'] > 0]
    
    # Encode
    le = LabelEncoder()
    df['item type'] = le.fit_transform(df['item type'])
    
    # Regression
    X_reg = df.drop(columns=['selling_price', 'status'])
    y_reg = df['selling_price']
    X_reg_train, X_reg_test, y_reg_train, _ = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg_model = XGBRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_reg_train, y_reg_train)
    
    # Classification
    df_class = df[df['status'].isin(['Won', 'Lost'])].copy()
    df_class['status'] = df_class['status'].map({'Won': 1, 'Lost': 0})
    X_class = df_class.drop(columns=['selling_price', 'status'])
    y_class = df_class['status']
    X_class_train, _, y_class_train, _ = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    class_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    class_model.fit(X_class_train, y_class_train)
    
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
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets["GEMINI_API_KEY"]
            except:
                return "AI explanation unavailable."
        
        client = google_genai.Client(api_key=api_key)
        
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