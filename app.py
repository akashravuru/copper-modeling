from copper_functions import *
import streamlit as st

st.title("Industrial Copper Modeling")
st.subheader("Predict Selling Price or Lead Status")

mode = st.radio("Select Mode", ["Predict Selling Price", "Predict Lead Status"])

quantity_tons = st.number_input("Quantity (Tons)", min_value=0.1, value=100.0)
customer = st.number_input("Customer ID", min_value=0, value=30202938)
country = st.number_input("Country Code", min_value=0, value=28)
item_type = st.selectbox("Item Type", ["W", "WI", "S", "Others", "PL", "IPL", "SLAWR"])
application = st.number_input("Application", min_value=0, value=10)
thickness = st.number_input("Thickness", min_value=0.1, value=2.0)
width = st.number_input("Width", min_value=0.0, value=1250.0)
product_ref = st.number_input("Product Reference", min_value=0, value=1670798778)

inputs = {
    'quantity': quantity_tons,
    'thickness': thickness,
    'width': width,
    'application': application,
    'country': country
}

if mode == "Predict Selling Price":
    if st.button("Predict Price"):
        price = predict_price(quantity_tons, customer, country, item_type,
                              application, thickness, width, product_ref)
        st.success(f"Predicted Selling Price: ${price:,.2f}")
        with st.spinner("Getting AI explanation..."):
            explanation = explain_prediction("price", price, inputs)
        st.info(f"💡 {explanation}")

elif mode == "Predict Lead Status":
    if st.button("Predict Status"):
        status = predict_status(quantity_tons, customer, country, item_type,
                                application, thickness, width, product_ref)
        if status == "WON":
            st.success("Lead Status: WON ✅")
        else:
            st.error("Lead Status: LOST ❌")
        with st.spinner("Getting AI explanation..."):
            explanation = explain_prediction("status", status, inputs)
        st.info(f"💡 {explanation}")