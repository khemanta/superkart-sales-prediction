"""Streamlit front end for the SuperKart sales forecasting model."""
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO = "khemanta/superkart-sales-model"
MODEL_FILE = "best_superkart_model_v1.joblib"
CURRENT_YEAR = 2026

# The four real stores, so the form cannot produce a combination that never existed.
STORES = {
    "OUT001 - Supermarket Type1, Tier 2, High, est. 1987": {
        "Store_Id": "OUT001", "Store_Size": "High",
        "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type1", "year": 1987},
    "OUT002 - Food Mart, Tier 3, Small, est. 1998": {
        "Store_Id": "OUT002", "Store_Size": "Small",
        "Store_Location_City_Type": "Tier 3", "Store_Type": "Food Mart", "year": 1998},
    "OUT003 - Departmental Store, Tier 1, Medium, est. 1999": {
        "Store_Id": "OUT003", "Store_Size": "Medium",
        "Store_Location_City_Type": "Tier 1", "Store_Type": "Departmental Store", "year": 1999},
    "OUT004 - Supermarket Type2, Tier 2, Medium, est. 2009": {
        "Store_Id": "OUT004", "Store_Size": "Medium",
        "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type2", "year": 2009},
}

PRODUCT_TYPES = ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
                 "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
                 "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"]


@st.cache_resource
def load_model():
    """Download the registered model once and keep it in memory."""
    return joblib.load(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE))


model = load_model()

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒")
st.title("SuperKart — Product Store Sales Forecast")
st.write(
    "Estimate the revenue a product will generate in a given store. Use it to plan procurement "
    "and shelf allocation before the stock is ordered."
)

st.subheader("Product")
col1, col2 = st.columns(2)
with col1:
    product_type = st.selectbox("Product type", PRODUCT_TYPES, index=5)
    product_category = st.selectbox(
        "Product category", ["FD", "NC", "DR"],
        format_func=lambda c: {"FD": "FD - Food", "NC": "NC - Non-consumable",
                               "DR": "DR - Drinks"}[c])
    sugar_content = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
with col2:
    product_weight = st.number_input("Product weight", min_value=4.0, max_value=22.0,
                                     value=12.65, step=0.1)
    product_mrp = st.number_input("Maximum retail price (MRP)", min_value=30.0, max_value=270.0,
                                  value=147.0, step=1.0)
    allocated_area = st.number_input("Allocated display area (ratio)", min_value=0.0,
                                     max_value=0.30, value=0.07, step=0.005, format="%.3f")

st.subheader("Store")
store_label = st.selectbox("Store", list(STORES.keys()))
store = STORES[store_label]

# One row, with exactly the columns the pipeline was trained on. Store_Age_Years and
# Product_Category are derived here the same way prep.py derived them.
input_data = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Sugar_Content": sugar_content,
    "Product_Allocated_Area": allocated_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Id": store["Store_Id"],
    "Store_Size": store["Store_Size"],
    "Store_Location_City_Type": store["Store_Location_City_Type"],
    "Store_Type": store["Store_Type"],
    "Product_Category": product_category,
    "Store_Age_Years": CURRENT_YEAR - store["year"],
}])

with st.expander("See the row sent to the model"):
    st.dataframe(input_data)

if st.button("Forecast sales", type="primary"):
    prediction = float(model.predict(input_data)[0])
    st.subheader("Forecast")
    st.success(f"Expected sales: **{prediction:,.2f}**")
    st.caption(
        "Typical error for this model is around 290 on average sales of 3,464 (about 8%). "
        "Treat the figure as a planning estimate, not a guarantee."
    )

st.divider()
st.caption(
    "Model: tuned XGBoost pipeline (preprocessing + regressor), registered on the Hugging Face "
    "model hub and retrained automatically by the project's GitHub Actions pipeline."
)
