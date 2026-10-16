"""Gradio front end for the SuperKart sales forecasting model.

Identical inputs and identical model to the Streamlit app in ../deployment; only the UI framework
differs, because a free Hugging Face account can host a Gradio Space but not a Docker one.
"""
import joblib
import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download

# ZeroGPU is the only hardware a free account gets, and that runtime refuses to start unless it
# finds a function decorated with @spaces.GPU. This app is pure CPU, so the probe below exists only
# to satisfy that check and is never called.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_startup_probe():
        return "ok"

except ImportError:
    pass

MODEL_REPO = "khemanta/superkart-sales-model"
MODEL_FILE = "best_superkart_model_v1.joblib"
CURRENT_YEAR = 2026

model = joblib.load(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE))

STORES = {
    "OUT001 - Supermarket Type1, Tier 2, High (1987)":
        ("OUT001", "High", "Tier 2", "Supermarket Type1", 1987),
    "OUT002 - Food Mart, Tier 3, Small (1998)":
        ("OUT002", "Small", "Tier 3", "Food Mart", 1998),
    "OUT003 - Departmental Store, Tier 1, Medium (1999)":
        ("OUT003", "Medium", "Tier 1", "Departmental Store", 1999),
    "OUT004 - Supermarket Type2, Tier 2, Medium (2009)":
        ("OUT004", "Medium", "Tier 2", "Supermarket Type2", 2009),
}

PRODUCT_TYPES = ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
                 "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
                 "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"]


def forecast_sales(product_type, product_category, sugar_content, product_weight, product_mrp,
                   allocated_area, store_label):
    """Score one product-store combination and return a formatted forecast."""
    store_id, store_size, city_type, store_type, year = STORES[store_label]

    input_data = pd.DataFrame([{
        "Product_Weight": float(product_weight),
        "Product_Sugar_Content": sugar_content,
        "Product_Allocated_Area": float(allocated_area),
        "Product_Type": product_type,
        "Product_MRP": float(product_mrp),
        "Store_Id": store_id,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
        "Product_Category": product_category,
        "Store_Age_Years": CURRENT_YEAR - year,
    }])

    prediction = float(model.predict(input_data)[0])
    return (f"## Expected sales: {prediction:,.2f}\n\n"
            f"**Store:** {store_id} ({store_type}, {city_type})\n\n"
            f"**Product:** {product_type}, MRP {product_mrp:,.2f}, weight {product_weight}\n\n"
            "*Typical error for this model is around 290 on average sales of 3,464 (about 8%). "
            "Treat the figure as a planning estimate, not a guarantee.*")


with gr.Blocks(title="SuperKart Sales Forecast") as demo:
    gr.Markdown("# SuperKart - Product Store Sales Forecast")
    gr.Markdown(
        "Estimate the revenue a product will generate in a given store, so procurement and shelf "
        "allocation can be planned before stock is ordered. The model is a tuned XGBoost pipeline "
        "explaining about 93% of the variation in sales."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Product")
            product_type = gr.Dropdown(PRODUCT_TYPES, value="Frozen Foods", label="Product type")
            product_category = gr.Dropdown(["FD", "NC", "DR"], value="FD",
                                           label="Category (FD food, NC non-consumable, DR drinks)")
            sugar_content = gr.Dropdown(["Low Sugar", "Regular", "No Sugar"], value="Low Sugar",
                                        label="Sugar content")
            product_weight = gr.Number(label="Product weight", value=12.65, minimum=4, maximum=22)
            product_mrp = gr.Number(label="Maximum retail price (MRP)", value=147.0,
                                    minimum=30, maximum=270)
            allocated_area = gr.Number(label="Allocated display area (ratio)", value=0.07,
                                       minimum=0.0, maximum=0.30)

        with gr.Column():
            gr.Markdown("### Store")
            store_label = gr.Dropdown(list(STORES.keys()), value=list(STORES.keys())[3],
                                      label="Store")

    submit = gr.Button("Forecast sales", variant="primary")
    output = gr.Markdown()

    inputs = [product_type, product_category, sugar_content, product_weight, product_mrp,
              allocated_area, store_label]
    submit.click(fn=forecast_sales, inputs=inputs, outputs=output)

    gr.Examples(
        examples=[
            ["Dairy", "FD", "Low Sugar", 16.5, 171.0, 0.144,
             "OUT003 - Departmental Store, Tier 1, Medium (1999)"],
            ["Snack Foods", "FD", "Regular", 8.2, 95.0, 0.030,
             "OUT002 - Food Mart, Tier 3, Small (1998)"],
        ],
        inputs=inputs,
        label="Example product-store combinations (a strong one, and a weak one)",
    )

if __name__ == "__main__":
    demo.launch()
