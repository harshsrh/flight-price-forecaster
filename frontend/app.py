import streamlit as st
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.api_client import get_health, get_available_routes, predict_price, get_history
from frontend.utils import build_forecast_curve, load_model_metrics

st.set_page_config(
    page_title="Flight Price Forecaster",
    page_icon="✈️",
    layout="wide"
)

# ---------- Sidebar: API status ----------
st.sidebar.title("✈️ Flight Price Forecaster")
health = get_health()

if health and health.get("status") == "healthy":
    st.sidebar.success("API connected")
    st.sidebar.caption(f"Prophet routes: {len(health.get('prophet_routes', []))}")
    st.sidebar.caption(f"LSTM routes: {len(health.get('lstm_routes', []))}")
else:
    st.sidebar.error("API not reachable — start it with:\nuvicorn api.main:app --reload")
    st.stop()

# ---------- Main title ----------
st.title("Flight Price Forecasting Dashboard")
st.caption("Predict prices and get a buy-now-or-wait recommendation")

# ---------- Input form ----------
col1, col2, col3 = st.columns(3)

with col1:
    source_city = st.selectbox("From", ["Delhi", "Mumbai", "Bangalore"])
with col2:
    destination_city = st.selectbox("To", ["Mumbai", "Delhi", "Bangalore"])

st.caption("⚠️ Demo currently supports Delhi↔Mumbai and Delhi→Bangalore routes (production model coverage)")

with col3:
    airline = st.selectbox("Airline", ["Indigo", "Air_India", "Vistara", "GO_FIRST", "SpiceJet", "AirAsia"])

col4, col5, col6 = st.columns(3)

with col4:
    days_left = st.slider("Days until departure", 1, 60, 15)
with col5:
    stops = st.selectbox("Stops", ["zero", "one", "two_or_more"])
with col6:
    travel_class = st.selectbox("Class", ["Economy", "Business"])

predict_btn = st.button("Predict price", type="primary", use_container_width=True)

# ---------- Validation ----------
if source_city == destination_city:
    st.warning("Source and destination cities must be different.")
    st.stop()

# ---------- Run prediction ----------
if predict_btn:
    with st.spinner("Getting prediction..."):
        result, error = predict_price(
            source_city, destination_city, airline, days_left,
            stops=stops, travel_class=travel_class
        )

    if error:
        st.error(f"Prediction failed: {error}")
    else:
        # --- Top metrics row ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted price", f"₹{result['predicted_price']:,.0f}")
        m2.metric("Model used", result["model_used"])
        m3.metric("Trend", result["trend"].capitalize())

        # --- Buy/Wait signal banner ---
        if result["recommendation"] == "wait":
            st.info(f"🟢 **Wait** — prices on this route are trending down. Booking later could save money.")
        else:
            st.warning(f"🔴 **Buy now** — prices are trending up or stable. Waiting may cost more.")

        if result.get("confidence_low") and result.get("confidence_high"):
            st.caption(f"95% confidence range: ₹{result['confidence_low']:,.0f} – ₹{result['confidence_high']:,.0f}")

        st.divider()

        # --- 30-day forecast chart ---
        st.subheader("30-day price forecast")
        with st.spinner("Building forecast curve (this calls the API 30 times)..."):
            days_range, prices = build_forecast_curve(
                source_city, destination_city, airline,
                stops=stops, travel_class=travel_class, max_days=30
            )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=days_range, y=prices,
            mode='lines+markers',
            name='Forecasted price',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        fig.add_vline(x=days_left, line_dash="dash", line_color="red",
                       annotation_text="Your selected date")
        fig.update_layout(
            xaxis_title="Days until departure",
            yaxis_title="Price (₹)",
            xaxis=dict(autorange="reversed"),  # 60 days -> 1 day, left to right = time passing
            height=400,
            margin=dict(t=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Curve reversed: right side = closer to departure date")

        st.divider()

        # --- Model performance section ---
        st.subheader("Model performance")
        st.caption("Based on held-out test data from training")

        metrics = load_model_metrics()
        p1, p2, p3 = st.columns(3)
        p1.metric("RMSE", f"₹{metrics['rmse']:,.0f}" if metrics['rmse'] else "N/A")
        p2.metric("MAE", f"₹{metrics['mae']:,.0f}" if metrics['mae'] else "N/A")
        p3.metric("Directional accuracy", f"{metrics['directional_acc']:.1f}%" if metrics['directional_acc'] else "N/A")
else:
    st.info("Fill in the flight details above and click **Predict price** to get started.")

# ---------- Footer ----------
st.divider()
st.caption("Built with FastAPI + Prophet/LSTM + Streamlit | Data: Kaggle Flight Price Prediction Dataset")