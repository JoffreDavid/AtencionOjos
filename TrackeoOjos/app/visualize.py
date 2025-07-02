import streamlit as st
import pandas as pd
import os
import json

st.set_page_config(page_title="Tracker de Atención", layout="wide")
st.title("📊 Dashboard de Atención en Tiempo Real")

col1, col2 = st.columns(2)

# Mostrar el video en vivo
with col1:
    st.header("🎥 Video en vivo (DroidCam)")
    st.markdown(
        """
        <img src="http://localhost:5000/video_feed" width="100%" style="border: 2px solid #999; border-radius: 10px;">
        """,
        unsafe_allow_html=True
    )

# Cargar y mostrar datos de atención
data_path = "/app/data/attention_logs.json"

if not os.path.exists(data_path):
    col2.warning("⏳ Aún no se han generado datos de atención.")
else:
    with open(data_path, "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    if not data:
        col2.warning("⚠️ El archivo de atención existe, pero está vacío.")
    else:
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        with col2:
            st.header("📈 Atención detectada")
            st.line_chart(df.set_index("timestamp")["nivel_atencion"])

            promedio = df.groupby("usuario")["nivel_atencion"].mean().round(2)
            st.subheader("📊 Promedio de atención por usuario")
            st.dataframe(promedio)
