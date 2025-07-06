import streamlit as st
import pandas as pd
import json
import os
import requests
import time

st.set_page_config(page_title="Tracker de Atención Multi-Cámara", layout="wide")
st.title("📊 Dashboard de Atención Multi-Cámara")

# Configuración de la API
API_BASE_URL = "http://app:5000"
TIMEOUT = 10

def hacer_request(endpoint, intentos=3):
    """Función auxiliar para hacer requests con reintentos"""
    for i in range(intentos):
        try:
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=TIMEOUT)
            if response.status_code == 200:
                return response.json()
            else:
                st.warning(f"Intento {i+1}: Error {response.status_code}")
        except requests.exceptions.RequestException as e:
            if i == intentos - 1:  # Último intento
                raise e
            st.warning(f"Intento {i+1} fallido, reintentando...")
            time.sleep(2)
    
    return None

# Estado de conexión
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("🔌 Estado de Conexión")
    with col2:
        if st.button("🔄 Actualizar"):
            st.rerun()

# Obtener lista de cámaras
try:
    st.info("Conectando con el servidor...")
    camaras_data = hacer_request("/camaras")
    
    if camaras_data and "camaras" in camaras_data:
        camaras = camaras_data["camaras"]
        st.success(f"✅ Conectado - {len(camaras)} cámaras encontradas")
        
        # Crear columnas para cada cámara
        cols = st.columns(len(camaras))
        
        for i, camara in enumerate(camaras):
            with cols[i]:
                st.header(f"🎥 {camara['usuario']}")
                st.subheader(f"📍 {camara['posicion']}")
                st.markdown(f"**Estado:** {'🟢 Activa' if camara['status'] == 'Activa' else '🔴 Inactiva'}")
                
                if camara['status'] == 'Activa':
                    # Para imágenes, usar localhost para que el navegador pueda acceder
                    st.markdown(
                        f'<img src="http://localhost:5000/video_feed/{camara["id"]}" width="100%" style="border: 2px solid #999; border-radius: 10px;">',
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Cámara no disponible")
    else:
        st.error("No se pudo obtener información de las cámaras")

except Exception as e:
    st.error(f"Error conectando con el servidor: {e}")
    st.info("💡 Asegúrate de que el servidor esté ejecutándose en el puerto 5000")

# Mostrar estadísticas por usuario
st.header("📈 Estadísticas por Usuario")

try:
    stats_data = hacer_request("/estadisticas")
    
    if stats_data and "estadisticas" in stats_data:
        for usuario, data in stats_data["estadisticas"].items():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"👤 {usuario}", f"{data['promedio_atencion']:.2f}")
            with col2:
                st.metric("📊 Registros", data['total_registros'])
            with col3:
                st.metric("⬆️ Máxima", f"{data['atencion_maxima']:.2f}")
            with col4:
                st.metric("⬇️ Mínima", f"{data['atencion_minima']:.2f}")
            st.divider()
    else:
        st.info("No hay estadísticas disponibles")

except Exception as e:
    st.error(f"Error cargando estadísticas: {e}")

# Información de debugging
with st.expander("🔍 Información de Debug"):
    st.write(f"**URL Base:** {API_BASE_URL}")
    st.write(f"**Timeout:** {TIMEOUT} segundos")
    
    # Probar endpoints
    endpoints = ["/camaras", "/estadisticas"]
    for endpoint in endpoints:
        try:
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
            st.write(f"**{endpoint}:** ✅ {response.status_code}")
        except Exception as e:
            st.write(f"**{endpoint}:** ❌ {str(e)}")