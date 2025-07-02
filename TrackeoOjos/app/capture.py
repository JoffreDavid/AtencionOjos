import cv2
import mediapipe as mp
import time
import json
import threading
from datetime import datetime
from flask import Flask, Response
import os

# Flask app para servir video
app = Flask(__name__)

# Abrimos la cámara DroidCam
cap = cv2.VideoCapture("http://192.168.137.223:4747/video")

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara desde DroidCam.")
    exit(1)

print("✅ Cámara abierta exitosamente.")

# Face Mesh de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5)

attention_data = []
start_time = time.time()
last_frame = None

def calcular_nivel_atencion(landmarks):
    return 1.0  # Simulación de atención alta

def procesar_video():
    global last_frame, attention_data, start_time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No se pudo leer el frame.")
            time.sleep(1)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            for landmarks in result.multi_face_landmarks:
                nivel = calcular_nivel_atencion(landmarks)
                timestamp = datetime.utcnow().isoformat()
                attention_data.append({
                    "usuario": "usuario1",
                    "timestamp": timestamp,
                    "nivel_atencion": nivel
                })
                print(f"👁️ Atención: {nivel} @ {timestamp}")
        else:
            print("😐 Rostro no detectado.")

        # Guardar JSON cada minuto
        if time.time() - start_time > 5:
            os.makedirs("/app/data", exist_ok=True)
            with open("/app/data/attention_logs.json", "a") as f:
                for row in attention_data:
                    f.write(json.dumps(row) + "\n")
            print(f"💾 {len(attention_data)} registros guardados.")
            attention_data = []
            start_time = time.time()

        # Guardamos último frame para mostrar
        last_frame = frame

# Ruta del video en vivo
def generar_frames():
    global last_frame
    while True:
        if last_frame is None:
            continue
        _, buffer = cv2.imencode('.jpg', last_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ejecutamos todo
if __name__ == "__main__":
    # Ejecuta la detección en un hilo
    hilo = threading.Thread(target=procesar_video)
    hilo.daemon = True
    hilo.start()

    # Inicia Flask en el puerto 5000
    app.run(host='0.0.0.0', port=5000)
