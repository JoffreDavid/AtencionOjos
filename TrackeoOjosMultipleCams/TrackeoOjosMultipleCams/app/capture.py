import cv2
import mediapipe as mp
import time
import json
import threading
from datetime import datetime
from flask import Flask, Response
import os
import numpy as np

# Flask app para servir video
app = Flask(__name__)

# Configuración de múltiples cámaras
# Cambiar segun las URL's que aparezcan en los celulares
CAMARAS = {
    "camara1": {
        "url": "http://192.168.100.156:4747/video",
        "usuario": "camara1",
        "posicion": "Escritorio 1"
    },
    "camara2": {
        "url": "http://192.168.100.117:4747/video",
        "usuario": "camara2",
        "posicion": "Escritorio 2"
    }
}
# Diccionario para almacenar las capturas
capturas = {}
last_frames = {}
attention_data = []
start_time = time.time()

# Face Mesh de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    static_image_mode=False
)

# Inicializar cámaras
def inicializar_camaras():
    """Inicializa todas las cámaras configuradas"""
    for cam_id, config in CAMARAS.items():
        print(f"🎥 Inicializando {cam_id} - Usuario: {config['usuario']}")
        cap = cv2.VideoCapture(config['url'])
        
        # Configurar propiedades
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if cap.isOpened():
            capturas[cam_id] = cap
            last_frames[cam_id] = None
            print(f"✅ {cam_id} inicializada correctamente")
        else:
            print(f"❌ Error al inicializar {cam_id}")
            capturas[cam_id] = None

# Reconectar cámara específica
def reconectar_camara(cam_id):
    """Reconectar una cámara específica"""
    if cam_id not in CAMARAS:
        return False
    
    config = CAMARAS[cam_id]
    print(f"🔄 Reconectando {cam_id} - Usuario: {config['usuario']}")
    
    if capturas.get(cam_id):
        capturas[cam_id].release()
    
    time.sleep(2)
    
    cap = cv2.VideoCapture(config['url'])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if cap.isOpened():
        capturas[cam_id] = cap
        print(f"✅ {cam_id} reconectada exitosamente")
        return True
    else:
        print(f"❌ No se pudo reconectar {cam_id}")
        capturas[cam_id] = None
        return False


# 🔍 Método 1: EAR (Eye Aspect Ratio) - Más preciso
def calcular_nivel_atencion_ear(landmarks):
    """Calcula atención usando Eye Aspect Ratio (EAR)"""
    try:
        # Puntos del ojo derecho
        right_eye = [
            landmarks.landmark[33],   # Esquina externa
            landmarks.landmark[7],    # Punto superior 1
            landmarks.landmark[163],  # Punto superior 2
            landmarks.landmark[144],  # Esquina interna
            landmarks.landmark[145],  # Punto inferior 1
            landmarks.landmark[153]   # Punto inferior 2
        ]
        
        # Puntos del ojo izquierdo
        left_eye = [
            landmarks.landmark[362],  # Esquina externa
            landmarks.landmark[382],  # Punto superior 1
            landmarks.landmark[381],  # Punto superior 2
            landmarks.landmark[380],  # Esquina interna
            landmarks.landmark[374],  # Punto inferior 1
            landmarks.landmark[373]   # Punto inferior 2
        ]
        
        # Calcular EAR para cada ojo
        def calcular_ear(eye_points):
            # Distancias verticales
            A = abs(eye_points[1].y - eye_points[5].y)
            B = abs(eye_points[2].y - eye_points[4].y)
            # Distancia horizontal
            C = abs(eye_points[0].x - eye_points[3].x)
            
            # EAR formula
            ear = (A + B) / (2.0 * C)
            return ear
        
        right_ear = calcular_ear(right_eye)
        left_ear = calcular_ear(left_eye)
        
        # Promedio de ambos ojos
        avg_ear = (right_ear + left_ear) / 2.0
        
        # Clasificar nivel de atención basado en EAR
        if avg_ear < 0.20:
            return 0.1, "Somnoliento"
        elif avg_ear < 0.25:
            return 0.4, "Cansado"
        elif avg_ear < 0.30:
            return 0.7, "Alerta"
        else:
            return 1.0, "Muy atento"
            
    except Exception as e:
        print(f"❌ Error EAR: {e}")
        return 0.0, "Error"

#👀 Método 2: Dirección de la mirada
def calcular_nivel_atencion_mirada(landmarks):
    """Calcula atención basado en dirección de la mirada"""
    try:
        # Puntos de las pupilas (aproximados)
        right_pupil = landmarks.landmark[468]  # Centro ojo derecho
        left_pupil = landmarks.landmark[473]   # Centro ojo izquierdo
        
        # Punto central de la cara
        nose_tip = landmarks.landmark[1]
        
        # Calcular desviación de la mirada
        right_deviation = abs(right_pupil.x - nose_tip.x)
        left_deviation = abs(left_pupil.x - nose_tip.x)
        
        avg_deviation = (right_deviation + left_deviation) / 2
        
        # Clasificar atención según desviación
        if avg_deviation < 0.02:
            return 0.9, "Mirando al frente"
        elif avg_deviation < 0.05:
            return 0.6, "Ligeramente distraído"
        elif avg_deviation < 0.08:
            return 0.3, "Distraído"
        else:
            return 0.1, "Muy distraído"
            
    except Exception as e:
        print(f"❌ Error mirada: {e}")
        return 0.0, "Error"
#🧠 Método 3: Combinado (EAR + Mirada + Parpadeo)

# Contador global de parpadeos por usuario
parpadeo_contador = {}
ultimo_ear = {}

def calcular_nivel_atencion_combinado(landmarks, usuario_id):
    """Método combinado con identificación de usuario"""
    global parpadeo_contador, ultimo_ear
    
    # Inicializar contadores para usuario nuevo
    if usuario_id not in parpadeo_contador:
        parpadeo_contador[usuario_id] = 0
        ultimo_ear[usuario_id] = 0.25
    
    try:
        # 1. Calcular EAR
        def calcular_ear_simple(landmarks):
            right_eye_top = landmarks.landmark[159]
            right_eye_bottom = landmarks.landmark[145]
            left_eye_top = landmarks.landmark[386]
            left_eye_bottom = landmarks.landmark[374]
            
            right_ear = abs(right_eye_top.y - right_eye_bottom.y)
            left_ear = abs(left_eye_top.y - left_eye_bottom.y)
            
            return (right_ear + left_ear) / 2.0
        
        ear_actual = calcular_ear_simple(landmarks)
        
        # 2. Detectar parpadeo
        if ear_actual < 0.015 and ultimo_ear[usuario_id] > 0.020:
            parpadeo_contador[usuario_id] += 1
        
        ultimo_ear[usuario_id] = ear_actual
        
        # 3. Calcular dirección de mirada
        nose_tip = landmarks.landmark[1]
        right_eye_center = landmarks.landmark[159]
        left_eye_center = landmarks.landmark[386]
        
        eye_center_x = (right_eye_center.x + left_eye_center.x) / 2
        mirada_desviacion = abs(eye_center_x - nose_tip.x)
        
        # 4. Calcular puntuación combinada
        ear_score = min(ear_actual * 20, 1.0)
        mirada_score = max(0, 1.0 - mirada_desviacion * 10)
        
        # Frecuencia de parpadeo
        if parpadeo_contador[usuario_id] < 5:
            parpadeo_score = 0.3
        elif parpadeo_contador[usuario_id] < 15:
            parpadeo_score = 1.0
        else:
            parpadeo_score = 0.5
        
        # Puntuación final
        atencion_final = (ear_score * 0.5) + (mirada_score * 0.3) + (parpadeo_score * 0.2)
        
        # Determinar estado
        if atencion_final > 0.8:
            estado = "Muy atento"
        elif atencion_final > 0.6:
            estado = "Atento"
        elif atencion_final > 0.4:
            estado = "Distraído"
        else:
            estado = "Somnoliento"
        
        return round(atencion_final, 2), estado
        
    except Exception as e:
        print(f"❌ Error combinado para {usuario_id}: {e}")
        return 0.0, "Error"

def calcular_nivel_atencion(landmarks):
    """Calcula nivel de atencion basado en apertura de ojos"""
    try:
        # Puntos del ojo derecho (más precisos)
        eye_top = landmarks.landmark[159]
        eye_bottom = landmarks.landmark[145]
        
        # Puntos del ojo izquierdo para comparar
        left_eye_top = landmarks.landmark[386]
        left_eye_bottom = landmarks.landmark[374]
        
        # Calcular apertura de ambos ojos
        right_eye_opening = abs(eye_top.y - eye_bottom.y)
        left_eye_opening = abs(left_eye_top.y - left_eye_bottom.y)
        
        # Promedio de ambos ojos
        avg_eye_opening = (right_eye_opening + left_eye_opening) / 2
         # Normalizar y clasificar
        if avg_eye_opening < 0.012:
            return 0.1  # Ojos muy cerrados
        elif avg_eye_opening < 0.018:
            return 0.3  # Ojos medio cerrados
        elif avg_eye_opening < 0.025:
            return 0.6  # Ojos medio abiertos
        else:
            return 0.9  # Ojos bien abiertos
    except Exception as e:
        print(f"❌ Error al calcular nivel de atención: {e}")
        return 0.0  

def crear_directorio_datos():
    """Crear directorio de datos compatible con Windows/Linux"""
    if os.name == 'nt':  # Windows
        data_dir = os.path.join(os.getcwd(), "data")
    else:  # Linux/Docker
        data_dir = "/app/data"
    
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def reconectar_camara():
    """Reconectar cámara en caso de error"""
    global cap
    print("🔄 Intentando reconectar cámara...")
    
    if cap:
        cap.release()
    
    time.sleep(2)  # Esperar antes de reconectar
    
    cap = cv2.VideoCapture("http://192.168.100.117:4747/video")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if cap.isOpened():
        print("✅ Cámara reconectada exitosamente.")
        return True
    else:
        print("❌ No se pudo reconectar la cámara.")
        return False

def procesar_camara(cam_id):
    """Procesa una cámara específica"""
    global last_frames, attention_data, start_time
    
    if cam_id not in CAMARAS or not capturas.get(cam_id):
        return
    
    config = CAMARAS[cam_id]
    usuario = config['usuario']
    posicion = config['posicion']
    cap = capturas[cam_id]
    
    frame_skip = 0
    error_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            error_count += 1
            print(f"⚠️ Frame perdido en {cam_id} ({error_count})")
            
            if error_count > 10:
                if reconectar_camara(cam_id):
                    error_count = 0
                    cap = capturas[cam_id]
                    continue
                else:
                    time.sleep(5)
                    continue
            
            time.sleep(0.1)
            continue
        
        error_count = 0
        
        # Saltar frames para reducir carga
        frame_skip += 1
        if frame_skip % 3 != 0:
            last_frames[cam_id] = frame.copy()
            continue
        
        # Redimensionar frame
        frame_small = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        
        if result.multi_face_landmarks:
            for i, landmarks in enumerate(result.multi_face_landmarks):
                # Usar el método combinado con identificación de usuario
                nivel, estado = calcular_nivel_atencion_combinado(landmarks, usuario)
                
                timestamp = datetime.utcnow().isoformat()
                attention_data.append({
                    "camara_id": cam_id,
                    "usuario": usuario,
                    "posicion": posicion,
                    "timestamp": timestamp,
                    "nivel_atencion": round(nivel, 2),
                    "estado": estado
                })
                print(f"👁️ {usuario} ({posicion}) - Atención: {nivel:.2f} ({estado})")
        else:
            print(f"😐 Rostro no detectado en {cam_id} ({usuario})")
        
        # Guardar último frame
        last_frames[cam_id] = frame.copy()
        time.sleep(0.01)

def procesar_todas_camaras():
    """Inicia el procesamiento de todas las cámaras"""
    threads = []
    
    for cam_id in CAMARAS.keys():
        if capturas.get(cam_id):
            thread = threading.Thread(target=procesar_camara, args=(cam_id,))
            thread.daemon = True
            threads.append(thread)
            thread.start()
    
    return threads

def procesar_video():
    global last_frame, attention_data, start_time
    data_dir = crear_directorio_datos()
    frame_skip = 0
    error_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            error_count += 1
            print(f"⚠️ Frame perdido ({error_count})")
            
            # Si hay muchos errores consecutivos, reconectar
            if error_count > 10:
                if reconectar_camara():
                    error_count = 0
                    continue
                else:
                    time.sleep(5)  # Esperar más tiempo antes de reintentar
                    continue
            
            time.sleep(0.1)
            continue
        
        # Resetear contador de errores cuando se recibe un frame
        error_count = 0
        # Saltar frames para reducir carga de procesamiento
        frame_skip += 1
        if frame_skip % 3 != 0:  # Procesar solo cada 2 frames
            last_frame = frame.copy()
            continue
        
        # Redimensionar frame para procesamiento más rápido
        frame_small = cv2.resize(frame, (320, 240))

        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            for i, landmarks in enumerate(result.multi_face_landmarks):
                # Método 0: Calcular nivel de atención
                #nivel = calcular_nivel_atencion(landmarks)
                
                # Método 1: EAR
                #nivel, estado = calcular_nivel_atencion_ear(landmarks)
                
                # Método 2: Mirada
                # nivel, estado = calcular_nivel_atencion_mirada(landmarks)
                
                # Método 3: Combinado
                nivel, estado = calcular_nivel_atencion_combinado(landmarks)
                
                
                timestamp = datetime.utcnow().isoformat()
                attention_data.append({
                    "usuario": f"usuario{i+1}",
                    "timestamp": timestamp,
                    "nivel_atencion": round(nivel, 2)
                })
                print(f"👁️ Usuario {i+1} - Atención: {nivel:.2f}")
        else:
            print("😐 Rostro no detectado.")

        # Guardar JSON cada minuto
        if time.time() - start_time > 5:
            if attention_data:
                log_file = os.path.join(data_dir, "attention_logs.json")
                try:
                    with open(log_file, "a", encoding='utf-8') as f:
                        for row in attention_data:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"💾 {len(attention_data)} registros guardados en {log_file}")
                except Exception as e:
                    print(f"❌ Error guardando datos: {e}")
                
                attention_data = []
            start_time = time.time()

        # Guardamos último frame para mostrar
        last_frame = frame.copy()
        # Pequeña pausa para evitar sobrecargar el sistema
        time.sleep(0.01)
        
def guardar_datos_periodicamente():
    """Guarda datos cada 5 segundos"""
    global attention_data, start_time
    
    while True:
        time.sleep(5)
        
        if attention_data:
            data_dir = crear_directorio_datos()
            log_file = os.path.join(data_dir, "attention_logs.json")
            
            try:
                with open(log_file, "a", encoding='utf-8') as f:
                    for row in attention_data:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                
                print(f"💾 {len(attention_data)} registros guardados de {len(set(r['usuario'] for r in attention_data))} usuarios")
                
                # Generar estadísticas por usuario
                estadisticas = {}
                for row in attention_data:
                    usuario = row['usuario']
                    if usuario not in estadisticas:
                        estadisticas[usuario] = {'total': 0, 'suma': 0}
                    estadisticas[usuario]['total'] += 1
                    estadisticas[usuario]['suma'] += row['nivel_atencion']
                
                for usuario, stats in estadisticas.items():
                    promedio = stats['suma'] / stats['total']
                    print(f"📊 {usuario}: Promedio {promedio:.2f} ({stats['total']} registros)")
                
            except Exception as e:
                print(f"❌ Error guardando datos: {e}")
            
            attention_data = []

# Ruta del video en vivo
def generar_frames():
    global last_frame
    while True:
        if last_frame is None:
            continue
        _, buffer = cv2.imencode('.jpg', last_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
def cleanup():
    """Limpieza al cerrar la aplicación"""
    global cap
    if cap:
        cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    """Feed de video para una cámara específica"""
    def generar_frames():
        while True:
            if last_frames.get(cam_id) is None:
                continue
            
            frame = last_frames[cam_id]
            
            # Agregar información del usuario en el frame
            if cam_id in CAMARAS:
                usuario = CAMARAS[cam_id]['usuario']
                posicion = CAMARAS[cam_id]['posicion']
                cv2.putText(frame, f"{usuario} - {posicion}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camaras')
def listar_camaras():
    """Lista todas las cámaras disponibles"""
    camaras_info = []
    for cam_id, config in CAMARAS.items():
        status = "Activa" if capturas.get(cam_id) else "Inactiva"
        camaras_info.append({
            "id": cam_id,
            "usuario": config['usuario'],
            "posicion": config['posicion'],
            "status": status,
            "url": config['url']
        })
    return {"camaras": camaras_info}

@app.route('/estadisticas')
def estadisticas_usuarios():
    """Estadísticas por usuario"""
    data_dir = crear_directorio_datos()
    log_file = os.path.join(data_dir, "attention_logs.json")
    
    if not os.path.exists(log_file):
        return {"mensaje": "No hay datos disponibles"}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        usuarios_stats = {}
        for line in lines:
            try:
                data = json.loads(line.strip())
                usuario = data.get('usuario', 'desconocido')
                nivel = data.get('nivel_atencion', 0)
                
                if usuario not in usuarios_stats:
                    usuarios_stats[usuario] = {
                        'total_registros': 0,
                        'suma_atencion': 0,
                        'niveles': []
                    }
                
                usuarios_stats[usuario]['total_registros'] += 1
                usuarios_stats[usuario]['suma_atencion'] += nivel
                usuarios_stats[usuario]['niveles'].append(nivel)
            except:
                continue
        
        # Calcular estadísticas
        resultado = {}
        for usuario, stats in usuarios_stats.items():
            promedio = stats['suma_atencion'] / stats['total_registros']
            resultado[usuario] = {
                'promedio_atencion': round(promedio, 2),
                'total_registros': stats['total_registros'],
                'atencion_maxima': max(stats['niveles']),
                'atencion_minima': min(stats['niveles'])
            }
        
        return {"estadisticas": resultado}
        
    except Exception as e:
        return {"error": str(e)} 
    
# Ejecutamos todo
if __name__ == "__main__":
    try:
        print("🎥 Iniciando sistema de múltiples cámaras...")
        
        # Inicializar cámaras
        inicializar_camaras()
        
        # Iniciar procesamiento de cámaras
        camera_threads = procesar_todas_camaras()
        
        # Iniciar guardado automático
        save_thread = threading.Thread(target=guardar_datos_periodicamente)
        save_thread.daemon = True
        save_thread.start()
        
        print("✅ Sistema iniciado. Cámaras activas:")
        for cam_id, config in CAMARAS.items():
            if capturas.get(cam_id):
                print(f"   - {cam_id}: {config['usuario']} ({config['posicion']})")
        
        # Inicia Flask
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        print("🔴 Cerrando aplicación...")
        cleanup()
    except Exception as e:
        print(f"❌ Error: {e}")
        cleanup()

def cleanup():
    """Limpieza al cerrar la aplicación"""
    for cam_id, cap in capturas.items():
        if cap:
            cap.release()
    cv2.destroyAllWindows()