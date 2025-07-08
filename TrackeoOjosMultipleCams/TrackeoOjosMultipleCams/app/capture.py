import cv2
import mediapipe as mp
import time
import json
import threading
from datetime import datetime
from flask import Flask, Response, jsonify
import os
import numpy as np

# Flask app para servir video
app = Flask(__name__)

# Variables globales para controlar el estado del sistema
sistema_activo = False
threads_camaras = []
thread_guardado = None
stop_event = threading.Event()

# Configuración de múltiples cámaras
CAMARAS = {
    "camara1": {
        "url": "http://192.168.100.117:4747/video",
        "usuario": "camara1",
        "posicion": "Escritorio 1"
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

# Contador global de parpadeos por usuario
parpadeo_contador = {}
ultimo_ear = {}

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

def crear_directorio_datos():
    """Crear directorio de datos dentro del proyecto"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    
    try:
        os.makedirs(data_dir, exist_ok=True)
        if os.path.exists(data_dir):
            print(f"✅ Directorio confirmado: {data_dir}")
        else:
            print(f"❌ Error: No se pudo crear el directorio")
    except Exception as e:
        print(f"❌ Error creando directorio: {e}")
        data_dir = os.path.join(os.getcwd(), "temp_data")
        os.makedirs(data_dir, exist_ok=True)
        print(f"🔄 Usando directorio alternativo: {data_dir}")
    
    return data_dir

def procesar_camara_controlado(cam_id):
    """Procesa una cámara específica con control de parada"""
    global last_frames, attention_data, stop_event
    
    if cam_id not in CAMARAS or not capturas.get(cam_id):
        return
    
    config = CAMARAS[cam_id]
    usuario = config['usuario']
    posicion = config['posicion']
    cap = capturas[cam_id]
    
    frame_skip = 0
    error_count = 0
    
    print(f"🎬 Iniciando procesamiento de {cam_id}")
    
    try:
        while not stop_event.is_set():
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
                        time.sleep(1)
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
            
            last_frames[cam_id] = frame.copy()
            time.sleep(0.01)
    except Exception as e:
        print(f"❌ Error en procesamiento de {cam_id}: {e}")
    finally:
        print(f"🛑 Procesamiento de {cam_id} detenido")

def procesar_todas_camaras_controlado():
    """Inicia el procesamiento controlado de todas las cámaras"""
    threads = []
    
    for cam_id in CAMARAS.keys():
        if capturas.get(cam_id):
            thread = threading.Thread(target=procesar_camara_controlado, args=(cam_id,))
            thread.daemon = True
            threads.append(thread)
            thread.start()
            print(f"🚀 Thread iniciado para {cam_id}")
    
    return threads

def guardar_datos_periodicamente_controlado():
    """Guarda datos cada 5 segundos con control de parada"""
    global attention_data, stop_event
    
    print("💾 Servicio de guardado iniciado")
    
    try:
        while not stop_event.is_set():
            if stop_event.wait(5):
                break
            
            if attention_data:
                data_dir = crear_directorio_datos()
                log_file = os.path.join(data_dir, "attention_logs.json")
                
                try:
                    with open(log_file, "a", encoding='utf-8') as f:
                        for row in attention_data:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    
                    print(f"💾 {len(attention_data)} registros guardados de {len(set(r['usuario'] for r in attention_data))} usuarios")
                    
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
    except Exception as e:
        print(f"❌ Error en servicio de guardado: {e}")
    finally:
        print("🛑 Servicio de guardado detenido")

def iniciar_sistema():
    """Inicia el sistema de captura"""
    global sistema_activo, threads_camaras, thread_guardado, stop_event
    
    if sistema_activo:
        return {"status": "error", "mensaje": "El sistema ya está activo"}
    
    try:
        print("🎥 Iniciando sistema de captura...")
        
        stop_event.clear()
        inicializar_camaras()
        
        camaras_activas = sum(1 for cap in capturas.values() if cap is not None)
        
        if camaras_activas == 0:
            return {"status": "error", "mensaje": "No hay cámaras disponibles"}
        
        threads_camaras = procesar_todas_camaras_controlado()
        
        thread_guardado = threading.Thread(target=guardar_datos_periodicamente_controlado)
        thread_guardado.daemon = True
        thread_guardado.start()
        
        sistema_activo = True
        
        print(f"✅ Sistema iniciado con {camaras_activas} cámara(s) activa(s)")
        
        return {
            "status": "success", 
            "mensaje": f"Sistema iniciado correctamente con {camaras_activas} cámara(s)",
            "camaras_activas": camaras_activas
        }
        
    except Exception as e:
        print(f"❌ Error iniciando sistema: {e}")
        return {"status": "error", "mensaje": f"Error al iniciar sistema: {str(e)}"}

def detener_captura_solamente():
    """Detiene solo la captura de video, mantiene el servidor activo"""
    global sistema_activo, stop_event
    
    try:
        print("🔴 Deteniendo solo la captura de video...")
        
        stop_event.set()
        sistema_activo = False
        
        # Esperar un poco para que los threads se detengan
        time.sleep(2)
        
        for cam_id, cap in list(capturas.items()):
            if cap:
                try:
                    cap.release()
                    print(f"📹 Cámara {cam_id} liberada")
                except Exception as e:
                    print(f"⚠️ Error liberando cámara {cam_id}: {e}")
        
        capturas.clear()
        last_frames.clear()
        
        if attention_data:
            try:
                data_dir = crear_directorio_datos()
                log_file = os.path.join(data_dir, "attention_logs.json")
                with open(log_file, "a", encoding='utf-8') as f:
                    for row in attention_data:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"💾 {len(attention_data)} registros finales guardados")
            except Exception as e:
                print(f"⚠️ Error guardando datos finales: {e}")
            
            attention_data.clear()
        
        print("✅ Captura detenida, servidor sigue activo")
        
        return {
            "status": "success", 
            "mensaje": "Captura detenida correctamente, servidor activo"
        }
        
    except Exception as e:
        print(f"❌ Error deteniendo captura: {e}")
        sistema_activo = False
        stop_event.set()
        capturas.clear()
        last_frames.clear()
        
        return {
            "status": "success", 
            "mensaje": f"Captura forzada a detenerse: {str(e)}"
        }

def cleanup_seguro():
    """Limpieza segura al cerrar la aplicación"""
    try:
        print("🧹 Iniciando limpieza segura...")
        detener_captura_solamente()
        time.sleep(1)
        cv2.destroyAllWindows()
        print("✅ Limpieza completada")
    except Exception as e:
        print(f"⚠️ Error en limpieza: {e}")

# RUTAS FLASK
@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    """Feed de video para una cámara específica"""
    def generar_frames():
        while True:
            if last_frames.get(cam_id) is None:
                time.sleep(0.1)
                continue
            
            frame = last_frames[cam_id]
            
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

@app.route('/sistema/iniciar', methods=['POST'])
def iniciar_captura():
    """Servicio para iniciar la captura"""
    try:
        resultado = iniciar_sistema()
        return jsonify(resultado), 200 if resultado["status"] == "success" else 400
    except Exception as e:
        return jsonify({"status": "error", "mensaje": f"Error interno: {str(e)}"}), 500

@app.route('/sistema/detener', methods=['POST'])
def detener_captura():
    """Servicio para detener solo la captura"""
    try:
        resultado = detener_captura_solamente()
        return jsonify(resultado), 200
    except Exception as e:
        return jsonify({"status": "error", "mensaje": f"Error interno: {str(e)}"}), 500

@app.route('/sistema/reiniciar', methods=['POST'])
def reiniciar_sistema():
    """Servicio para reiniciar solo la captura"""
    try:
        print("🔄 Reiniciando captura...")
        
        resultado_detener = detener_captura_solamente()
        print(f"Resultado detener captura: {resultado_detener}")
        
        time.sleep(3)
        
        global stop_event
        stop_event = threading.Event()
        
        resultado_iniciar = iniciar_sistema()
        print(f"Resultado iniciar: {resultado_iniciar}")
        
        if resultado_iniciar["status"] == "success":
            return jsonify({
                "status": "success",
                "mensaje": f"Captura reiniciada: {resultado_iniciar['mensaje']}",
                "detener": resultado_detener["mensaje"],
                "iniciar": resultado_iniciar["mensaje"]
            }), 200
        else:
            return jsonify({
                "status": "error",
                "mensaje": f"Error reiniciando captura: {resultado_iniciar['mensaje']}",
                "detener": resultado_detener["mensaje"],
                "iniciar": resultado_iniciar["mensaje"]
            }), 400
            
    except Exception as e:
        print(f"❌ Error en reinicio: {e}")
        return jsonify({
            "status": "error",
            "mensaje": f"Error crítico en reinicio: {str(e)}"
        }), 500

@app.route('/sistema/estado', methods=['GET'])
def estado_sistema():
    """Servicio para consultar el estado del sistema"""
    try:
        camaras_activas = sum(1 for cap in capturas.values() if cap is not None)
        
        return jsonify({
            "sistema_activo": sistema_activo,
            "servidor_activo": True,
            "camaras_configuradas": len(CAMARAS),
            "camaras_activas": camaras_activas,
            "datos_pendientes": len(attention_data),
            "threads_activos": threading.active_count(),
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "servidor_activo": True
        }), 200

@app.route('/')
def index():
    """Página principal con información de servicios"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sistema de Monitoreo de Atención</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
            .start { background: #4CAF50; color: white; }
            .stop { background: #f44336; color: white; }
            .restart { background: #2196F3; color: white; }
            .info { background: #FF9800; color: white; }
            #resultado { margin-top: 20px; padding: 10px; background: #f5f5f5; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🎥 Sistema de Monitoreo de Atención</h1>
        <h2>Estado del Sistema</h2>
        <p><strong>Servidor:</strong> <span style="color: green;">Activo ✅</span></p>
        <p><strong>Captura:</strong> <span id="estado">Cargando...</span></p>
        <p><strong>Cámaras Activas:</strong> <span id="camaras">Cargando...</span></p>
        <p><strong>Threads Activos:</strong> <span id="threads">Cargando...</span></p>
        
        <h2>Controles de Captura</h2>
        <button class="start" onclick="iniciarSistema()">🟢 Iniciar Captura</button>
        <button class="stop" onclick="detenerSistema()">🔴 Detener Captura</button>
        <button class="restart" onclick="reiniciarSistema()">🔄 Reiniciar Captura</button>
        <button class="info" onclick="actualizarEstado()">🔍 Actualizar Estado</button>
        
        <h2>APIs Disponibles</h2>
        <ul>
            <li><strong>POST</strong> /sistema/iniciar - Iniciar captura</li>
            <li><strong>POST</strong> /sistema/detener - Detener captura (mantiene servidor)</li>
            <li><strong>POST</strong> /sistema/reiniciar - Reiniciar captura</li>
            <li><strong>GET</strong> /sistema/estado - Estado del sistema</li>
            <li><strong>GET</strong> /camaras - Lista de cámaras</li>
            <li><strong>GET</strong> /estadisticas - Estadísticas de atención</li>
            <li><strong>GET</strong> /video_feed/&lt;cam_id&gt; - Video en vivo</li>
        </ul>
        
        <div id="resultado"></div>
        
        <script>
            function actualizarEstado() {
                fetch('/sistema/estado')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('estado').innerHTML = data.sistema_activo ? 
                            '<span style="color: green;">Activa ✅</span>' : 
                            '<span style="color: red;">Inactiva ❌</span>';
                        document.getElementById('camaras').textContent = data.camaras_activas + '/' + data.camaras_configuradas;
                        document.getElementById('threads').textContent = data.threads_activos;
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        document.getElementById('estado').innerHTML = '<span style="color: red;">Error ❌</span>';
                    });
            }
            
            function iniciarSistema() {
                fetch('/sistema/iniciar', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('resultado').innerHTML = '<strong>Iniciar Captura:</strong> ' + data.mensaje;
                        actualizarEstado();
                    })
                    .catch(error => {
                        document.getElementById('resultado').innerHTML = '<strong>Error:</strong> ' + error;
                    });
            }
            
            function detenerSistema() {
                fetch('/sistema/detener', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('resultado').innerHTML = '<strong>Detener Captura:</strong> ' + data.mensaje;
                        actualizarEstado();
                    })
                    .catch(error => {
                        document.getElementById('resultado').innerHTML = '<strong>Error:</strong> ' + error;
                    });
            }
            
            function reiniciarSistema() {
                fetch('/sistema/reiniciar', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('resultado').innerHTML = '<strong>Reiniciar Captura:</strong> ' + data.mensaje;
                        actualizarEstado();
                    })
                    .catch(error => {
                        document.getElementById('resultado').innerHTML = '<strong>Error:</strong> ' + error;
                    });
            }
            
            // Actualizar estado cada 5 segundos
            setInterval(actualizarEstado, 5000);
            
            // Cargar estado inicial
            actualizarEstado();
        </script>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    try:
        print("🎥 Sistema de Monitoreo de Atención - Múltiples Cámaras")
        print("=" * 60)
        print("✅ Servidor iniciado en modo manual")
        print("   Usa los servicios REST para controlar la captura:")
        print("   - POST /sistema/iniciar - Para iniciar captura")
        print("   - POST /sistema/detener - Para detener captura (servidor activo)")
        print("   - GET /sistema/estado - Para ver el estado")
        print(f"\n🌐 Interfaz web: http://localhost:5000")
        print("\n⚠️  Presiona Ctrl+C para detener el servidor")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n🔴 Cerrando servidor...")
        if sistema_activo:
            detener_captura_solamente()
        cleanup_seguro()
    except Exception as e:
        print(f"❌ Error: {e}")
        if sistema_activo:
            detener_captura_solamente()
        cleanup_seguro()