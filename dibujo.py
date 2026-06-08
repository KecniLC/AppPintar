import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os

# --- CONFIGURACIÓN DE MEDIAPIPE ---
model_path = "hand_landmarker.task"

if not os.path.exists(model_path):
    print(f"ERROR: No se encuentra el archivo '{model_path}'.")
    print("Descárgalo de: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models")
    exit()

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO, # Modo optimizado para video en tiempo real
    num_hands=2,                          # Aumentamos a 2 manos
    min_hand_detection_confidence=0.8, # Mayor confianza inicial para evitar errores
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8       # Mayor estabilidad en el seguimiento del movimiento
)
detector = vision.HandLandmarker.create_from_options(options)

# Captura el video de la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: No se pudo acceder a la cámara.")
    exit()

# Inicializamos el lienzo como None para crearlo con el tamaño correcto de la cámara
canvas = None

# --- CONFIGURACIÓN DE ELEMENTOS "ANIME" ---
# Intenta cargar una imagen (asegúrate de tener un archivo 'personaje.png' en la carpeta)
anime_sprite = cv2.imread("personaje.png", cv2.IMREAD_UNCHANGED)
if anime_sprite is not None:
    anime_sprite = cv2.resize(anime_sprite, (100, 100)) # Ajustar tamaño

def overlay_image(background, overlay, x, y):
    """Superpone una imagen PNG con transparencia sobre un fondo."""
    h, w = overlay.shape[:2]
    # Calcular coordenadas para centrar la imagen en la punta del dedo
    x_offset, y_offset = x - w // 2, y - h // 2
    
    # Verificar límites de la pantalla
    if x_offset < 0 or y_offset < 0 or x_offset + w > background.shape[1] or y_offset + h > background.shape[0]:
        return background

    # Extraer canales
    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3] / 255.0  # Canal Alfa

    # Mezclar imágenes
    for c in range(0, 3):
        background[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
            (mask * overlay_img[:, :, c] + (1 - mask) * background[y_offset:y_offset+h, x_offset:x_offset+w, c])
    return background

colors = [
    (0, 0, 255),    # Rojo
    (0, 255, 0),    # Verde
    (255, 0, 0),    # Azul
    (0, 255, 255)   # Amarillo
]

# Estado independiente para cada mano
hand_states = {
    "Left": {
        "prev_x": 0, "prev_y": 0, "smooth_x": 0, "smooth_y": 0,
        "color_idx": 0, "last_time": 0
    },
    "Right": {
        "prev_x": 0, "prev_y": 0, "smooth_x": 0, "smooth_y": 0,
        "color_idx": 2, "last_time": 0 # La derecha empieza en azul por ejemplo
    }
}

cooldown_duration = 0.8 # Esperar casi 1 segundo antes de permitir otro cambio de color o borrado

# Función para contar qué dedos están levantados
def get_raised_fingers(hand_landmarks):
    # Puntas: Pulgar(4), Índice(8), Medio(12), Anular(16), Meñique(20)
    tips = [4, 8, 12, 16, 20]
    fingers = []
    
    for i, tip in enumerate(tips):
        if i == 0: # Lógica especial para el Pulgar
            # Si el pulgar está a la derecha o izquierda de su base (depende de la mano)
            # Simplificado: comparamos x del tip con x de la base del pulgar
            # Un método más robusto es comparar la distancia del pulgar al meñique
            dist_thumb_pinky = abs(hand_landmarks[4].x - hand_landmarks[17].x)
            dist_base_pinky = abs(hand_landmarks[2].x - hand_landmarks[17].x)
            fingers.append(dist_thumb_pinky > dist_base_pinky)
        else:
            # Otros dedos: comparar punta con la falange media
            fingers.append(hand_landmarks[tip].y < hand_landmarks[tip - 2].y)
        
    return fingers # Retorna [Pulgar, Índice, Medio, Anular, Meñique]

def draw_ui(img):
    # Mostrar el color de cada mano en las esquinas superiores
    # Mano Izquierda (L)
    cv2.rectangle(img, (10, 10), (100, 50), (255, 255, 255), -1)
    cv2.putText(img, "L", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[hand_states["Left"]["color_idx"]], 3)
    # Mano Derecha (R)
    cv2.rectangle(img, (w - 110, 10), (w - 10, 50), (255, 255, 255), -1)
    cv2.putText(img, "R", (w - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[hand_states["Right"]["color_idx"]], 3)

start_time_ms = int(time.time() * 1000)

# Bucle principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Espejo
    frame_copy = frame.copy()
    h, w, _ = frame.shape

    # Si el lienzo no existe o la cámara cambió de resolución, lo creamos/ajustamos
    if canvas is None or canvas.shape[:2] != (h, w):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Preparar imagen para MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Procesar detección de manos con timestamp para modo VIDEO
    frame_timestamp_ms = int(time.time() * 1000) - start_time_ms
    result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    current_time = time.time()

    # Iteramos sobre todas las manos detectadas en lugar de solo la primera
    active_labels = []
    for i, landmarks in enumerate(result.hand_landmarks):
        # Identificar si es la mano izquierda o derecha según MediaPipe
        label = result.handedness[i][0].category_name 
        active_labels.append(label)
        state = hand_states[label]

        cx = int(landmarks[8].x * w)
        cy = int(landmarks[8].y * h)

        # Obtenemos el estado de los 5 dedos
        raised = get_raised_fingers(landmarks)
        finger_count = sum(raised)

        # --- GESTO 1: SÓLO DEDO ÍNDICE (raised[1]) LEVANTADO ---
        if raised[1] == True and finger_count == 1:
            if state["prev_x"] == 0 and state["prev_y"] == 0:
                state["prev_x"], state["prev_y"] = cx, cy
                state["smooth_x"], state["smooth_y"] = cx, cy

            alpha = 0.5
            state["smooth_x"] = int(alpha * cx + (1 - alpha) * state["smooth_x"])
            state["smooth_y"] = int(alpha * cy + (1 - alpha) * state["smooth_y"])
            
            current_color = colors[state["color_idx"]]
            if abs(state["smooth_x"] - state["prev_x"]) > 1 or abs(state["smooth_y"] - state["prev_y"]) > 1:
                cv2.line(canvas, (state["prev_x"], state["prev_y"]), (state["smooth_x"], state["smooth_y"]), current_color, 6)
                state["prev_x"], state["prev_y"] = state["smooth_x"], state["smooth_y"]

            # Si tenemos un sprite de anime, lo usamos como cursor
            if anime_sprite is not None:
                frame_copy = overlay_image(frame_copy, anime_sprite, state["smooth_x"], state["smooth_y"])
            else:
                # Cursor por defecto si no hay imagen
                cv2.circle(frame_copy, (state["smooth_x"], state["smooth_y"]), 12, current_color, cv2.FILLED)

        # --- GESTO 2: ÍNDICE Y MEDIO LEVANTADOS (Selección de Color) ---
        elif raised[1] == True and raised[2] == True and finger_count == 2:
            state["prev_x"], state["prev_y"] = 0, 0
            
            if current_time - state["last_time"] > cooldown_duration:
                state["color_idx"] = (state["color_idx"] + 1) % len(colors)
                state["last_time"] = current_time
                print(f"Mano {label} cambió a color {state['color_idx']}")
                
            cv2.circle(frame_copy, (cx, cy), 15, (255, 255, 255), cv2.FILLED)

        # --- GESTO 3: PALMA ABIERTA (5 DEDOS) PARA BORRAR ---
        elif finger_count == 5:
            state["prev_x"], state["prev_y"] = 0, 0
            if current_time - state["last_time"] > cooldown_duration:
                canvas = np.zeros_like(canvas) # Limpiar lienzo
                state["last_time"] = current_time
                print(f"Lienzo borrado por mano {label}!")

        else:
            state["prev_x"], state["prev_y"] = 0, 0

        for lm in landmarks:
            lx, ly = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_copy, (lx, ly), 4, (0, 255, 0), cv2.FILLED)

    # Resetear manos que salieron de cuadro
    for label in hand_states:
        if label not in active_labels:
            hand_states[label]["prev_x"] = 0
            hand_states[label]["prev_y"] = 0

    # Mezclamos la cámara real con el lienzo de dibujo de manera más clara (0.8 y 1.0)
    combined = cv2.addWeighted(frame_copy, 0.8, canvas, 1.0, 0)
    draw_ui(combined)

    # Mostrar la interfaz del programa
    cv2.imshow("App para Pintar", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()