import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# --- CONFIGURACIÓN DE MEDIAPIPE ---
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1) # Usamos 1 mano principal para pintar
detector = vision.HandLandmarker.create_from_options(options)

# Captura el video de la webcam
cap = cv2.VideoCapture(0)

# Lienzo (640x480)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Variables de control
prev_x, prev_y = 0, 0
draw_color = (0, 0, 255) # Empieza en Rojo

colors = [
    (0, 0, 255),    # Rojo
    (0, 255, 0),    # Verde
    (255, 0, 0),    # Azul
    (0, 255, 255)   # Amarillo
]
color_index = 0

# Variables para evitar que los gestos se repitan súper rápido (Cooldown)
last_action_time = 0
cooldown_duration = 0.8 # Esperar casi 1 segundo antes de permitir otro cambio de color o borrado

# Función para contar qué dedos están levantados
def get_raised_fingers(hand_landmarks):
    # Puntas de los dedos (índice, medio, anular, meñique)
    tips = [8, 12, 16, 20]
    fingers = []
    
    # Comparamos la punta (tip) con la falange media (tip - 2)
    for tip in tips:
        # En coordenadas de imagen, un valor de Y menor significa que está más arriba
        fingers.append(hand_landmarks[tip].y < hand_landmarks[tip - 2].y)
        
    return fingers # Retorna una lista de True/False (ej: [True, False, False, False] para el índice)

# Bucle principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Espejo
    frame_copy = frame.copy()
    h, w, _ = frame.shape

    # Preparar imagen para MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Procesar detección de manos
    result = detector.detect(mp_image)

    current_time = time.time()

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        # Coordenadas de la punta del dedo índice (ID 8)
        cx = int(landmarks[8].x * w)
        cy = int(landmarks[8].y * h)

        # Obtenemos el estado de los dedos [Índice, Medio, Anular, Meñique]
        raised = get_raised_fingers(landmarks)
        finger_count = sum(raised)

        # --- GESTO 1: SÓLO DEDO ÍNDICE LEVANTADO (DIBUJAR) ---
        if raised[0] == True and raised[1] == False and raised[2] == False and raised[3] == False:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy

            # Dibujamos una línea continua en el lienzo
            cv2.line(canvas, (prev_x, prev_y), (cx, cy), draw_color, 6)
            prev_x, prev_y = cx, cy
            
            # Dibujamos un indicador visual en el dedo
            cv2.circle(frame_copy, (cx, cy), 12, draw_color, cv2.FILLED)

        # --- GESTO 2: ÍNDICE Y MEDIO LEVANTADOS (CAMBIAR DE COLOR con Cooldown) ---
        elif raised[0] == True and raised[1] == True and raised[2] == False and raised[3] == False:
            prev_x, prev_y = 0, 0 # Reseteamos trazo para no conectar líneas al cambiar color
            
            if current_time - last_action_time > cooldown_duration:
                color_index = (color_index + 1) % len(colors)
                draw_color = colors[color_index]
                last_action_time = current_time
                print(f"Color cambiado! Siguiente color ID: {color_index}")
                
            # Círculo blanco para indicar que estás en modo de selección de color
            cv2.circle(frame_copy, (cx, cy), 15, (255, 255, 255), cv2.FILLED)

        # --- GESTO 3: TODOS LOS DEDOS LEVANTADOS (BORRAR LIENZO con Cooldown) ---
        elif finger_count == 4:
            prev_x, prev_y = 0, 0
            if current_time - last_action_time > cooldown_duration:
                canvas = np.zeros_like(canvas) # Limpiar lienzo
                last_action_time = current_time
                print("Lienzo borrado!")

        # Si haces cualquier otra pose, detén el trazo de dibujo
        else:
            prev_x, prev_y = 0, 0

        # Dibujar los puntos de la mano en pantalla para saber que te está leyendo bien
        for lm in landmarks:
            lx, ly = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_copy, (lx, ly), 4, (0, 255, 0), cv2.FILLED)

    else:
        # Si la cámara no ve ninguna mano, reseteamos las coordenadas del trazo
        prev_x, prev_y = 0, 0

    # Mezclamos la cámara real con el lienzo de dibujo de manera más clara (0.8 y 1.0)
    combined = cv2.addWeighted(frame_copy, 0.8, canvas, 1.0, 0)

    # Mostrar la interfaz del programa
    cv2.imshow("App para Pintar", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()