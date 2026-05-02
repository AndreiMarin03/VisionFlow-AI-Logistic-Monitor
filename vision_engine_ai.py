import cv2
from ultralytics import YOLO
import sqlite3
from datetime import datetime

# 1. INITIALIZARE MODEL AI
# Folosim YOLOv8n (varianta Nano) - optimizat pentru viteză și procesare în timp real (Real-time Inference)
model = YOLO('yolov8n.pt')


# 2. FUNCTIE INITIALIZARE BAZA DE DATE
def init_db():
    conn = sqlite3.connect("warehouse_inventory.db")
    cursor = conn.cursor()
    # Creăm un tabel specific pentru modulul de AI pentru a păstra integritatea datelor (Schema separation)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections_ai (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            object_type TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    return conn


# Inițializăm conexiunea la DB și un timer pentru cooldown-ul logării
db_conn = init_db()
db_cursor = db_conn.cursor()
last_logged_time = 0

# 3. CAPTURĂ VIDEO
# Poate fi înlocuit cu 0 pentru camera web sau cu un stream IP în condiții industriale
cap = cv2.VideoCapture("Adauga videclipul aici")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- OPTIMIZARE ROI (Region of Interest) ---
    # Procesăm doar zona centrală a benzii/culoarului pentru a economisi resurse GPU/CPU
    height, width, _ = frame.shape
    roi = frame[int(height * 0.4):height, int(width * 0.1):int(width * 0.8)]

    # 4. DETECTIE AI CU YOLO
    # Rulăm modelul doar pe zona ROI; conf=0.5 elimină detecțiile incerte (zgomotul vizual)
    results = model(roi, conf=0.5, verbose=False)

    # Statusul "Scanning" este afișat implicit dacă niciun obiect nu îndeplinește criteriile de stocare
    current_status = "SCANNING..."

    for r in results:
        for box in r.boxes:
            # Preluăm coordonatele și datele despre clasa obiectului
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            # Analizăm poziția obiectului față de axa centrală a cadrului
            obj_center = (x1 + x2) // 2
            roi_center = roi.shape[1] // 2

            color = (0, 255, 0) # Verde pentru obiecte aflate în tranzit

            # --- LOGICĂ DE GATEKEEPING (Zona de Stocare/Înregistrare) ---
            # Dacă obiectul se află într-un interval de 70px față de centru, este gata de logare
            if abs(obj_center - roi_center) < 70:
                color = (0, 255, 255) # Schimbăm culoarea în galben (vizualizare status activ)
                current_status = f"STORING: {label.upper()}"

                current_time = datetime.now()
                # Cooldown de 3 secunde pentru a asigura că un obiect este logat o singură dată
                if (current_time.timestamp() - last_logged_time) > 3:
                    db_cursor.execute(
                        "INSERT INTO detections_ai (timestamp, object_type, confidence) VALUES (?, ?, ?)",
                        (current_time.strftime("%Y-%m-%d %H:%M:%S"), label.upper(), round(conf, 2))
                    )
                    db_conn.commit()
                    last_logged_time = current_time.timestamp()
                    print(f"[DATABASE] Inregistrare noua: {label.upper()} ({conf:.2%})")

            # Desenăm elementele vizuale pe ROI
            cv2.rectangle(roi, (x1, y1), (x2, y2), color, 2)
            cv2.putText(roi, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Afișăm statusul general al sistemului (Sincronizat cu logica de business)
    cv2.putText(roi, f"SYSTEM: {current_status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Titlul ferestrei este neutru și profesionist
    cv2.imshow("SmartFlow AI Monitor", roi)

    # Ieșire de urgență prin tasta ESC
    if cv2.waitKey(20) & 0xFF == 27:
        break

# Eliberare resurse
cap.release()
db_conn.close()
cv2.destroyAllWindows()