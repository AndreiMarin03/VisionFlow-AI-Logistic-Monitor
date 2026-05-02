import cv2
import sqlite3
from datetime import datetime


# --- CONFIGURARE BAZĂ DE DATE ---
# Inițializăm SQLite pentru a stoca istoricul detecțiilor fără a avea nevoie de un server extern
def init_db():
    conn = sqlite3.connect("warehouse_inventory.db")
    cursor = conn.cursor()
    # Aici am scris toata comanda, fara "..."
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections_classic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            action TEXT,
            area REAL
        )
    ''')
    conn.commit()
    return conn


db_conn = init_db()
db_cursor = db_conn.cursor()

# Variabilă pentru a controla frecvența logării (evităm duplicarea aceluiași obiect în DB)
last_logged_time = 0

# Încărcăm fluxul video (în cazul acesta un fișier local care simulează o cameră de supraveghere)
cap = cv2.VideoCapture("Adauga videoclipul aici")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # --- PROCESARE ROI (Region of Interest) ---
    # Decupăm zona relevantă unde se află banda transportoare/fluxul principal
    # Această optimizare reduce consumul de resurse prin ignorarea zonelor statice (tavan, margini)
    roi = frame[int(height * 0.4):height, int(width * 0.1):int(width * 0.8)]

    # --- PRELUCRARE IMAGINE ---
    # Convertim în tonuri de gri pentru a simplifica procesarea matematică
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Aplicăm Gaussian Blur pentru a elimina zgomotul (pixelii paraziți) înainte de detectarea marginilor
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Algoritmul Canny pentru detectarea marginilor obiectelor
    edges = cv2.Canny(blur, 100, 200)

    # Dilatăm marginile pentru a închide contururile incomplete (le facem mai "groase")
    dilated = cv2.dilate(edges, (5, 5), iterations=3)

    # Identificăm formele geometrice (contururile) din imaginea procesată
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    max_area = 0

    # Analizăm fiecare contur găsit pentru a filtra doar obiectele care ne interesează
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # 1. Filtru de mărime: eliminăm obiectele prea mici (zgomot) sau prea mari (erori de lumină)
        if area < 2000 or area > 50000:
            continue

        # 2. Filtru formă (Aspect Ratio): ignorăm obiectele extrem de subțiri sau lungi
        aspect_ratio = w / float(h)
        if aspect_ratio < 1 or aspect_ratio > 4:
            continue

        # 3. Filtru margini: ignorăm obiectele care "ating" marginea ROI (sunt parțial ieșite din cadru)
        # Acest lucru asigură că logăm obiectul doar când este vizibil complet
        if x < 50 or x + w > roi.shape[1] - 50:
            continue

        # Reținem cel mai mare contur valid din cadrul curent
        if area > max_area:
            max_area = area
            best_contour = contour

    # Dacă am găsit un obiect valid, trecem la logica de business
    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        # Vizualizăm obiectul detectat cu un dreptunghi verde
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Calculăm centrul obiectului pentru a determina poziția pe bandă
        obj_center = x + w // 2
        frame_center = roi.shape[1] // 2

        # Logică de sortare/poziționare
        if obj_center < frame_center - 50:
            direction = "LEFT"
        elif obj_center > frame_center + 50:
            direction = "RIGHT"
        else:
            direction = "MIDDLE"

            # --- LOGARE ÎN BAZA DE DATE ---
            # Dacă obiectul este pe centru (Ready to Store), îl salvăm în DB
            current_time = datetime.now()
            # Setăm un cooldown de 2 secunde pentru a nu loga același obiect de mai multe ori
            if (current_time.timestamp() - last_logged_time) > 2:
                # Si aici am scris toata comanda SQL
                db_cursor.execute("INSERT INTO detections_classic (timestamp, action, area) VALUES (?, ?, ?)",
                                  (current_time.strftime("%Y-%m-%d %H:%M:%S"), "OBJECT_STORED", max_area))
                db_conn.commit()
                last_logged_time = current_time.timestamp()
                print(f"[DB] Inregistrare noua: Obiect detectat pe centru.")

        # Afișăm pe video direcția curentă a obiectului
        cv2.putText(roi, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Afișăm fereastra cu rezultatul procesării și cea cu marginile detectate (pentru debug)
    cv2.imshow("VisionCore Classic - ROI", roi)
    cv2.imshow("VisionCore Debug - Canny Edges", edges)

    # Tasta ESC pentru a închide aplicația
    if cv2.waitKey(25) & 0xFF == 27:
        break

# Eliberăm resursele la final
cap.release()
db_conn.close()
cv2.destroyAllWindows()