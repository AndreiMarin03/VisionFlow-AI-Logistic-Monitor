import streamlit as st
import sqlite3
import pandas as pd
import time

# Configurarea paginii: "wide" permite folosirea întregii lățimi a ecranului pentru graficele de monitorizare
st.set_page_config(page_title="VisionCore AI Dashboard", layout="wide")

st.title("📊 VisionCore - Intelligent Logistics Dashboard")
st.markdown("Interfață de monitorizare în timp real pentru sistemul autonom de sortare și inventariere.")
st.markdown("---")


# Funcție pentru extragerea datelor din baza de date SQLite
def load_data():
    try:
        # Stabilim conexiunea cu baza de date partajată cu motorul de AI
        conn = sqlite3.connect("warehouse_inventory.db")

        # Extragem datele din tabelul specific AI, ordonate descrescător după ID pentru a avea noutățile sus
        # Utilizăm Pandas pentru a converti rezultatul SQL direct într-un format tabelar (DataFrame)
        df = pd.read_sql_query("SELECT * FROM detections_ai ORDER BY id DESC", conn)

        conn.close()
        return df
    except Exception as e:
        # Returnăm un obiect gol dacă tabelul nu a fost încă creat de scriptul principal
        return pd.DataFrame()


# Utilizăm un placeholder pentru a permite actualizarea conținutului fără a reîncărca întreaga pagină (fără flicker)
placeholder = st.empty()

# --- BUCLA DE MONITORIZARE LIVE ---
# Această buclă rulează continuu, transformând aplicația într-un tablou de bord industrial (Real-time Dashboard)
while True:
    # Citim versiunea curentă a baze de date
    df = load_data()

    with placeholder.container():
        if not df.empty:
            # Indicatori de tip KPI (Key Performance Indicators) pentru o vizualizare rapidă a volumului de muncă
            st.metric("Unități Inventariate Total", len(df))

            # Organizăm vizualizarea pe două coloane pentru o ergonomie mai bună a datelor
            c1, c2 = st.columns(2)

            with c1:
                st.write("### 📊 Distribuție Categorii (AI Analytics)")
                # Agregăm datele pentru a vedea frecvența fiecărui tip de obiect (ex: Person vs Car)
                counts = df['object_type'].value_counts()
                st.bar_chart(counts, color="#FF4B4B")  # Roșu Nokia-style pentru identitate vizuală

            with c2:
                st.write("### 📈 Stabilitate Model (Confidence Score)")
                # Monitorizăm ultimele 20 de detecții pentru a evalua cât de sigur este AI-ul pe deciziile sale
                # Un scor de încredere constant ridicat indică un sistem bine calibrat
                df_chart = df.head(20).copy()
                st.line_chart(df_chart.set_index('timestamp')['confidence'])

            # Secțiunea de jurnal (Log-ul brut): util pentru audit și trasabilitatea coletelor/persoanelor
            st.write("### 📋 Jurnal Detecții - Istoric Recent (Top 100)")
            st.dataframe(df.head(100), use_container_width=True, height=300)

        else:
            # Mesaj de feedback în cazul în care motorul de AI nu trimite date
            st.warning(
                "Sistemul de monitorizare este activ, dar nu recepționează date. Verifică motorul VisionCore AI!")

    # Interval de refresh de 2 secunde: echilibru între latență scăzută și consum redus de resurse (I/O)
    time.sleep(2)