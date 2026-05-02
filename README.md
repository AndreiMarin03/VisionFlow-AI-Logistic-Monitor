# VisionFlow AI - Intelligent Logistics Monitor

An end-to-end Computer Vision system designed for autonomous logistics hubs, featuring real-time object detection and inventory management.

## 🚀 Features
- **Dual Processing Engines**: Includes both a classic OpenCV approach (Canny Edge/Contours) and a Deep Learning approach (YOLOv8).
- **Automated Inventory**: Objects detected in the central "Storage Zone" are automatically logged into an SQLite database.
- **Live Dashboard**: A real-time web interface built with Streamlit to monitor flow statistics and AI confidence scores.

## 🛠️ Tech Stack
- **Languages**: Python
- **AI/Vision**: OpenCV, Ultralytics YOLOv8
- **Data**: SQLite, Pandas
- **Visualization**: Streamlit, Plotly

## 📦 Installation & Usage
1. Clone the repository.
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Run the AI Engine:  
   `python vision_engine_ai.py`
4. Launch the Dashboard:  
   `streamlit run dashboard.py`
