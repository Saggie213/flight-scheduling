# ✈️ Flight Scheduling Analysis and AI-Driven Decision Support

## 📌 Problem Statement
Mumbai (CSMIA) and Delhi (IGIA) airports are among the busiest in India.  
Due to **capacity constraints, weather disruptions, and peak-hour clustering**, flight scheduling often suffers from cascading delays.  

This project analyzes real-world flight data, identifies congestion patterns, and builds **AI-driven tools** to support controllers and operators in making better scheduling decisions.

---

## 🚀 Features
- 📊 **Traffic & Delay Analysis** – Detects busiest hours and quantifies delay severity.  
- 🤖 **Predictive Delay Model** – Linear regression baseline to estimate departure delays.  
- 🕒 **Schedule-Tuning Tool** – Suggests alternative slots to reduce congestion.  
- 💬 **NLP Query Interface** – Natural language queries to explore patterns and get insights.  
- 📈 **Visualization Dashboards** – Flight counts by hour, average delays, delay distributions.

---

## 🛠️ Tech Stack
- **Python** – Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Next.js (TypeScript)** – API + UI layer  
- **React + Tailwind CSS + shadcn/ui** – Interactive dashboards  
- **SQLite + Prisma** – Data persistence  
- **Open-Source Flight Data** – Flightradar24, FlightAware  

---

## 📂 Project Structure
flight-scheduling-ai/
├── analysis.py # Python script (data cleaning, modeling, tuning)
├── flight_data_cleaned_full.csv # Cleaned dataset
├── report.pdf # Detailed project report
├── requirements.txt # Python dependencies
└── README.md # Project documentation
