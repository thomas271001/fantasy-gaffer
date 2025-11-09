

# ⚽ Fantasy Premier League Lineup Prediction System

![Streamlit](https://img.shields.io/badge/Deployed%20on-Streamlit-blue)
![Python](https://img.shields.io/badge/Built%20with-Python%203.10+-brightgreen)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

### 🔗 Live App  
👉 [Fantasy Gaffer — FPL Lineup Explorer](https://fantasy-gaffer-fpl.streamlit.app)

---

## 🧩 Overview

This project is an **AI-powered Fantasy Premier League (FPL) lineup prediction and optimization system**.  
It predicts player performances for upcoming English Premier League gameweeks using a machine learning model, then automatically selects the **optimal 15-player squad** within FPL constraints.

A **Streamlit dashboard** visualizes model predictions, recommended lineups, and player value analysis.

---

## 🎯 Objectives

1. **Predict Expected Points** — for each player in the next gameweek.  
2. **Optimize Squad Selection** — under the £100M budget, positional and team constraints.  
3. **Automate Weekly Updates** — using GitHub Actions for CI/CD.

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | LightGBM, scikit-learn |
| Optimization | PuLP (Linear Programming) |
| Visualization | Plotly, Streamlit |
| Automation | GitHub Actions |
| Deployment | Streamlit Cloud |

---

## 🏗️ Project Architecture

```text
📁 fpl-pipeline/
├── data/                         # generated CSVs (predictions, squad)
├── run_fpl_pipeline.py           # main ML + optimization pipeline
├── app_streamlit.py              # Streamlit dashboard
├── .github/workflows/weekly.yml  # GitHub Actions CI/CD
├── requirements.txt
└── README.md
````

---

## 📡 Data Pipeline

Data is fetched directly from the **official FPL API**:

* `bootstrap-static` → all players, teams, positions
* `element-summary/{player_id}` → per-player gameweek history

### 🧮 Feature Engineering

For each player, rolling averages are calculated:

* `pts_roll3`: average points over last 3 matches
* `min_roll3`: average minutes
* `goals_roll3`: average goals
* `value`: current player price

---

## 🤖 Model Training

A **LightGBM Regressor** is trained using `TimeSeriesSplit` to handle sequential FPL data.

**Objective:** regression (L1/MAE loss)
**Features:** rolling averages, position, team, value, recent form
**Target:** next gameweek points

After training, predictions are generated for all active players.

---

## 🧠 Squad Optimization

A **Linear Programming model** (PuLP) selects the optimal 15-man squad:

**Objective:**
Maximize Σ(predicted_points × player_selected)

**Constraints:**

* Total cost ≤ 100M
* Squad size = 15
* 2 GK, 5 DEF, 5 MID, 3 FWD
* ≤ 3 players per team

The CBC solver finds the best valid combination automatically.

---

## 🖥️ Streamlit Dashboard

### 🔗 [Live Dashboard](https://fantasy-gaffer-fpl.streamlit.app)

Features:

* Interactive player table with filters (position, team, points)
* Optimal 15-man squad and **auto-suggested starting XI**
* Visual analytics: *Predicted Points vs Player Value*
* Download CSVs for both predictions and optimized squads

---

## 🔁 Automation — GitHub Actions

The project runs automatically every week via CI/CD:

1. Fetches new FPL data
2. Retrains the LightGBM model
3. Generates updated predictions and squad
4. Commits the CSVs to the repository
5. Streamlit automatically redeploys the dashboard

You can trigger it manually from the **Actions** tab on GitHub.

---

## ⚡ Setup Instructions (Local or Colab)

### 🧰 1. Clone the repository

```bash
git clone https://github.com/thomas271001/fpl-pipeline.git
cd fpl-pipeline
```

### 🐍 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 🧠 3. Run the prediction pipeline

```bash
python run_fpl_pipeline.py
```

This generates:

```
data/predictions_next_gw.csv
data/selected_squad.csv
```

### 🖥️ 4. Launch Streamlit app

```bash
streamlit run app_streamlit.py
```

The dashboard will open at:

```
http://localhost:8501
```

---

## 📊 Example Output

| Name     | Team    | Position | Value | Predicted Points |
| -------- | ------- | -------- | ----- | ---------------- |
| Gabriel  | Arsenal | DEF      | 6.6   | 6.96             |
| Rice     | Arsenal | MID      | 6.8   | 5.21             |
| Raya     | Arsenal | GK       | 5.9   | 2.90             |
| Trossard | Arsenal | MID      | 6.9   | 3.64             |

---

## 🧩 Future Enhancements

* Add **start/minutes probability** model
* Handle **double gameweeks**
* Integrate **fixture difficulty** & opponent stats
* Add **captaincy simulation**
* Historical **backtesting** of predicted vs actual points

---

## 🧠 Learnings

* Time series feature engineering for sports data
* Optimization under combinatorial constraints
* Automating ML workflows with GitHub Actions
* Deploying real-time dashboards using Streamlit

---

## 👤 Author

**Thomas**
Machine Learning & Data Science Enthusiast
🔗 [LinkedIn](https://www.linkedin.com/in/thomas271001/)
📧 [thomas.anto.moothedan@gmail.com](mailto:thomas.anto.moothedan@gmail.com) 

---

## 📝 License

This project is released under the **MIT License** — feel free to fork and build upon it.

---

### 🌟 Support

If you like this project, consider giving it a ⭐ on GitHub — it helps others find it and encourages further development!

