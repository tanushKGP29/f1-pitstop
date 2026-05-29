# 🏎️ F1 Pit Stop Prediction Strategy Center

A high-performance machine learning system built to predict whether an F1 driver will pit on their next lap. 

Utilizing a vectorized feature engineering pipeline and a blended model ensemble (**LightGBM + CatBoost + XGBoost**), the system processes live or batch telemetry and delivers real-time strategic recommendations through an interactive Streamlit dashboard and a robust FastAPI REST service.

---

## 📂 Repository Structure

The project follows a clean, decoupled production-grade directory layout:

```
f1-pitstop/
├── src/                         # Core Machine Learning Pipeline
│   ├── data_preparation.py      # Core data loader
│   ├── data_cleaning.py         # Outlier clipping & compound ordinal flags
│   ├── feature_engg.py          # Vectorized rolling & stint interaction engineering
│   ├── model_development.py     # Cross-validation baseline & OOF ensemble training
│   └── pipeline.py              # Main training execution & serialization runner
├── api/                         # REST API Interface
│   ├── main.py                  # FastAPI application exposing CSV/Single endpoints
│   ├── schemas.py               # Pydantic validation schemas
│   └── predictor.py             # Inference engine (loads GBDT assets & maps categories)
├── dashboard/                   # Interactive Dashboard Interface
│   └── app.py                   # Custom-styled Streamlit strategy application
├── models/                      # Saved Artifact Store (Generated on training)
│   ├── target_encoders.pkl      # Pre-learned category maps (Driver, Race, Race_Compound)
│   ├── ensemble_metadata.json   # Optimal thresholds, weights, global means, features list
│   └── ensemble/                # Serialized model folds
├── configs/                     # Hyperparameter & Path Configurations
│   └── config.yaml              # Centralized configuration YAML
├── tests/                       # Automated Test Suites
│   ├── test_features.py         # Unit tests validating feature engineering logic
│   └── test_api.py              # Integration tests validating API schemas & routing
├── Datasets/                    # Train and Test CSV Data Store
│   ├── train.csv
│   └── test.csv
├── requirements.txt             # Python libraries list
├── Dockerfile                   # Docker container builder definition
└── README.md                    # System documentation (this file)
```

---

## 🛠️ Local Setup & Installation

### 1. Prerequisite Packages
Ensure you have Python 3.10+ installed. Clone or copy this repository into your workspace, and install the package dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### Step 1: Execute Model Training (Offline Phase)
Before launching the server or dashboard, you must run the training pipeline once. This will train the multi-model ensemble (15 models across 5 folds), optimize prediction weights and thresholds, and export all preprocessors and models to the `/models` directory:

```bash
python -m src.pipeline
```

### Step 2: Start the Streamlit Dashboard (Visual Interface)
Once the models are generated in the `/models` directory, start the interactive F1 Strategy Center:

```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
*   **Batch Prediction Tab**: Drag-and-drop a raw CSV (matching the test dataset schema). View interactive Plotly scatter plots mapping tyre life to pit probability (colored by Pirelli compound colors) and download a Kaggle-compliant `submission.csv`.
*   **Simulator Tab**: Enter telemetry values manually for a driver, race, and lap. Instantly calculate their pit probability and view dynamic green/orange/red warning strategy recommendations.

### Step 3: Run the FastAPI Server (Production REST API)
If you wish to serve predictions programmatically, start the FastAPI uvicorn server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Access the interactive Swagger API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).
*   `POST /predict/single`: Accepts a telemetry JSON payload and returns the probability and binary decision.
*   `POST /predict/csv`: Accepts a CSV file upload, runs batch inference, and streams back the prediction results.

---

## 🐳 Docker Deployment

To build and run the entire application inside a container (defaulting to the Streamlit UI):

```bash
# Build the Docker image
docker build -t f1-pitstop .

# Run the container (mapping port 8501 for Streamlit and 8000 for FastAPI)
docker run -p 8501:8501 -p 8000:8000 f1-pitstop
```

---

## 🧪 Running Automated Tests

We maintain an active test suite to prevent logic regression. Run the test cases using `pytest`:

```bash
pytest tests/
```
