import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import TelemetryInput, PredictionResponse
from api.predictor import F1Predictor

app = FastAPI(
    title="Formula 1 Pit Stop Predictor API",
    description="A high-performance ML ensemble API (LightGBM + CatBoost + XGBoost) predicting whether an F1 driver will pit on their next lap.",
    version="1.0.0"
)

# Enable CORS for frontend cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance, initialized at startup
predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    try:
        predictor = F1Predictor()
        print("API successfully initialized model assets.")
    except Exception as e:
        print(f"Startup error loading models: {e}")
        # We don't crash, but endpoints will report 503 until model is generated
        predictor = None


@app.get("/")
def read_root():
    """Return API metadata and status."""
    if predictor is None:
        return {
            "status": "Warning",
            "message": "API online but model assets are missing. Please run src/pipeline.py to train models.",
            "endpoints": ["/predict/single", "/predict/csv"]
        }
    
    return {
        "status": "Healthy",
        "message": "F1 Pit Stop Predictor API is fully active and loaded.",
        "ensemble_metadata": {
            "blend_weights": predictor.blend_weights,
            "optimal_threshold": predictor.optimal_threshold,
            "loaded_features_count": len(predictor.feature_names)
        }
    }


@app.post("/predict/single", response_model=PredictionResponse)
def predict_single(telemetry: TelemetryInput):
    """
    Accept telemetry parameters for a single lap, engineer features,
    and predict pit probability and binary decision.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model ensemble assets are not loaded on server. Run src/pipeline.py training first."
        )
    
    try:
        # Convert Pydantic object to dictionary, resolving field aliases
        input_data = telemetry.model_dump(by_alias=True)
        res = predictor.predict_single(input_data)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/csv")
def predict_csv(file: UploadFile = File(...)):
    """
    Upload a batch CSV file (matching F1 test.csv schema) and
    receive a streamed CSV of predictions containing 'pit_probability' and 'should_pit'.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model ensemble assets are not loaded on server. Run src/pipeline.py training first."
        )

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    try:
        # Read uploaded bytes into a Pandas DataFrame
        contents = file.file.read()
        raw_df = pd.read_csv(io.BytesIO(contents))
        
        # Verify required columns are present (rough check)
        required = ['Driver', 'Race', 'Compound', 'LapNumber', 'Stint', 'TyreLife']
        missing = [col for col in required if col not in raw_df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded CSV is missing mandatory columns for feature engineering: {missing}"
            )

        # Run predictions
        results_df = predictor.predict_csv(raw_df)
        
        # Combine ID if available, otherwise just output predictions
        if 'id' in raw_df.columns:
            final_df = pd.DataFrame({'id': raw_df['id']}).join(results_df)
        else:
            final_df = results_df

        # Save to buffer and stream response
        stream = io.StringIO()
        final_df.to_csv(stream, index=False)
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f"attachment; filename=predictions_{file.filename}"
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV prediction: {str(e)}")
