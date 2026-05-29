from pydantic import BaseModel, Field

class TelemetryInput(BaseModel):
    Driver: str = Field(..., description="Driver's name (e.g. Verstappen, Hamilton)", json_schema_extra={"example": "Hamilton"})
    Race: str = Field(..., description="Race venue (e.g. Monaco, Spa)", json_schema_extra={"example": "Monaco"})
    Compound: str = Field(..., description="Tyre compound (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)", json_schema_extra={"example": "SOFT"})
    LapNumber: int = Field(..., ge=1, description="Current lap number in the race", json_schema_extra={"example": 15})
    Stint: int = Field(..., ge=1, description="Current stint number for the driver", json_schema_extra={"example": 1})
    TyreLife: float = Field(..., ge=0.0, description="Age of the current tyre set in laps", json_schema_extra={"example": 12.0})
    Position: float = Field(..., ge=1.0, le=20.0, description="Current position in the race", json_schema_extra={"example": 3.0})
    LapTime_s: float = Field(..., alias="LapTime (s)", ge=0.0, description="Lap time in seconds", json_schema_extra={"example": 78.5})
    LapTime_Delta: float = Field(..., description="Change in lap time relative to previous lap", json_schema_extra={"example": 0.2})
    Cumulative_Degradation: float = Field(..., ge=0.0, description="Cumulative tyre degradation score", json_schema_extra={"example": 0.08})
    RaceProgress: float = Field(..., ge=0.0, le=1.0, description="Race progress percentage (0.0 to 1.0)", json_schema_extra={"example": 0.25})
    Position_Change: float = Field(..., description="Position change relative to previous lap", json_schema_extra={"example": 0.0})
    PitStop: float = Field(default=0.0, description="Whether the driver pitted on the current lap", json_schema_extra={"example": 0.0})
    Year: int = Field(default=2025, description="Race year", json_schema_extra={"example": 2025})

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    pit_probability: float = Field(..., description="Predicted probability of pitting on the next lap (0.0 to 1.0)")
    should_pit: bool = Field(..., description="Ensemble binary classification based on the optimized decision threshold")
    optimal_threshold: float = Field(..., description="The OOF optimized decision threshold used for classification")
