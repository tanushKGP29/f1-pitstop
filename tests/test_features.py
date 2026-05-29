import pandas as pd
import pytest
from src.feature_engg import F1FeatureEngineer

def test_feature_engineering_columns():
    """Verify that F1FeatureEngineer runs cleanly and produces all core engineered columns."""
    # Create simple mock telemetry DataFrame representing 3 laps for Hamilton at Monaco
    data = {
        'id': [1, 2, 3],
        'Driver': ['Hamilton', 'Hamilton', 'Hamilton'],
        'Race': ['Monaco', 'Monaco', 'Monaco'],
        'Compound': ['SOFT', 'SOFT', 'SOFT'],
        'LapNumber': [1, 2, 3],
        'Stint': [1, 1, 1],
        'TyreLife': [1.0, 2.0, 3.0],
        'Position': [3.0, 3.0, 2.0],
        'LapTime (s)': [78.5, 78.2, 77.9],
        'LapTime_Delta': [0.0, -0.3, -0.3],
        'Cumulative_Degradation': [0.01, 0.02, 0.03],
        'RaceProgress': [0.02, 0.04, 0.06],
        'Position_Change': [0.0, 0.0, -1.0],
        'PitStop': [0.0, 0.0, 0.0],
        'Year': [2025, 2025, 2025]
    }
    df = pd.DataFrame(data)
    
    # Run engineering pipeline
    engineer = F1FeatureEngineer(df)
    df_eng = engineer.engineer()
    
    # 1. Assert size is unchanged
    assert len(df_eng) == 3
    
    # 2. Assert basic engineered features are created correctly
    assert 'TyreLife_norm' in df_eng.columns
    assert 'Degrad_rate' in df_eng.columns
    assert 'is_fresh_tyre' in df_eng.columns
    assert 'is_in_points' in df_eng.columns
    assert 'TyreLife_squared' in df_eng.columns
    assert 'TyreLife_cubed' in df_eng.columns
    
    # Check specific logic (e.g. Tyres are SOFT compound typical stint = 12 laps, so TyreLife_norm = TyreLife / 12)
    assert df_eng.loc[0, 'TyreLife_norm'] == 1.0 / 12.0
    assert df_eng.loc[2, 'TyreLife_norm'] == 3.0 / 12.0
    
    # Check is_in_points flag (Position <= 10)
    assert df_eng.loc[0, 'is_in_points'] == 1
    
    # Check fresh tyre flag (TyreLife <= 2)
    assert df_eng.loc[0, 'is_fresh_tyre'] == 1
    assert df_eng.loc[1, 'is_fresh_tyre'] == 1
    assert df_eng.loc[2, 'is_fresh_tyre'] == 0
