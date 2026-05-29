import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from src.data_cleaning import F1DataCleaner
from src.feature_engg import F1FeatureEngineer


class F1Predictor:
    def __init__(self, models_dir=None):
        if models_dir is None:
            # Resolve models/ directory in project root
            self.models_dir = Path(__file__).resolve().parent.parent / 'models'
        else:
            self.models_dir = Path(models_dir)

        self.metadata_path = self.models_dir / 'ensemble_metadata.json'
        self.encoders_path = self.models_dir / 'target_encoders.pkl'

        if not self.metadata_path.exists() or not self.encoders_path.exists():
            raise FileNotFoundError(
                f"Model assets not found in {self.models_dir.resolve()}. "
                f"Please run model training (src/pipeline.py) first to generate them."
            )

        # 1. Load ensemble metadata
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.blend_weights = self.metadata['blend_weights']
        self.optimal_threshold = self.metadata['optimal_threshold']
        self.global_mean = self.metadata['global_mean']
        self.feature_names = self.metadata['feature_names']

        # 2. Load target encoders
        with open(self.encoders_path, 'rb') as f:
            self.target_encoders = pickle.load(f)
        
        self.avg_stint_map = self.target_encoders.get('avg_stint_map', {})

        # 3. Load ensemble models
        self.models = {'lgb': [], 'cb': [], 'xgb': []}
        ensemble_path = self.models_dir / 'ensemble'
        
        print(f"Loading ensemble models from {ensemble_path.resolve()}...")
        for i in range(5):
            # LightGBM
            lgb_model = lgb.Booster(model_file=str(ensemble_path / f'lgb_fold_{i}.txt'))
            self.models['lgb'].append(lgb_model)
            
            # CatBoost
            cb_model = cb.CatBoostClassifier()
            cb_model.load_model(str(ensemble_path / f'cb_fold_{i}.cbm'))
            self.models['cb'].append(cb_model)
            
            # XGBoost
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(str(ensemble_path / f'xgb_fold_{i}.json'))
            self.models['xgb'].append(xgb_model)
            
        print("All 15 models (LightGBM, CatBoost, XGBoost) loaded successfully.")

    def predict_csv(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch raw DataFrame, run it through the cleaning and engineering pipeline,
        apply pre-learned target encodings, run the ensemble, and return prediction results.
        """
        df = raw_df.copy()
        
        # 1. Cleaning (Inference mode)
        cleaner = F1DataCleaner(df, is_train=False)
        df_clean = cleaner.clean()

        # 2. Feature Engineering
        engineer = F1FeatureEngineer(df_clean)
        df_eng = engineer.engineer()

        # 3. Handle features added in Phase B that are not part of the standard engineer()
        if 'TyreLife_cubed' not in df_eng.columns:
            df_eng['TyreLife_cubed'] = df_eng['TyreLife'] ** 3
            df_eng['Stint_x_RaceProgress'] = df_eng['Stint'] * df_eng['RaceProgress']
        if 'Race_Compound' not in df_eng.columns:
            df_eng['Race_Compound'] = df_eng['Race'] + '_' + df_eng['Compound']

        # 4. Apply pre-learned Target Encodings
        enc_cols = [
            ('Driver', 'driver_enc'),
            ('Race', 'race_enc'),
            ('Race_Compound', 'race_compound_enc'),
        ]
        
        for col, enc_name in enc_cols:
            enc_info = self.target_encoders.get(col, {})
            enc_map = enc_info.get('map', {})
            df_eng[enc_name] = df_eng[col].map(enc_map).fillna(self.global_mean)

        # 5. Historical Stint Lookup Overwrite (Guarantees accuracy for small batches)
        # Check if we can overwrite AvgStintLen with historical pre-computed stint lengths
        for idx, row in df_eng.iterrows():
            key = (row['Driver'], row['Race'])
            if key in self.avg_stint_map:
                df_eng.at[idx, 'AvgStintLen'] = self.avg_stint_map[key]
                df_eng.at[idx, 'TyreLife_vs_AvgStint'] = row['TyreLife'] / (self.avg_stint_map[key] + 1)

        # 6. Extract target features in the exact same training order
        X = df_eng[self.feature_names]

        # 7. Ensemble Model Predictions
        n_splits = 5
        preds = {'lgb': np.zeros(len(X)), 'cb': np.zeros(len(X)), 'xgb': np.zeros(len(X))}

        for i in range(n_splits):
            # LightGBM booster uses predict directly for probabilities
            preds['lgb'] += self.models['lgb'][i].predict(X) / n_splits
            # CatBoost
            preds['cb'] += self.models['cb'][i].predict_proba(X)[:, 1] / n_splits
            # XGBoost
            preds['xgb'] += self.models['xgb'][i].predict_proba(X)[:, 1] / n_splits

        # Blend predictions using OOF optimized weights
        w = self.blend_weights
        final_proba = w[0] * preds['lgb'] + w[1] * preds['cb'] + w[2] * preds['xgb']
        should_pit = final_proba >= self.optimal_threshold

        results = pd.DataFrame({
            'pit_probability': final_proba,
            'should_pit': should_pit.astype(bool)
        })
        
        return results

    def predict_single(self, telemetry: dict) -> dict:
        """
        Process a single lap telemetry dictionary, run it through the pipeline,
        and return probability and classification values.
        """
        # Ensure LapTime (s) is mapped correctly if passed without space
        raw_dict = telemetry.copy()
        if 'LapTime_s' in raw_dict and 'LapTime (s)' not in raw_dict:
            raw_dict['LapTime (s)'] = raw_dict.pop('LapTime_s')

        # Convert to single-row dataframe
        df = pd.DataFrame([raw_dict])
        
        # Ensure standard schema features exist for cleaning/engineering
        defaults = {
            'id': 0,
            'Driver_Race': f"{raw_dict.get('Driver', '')}_{raw_dict.get('Race', '')}",
            'Driver_Compound': f"{raw_dict.get('Driver', '')}_{raw_dict.get('Compound', '')}",
            'Race_Compound': f"{raw_dict.get('Race', '')}_{raw_dict.get('Compound', '')}",
            'Position_Change': 0.0,
            'PitStop': 0.0,
            'Year': 2025
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        # Run batch processor on this single row
        results = self.predict_csv(df)
        
        proba = float(results.loc[0, 'pit_probability'])
        decision = bool(results.loc[0, 'should_pit'])

        return {
            'pit_probability': proba,
            'should_pit': decision,
            'optimal_threshold': self.optimal_threshold
        }
