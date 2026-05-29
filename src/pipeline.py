import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from .data_preparation import load_data
from .data_cleaning import F1DataCleaner, F1TargetEncoder, target_encode_with_smoothing
from .feature_engg import F1FeatureEngineer, get_engineered_features
from .model_development import run_ensemble


def run_pipeline():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / 'Datasets' / 'train.csv'
    comp_test_path = base_dir / 'Datasets' / 'test.csv'
    cache_path = base_dir / 'Datasets' / 'train_preprocessed_cache.pkl'
    models_dir = base_dir / 'models'

    # Ensure models directory exists
    models_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as handle:
                cached = pickle.load(handle)
            print(f'Loaded cached preprocessing from {cache_path.name}')
            train_df = cached['train_df']
            val_df = cached['val_df']
            test_df = cached['test_df']
            df_eng = cached['df_eng']
        except Exception:
            cached = None
    else:
        cached = None

    if cached is None:
        df = load_data(data_path)

        # Clean
        df_clean = F1DataCleaner(df).clean()

        # Engineer
        engineer = F1FeatureEngineer(df_clean)
        df_eng = engineer.engineer()
        print('Feature engineering complete.')

        # Split (safe)
        train_df, val_df, test_df = engineer.split_data()
        print('Split sizes:', len(train_df), len(val_df), len(test_df))

        # Target-encode Driver and Race using training stats only
        try:
            train_df['driver_enc'], val_df['driver_enc'], _ = target_encode_with_smoothing(
                train_df, val_df, 'Driver', 'PitNextLap', smoothing=10
            )
            driver_encoder = F1TargetEncoder(train_df, smoothing=10)
            driver_encoder.encode('Driver', 'PitNextLap')
            test_df['driver_enc'] = driver_encoder.apply(test_df, 'Driver')

            train_df['race_enc'], val_df['race_enc'], _ = target_encode_with_smoothing(
                train_df, val_df, 'Race', 'PitNextLap', smoothing=10
            )
            race_encoder = F1TargetEncoder(train_df, smoothing=10)
            race_encoder.encode('Race', 'PitNextLap')
            test_df['race_enc'] = race_encoder.apply(test_df, 'Race')
        except Exception as e:
            print('Target encoding failed:', e)

        # Save cache
        with open(cache_path, 'wb') as handle:
            pickle.dump({
                'train_df': train_df,
                'val_df': val_df,
                'test_df': test_df,
                'df_eng': df_eng,
            }, handle)

    # ---- Features that can be added without cache rebuild ----
    for df_part in [train_df, val_df, test_df]:
        if 'TyreLife_cubed' not in df_part.columns:
            df_part['TyreLife_cubed'] = df_part['TyreLife'] ** 3
            df_part['Stint_x_RaceProgress'] = df_part['Stint'] * df_part['RaceProgress']
        if 'Race_Compound' not in df_part.columns:
            df_part['Race_Compound'] = df_part['Race'] + '_' + df_part['Compound']

    # Target-encode Race_Compound
    if 'race_compound_enc' not in train_df.columns:
        train_df['race_compound_enc'], val_df['race_compound_enc'], _ = target_encode_with_smoothing(
            train_df, val_df, 'Race_Compound', 'PitNextLap', smoothing=15
        )
        enc = F1TargetEncoder(train_df, smoothing=15)
        enc.encode('Race_Compound', 'PitNextLap')
        test_df['race_compound_enc'] = enc.apply(test_df, 'Race_Compound')

    # Build the final model-ready feature matrices
    X_train, y_train, X_val, y_val, X_test, y_test = get_engineered_features(train_df, val_df, test_df)

    # ---- Competition Test Handling ----
    comp_test_ready = False
    if comp_test_path.exists():
        print(f'\nProcessing competition test set: {comp_test_path.name}...')
        comp_test_df = load_data(comp_test_path)
        
        comp_test_clean = F1DataCleaner(comp_test_df, is_train=False).clean()
        comp_test_eng = F1FeatureEngineer(comp_test_clean).engineer()
        comp_test_ready = True

    # ---- Phase B: combine 2022+2024, re-encode, run ensemble ----
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    # Re-encode on the combined 2022+2024 set (only low-cardinality features)
    enc_cols = [
        ('Driver', 'driver_enc', 10),
        ('Race', 'race_enc', 10),
        ('Race_Compound', 'race_compound_enc', 15),
    ]
    
    target_encoder_maps = {}
    global_mean = float(trainval_df['PitNextLap'].mean())
    target_encoder_maps['global_mean'] = global_mean

    # Process both internal test_df and competition comp_test_eng
    for col, enc_name, smooth in enc_cols:
        stats = trainval_df.groupby(col)['PitNextLap'].agg(['mean', 'count'])
        stats['smoothed'] = (
            (stats['mean'] * stats['count'] + global_mean * smooth)
            / (stats['count'] + smooth)
        )
        enc_map = stats['smoothed'].to_dict()
        target_encoder_maps[col] = {
            'map': enc_map,
            'smoothing': smooth
        }
        
        X_trainval[enc_name] = trainval_df[col].map(enc_map).fillna(global_mean).values
        X_test[enc_name] = test_df[col].map(enc_map).fillna(global_mean).values
        if comp_test_ready:
            comp_test_eng[enc_name] = comp_test_eng[col].map(enc_map).fillna(global_mean)

    # Pre-compute average stint lengths per (Driver, Race) using the full combined trainval set
    try:
        avg_stint_df = (
            trainval_df.groupby(['Driver', 'Race', 'Year', 'Stint'])['TyreLife']
            .max()
            .groupby(level=['Driver', 'Race'])
            .mean()
            .rename('AvgStintLen')
            .reset_index()
        )
        avg_stint_map = avg_stint_df.set_index(['Driver', 'Race'])['AvgStintLen'].to_dict()
        target_encoder_maps['avg_stint_map'] = avg_stint_map
        print("Historical average stint lengths mapped successfully.")
    except Exception as e:
        print(f"Failed to map average stint lengths: {e}")
        target_encoder_maps['avg_stint_map'] = {}

    # Save target encoders mappings for deployment API loading
    encoders_path = models_dir / 'target_encoders.pkl'
    with open(encoders_path, 'wb') as handle:
        pickle.dump(target_encoder_maps, handle)
    print(f'Target encoders serialized and saved to {encoders_path.name}')

    if comp_test_ready:
        X_comp_test = comp_test_eng[X_trainval.columns.tolist()]
        print("Using competition test set for final ensemble predictions...")
        final_proba, weights, threshold = run_ensemble(
            X_trainval, y_trainval, X_comp_test, np.zeros(len(X_comp_test)), n_splits=5, models_dir=models_dir
        )
        
        # Create submission file
        submission = pd.DataFrame({
            'id': comp_test_eng['id'].values,
            'PitNextLap': final_proba
        }).sort_values('id').reset_index(drop=True)
        
        submission_path = base_dir / 'submission.csv'
        submission.to_csv(submission_path, index=False)
        print(f'\nSUCCESS: Submission file saved to {submission_path.name}. Total rows: {len(submission)}')
    else:
        # Normal run if test.csv doesn't exist
        test_blend, weights, threshold = run_ensemble(
            X_trainval, y_trainval, X_test, y_test, n_splits=5, models_dir=models_dir
        )


if __name__ == '__main__':
    run_pipeline()
