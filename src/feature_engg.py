import numpy as np
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple

from .data_preparation import load_data
from .data_cleaning import F1DataCleaner
from .data_cleaning import F1DataSplitStrategy
from .data_cleaning import F1TargetEncoder
from .data_cleaning import target_encode_with_smoothing


class BaseFeatureEngineer(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @abstractmethod
    def engineer(self) -> pd.DataFrame:
        raise NotImplementedError


class F1FeatureEngineer(BaseFeatureEngineer):
    """Feature engineering pipeline for the F1 pit prediction problem."""

    GROUP_KEYS = ['Driver', 'Race', 'Year']

    @staticmethod
    def _slope_weights(window: int) -> np.ndarray:
        x = np.arange(window, dtype=float)
        x_centered = x - x.mean()
        return x_centered / np.sum(x_centered ** 2)

    def engineer(self) -> pd.DataFrame:
        # Sort within groups and compute race max laps
        self.df = self.df.sort_values(['Driver', 'Race', 'Year', 'LapNumber']).reset_index(drop=True)
        race_max_laps = self.df.groupby(['Race', 'Year'])['LapNumber'].max().rename('RaceMaxLap')
        self.df = self.df.join(race_max_laps, on=['Race', 'Year'])

        # Sequential feature construction (keeps same order as original)
        self.TyreLife_norm()
        self.Degrad_rate()
        self.Rolling_LapTime_Trend()
        self.DeltaMA_3()
        self.is_on_cliff()
        self.is_fresh_tyre()
        self.LapsRemaining()
        self.Tyre_vs_remaining()
        self.Strategic_Pit_Window()
        self.Compound_TyreLife_Interaction()
        self.TyreLifeNorm_x_LapTimeSlope()
        self.Position_Trend()
        self.is_in_points()
        self.Stint_Completion()
        self.is_final_stint()
        self.TyreLife_squared()
        self.Stint_x_TyreLife()
        self.LapTime_ratio_to_race_median()
        self.Laps_since_last_pit()
        self.TyreLife_x_RaceProgress()
        self.TyreLife_cubed()
        self.Stint_x_RaceProgress()
        self.create_bigram_columns()
        self.AvgStintLength()
        self.PitCount_so_far()

        return self.df

    # --- Feature methods (converted from functions) ---
    def TyreLife_norm(self):
        COMPOUND_TYPICAL_STINT = {
            'SOFT': 12, 'MEDIUM': 16, 'HARD': 20,
            'INTERMEDIATE': 17, 'WET': 11
        }
        compound_scale = self.df['Compound'].map(COMPOUND_TYPICAL_STINT).fillna(15)
        self.df['TyreLife_norm'] = self.df['TyreLife'] / compound_scale
        return self.df

    def Degrad_rate(self):
        self.df['Degrad_rate'] = self.df['Cumulative_Degradation'] / (self.df['TyreLife'].replace(0, np.nan))
        self.df['Degrad_rate'] = self.df['Degrad_rate'].fillna(0)
        return self.df

    def rolling_slope(self, series: pd.Series, window: int):
        weights = self._slope_weights(window)

        def _apply(window_values: np.ndarray) -> float:
            return float(np.dot(window_values, weights))

        return series.rolling(window=window, min_periods=window).apply(_apply, raw=True).to_numpy()

    def Rolling_LapTime_Trend(self):
        """Vectorized lap time trend using rolling mean of differences (much faster)."""
        grouped = self.df.groupby(['Driver', 'Race', 'Year', 'Stint'])['LapTime (s)']
        
        # Use diff (lap-over-lap change) as a fast proxy for slope
        lap_diff = grouped.diff().fillna(0)
        
        # Global rolling is vectorized and very fast
        self.df['LapTime_slope_3'] = lap_diff.rolling(3, min_periods=1).mean().fillna(0)
        self.df['LapTime_slope_5'] = lap_diff.rolling(5, min_periods=1).mean().fillna(0)
        
        # Mask the leakage at group boundaries (TyreLife tracks laps within a stint)
        self.df.loc[self.df['TyreLife'] < 2, 'LapTime_slope_3'] = lap_diff
        self.df.loc[self.df['TyreLife'] < 4, 'LapTime_slope_5'] = lap_diff
        
        return self.df

    def _col(self, *candidates):
        for c in candidates:
            if c in self.df.columns:
                return c
        return None

    def DeltaMA_3(self):
        # support both 'LapTimeDelta' and 'LapTime_Delta' column names
        col = self._col('LapTimeDelta', 'LapTime_Delta')
        if col is None:
            # if missing, create column with zeros to keep pipeline robust
            self.df['DeltaMA_3'] = 0
            print('DeltaMA_3 created (default 0) — no LapTimeDelta column found')
            return self.df

        # Optimized vectorized rolling mean
        self.df['DeltaMA_3'] = self.df[col].rolling(window=3, min_periods=1).mean().fillna(0)
        # Mask boundary leakage (TyreLife resets at each stint)
        self.df.loc[self.df['TyreLife'] < 2, 'DeltaMA_3'] = self.df[col]
        return self.df

    def is_on_cliff(self, threshold: float = 0.4):
        self.df['is_on_cliff'] = (self.df['LapTime_slope_3'] > threshold).astype(int)
        return self.df

    def is_fresh_tyre(self):
        self.df['is_fresh_tyre'] = (self.df['TyreLife'] <= 2).astype(int)
        return self.df

    def LapsRemaining(self):
        self.df['Laps_remaining'] = self.df['RaceMaxLap'] - self.df['LapNumber']
        # keep RaceMaxLap removal optional; preserve original behavior
        if 'RaceMaxLap' in self.df.columns:
            self.df = self.df.drop(columns=['RaceMaxLap'])
        return self.df

    def Tyre_vs_remaining(self):
        self.df['Tyre_vs_remaining'] = self.df['TyreLife'] / (self.df['Laps_remaining'] + 1)
        return self.df

    def Strategic_Pit_Window(self):
        self.df['In_pit_window_1'] = ((self.df['RaceProgress'] >= 0.20) & (self.df['RaceProgress'] <= 0.45)).astype(int)
        self.df['In_pit_window_2'] = ((self.df['RaceProgress'] >= 0.52) & (self.df['RaceProgress'] <= 0.75)).astype(int)
        return self.df

    def Compound_TyreLife_Interaction(self):
        self.df['TyreLife_x_Soft'] = self.df['TyreLife'] * self.df.get('is_soft', 0)
        self.df['TyreLife_x_Hard'] = self.df['TyreLife'] * self.df.get('is_hard', 0)
        self.df['TyreLife_x_Medium'] = self.df['TyreLife'] * self.df.get('is_medium', 0)
        return self.df

    def TyreLifeNorm_x_LapTimeSlope(self):
        # use slope_3
        self.df['TyreLifeNorm_x_LapTimeSlope_3'] = self.df.get('TyreLife_norm', 0) * self.df.get('LapTime_slope_3', 0)
        return self.df

    def Position_Trend(self):
        # Optimized vectorized rolling mean
        self.df['Position_trend_3'] = self.df['Position_Change'].rolling(window=3, min_periods=1).mean().fillna(0)
        # Mask boundary leakage (LapNumber resets at each race)
        self.df.loc[self.df['LapNumber'] < 3, 'Position_trend_3'] = self.df['Position_Change']
        return self.df

    def is_in_points(self):
        self.df['is_in_points'] = (self.df['Position'] <= 10).astype(int)
        return self.df

    def Stint_Completion(self):
        self.df['Stint_Completion'] = self.df['TyreLife'] / (self.df['TyreLife'] + self.df['Laps_remaining'])
        return self.df

    def is_final_stint(self):
        self.df['is_likely_final_stint'] = (
            (self.df['Laps_remaining'] <= 20) &
            (self.df['Compound'].isin(['HARD', 'MEDIUM'])) &
            (self.df['Stint'] >= 2)
        ).astype(int)
        return self.df

    def TyreLife_squared(self):
        """Non-linear degradation: penalty accelerates with age."""
        self.df['TyreLife_squared'] = self.df['TyreLife'] ** 2
        return self.df

    def Stint_x_TyreLife(self):
        """Later stints + high tyre life = much higher pit probability."""
        self.df['Stint_x_TyreLife'] = self.df['Stint'] * self.df['TyreLife']
        return self.df

    def LapTime_ratio_to_race_median(self):
        """Relative pace: normalizes lap times across different circuits."""
        race_median = self.df.groupby(['Race', 'Year'])['LapTime (s)'].transform('median')
        self.df['LapTime_ratio'] = self.df['LapTime (s)'] / race_median
        return self.df

    def Laps_since_last_pit(self):
        """How many laps since the driver's last pit stop in this race."""
        self.df['Laps_since_last_pit'] = self.df.groupby(
            self.GROUP_KEYS
        )['PitStop'].cumsum()
        # Convert cumulative pit count into laps-since-last by using TyreLife as proxy
        # TyreLife already tracks this well, so we create an interaction:
        # high cumulative pits + high current tyre life = overdue
        self.df['Laps_since_last_pit'] = self.df['TyreLife'] * (self.df['Laps_since_last_pit'] + 1)
        return self.df

    def TyreLife_x_RaceProgress(self):
        """Late-race + old tyres = critical interaction."""
        self.df['TyreLife_x_RaceProgress'] = self.df['TyreLife'] * self.df['RaceProgress']
        return self.df

    def TyreLife_cubed(self):
        """Stronger non-linearity for degradation cliff."""
        self.df['TyreLife_cubed'] = self.df['TyreLife'] ** 3
        return self.df

    def Stint_x_RaceProgress(self):
        """Stint number relative to race progress — captures multi-stop vs single-stop."""
        self.df['Stint_x_RaceProgress'] = self.df['Stint'] * self.df['RaceProgress']
        return self.df

    def create_bigram_columns(self):
        """Create categorical bigram columns for target encoding later."""
        self.df['Driver_Race'] = self.df['Driver'] + '_' + self.df['Race']
        self.df['Driver_Compound'] = self.df['Driver'] + '_' + self.df['Compound']
        self.df['Race_Compound'] = self.df['Race'] + '_' + self.df['Compound']
        return self.df

    def AvgStintLength(self):
        """Average stint length for this Driver+Race group, computed without future leakage.
        Uses cummax of TyreLife per stint (visible up to current lap) to estimate stint lengths seen so far.
        TyreLife_vs_AvgStint > 1.0 means the driver is running longer than their typical stint."""
        # Max TyreLife seen so far in each stint (no future leakage)
        self.df['_stint_max_so_far'] = self.df.groupby(
            ['Driver', 'Race', 'Year', 'Stint']
        )['TyreLife'].cummax()

        # Average of completed-stint estimates per Driver+Race (across all stints and laps seen)
        # We use the running cummax as the best proxy for "how long this stint will be"
        avg_stint = (
            self.df.groupby(['Driver', 'Race', 'Year', 'Stint'])['TyreLife']
            .max()
            .groupby(level=['Driver', 'Race'])
            .mean()
            .rename('AvgStintLen')
            .reset_index()
        )

        # Merge back on Driver+Race (safe, handles duplicate index correctly)
        self.df = self.df.merge(avg_stint[['Driver', 'Race', 'AvgStintLen']], on=['Driver', 'Race'], how='left')
        self.df['AvgStintLen'] = self.df['AvgStintLen'].fillna(self.df['TyreLife'].median())
        self.df['TyreLife_vs_AvgStint'] = self.df['TyreLife'] / (self.df['AvgStintLen'] + 1)
        self.df.drop(columns=['_stint_max_so_far'], inplace=True)
        return self.df

    def PitCount_so_far(self):
        """Cumulative pit stops for this driver in this race so far.
        Captures multi-stop vs single-stop strategy."""
        self.df['PitCount_so_far'] = self.df.groupby(['Driver', 'Race', 'Year'])['PitStop'].cumsum()
        return self.df

    # utility
    def available_features(self, features: list) -> None:
        available = [f for f in features if f in self.df.columns]
        print(f'Features ready: {len(available)}')
        numeric_available = self.df[available].select_dtypes(include='number').columns.tolist()
        corr = self.df[numeric_available + ['PitNextLap']].corr()['PitNextLap'].drop('PitNextLap')
        print(corr.abs().sort_values(ascending=False).round(4).head(20))

    # safe splitting fallback (uses F1DataSplitStrategy if available, otherwise manual Year-split)
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            splitter = F1DataSplitStrategy()
            result = splitter.split(self.df)
            # strategy may return None or three-dataframes; handle both
            if isinstance(result, tuple) and len(result) == 3:
                return result
        except Exception:
            pass
        # fallback: year-based split
        train_df = self.df[self.df['Year'].isin([2022, 2023])].copy()
        val_df = self.df[self.df['Year'] == 2024].copy()
        test_df = self.df[self.df['Year'] == 2025].copy()
        return train_df, val_df, test_df
    


# features list (kept for compatibility)
engineered_features = [
    'Compound', 'LapNumber', 'Stint', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation',
    'RaceProgress', 'Position_Change', 'PitStop', 'Year',
    'compound_ordinal', 'is_soft', 'is_hard', 'is_medium', 'is_wet_tire',
    'TyreLife_norm', 'Degrad_rate', 'LapTime_slope_3', 'LapTime_slope_5',
    'DeltaMA_3', 'is_on_cliff', 'is_fresh_tyre',
    'Laps_remaining', 'Tyre_vs_remaining', 'In_pit_window_1', 'In_pit_window_2',
    'TyreLife_x_Soft', 'TyreLife_x_Hard', 'TyreLife_x_Medium', 'TyreLifeNorm_x_LapTimeSlope_3',
    'Position_trend_3', 'is_in_points',
    'Stint_Completion', 'is_likely_final_stint',
    'is_sc_lap',
    # New features
    'TyreLife_squared', 'Stint_x_TyreLife', 'LapTime_ratio',
    'Laps_since_last_pit', 'TyreLife_x_RaceProgress',
    'TyreLife_cubed', 'Stint_x_RaceProgress',
    # Bigram columns (string, for reference — encoded versions used in model)
    'Driver_Race', 'Driver_Compound', 'Race_Compound',
]


def available_features(df, features) -> None:
    available = [f for f in features if f in df.columns]
    print(f'Features ready: {len(available)}')
    numeric_available = df[available].select_dtypes(include='number').columns.tolist()
    corr = df[numeric_available + ['PitNextLap']].corr()['PitNextLap'].drop('PitNextLap')
    print(corr.abs().sort_values(ascending=False).round(4).head(20))

# final feature set (drop raw categoricals that have been encoded)
def get_engineered_features(train_df, val_df, test_df):
    
    FEATURES = [
    # Original numeric
    'LapNumber', 'Stint', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation',
    'RaceProgress', 'Position_Change', 'PitStop', 'Year',
    # Compound encoding
    'compound_ordinal', 'is_soft', 'is_hard', 'is_medium', 'is_wet_tire',
    # Group 1: Tyre degradation
    'TyreLife_norm', 'Degrad_rate', 'LapTime_slope_3', 'LapTime_slope_5',
    'DeltaMA_3', 'is_on_cliff', 'is_fresh_tyre',
    # Group 2: Race context
    'Laps_remaining', 'Tyre_vs_remaining', 'In_pit_window_1', 'In_pit_window_2',
    # Group 3: Interactions
    'TyreLife_x_Soft', 'TyreLife_x_Hard', 'TyreLife_x_Medium', 'TyreLifeNorm_x_LapTimeSlope_3',
    # Group 4: Race dynamics
    'Position_trend_3', 'is_in_points',
    # Group 5: Stint context
    'Stint_Completion', 'is_likely_final_stint',
    # Flags
    'is_sc_lap',
    # New features
    'TyreLife_squared', 'Stint_x_TyreLife', 'LapTime_ratio',
    'Laps_since_last_pit', 'TyreLife_x_RaceProgress',
    'TyreLife_cubed', 'Stint_x_RaceProgress',
    # Stint strategy features
    'AvgStintLen', 'TyreLife_vs_AvgStint', 'PitCount_so_far',
    # Target encoded (only low-cardinality encodings that generalize across years)
    'driver_enc', 'race_enc', 'race_compound_enc',
    ]

    TARGET = 'PitNextLap'

    # drop raw categorical columns that have been encoded
    drop_cols = ['Compound', 'Driver', 'Race', 'Driver_Race', 'Driver_Compound', 'Race_Compound']
    train_df = train_df.drop(columns=drop_cols, errors='ignore')
    val_df = val_df.drop(columns=drop_cols, errors='ignore')
    test_df = test_df.drop(columns=drop_cols, errors='ignore')

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_val = val_df[FEATURES]
    y_val = val_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET] if TARGET in test_df.columns else None

    print(f'X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}')
    print(f'Total features: {len(FEATURES)}')
    
    return X_train, y_train, X_val, y_val, X_test, y_test

