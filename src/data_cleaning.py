from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

class BaseDataSplitStrategy(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame):
        raise NotImplementedError

class F1DataSplitStrategy(BaseDataSplitStrategy):
    def split(self, df: pd.DataFrame):
        # Implementation for F1 data splitting
        # year-based split: 2022+2023 train, 2024 val, 2025 test
        self.train_df = df[df['Year'].isin([2022, 2023])].copy()
        self.val_df = df[df['Year'] == 2024].copy()
        self.test_df = df[df['Year'] == 2025].copy()

        print(f'Train (2022+2023): {len(self.train_df):>8,} rows | pit rate: {self.train_df["PitNextLap"].mean():.3f}')
        print(f'Validate (2024):   {len(self.val_df):>8,} rows | pit rate: {self.val_df["PitNextLap"].mean():.3f}')
        print(f'Test (2025):       {len(self.test_df):>8,} rows | pit rate: {self.test_df["PitNextLap"].mean():.3f}')

        return self.train_df, self.val_df, self.test_df



class BaseDataCleaner(ABC):
    def __init__(self, df: pd.DataFrame, is_train: bool = True):
        self.df = df.copy()
        self.is_train = is_train

    def clean(self, cols_to_drop=None):
        if cols_to_drop:
            self.drop_data(cols_to_drop)

        self.data_quality_solving()
        return self.df

    @abstractmethod
    def drop_data(self, cols_to_drop):
        raise NotImplementedError

    @abstractmethod
    def data_quality_solving(self):
        raise NotImplementedError


class F1DataCleaner(BaseDataCleaner):
    def drop_data(self, cols_to_drop):
        self.df = self.df.drop(columns=cols_to_drop)
        print(f'Dropped columns: {cols_to_drop}')
        return self.df

    def data_quality_solving(self):
        df_clean = self.df.copy()

        # only drop rows if we are in training mode
        if self.is_train:
            # NOTE: 2023 has a very low pit rate (~0.96%) compared to other years (~28%),
            # but we MUST keep it because 31% of the competition test set is from 2023.
            # The model needs to learn that 2023 = almost never pits.

            # outlier handling for lap time values above 200 seconds
            df_clean = df_clean[df_clean['LapTime (s)'] <= 200].copy()
            print('Dropped outliers in LapTime (s): now rows are', df_clean.shape[0])
        else:
            print('Inference mode: keeping all rows')

        # laptime delta: stint transition extremes
        low = df_clean['LapTime_Delta'].quantile(0.02)
        high = df_clean['LapTime_Delta'].quantile(0.98)
        df_clean['LapTime_Delta'] = df_clean['LapTime_Delta'].clip(lower=low, upper=high)
        print(f'Clipped LapTime_Delta to [{low:.2f}, {high:.2f}]')

        # outlier handling for cumulative degradation
        low_cd = df_clean['Cumulative_Degradation'].quantile(0.01)
        high_cd = df_clean['Cumulative_Degradation'].quantile(0.99)
        df_clean['Cumulative_Degradation'] = df_clean['Cumulative_Degradation'].clip(lower=low_cd, upper=high_cd)
        print(f'Clipped Cumulative_Degradation to [{low_cd:.2f}, {high_cd:.2f}]')

        # flag anomalous laps relative to the race median
        race_medians = df_clean.groupby('Race')['LapTime (s)'].median().rename('Median_LapTime')
        df_clean = df_clean.join(race_medians, on='Race')
        df_clean['is_sc_lap'] = (df_clean['LapTime (s)'] > 1.4 * df_clean['Median_LapTime']).astype(int)
        df_clean = df_clean.drop(columns=['Median_LapTime'])
        print(
            f'Flagged {df_clean["is_sc_lap"].sum()} potential safety car laps. '
            f'({df_clean["is_sc_lap"].mean():.2%})'
        )

        # categorical encoding for compound based on grip
        compound_grip_order = {'WET': 0, 'INTERMEDIATE': 1, 'HARD': 2, 'MEDIUM': 3, 'SOFT': 4}
        df_clean['compound_ordinal'] = df_clean['Compound'].map(compound_grip_order)

        # binary flags for compound types
        df_clean['is_soft'] = (df_clean['Compound'] == 'SOFT').astype(int)
        df_clean['is_hard'] = (df_clean['Compound'] == 'HARD').astype(int)
        df_clean['is_medium'] = (df_clean['Compound'] == 'MEDIUM').astype(int)
        df_clean['is_wet_tire'] = df_clean['Compound'].isin(['WET', 'INTERMEDIATE']).astype(int)

        print('Compound encoding done.')
        print(
            df_clean[
                ['Compound', 'compound_ordinal', 'is_soft', 'is_hard', 'is_wet_tire']
            ].drop_duplicates().sort_values('compound_ordinal')
        )        

        self.df = df_clean
        return self.df
    


class BaseTargetEncoder(ABC):
    """Abstract base class for target encoding with leakage prevention."""

    def __init__(self, train_df: pd.DataFrame, smoothing: float = 10):
        """
        Args:
            train_df: Training set used to learn encoding (only this set computes statistics)
            smoothing: Laplace smoothing weight — higher = more shrinkage toward global mean for rare categories
        """
        self.train_df = train_df.copy()
        self.smoothing = smoothing
        self.encoding_map = {}
        self.global_mean = None

    @abstractmethod
    def encode(self, col: str, target: str) -> dict:
        """Learn encoding from train_df. Returns encoding map."""
        raise NotImplementedError

    def apply(self, df: pd.DataFrame, col: str, col_name: str = None) -> pd.Series:
        """Apply learned encoding to any dataframe (train, val, or test)."""
        if not self.encoding_map:
            raise ValueError("No encoding learned. Call encode() first.")
        
        if col_name is None:
            col_name = col
        
        encoded = df[col].map(self.encoding_map).fillna(self.global_mean)
        return encoded


class F1TargetEncoder(BaseTargetEncoder):
    """Target encoder for F1 data with Laplace smoothing to prevent leakage."""

    def encode(self, col: str, target: str) -> dict:
        """
        Learn target encoding ONLY from training set.
        Compute smoothed category means using Laplace smoothing.

        Args:
            col: Column to encode (e.g., 'Driver', 'Race')
            target: Target variable (e.g., 'PitNextLap')

        Returns:
            Dictionary mapping category values to smoothed means
        """
        # Compute global mean from training set only
        self.global_mean = self.train_df[target].mean()

        # Compute category-level statistics from training set only
        stats = self.train_df.groupby(col)[target].agg(['mean', 'count'])

        # Laplace smoothing: blend category mean toward global mean based on count
        # Rare categories (low count) shrink more toward global_mean
        stats['smoothed'] = (
            (stats['mean'] * stats['count'] + self.global_mean * self.smoothing) /
            (stats['count'] + self.smoothing)
        )

        self.encoding_map = stats['smoothed'].to_dict()
        
        print(
            f'Encoded {col} on {len(stats)} unique values | '
            f'global {target} mean: {self.global_mean:.4f} | smoothing: {self.smoothing}'
        )
        
        return self.encoding_map


def target_encode_with_smoothing(train_df, test_df, col, target, smoothing=10):
    """
    Wrapper function for backward compatibility.
    Learns encoding from train_df only, applies to both test_df and returns encoding map.
    Prevents data leakage by computing statistics only from training set.

    Args:
        train_df: Training set (used to learn encoding statistics)
        test_df: Validation or test set (encoding applied, no stats computed from it)
        col: Column to encode (e.g., 'Driver', 'Race')
        target: Target variable (e.g., 'PitNextLap')
        smoothing: Laplace smoothing weight

    Returns:
        train_encoded: Encoded training set column
        test_encoded: Encoded test/validation set column
        enc_map: Dictionary mapping category values to smoothed means
    """
    encoder = F1TargetEncoder(train_df, smoothing=smoothing)
    enc_map = encoder.encode(col, target)
    
    train_encoded = encoder.apply(train_df, col)
    test_encoded = encoder.apply(test_df, col)
    
    print(f'Applied encoding to {len(train_encoded)} train rows and {len(test_encoded)} test rows.')
    
    return train_encoded, test_encoded, enc_map


def drop_data(df, cols_to_drop):
    return F1DataCleaner(df).drop_data(cols_to_drop)


def data_quality_solving(df):
    return F1DataCleaner(df).data_quality_solving()



    

    