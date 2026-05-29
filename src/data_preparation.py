from pathlib import Path

import pandas as pd

# load data

def load_data(file_path):
    try: 
        path = Path(file_path)
        if not path.is_absolute() and not path.exists():
            path = Path(__file__).resolve().parent / path

        df = pd.read_csv(path)
        print(f'Loaded: {df.shape[0]} rows and {df.shape[1]} columns')
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f'Could not find data file: {file_path}')
    except Exception as e:
        raise RuntimeError(f'An error occurred while loading {file_path}: {e}')

# df = load_data('./Datasets/train.csv')

