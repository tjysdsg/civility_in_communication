import pandas as pd
from config import DEV_SET_PATH, TRAIN_SET_PATH, TEST_SET_PATH, DEMO_DEV_PATH
from datasets import Dataset


def load_pd_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', header=0)
    if 'label' in df.columns:
        df['label'] = (df['label'] == 'OFF').astype(int)
    return df


def load_dev_df() -> pd.DataFrame:
    return load_pd_dataframe(DEV_SET_PATH)


def load_test_df() -> pd.DataFrame:
    return load_pd_dataframe(TEST_SET_PATH)


def load_train_df() -> pd.DataFrame:
    return load_pd_dataframe(TRAIN_SET_PATH)


def load_demographic_dev_df() -> pd.DataFrame:
    return load_pd_dataframe(DEMO_DEV_PATH)


def df2dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df)
