import os
import pandas as pd
from typing import Dict


def intersect_cohort(cohort_name_to_path: Dict[str, str]) -> pd.DataFrame:
    dfs = {}
    # Update label name
    for name, path in cohort_name_to_path.items():
        data = pd.read_csv(path)
        data = data.rename(columns={'label': f"{name}_label"})
        dfs[name] = data

    # Intersect by subject_id and hadm_id ( not duplicate the other common columns -- chose the first) -- preserve the label with names.
    # Start with the first dataframe
    result = dfs[list(dfs.keys())[0]]

    # Merge with remaining dataframes
    for name, df in list(dfs.items())[1:]:
        result = pd.merge(
            result,
            df[['subject_id', 'hadm_id', f'{name}_label']],
            on=['subject_id', 'hadm_id'],
            how='inner'
        )

    # Calculate the LOS labe >= 15
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    df['los_days'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / (60 * 60 * 24)
    df['los_label'] = (df['los_days'] >= 15).astype(int)

    # Create final formal label - any on the columns that ends with _label
    label_columns = [col for col in result.columns if col.endswith('_label')]
    result['label'] = result[label_columns].any(axis=1).astype(int)

    return result


def filter_to_real_cohort(df_cohort) -> pd.DataFrame:
    real_hids = [int(x) for x in os.listdir("data/csv") if x != "labels.csv"]
    df_cohort = df_cohort[df_cohort['hadm_id'].isin(real_hids)]
    return df_cohort


def create_multi_task_cohort(data_name_to_path: Dict[str, str], output_path: str):
    df_cohort = intersect_cohort(data_name_to_path)
    df_real_cohort = filter_to_real_cohort(df_cohort)

    df_real_cohort.to_csv(output_path)
