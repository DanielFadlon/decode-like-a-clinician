from datetime import datetime
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def split_train_test_by_years(data_path: str, ratio: int, plan_key: int):
    if ratio not in [25, 50, 75]:
        raise Exception("Error! Currently supported just - 25%, 50%, 75%")

    df = pd.read_csv(data_path)
    df['admittime'] = pd.to_datetime(df['admittime'])
    # TODO: use percentiles instead of describe.
    all_times = df['admittime'].describe()

    df_train = df[df['admittime'] < all_times[f'{ratio}%']]

    df_val_test = df[df['admittime'] >= all_times[f'{ratio}%']]
    valid_test_times = df_val_test['admittime'].describe()

    match plan_key:
        case 1:
            "Smaller (val/test) data - extract quarter from the remaining data"
            df_val = df_val_test[df_val_test['admittime'] <= valid_test_times['25%']]
            df_test = df_val_test[(df_val_test['admittime'] > valid_test_times['25%']) & (df_val_test['admittime'] <= valid_test_times['50%'])]
        case 2:
            "Smaller (train) data"
            pass
        case 3:
            "Large (val/test) data 50/50 on the remain 25%"
            df_val = df_val_test[df_val_test['admittime'] <= valid_test_times['50%']]
            df_test = df_val_test[df_val_test['admittime'] > valid_test_times['50%']]

    # TODO: Re-consider the return format
    return df_train, df_val, df_test


def concat_set_types(data_dir_path: str):
    dfs = [pd.read_csv(f"{data_dir_path}/{set}.csv.gz")for set in ['train', 'valid', 'test']]
    res = pd.concat(dfs)
    res.to_csv(f"{data_dir_path}/df_all.csv.gz")

    # Remove sets
    for set in ['train', 'valid', 'test']:
        os.remove(f"{data_dir_path}/{set}.csv.gz")


def build_admissions_dataset(data_path: str, output_dir_path: str, version: str, split_plan_key: int):
    cohort_train, cohort_val, cohort_test = split_train_test_by_years(data_path, 75, split_plan_key)

    def post_process_hid(hid, label, set_type, version):
        data_path = f'data/csv/{hid}/final_{version}.csv'
        try:
            hid_data = pd.read_csv(data_path)
            if hid_data.empty:
                print(f"{data_path} is empty")
                return None

            # fix fill forward
            hid_data.replace([0, 0.0, '0.0', '0'], np.nan, inplace=True)
            hid_data.ffill(inplace=True)

            if 'Unnamed: 0' in hid_data.columns:
                hid_data.drop(columns=['Unnamed: 0'], inplace=True)

            hid_data['label'] = label
            hid_data['set_type'] = set_type
            return hid_data
        except:
            print(f"   csv file is empty! - {data_path}")
            return None


    def process_set(cohort_df, set_type: str):
        print(f"   [START PROCESS {set_type} SET]")
        hids = cohort_df['hadm_id']
        label_series = cohort_df.set_index('hadm_id')['label']
        hids_dfs_list = []

        # for hid in hids:
        #     hids_dfs_list.append(post_process_hid(hid, label_series.get(hid, None), set_type, version))

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {
                executor.submit(post_process_hid, hid, label_series.get(hid, None), set_type, version): hid
                for hid in hids
            }

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    hids_dfs_list.append(result)

        if hids_dfs_list:
            hids_dfs = pd.concat(hids_dfs_list, ignore_index=True)
            hids_dfs.to_csv(f'{output_dir_path}/{set_type}.csv.gz')

    process_set(cohort_train, "train")
    process_set(cohort_val, "valid")
    process_set(cohort_test, "test")
    print("   [ Concat SETS ]")
    concat_set_types(output_dir_path)
    return
