from datetime import datetime, timedelta
import json
import os
import concurrent.futures
import numpy as np
import pandas as pd
from tqdm import tqdm



# --------------------------------- UTILS --------------------------------- #
def load_json(file_path: str):
    # Load the JSON content into a Python dictionary
    with open(file_path, 'r') as file:
        json_object = json.load(file)

    return json_object


# --------------------------------- DYNAMIC FEATS --------------------------------- #

def get_labs_map(ids_row: pd.Series, selected_labs):
    cols_mapping = {}
    for lab_obj in selected_labs:
        lab_key = lab_obj['label'].replace(" ", "_")
        itemid = lab_obj['id']
        if isinstance(itemid, int):
            cols_mapping[lab_key] = ids_row[ids_row == itemid].index[0]
        elif isinstance(itemid, list):
            cols_mapping[lab_key] = list(ids_row[ids_row.isin(itemid)].index)

    # Flatten single items list
    cols_mapping = {
        k: v[0] if isinstance(v, list) and len(v) == 1 else v
        for k, v in cols_mapping.items()
    }
    return cols_mapping


def get_proc_map(ids_row: pd.Series, selected_proc):
    cols_mapping = {}
    for proc in selected_proc:
        proc_key = proc['label'].replace(" ", "_")
        icd_code = proc['icd_code']
        cols_mapping[proc_key] = ids_row[ids_row == icd_code].index[0]

    return cols_mapping


def get_meds_map(ids_row: pd.Series, selected_meds):
    cols_mapping = {}
    for med_group_key in selected_meds:
        cols_mapping[med_group_key] = []
        for med_name in selected_meds[med_group_key]:
            try:
                cols_mapping[med_group_key].append(
                    ids_row[ids_row == med_name].index[0]
                )
            except Exception as e:
                print("Error for med name -- ", med_name)

    return cols_mapping


def get_all_feats(labs_map, proc_map, meds_map):
    def to_list_of_values(d):
        values_list = []
        for v in d.values():
            if isinstance(v, list):
                values_list.extend(v)
            else:
                values_list.append(v)
        return values_list

    labs_col_names = to_list_of_values(labs_map)
    meds_col_names = to_list_of_values(meds_map)
    proc_col_names = to_list_of_values(proc_map)

    return labs_col_names + meds_col_names + proc_col_names

def process_features(df: pd.DataFrame, map):
    def get_first_valid(row, union_list):
        for x in union_list:
            col = x['col_name'] if isinstance(x, dict) else x
            val = row[col]
            if pd.notna(val) and val != 0 and val != '' and val != '0.0' and val != '0': # TODO: parse values to numbers for 'proc' and maybe 'meds'
                return x['value'] if isinstance(x, dict) else val

        return None

    for k, v in map.items():
        if isinstance(v, list):
            df[k] = df.apply(lambda row: get_first_valid(row, v), axis=1)
            cols_to_remove = [x['col_name'] for x in v] if isinstance(v[0], dict) else v
            df = df.drop(columns=cols_to_remove)
        else:
            df = df.rename(columns={v: k})
    return df



def extract_admission_dynamics(hid_dir_path: str):
    df_dynamic = pd.read_csv(f"{hid_dir_path}/dynamic.csv")
    ids_row = df_dynamic.iloc[0]

    # Selected Features Configuration
    top40_selection  = load_json("data/myselection/top_40.json")

    # Get Feature Mapping
    labs_map = get_labs_map(ids_row, top40_selection['labs'])
    proc_map = get_proc_map(ids_row, top40_selection['procedures'])
    meds_map = get_meds_map(ids_row, top40_selection['medications'])

    relevant_features = get_all_feats(labs_map, proc_map, meds_map)
    df_dynamic = df_dynamic[relevant_features]

    df = process_features(df_dynamic.iloc[1:], labs_map | meds_map | proc_map)

    # Fill Forward
    return df.ffill()


def extract_admission_static(hid_dir_path: str):
    df_static = pd.read_csv(f"{hid_dir_path}/static.csv")
    # ids_row = df_dynamic.iloc[0]
    print(df_static.head())


# --------------------------------- Add Time --------------------------------- #
def add_time(df: pd.DataFrame):
    # TODO: Consider using the real time
    # Currently ignored -- since it is unavailable in real world scenario.
    df['TimeFromHosp'] = pd.to_timedelta(df.index * 4, unit='h')
    df['TimeFromHospFeat'] = df.index * 4

    return df


def get_admission_df(hid):
    hid_path = f'data/csv/{hid}'
    # df_los = pd.read_csv("data/cohort/cohort_non-icu_length_of_stay_10_REAL.csv.gz")
    df = extract_admission_dynamics(hid_path)
    # Add Demographics
    df_demo = pd.read_csv(f"{hid_path}/demo.csv")
    if df_demo.shape[0] != 1:
        print(f"Warning! hid={hid_path}, df_demo_feats shape is ", df_demo.shape[0])
    demo_feats = dict(df_demo.iloc[0])
    df = df.assign(
        hadm_id = hid,
        Age=demo_feats['Age'],
        gender=demo_feats['gender'],
        insurance=demo_feats['insurance'],
    )

    # Handle Time
    df = add_time(df)
    return df



def worker(hids, hid_to_subject_map, version):
    num_admissions = len(hids)
    for i, hid in enumerate(hids):
        if i % 500 == 0 and i != 0:
            print(f"{i}/{num_admissions}")
        hid_dir_path = f"data/csv/{hid}" # TODO: agnostic to real argument
        if os.path.exists(f"{hid_dir_path}/final_{version}.csv"):
            continue

        df = get_admission_df(hid)
        df['subject_id'] = hid_to_subject_map[hid]
        df.to_csv(f"{hid_dir_path}/final_{version}.csv")


def post_process_admissions(hids, hid_to_subject_map, version):
    num_cases = len(hids)
    print("Total num cases =", num_cases)
    num_threads = 32  # Number of threads to use
    range_per_thread = num_cases // num_threads  # Each thread works on an equal range


    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        print('   [ START PROCESS ADMISSIONS ]')
        futures = []

        for i in range(num_threads):
            start = i * range_per_thread
            end = (i + 1) * range_per_thread if i != num_threads - 1 else num_cases  # Handle last thread's range
            print(f'dispatch hids - {start} - {end}')
            futures.append(executor.submit(worker, hids[start:end], hid_to_subject_map, version))

        # Wait for all threads to finish
        concurrent.futures.wait(futures)

    print("   [ DONE ]")
