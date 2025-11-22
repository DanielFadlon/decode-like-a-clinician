from typing import Optional
import pandas as pd
from postprocessing.helpers.admission_feature_builder import post_process_admissions
from postprocessing.helpers.build_dataset import build_admissions_dataset
from postprocessing.helpers.data_cleaner import clean_data
from postprocessing.helpers.labeling_pipeline import composite_deterioration_label
from postprocessing.helpers.numeric_data_creator import to_numeric


def post_process_data(
        cohort_path: str,
        version: str,
        label_plan_key, # Currently, binary compose or not. But enabling future definition by key.
        split_plan_key: Optional[int] = 1,
        output_dir_path: Optional[str] = None
    ):

    df_cohort = pd.read_csv(cohort_path)
    admissions_to_subjects_map = dict(zip(df_cohort['hadm_id'], df_cohort['subject_id']))
    hids = list(set(df_cohort['hadm_id']))

    if output_dir_path is None:
        output_dir_path = f"data/multi_{version}/p{split_plan_key}"

    print(" [START POST-PROCESSING]")
    print(f" [ CREATE FINAL ADMISSION -- VERSION == {version}]")
    # 1- addmission feature builder
    post_process_admissions(hids, admissions_to_subjects_map, version=version)
    print(f" [ FINISH ADMISSION DATA CREATION ]")
    print()
    # 2 - build dataset - one dataframe with sets (train/valid/test)
    print("[CREATE DATA FRAME OF ALL ADMISSIONS]")
    build_admissions_dataset(cohort_path, output_dir_path, version=version, split_plan_key=split_plan_key)
    # 3 - Clean data
    clean_data(output_dir_path)
    # 4 - Label pipeline if needed
    if label_plan_key == 1:
        print('[ COMPOSITE LABEL ]')
        composite_deterioration_label(cohort_path, output_dir_path)
    else:
        print(" ---- Skip Label composition -- ")
    # 5 - Create Numeric Data
    print('[ CREATE NUMERIC DATA]')
    to_numeric(output_dir_path)
    print('[ FINISH ]')
