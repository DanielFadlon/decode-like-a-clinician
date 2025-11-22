
import pandas as pd

def composite_deterioration_label(cohort_path: str, data_dir_path: str):
    df_cohort = pd.read_csv(cohort_path)
    data_path = f"{data_dir_path}/df_all.csv.gz"
    print(df_cohort.head())
    print(df_cohort.shape)

    df = pd.read_csv(data_path, index_col=["Encrypted_PatNum", "Encrypted_CaseNum", "TimeFromHosp"])
    df = df.reset_index()
    # # assuming df is your dataframe
    df['label'] = 0  # start with all labels as 0

    # # Rule 1: col1 is True -> label = 1 for entire case
    hids_of_ihm = set(list(df_cohort[df_cohort['ihm_label'] == 1]['hadm_id']))
    print('   num_ ihm hids --- ', len(hids_of_ihm))
    df.loc[df['Encrypted_CaseNum'].isin(hids_of_ihm), 'label'] = 1

    # Rule 2: col1 is False, col2 is True, time >= 72 -> label = 1 for entire case
    hids_of_heart = set(list(df_cohort[(df_cohort['ihm_label'] == 0) & (df_cohort['heart_label'] == 1)]['hadm_id']))
    df.loc[((df['Encrypted_CaseNum'].isin(hids_of_heart)) & (df['TimeFromHospFeat'] > 70)), 'label'] = 1

    # # Rule 3: col1 is False, col3 is True, time >= 48 -> label = 1 for entire case
    hids_of_los = set(list(df_cohort[(df_cohort['ihm_label'] == 0) & (df_cohort['los15_label'] == 1)]['hadm_id']))
    df.loc[((df['Encrypted_CaseNum'].isin(hids_of_los)) & (df['TimeFromHospFeat'] > 47)), 'label'] = 1

    df = df.set_index(['Encrypted_PatNum', 'Encrypted_CaseNum', 'TimeFromHosp'])
    df.to_csv(data_path)
