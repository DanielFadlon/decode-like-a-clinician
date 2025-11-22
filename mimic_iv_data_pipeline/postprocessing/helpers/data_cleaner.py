import pandas as pd

def clean_features(data_path: str, th: int = 100):
    print("   [Clean Data Features]")
    df = pd.read_csv(f"{data_path}")
    print('   data before cleaning')
    print(df.head())
    desc = df.describe()
    print("   Shape Before Cleaning - ", df.shape)

    empty_features = []
    for key, count in dict(desc.loc['count', :]).items():
        if count < th:
            empty_features.append(key)

    print(f"   Remove EMPTY FEATURES  (c<{th}): ")
    print(empty_features)
    df = df.drop(columns=empty_features)

    print(f"   SET Index (if needed)")
    if 'hadm_id' in df.columns:
        df = df.rename(columns={'subject_id': 'Encrypted_PatNum', 'hadm_id': 'Encrypted_CaseNum'})
        df = df.set_index(['Encrypted_PatNum', 'Encrypted_CaseNum', 'TimeFromHosp'])

    print("   Remove UNNamed cols")
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
    num_unnamed_cols = len(unnamed_cols)
    if num_unnamed_cols > 0:
        print(f"Drop {num_unnamed_cols} 'Unnamed' cols")
        df = df.drop(columns=unnamed_cols)

    print("   Shape After Cleaning - ", df.shape)

    return df


def remove_cross_set_patients(df: pd.DataFrame):
    """
    Removes overlapping subjects (patients) across dataset splits to ensure strict separation.

    Specifically:
    1. Removes patients from the validation and test sets if they appear in the training set.
    2. Removes patients from the test set if they appear in the validation set.
    """
    def get_set_data(df: pd.DataFrame, set_type: str):
        return df[df['set_type'] == set_type]

    train = get_set_data(df, "train")
    valid = get_set_data(df, "valid")
    test = get_set_data(df, "test")

    train_subjects = set(list(train.reset_index()['Encrypted_PatNum']))
    valid_subjetcs =set(list(valid.reset_index()['Encrypted_PatNum']))
    test_subjects =set(list(test.reset_index()['Encrypted_PatNum']))

    cross_train_valid = train_subjects.intersection(valid_subjetcs)
    cross_train_test = train_subjects.intersection(test_subjects)
    cross_valid_test = valid_subjetcs.intersection(test_subjects)

    cross_subjects = cross_train_valid | cross_train_test | cross_valid_test
    print("   Num. duplicates subjects -- ", len(cross_subjects))

    hids = set([])
    for subject_id in cross_subjects:
        subject_df = df.xs(subject_id, level='Encrypted_PatNum').reset_index()
        subject__val_test = subject_df[subject_df['set_type'].isin(['test', 'valid'])]
        hids = hids | set(list(subject__val_test['Encrypted_CaseNum']))

    print("   Num admissions to remove (from valid and test sets) - ", len(hids))

    df = df[~df.index.get_level_values('Encrypted_CaseNum').isin(hids)]
    return df


def clean_data(data_dir_path:str):
    data_path = f"{data_dir_path}/df_all.csv.gz"
    df = clean_features(data_path)
    df = remove_cross_set_patients(df)
    df.to_csv(data_path)

