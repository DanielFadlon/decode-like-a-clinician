import pandas as pd


def to_numeric(data_dir_path: str):
    df = pd.read_csv(f"{data_dir_path}/df_all.csv.gz", index_col=["Encrypted_PatNum", "Encrypted_CaseNum", "TimeFromHosp"])

    start_with_class_number_features = set(['ECMO', 'resp_vent_severity', 'cardiac_support', 'tracheostomy_approach',
                                            'coronary_bypass', 'autologous_blood_transfusion'])
    medications_boolean_str = set(['restore_cardiac'])
    uniques = set(["gender", "insurance", "set_type"])
    known_non_numeric_cols = start_with_class_number_features.union(medications_boolean_str, uniques)

    non_numeric_cols = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype not in ['float', 'int']:
            if col not in known_non_numeric_cols:
                print('_' * 20)
                print()
                print(f"Warning! the col {col} type is not numeric. dtype={dtype}. BUT it doesn't exist in 'known_non_numeric_cols'")
                print()
                print("_" * 20)
            non_numeric_cols.append(col)
            print(df[df[col].notna()][col])

    for col in non_numeric_cols:
        if col == 'set_type':
            continue

        if col in start_with_class_number_features:
            df[col] = df[col].apply(lambda x: int(x[0]) if pd.notna(x) else x)

        if col in medications_boolean_str:
            df[col] = df[col].apply(lambda x: 1 if pd.notna(x) else 0)

        # Uniques
        if col == "gender":
            df[col] = df[col].apply(lambda x: 1 if x == "M" else 0)

        if col == "insurance":
            map_insurance = {
                "Medicaid": 0,
                "Medicare": 1,
                "Other": 2
            }
            df[col] = df[col].apply(lambda x: map_insurance[x])

    df.to_csv(f"{data_dir_path}/df_all_numeric.csv.gz")
