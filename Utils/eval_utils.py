# Check Data leakage between (Test) and (Train,Dev) Dataframes
def check_data_leakage(df1, df2, patient_col):
    """
    Return: True if leakage, false otherwise (bool)

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_key (str): string name of column with patient IDs
    """
   
    df1_uniq_pid = set(df1[patient_col])
    df2_uniq_pid = set(df2[patient_col])
    
    # leakage = True if at_least overlaping pid
    pid_intersection = df1_uniq_pid.intersection(df2_uniq_pid)
    if pid_intersection:
        leakage = True 
    else:
        leakage = False
   
    return leakage