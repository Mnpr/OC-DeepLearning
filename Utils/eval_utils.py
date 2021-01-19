
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


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

def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals