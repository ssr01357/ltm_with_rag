import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.utils import resample
from scipy import stats

# Function to calculate metrics for a model on a dataset
def _calculate_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return precision, recall, f1

def _boostrap_resampling_report(y_true, y_pred):
    # Bootstrapping for each LLM
    n_bootstraps = 500  # Number of bootstrap samples
    results = []

    for _ in range(n_bootstraps):
        # Resample data with replacement
        y_true_boot, y_pred_boot = resample(y_true, y_pred, replace=True, random_state=_)

        # Calculate metrics for each LLM on the bootstrapped dataset
        precision_boot, recall_boot, f1_boot = _calculate_metrics(y_true_boot, y_pred_boot)

        # Store results
        results.append([precision_boot, recall_boot, f1_boot])

    # Calculate confidence intervals
    results = np.array(results)

    ci_lower = np.percentile(results, 2.5, axis=0)
    ci_upper = np.percentile(results, 97.5, axis=0)

    # Print results and confidence intervals
    print("Precision:", np.mean(results[:, 0]), "(CI:", ci_lower[0], ci_upper[0], ")")
    print("Recall:", np.mean(results[:, 1]), "(CI:", ci_lower[1], ci_upper[1], ")")
    print("F1:", np.mean(results[:, 2]), "(CI:", ci_lower[2], ci_upper[2], ")")

    return results

def ttest_bootstrapping_results(boot_rs1, boot_rs2):
    # 2 for f1
    f1_values_rs1 = boot_rs1[:, 2]
    f1_values_rs2 = boot_rs2[:, 2]

    t_statistic, p_value = stats.ttest_rel(f1_values_rs1, f1_values_rs2)
    print("F1 difference -> T-statistic: {} with p-value: {}".format(t_statistic,p_value))

def m01_performance_report(df, ans_col="ans_str"):
    df['coded_pred'] = df[ans_col].str.contains('Yes', case=False)
    df['Has_Yes_No'] = df[ans_col].str.contains('Yes|No', case=False)
    effective_index = df["Has_Yes_No"] == True
    coded_pred = df[effective_index]['coded_pred'].to_list()
    m_labels = df[effective_index]["m"].to_list()

    tn, fp, fn, tp = confusion_matrix(m_labels, coded_pred).ravel()
    print("tn={}, fp={}, fn={}, tp={}".format(tn, fp, fn, tp))
    target_names = ['M0', 'M1']
    print(classification_report(m_labels, coded_pred, target_names=target_names))

    boot_rs = _boostrap_resampling_report(m_labels, coded_pred)
    return boot_rs
    
def t14_performance_report(df, ans_col="ans_str"):
    # check if the ans_col contain any valid prediction (e.g., T1, T2, T3, T4)
    df['Has_Valid_Prediction'] = df[ans_col].str.contains('T1|T2|T3|T4', case=False)
    # transform the prediction string to code
    # note that following the t column we set T1 = 0, ... T4 = 3 
    coded_pred_list = []
    for _, row in df.iterrows():
        if "T1" in row[ans_col]:
            coded_pred_list.append(0)
        elif "T2" in row[ans_col]:
            coded_pred_list.append(1)
        elif "T3" in row[ans_col]:
            coded_pred_list.append(2)
        elif "T4" in row[ans_col]:
            coded_pred_list.append(3)
        else:
            # unvalid answers 
            # Has_Valid_Prediction == False
            coded_pred_list.append(-1)
    df['coded_pred'] = coded_pred_list

    effective_index = df["Has_Valid_Prediction"] == True
    coded_pred = df[effective_index]['coded_pred'].to_list()
    t_labels = df[effective_index]["t"].to_list()

    target_names = ['T1', 'T2', 'T3', 'T4']
    print(classification_report(t_labels, coded_pred, target_names=target_names))

    boot_rs = _boostrap_resampling_report(t_labels, coded_pred)
    return boot_rs

def n03_performance_report(df, ans_col="ans_str"):
    # check if the ans_col contain any valid prediction (e.g., T1, T2, T3, T4)
    df['Has_Valid_Prediction'] = df[ans_col].str.contains('N0|N1|N2|N3', case=False)
    # transform the prediction string to code
    coded_pred_list = []
    for _, row in df.iterrows():
        row[ans_col] = str(row[ans_col])
        if "N0" in row[ans_col]:
            coded_pred_list.append(0)
        elif "N1" in row[ans_col]:
            coded_pred_list.append(1)
        elif "N2" in row[ans_col]:
            coded_pred_list.append(2)
        elif "N3" in row[ans_col]:
            coded_pred_list.append(3)
        else:
            # unvalid answers 
            # Has_Valid_Prediction == False
            coded_pred_list.append(-1)
    df['coded_pred'] = coded_pred_list

    effective_index = df["Has_Valid_Prediction"] == True
    coded_pred = df[effective_index]['coded_pred'].to_list()
    n_labels = df[effective_index]["n"].to_list()

    target_names = ['N0', 'N1', 'N2', 'N3']
    print(classification_report(n_labels, coded_pred, target_names=target_names))

    boot_rs = _boostrap_resampling_report(n_labels, coded_pred)
    return boot_rs