import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.utils import resample
from scipy import stats

# new metrics
def t14_calculate_metrics(true_labels: pd.Series, predictions: pd.Series) -> dict:

    # Check for valid inputs
    if len(true_labels) != len(predictions):
        raise ValueError("The length of true_labels and predictions must be the same.")
    
    if any(not isinstance(pred, str) for pred in predictions):
        print(pred)
        # raise ValueError("All predictions must be non-null strings.")
    
    true_labels = true_labels.apply(lambda x: f'T{x+1}')

    metrics = {}
    label_counts = {}
    
    for label in set(true_labels):
        metrics[label] = {'tp': 0, 'fp': 0, 'fn': 0}
        label_counts[label] = 0

    for true_label, prediction in zip(true_labels, predictions):
        prediction = prediction.upper()
        label_counts[true_label] += 1
        if true_label in prediction:
            metrics[true_label]['tp'] += 1
        else:
            metrics[true_label]['fn'] += 1
        
        for label in metrics:
            if label in prediction and label != true_label:
                metrics[label]['fp'] += 1
    
    results = {}
    total_tp = total_fp = total_fn = 0
    macro_precision = macro_recall = macro_f1 = 0
    total_instances = len(true_labels)
    
    for label, counts in metrics.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = label_counts[label]
        
        results[label] = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'support': support,
            'num_errors': fp + fn
        }
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    # Calculate macro-averaged metrics
    num_labels = len(metrics)
    macro_precision /= num_labels
    macro_recall /= num_labels
    macro_f1 /= num_labels

    # Calculate overall (micro-averaged) precision, recall, and F1 score
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

    # Calculate weighted (balanced) F1 score
    weighted_f1 = sum(results[label]['f1'] * label_counts[label] for label in metrics) / total_instances

    results['overall'] = {
        # 'precision': round(total_precision, 3),
        # 'recall': round(total_recall, 3),
        # 'f1': round(total_f1, 3),
        'macro_precision': round(macro_precision, 3),
        'macro_recall': round(macro_recall, 3),
        'macro_f1': round(macro_f1, 3),
        # 'weighted_f1': round(weighted_f1, 3),
        'support': total_instances,
        'num_errors': total_fp + total_fn
    }
    
    return results

def n03_calculate_metrics(true_labels: pd.Series, predictions: pd.Series) -> dict:

    # Check for valid inputs
    if len(true_labels) != len(predictions):
        raise ValueError("The length of true_labels and predictions must be the same.")
    
    if any(not isinstance(pred, str) for pred in predictions):
        raise ValueError("All predictions must be non-null strings.")
    
    true_labels = true_labels.apply(lambda x: f'N{x}')

    metrics = {}
    label_counts = {}
    
    for label in set(true_labels):
        metrics[label] = {'tp': 0, 'fp': 0, 'fn': 0}
        label_counts[label] = 0

    for true_label, prediction in zip(true_labels, predictions):
        prediction = prediction.upper()
        prediction = prediction.replace("NO", "N0").replace("NL", "N1")
        label_counts[true_label] += 1
        if true_label in prediction:
            metrics[true_label]['tp'] += 1
        else:
            metrics[true_label]['fn'] += 1
        
        for label in metrics:
            if label in prediction and label != true_label:
                metrics[label]['fp'] += 1
    
    results = {}
    total_tp = total_fp = total_fn = 0
    macro_precision = macro_recall = macro_f1 = 0
    total_instances = len(true_labels)
    
    for label, counts in metrics.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = label_counts[label]
        
        results[label] = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'support': support,
            'num_errors': fp + fn
        }
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    # Calculate macro-averaged metrics
    num_labels = len(metrics)
    macro_precision /= num_labels
    macro_recall /= num_labels
    macro_f1 /= num_labels

    # Calculate overall (micro-averaged) precision, recall, and F1 score
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

    # Calculate weighted (balanced) F1 score
    weighted_f1 = sum(results[label]['f1'] * label_counts[label] for label in metrics) / total_instances

    results['overall'] = {
        # 'precision': round(total_precision, 3),
        # 'recall': round(total_recall, 3),
        # 'f1': round(total_f1, 3),
        'macro_precision': round(macro_precision, 3),
        'macro_recall': round(macro_recall, 3),
        'macro_f1': round(macro_f1, 3),
        # 'weighted_f1': round(weighted_f1, 3),
        'support': total_instances,
        'num_errors': total_fp + total_fn
    }
    
    return results