import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def get_alignment_mapping(reference_archetypes, current_archetypes):
    n_arch = reference_archetypes.shape[0]
    dists = np.zeros((n_arch, n_arch))
    for i in range(n_arch):
        for j in range(n_arch):
            dists[i, j] = np.linalg.norm(reference_archetypes[i] - current_archetypes[j])

    row_ind, col_ind = linear_sum_assignment(dists)
    mapping = np.zeros(n_arch, dtype=int)
    for ref_idx, curr_idx in zip(row_ind, col_ind):
        mapping[curr_idx] = ref_idx
    return mapping


def apply_mapping(labels, mapping):
    labels_0 = np.asarray(labels) - 1
    return (np.array([mapping[p] for p in labels_0]) + 1).tolist()


def align_predictions(reference_archetypes, current_archetypes, predictions):
    mapping = get_alignment_mapping(reference_archetypes, current_archetypes)
    return apply_mapping(predictions, mapping)


def accumulate_results(all_true, all_preds_by_model, y_true, loocv_results):
    all_true.append(list(y_true))
    entry = {}
    for mname, mres in loocv_results.items():
        entry[mname] = list(mres['predictions'])
    all_preds_by_model.append(entry)
    return all_true, all_preds_by_model


def compute_aggregate_confusion(all_true, all_preds_by_model, n_classes=3):
   
    agg = {}
    for model_name in all_preds_by_model[0].keys():
        cm_sum = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(len(all_true)):
            cm_sum += confusion_matrix(all_true[i], all_preds_by_model[i][model_name],
                                       labels=list(range(1, n_classes + 1)))
        agg[model_name] = cm_sum
    return agg
