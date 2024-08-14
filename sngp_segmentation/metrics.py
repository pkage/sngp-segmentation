import numpy as np
from torcheval import metrics


def expected_calibration_error(predicted, true_labels, percentiles = 10):
    bin_boundaries = np.linspace(0, 1, percentiles + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(predicted, axis=1) # could use Dempster-Shafer u(xi) instead? See appendix C.2 of https://arxiv.org/pdf/2006.10108.pdf
    predicted_label = np.argmax(predicted, axis=1) 
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        prob_in_bin = in_bin.mean()
        if prob_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece

def negative_log_liklihood(predicted, true_labels):
    return -np.mean(np.log(predicted[np.arange(len(true_labels)), true_labels]))

def brier_score(predicted, true_labels):
    return np.mean(np.sum((predicted - np.eye(predicted.shape[1])[true_labels])**2, axis=1))



def dempster_shafer_OOD(predicted, true_labels):
    
    K = len(true_labels)
    ds = K / K - np.sum(np.exp(predicted), axis=1)
    return ds
    
# TODO : Needs work. Not tested.  ///////////////////////////////////
def auprc_OOD(predicted, true_labels, n_classes):
    # returns Area Under the Precision-Recall Curve for either binary or multiclass.
    if n_classes == 2:
        metric = BinaryAUPRC()
        return metric(predicted, true_labels)
    else:
        metric = MultiClassAUPRC(num_classes = n_classes)
        return metric(predicted, true_labels)
