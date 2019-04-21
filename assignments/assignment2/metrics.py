import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    #accuracy = np.mean(prediction == ground_truth)

    count = len(prediction)
    bool_idx = np.equal(prediction, ground_truth)
    accuracy = np.sum(bool_idx) / count

    return accuracy
