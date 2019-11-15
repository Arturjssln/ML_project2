import numpy as np

def mse(image1, image2):
    """
    This function calculate the mean square error between 2 images
    """
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(np.size(image1))
    assert(np.size(image1) == np.size(image2))
    return err

def result_scores(y_pred, y_gt):
    y_pred_n = np.nonzero(y_pred)[0]
    y_gt_n = np.nonzero(y_gt)[0]
    y_pred_z = np.where(y_pred == 0)[0]
    y_gt_z = np.where(y_gt == 0)[0]

    TP = len(list(set(y_pred_n) & set(y_gt_n)))
    TN = len(list(set(y_pred_z) & set(y_gt_z)))
    FP = len(list(set(y_pred_n) & set(y_gt_z)))
    FN = len(list(set(y_pred_z) & set(y_gt_n)))

    TPR = TP / float(TP + FN)
    FPR = FP / float(FP + TN)
    F1 = 2 * TP / float(2 * TP + FN + FP)

    return TPR, FPR, F1
