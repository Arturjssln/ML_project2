import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          log = False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if log:
            print("Normalized confusion matrix")
    else:
        if log:
            print('Confusion matrix, without normalization')
    if log:
        print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def results_statistics(y_pred, y_gt, log = False):
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
    if log:
        print('True positive rate = ' + str(TPR))
        print('False positive rate = ' + str(FPR))
        print('F1 score = ' + str(F1))

        # Plot normalized confusion matrix
        plot_confusion_matrix(y_gt, y_pred, normalize = True)
        plt.show()

    return TPR, FPR, F1
