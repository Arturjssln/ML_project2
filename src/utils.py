import numpy as np
import matplotlib.pyplot as plt
from plot_functions import plot_confusion_matrix
from errors import result_scores

def results_statistics(y_pred, y_gt):

    TPR, FPR, F1, PREC = result_scores(y_pred, y_gt)

    print('True positive rate = ' + str(TPR))
    print('False positive rate = ' + str(FPR))
    print('F1 score = ' + str(F1))
    print('Precision  = ' + str(PREC))
    # Plot normalized confusion matrix
    plot_confusion_matrix(y_gt, y_pred, normalize = True)
    plt.show()
