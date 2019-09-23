import numpy as np


def calMetric(X1, Y1):
    """
    Args:
         X1 (list): Prediction values for inliers
         Y1 (list): Prediction values for outliers
    """

    thre = 0.5
    tp = sum(np.array(X1) >= thre)
    fn = sum(np.array(X1) < thre)
    tn = sum(np.array(Y1) < thre)
    fp = sum(np.array(Y1) >= thre)

    print('\ntp: %d, fn: %d , tn: %d, fp: %d\n' % (tp, fn, tn, fp))

    min_delta = 0.0
    max_delta = 1.0

    ##################################################################
    # FPR at TPR 95
    ##################################################################
    fpr95 = 0.0
    clothest_tpr = 1.0
    dist_tpr = 1.0
    for delta in np.arange(min_delta, max_delta, 0.05):
        tpr = np.sum(np.greater_equal(X1, delta)) / np.float(len(X1))
        fpr = np.sum(np.greater_equal(Y1, delta)) / np.float(len(Y1))
        if abs(tpr - 0.95) < dist_tpr:
            dist_tpr = abs(tpr - 0.95)
            clothest_tpr = tpr
            fpr95 = fpr

    print("tpr: %.3f" % clothest_tpr)
    print("fpr95: %.3f" % fpr95)

    ##################################################################
    # Detection error
    ##################################################################
    detect_error = 1.0
    for delta in np.arange(min_delta, max_delta, 0.05):
        tpr = np.sum(np.less(X1, delta)) / np.float(len(X1))
        fpr = np.sum(np.greater_equal(Y1, delta)) / np.float(len(Y1))
        detect_error = np.minimum(detect_error, (tpr + fpr) / 2.0)

    print("Detection error: %.3f " % detect_error)

    ##################################################################
    # AUPR IN
    ##################################################################
    auprin = 0.0
    recallTemp = 1.0
    for delta in np.arange(min_delta, max_delta, 0.05):
        tp = np.sum(np.greater_equal(X1, delta))
        fp = np.sum(np.greater_equal(Y1, delta))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(X1))
        auprin += (recallTemp - recall) * precision
        recallTemp = recall
    auprin += recall * precision

    print("auprin: %.3f" % auprin)

    ##################################################################
    # AUPR OUT
    ##################################################################
    min_delta, max_delta = -max_delta, -min_delta
    X1 = [-x for x in X1]
    Y1 = [-x for x in Y1]
    auprout = 0.0
    recallTemp = 1.0
    for delta in np.arange(min_delta, max_delta, 0.05):
        tp = np.sum(np.greater_equal(Y1, delta))
        fp = np.sum(np.greater_equal(X1, delta))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(Y1))
        auprout += (recallTemp - recall) * precision
        recallTemp = recall
    auprout += recall * precision

    print("auprout: %.3f" % auprout)

    return fpr95, detect_error, auprin, auprout
