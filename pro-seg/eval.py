import numpy as np
from medpy import metric


def multi_classes_dice(name, pred, label, num_classes=16):
    all_dc = []
    all_dc.append(name)

    for i in range(num_classes):
        label_one = label == i
        label_one.astype(np.int32)

        pred_one = pred == i
        pred_one.astype(np.int32)

        dc = metric.binary.dc(pred_one, label_one)
        if dc == 0:
            all_dc.append(-1)
        else:
            all_dc.append(dc)
    return all_dc

