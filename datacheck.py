import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import torch
def binary_mean_iou_eval(y_true, y_pred,t=0.5):
    tp = ((y_pred > t) * (y_true > t)).sum()
    tn = ((y_pred <= t) * (y_true <= t)).sum()
    fp = ((y_pred > t) * (y_true <= t)).sum()
    fn = ((y_pred <= t) * (y_true > t)).sum()
    ret_list = []
    if tp+fp+fn != 0:
        ret_list.append(tp/(tp+fp+fn))
    if tn+fp+fn != 0:
        ret_list.append(tn/(tn+fp+fn))
    miou = torch.mean(torch.stack(ret_list))
    return miou

c = torch.tensor([[0,1,1,1,1,1,1,1,1,1,1,1]])
d = torch.tensor([[0,0,1,1,1,1,1,1,1,1,1,1]])
e = binary_mean_iou_eval(c,d)
print(e)
# f = mean_iou_eval(c,d)
# g = (e+f)/2
# print(g)
