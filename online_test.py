import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
predictions = np.load('predictions.npz')['data']

data = np.load('example_test.npz')['data']
test_l = [x[b'targets'] for x in data]
test_label_nohot = np.array(test_l, dtype=np.int32)
labels = np.eye(11)[test_label_nohot.reshape(-1)]
del data
del test_l

threshold_num = 100
thresholds = []
fpr_list = [] #false positive rate
tpr_list = [] #true positive rate
predictions_result = np.argmax(predictions,axis=1)
confusion = confusion_matrix(test_label_nohot,predictions_result)
plt.pcolormesh(confusion)
plt.show()

# for i in range(len(labels)):
#     threshold_step = 1. / threshold_num
#     for t in range(threshold_num+1):
#         th = 1 - threshold_num * t
#         fp = 0
#         tp = 0
#         tn = 0
#         for j in range(len(labels)):
#             for k in range(11):
#                 if not labels[j][k]:
#                     if predictions[j][k] >= t:
#                         fp += 1
#                     else:
#                         tn += 1
#                 elif predictions[j][k].any() >= t:
#                     tp += 1
#         fpr = fp / float(fp + tn)
#         tpr = tp / float(len(labels))
#         fpr_list.append(fpr)
#         tpr_list.append(tpr)
#         thresholds.append(th)
