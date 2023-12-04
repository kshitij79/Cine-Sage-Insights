# Epoch	Training Loss	Validation Loss	Accuracy	F1	Roc Auc
# 1	0.243600	0.192742	0.216729	0.574738	0.926039
# 2	0.173500	0.185972	0.230961	0.622173	0.932597
# 3	0.140900	0.189130	0.236704	0.627551	0.933453
# 4	0.115800	0.199256	0.232210	0.626959	0.930859
# 5	0.099400	0.204741	0.221473	0.624435	0.928767
import matplotlib.pyplot as plt

loss = [0.243600, 0.173500, 0.140900, 0.115800, 0.099400]
val_loss = [0.192742, 0.185972, 0.189130, 0.199256, 0.204741]
accuracy = [0.216729, 0.230961, 0.236704, 0.232210, 0.221473]
f1 = [0.574738, 0.622173, 0.627551, 0.626959, 0.624435]
roc_auc = [0.926039, 0.932597, 0.933453, 0.930859, 0.928767]

epochs = [i for i in range(1, 6)]

plt.plot(epochs, loss, 'o-', label='Training Loss', markersize=3)
plt.plot(epochs, val_loss, 'o-', label='Validation Loss', markersize=3)
plt.plot(epochs, accuracy, 'o-', label='Accuracy', markersize=3)
plt.plot(epochs, f1, 'o-', label='Micro F1', markersize=3)
plt.plot(epochs, roc_auc, 'o-', label='Micro ROC AUC', markersize=3)

plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()
plt.show()