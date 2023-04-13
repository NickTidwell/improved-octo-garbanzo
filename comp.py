import pandas as pd
import numpy as np
k = 10
labels = pd.read_csv("kmeans_data/label.csv", header=None).to_numpy()
preds = pd.read_csv("cosine.csv", header=None).to_numpy()

matrix = np.zeros((k,k))
for pred, label in zip(preds, labels):
    x = label[0]
    y = pred[0]
    matrix[x][y] += 1
print(matrix)
cm_argmax = matrix.argmax(axis=1)

cm_map = {}
for i in range(k):
    cm_map[i] = cm_argmax[i]
acc = 0
print(cm_map)
for pred, label in zip(preds, labels):
    if pred[0] == cm_map[label[0]]:
        acc += 1
print(cm_argmax)
print(float(acc)/float(len(labels))*100)