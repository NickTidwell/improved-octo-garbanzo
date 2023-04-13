import pandas as pd
import numpy as np
#Creates Confusion Matrix to match True labels with our labels, then computes accuracy
k = 10
labels = pd.read_csv("kmeans_data/label.csv", header=None).to_numpy()

def get_accuracy(metric):
    preds = pd.read_csv(f"{metric}.csv", header=None).to_numpy()
    matrix = np.zeros((k,k))
    for pred, label in zip(preds, labels):
        x = label[0]
        y = pred[0]
        matrix[x][y] += 1
    cm_argmax = matrix.argmax(axis=1)

    cm_map = {}
    for i in range(k):
        cm_map[i] = cm_argmax[i]
    acc = 0
    for pred, label in zip(preds, labels):
        if pred[0] == cm_map[label[0]]:
            acc += 1
    print(metric, ":\t", float(acc)/float(len(labels))*100)

metrics = ['jaccard', 'cosine', 'euclidean']
for metric in metrics:
    get_accuracy(metric)