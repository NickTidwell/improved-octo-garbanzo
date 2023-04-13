import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys

X = pd.read_csv("./kmeans_data/data.csv", header=None).to_numpy()
k = 10



# function to compute euclidean distance
def euclidean_distance(a, b):
    return np.sum((a - b)**2)

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def jaccard_distance(x,y):
    x = np.array(x)
    y = np.array(y)
    min_sum = np.sum(np.minimum(x,y))
    max_sum = np.sum(np.maximum(x,y))

    jaccard_index = min_sum / max_sum
    jaccard_dist = 1 - jaccard_index
    return jaccard_dist


def Kmean(X, k, op="cosine"):
    MAX_ITER = 100
    prev_sse = sys.maxsize
    operations = {
    "cosine": cosine_distance,
    "jaccard": jaccard_distance,
    "euclidean": euclidean_distance
    }
    print(f"Computing {op} with {k} clusters")
    centroids = np.array(X[np.random.choice(X.shape[0], k)])
    counter = 0
    while counter < MAX_ITER:
        C = np.array([np.argmin([operations[op](x_i,y_k) for y_k in centroids]) for x_i in X])
        new_centroids = [X[C == K].mean(axis = 0) for K in range(k)]
        counter += 1
        print(f"ITER: {counter}")
        if np.array_equal(centroids,new_centroids):
            break #Centroids not change stop
        centroids = new_centroids
        clusterwise_sse = [ 0 for _ in range(k)]
        for K in range(k):
            clusterwise_sse[K] += np.square(X[C == K] - centroids[K]).sum()
        sse = sum(clusterwise_sse)
        if sse >= prev_sse:
            break #SSE not shrunk
        prev_sse = sse
    return np.array(centroids), C, counter,clusterwise_sse


def run_kmeans_all():
    metrics = ['jaccard', 'cosine', 'euclidean']
    for metric in metrics:
        centroids,C,counter,sse = Kmean(X, k, metric)
        print(f"{metric} Converged in {counter} iterations with sse {sum(sse)}\n")

        with open(f"{metric}.csv", 'w') as f:
            for val in C:
                f.write(str(val) + "\n")


def run_five_fold():
    euclid_cnt_total = 0
    cos_cnt_total = 0
    jac_cnt_total = 0

    euclid_sse_total = 0
    cos_sse_total = 0
    jac_sse_total = 0

    STEPS = 5
    for i in range(STEPS):
        centroids,C,counter,sse = Kmean(X, k, "euclidean")
        print(f"Euclidean Converged in {counter} iterations\n")
        euclid_cnt_total += counter
        euclid_sse_total += sum(sse)

        centroids,C,counter,sse = Kmean(X, k, "cosine")
        print(f"Cosine Converged in {counter} iterations\n")
        cos_cnt_total += counter
        cos_sse_total += sum(sse)

        centroids,C,counter,sse = Kmean(X, k, "jaccard")
        print(f"jaccard Converged in {counter} iterations {sum(sse)}\n====================================\n")
        jac_cnt_total += counter
        jac_sse_total += sum(sse)
    print(f"euclidean avg convg: {float(euclid_cnt_total) / float(STEPS)}")
    print(f"euclidean avg sse: {float(euclid_sse_total) / float(STEPS)}")

    print(f"cosine avg convg: {float(cos_cnt_total) / float(STEPS)}")
    print(f"cosine avg sse: {float(cos_sse_total) / float(STEPS)}")

    print(f"jaccard avg convg: {float(jac_cnt_total) / float(STEPS)}")
    print(f"jaccard avg sse: {float(jac_sse_total) / float(STEPS)}")
