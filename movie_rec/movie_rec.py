import pandas as pd
import numpy as np
from surprise import Dataset, SVD
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise import KNNWithMeans

ratings_pd = pd.read_csv("data/ratings_small.csv")
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(ratings_pd[['userId', 'movieId', 'rating']], reader)


def find_best_k():
    user_out = []
    item_out = []

    for i in range(1, 100):

        #USER COSINE TEST
        user_cosine = KNNWithMeans(k=i, sim_options={'name': 'cosine', 'user_based': True})
        res = cross_validate(user_cosine, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
        user_out.append(["user_cosine",np.mean(res['test_rmse']), np.mean(res['test_mae']), i])
        #USER MSD TEST
        user_msd = KNNWithMeans(k=i, sim_options={'name': 'msd', 'user_based': True})
        res = cross_validate(user_msd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
        user_out.append(["user_msd",np.mean(res['test_rmse']), np.mean(res['test_mae']), i])
        #USER PEARSON TEST
        user_pearson = KNNWithMeans(k=i, sim_options={'name': 'pearson', 'user_based': True})
        res = cross_validate(user_pearson, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
        user_out.append(["user_pearson",np.mean(res['test_rmse']), np.mean(res['test_mae']), i])

        #ITEM COSINE TEST
        item_cosine = KNNWithMeans(k=i, sim_options={'name': 'cosine', 'user_based': False})
        res = cross_validate(item_cosine, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
        item_out.append(["item_cosine",np.mean(res['test_rmse']), np.mean(res['test_mae']), i])
        #ITEM MSD TEST
        item_msd = KNNWithMeans(k=i, sim_options={'name': 'msd', 'user_based': False})
        res = cross_validate(item_msd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
        item_out.append(["item_msd",np.mean(res['test_rmse']), np.mean(res['test_mae']), i])
        #ITEM PEARSON TEST
        item_pearson = KNNWithMeans(k=i, sim_options={'name': 'pearson', 'user_based': False})
        res = cross_validate(item_pearson, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
        item_out.append(["item_pearson",np.mean(res['test_rmse']), np.mean(res['test_mae']), i])

    user_out_pd = pd.DataFrame(user_out, columns=["Algo_Name", "RMSE", "MAE", "K"])
    user_out_pd.to_csv("./output/user_out.csv")
    item_out_pd = pd.DataFrame(item_out, columns=["Algo_Name", "RMSE", "MAE", "K"])
    item_out_pd.to_csv("./output/item_out.csv")

def evaluate_all_sims():
    # Run 5-fold cross-validation and print results
    pmf = SVD(biased=False)

    user_cosine = KNNWithMeans(k=50, sim_options={'name': 'cosine', 'user_based': True})
    user_msd = KNNWithMeans(k=50, sim_options={'name': 'msd', 'user_based': True})
    user_pearson = KNNWithMeans(k=50, sim_options={'name': 'pearson', 'user_based': True})

    item_cosine = KNNWithMeans(k=50, sim_options={'name': 'cosine', 'user_based': False})
    item_msd = KNNWithMeans(k=50, sim_options={'name': 'msd', 'user_based': False})
    item_pearson = KNNWithMeans(k=50, sim_options={'name': 'pearson', 'user_based': False})


    output = []
    res = cross_validate(pmf, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
    output.append(["PMF",np.mean(res['test_rmse']), np.mean(res['test_mae'])])

    res = cross_validate(user_cosine, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
    output.append(["user_cosine",np.mean(res['test_rmse']), np.mean(res['test_mae'])])
    res = cross_validate(user_msd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
    output.append(["user_msd", np.mean(res["test_rmse"]), np.mean(res["test_mae"])])
    res = cross_validate(user_pearson, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
    output.append(["user_pearson", np.mean(res["test_rmse"]), np.mean(res["test_mae"])])

    res = cross_validate(item_cosine, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
    output.append(["item_cosine", np.mean(res["test_rmse"]), np.mean(res["test_mae"])])
    res = cross_validate(item_msd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
    output.append(["item_msd", np.mean(res["test_rmse"]), np.mean(res["test_mae"])])
    res = cross_validate(item_pearson, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
    output.append(["item_pearson", np.mean(res["test_rmse"]), np.mean(res["test_mae"])])

    out_fd = pd.DataFrame(output, columns=["Algo_Name", "RMSE", "MAE"])
    out_fd.to_csv("./output/results.csv")