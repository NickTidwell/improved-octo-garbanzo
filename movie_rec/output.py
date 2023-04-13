import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def filter_df(df, metric, func):
    return df.loc[(df['metric'] == metric) ][func].to_list()

def plot_bars():
    results_pd = pd.read_csv("./output/res_colab.csv").sort_values(by=['metric'])

    X = ['User', 'Item']
    X_axis = np.arange(len(X))
    fig, ax = plt.subplots(2)

    ax[0].set_xticks(X_axis, X)
    ax[0].bar(X_axis - 0.1,  filter_df(results_pd, 'cosine', 'RMSE') , width = 0.2, label = 'cosine')
    ax[0].bar(X_axis +  0.1,  filter_df(results_pd, 'msd', 'RMSE') , width = 0.2, label = 'msd')
    ax[0].bar(X_axis +  0.3,  filter_df(results_pd, 'pearson', 'RMSE') , width = 0.2, label = 'pearson')
    ax[0].set_ylabel("RMSE")

    ax[1].set_xticks(X_axis, X)
    ax[1].bar(X_axis - 0.1,  filter_df(results_pd, 'cosine', 'MAE') , width = 0.2, label = 'cosine')
    ax[1].bar(X_axis +  0.1,  filter_df(results_pd, 'msd', 'MAE') , width = 0.2, label = 'msd')
    ax[1].bar(X_axis +  0.3,  filter_df(results_pd, 'pearson', 'MAE') , width = 0.2, label = 'pearson')
    ax[1].set_ylabel("MAE")

    plt.legend()
    plt.show()


def plot_k():
    item_pd = pd.read_csv("./output/item_out.csv")
    user_pd = pd.read_csv("./output/user_out.csv")

    rmse_i = item_pd.loc[item_pd['metric'] == 'cosine']['RMSE'].to_list()
    mae_i = item_pd.loc[item_pd['metric'] == 'cosine']['MAE'].to_list()
    rmse_u = user_pd.loc[user_pd['metric'] == 'cosine']['RMSE'].to_list()
    mae_u = user_pd.loc[user_pd['metric'] == 'cosine']['MAE'].to_list()
    k = item_pd.loc[item_pd['metric'] == 'cosine']['K'].to_list()

    fig, ax = plt.subplots(2)
    ax[0].set_title("Item Colab filtering")
    ax[0].plot(k, rmse_i, label="RMSE")
    ax[0].set_ylabel("Error")
    ax[0].plot(k, mae_i, label="MAE")

    ax[1].set_title("User Colab filtering")
    ax[1].plot(k, rmse_u, label="RMSE")
    ax[1].set_ylabel("Error")
    ax[1].plot(k, mae_u, label="MAE")

    plt.xlabel("K")
    plt.legend()
    plt.show()

    print("ITEM RMSE MIN: ", np.argmin(rmse_i))
    print("ITEM MAE MIN: ", np.argmin(mae_i))
    print("USER RMSE MIN: ", np.argmin(rmse_u))
    print("USER MAE MIN: ", np.argmin(mae_u))
    
plot_k()
plot_bars()