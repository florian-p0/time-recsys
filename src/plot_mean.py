from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import pandas as pd

prefix = r"C:\Users\flerp\repos\time-recsys\Results"
folders = [     r"Amazon Electronics",
                r"Amazon Instant Video",
                r"Amazon Software",
                r"Amazon Video Games",
                r"Beer Advocate",
                r"Food Com",
                r"ML-1M",
                r"ML-100k",
                r"MovieTweetings"          ]

def plot_folder(path, metric, _algos, start, end, steps, dataset):
    os.chdir(path)
    paths = glob.glob("*.csv")
    for i, path in enumerate(paths):
        if i==0:
            df = pd.read_csv(path)
            continue
        df = pd.concat([df, pd.read_csv(path)])
    
    df.columns = ['algorithm', 'ndcg', 'recall', 'precision', 'nrecs']
    algos = df['algorithm']
    df = df[['ndcg', 'recall', 'precision', 'nrecs']].groupby(df.index)
    df = df.mean()
    algos = algos.tail(len(df))
    df["algorithm"] = algos
    print(df)
    plot(df, metric, _algos, start, end, steps, dataset)




def plot(results, metric, algos, start, end, steps, dataset):
    x = np.linspace(start, end, steps)
    for algo in algos:  
        plt.plot(x, results[results["algorithm"] == algo][metric])
    plt.legend(algos)
    plt.title('{} over time - {}'.format(metric, dataset))
    plt.xlabel('year')
    plt.ylabel(metric)
    plt.show()

plot_folder(prefix + "\\" + r"Amazon Electronics", "ndcg", ['Pop','HPF','Bias','II','UU','BiasedMF','SVD'], 2000, 2014, 15, "Amazon Electronics")
plot_folder(prefix + "\\" + r"Amazon Instant Video", "ndcg", ['Pop','HPF','Bias','II','UU','BiasedMF','SVD'], 2007, 2014, 7, "Amazon Instant Video")
plot_folder(prefix + "\\" + r"Amazon Software", "ndcg", ['Pop','HPF','Bias','II','UU','BiasedMF','SVD'], 1999, 2024, 5, "Amazon Software")
plot_folder(prefix + "\\" + r"Amazon Video Games", "ndcg", ['Pop','HPF','Bias','II','UU','BiasedMF','SVD'], 1998, 2024, 5, "Amazon Video Games")
plot_folder(prefix + "\\" + r"Beer Advocate", "ndcg", ['Pop','HPF','Bias','II','UU','BiasedMF','SVD'], 1998, 2011, 11, "Beer Advocate")
plot_folder(prefix + "\\" + r"Food Com", "ndcg", ['Pop','HPF','Bias','II','UU','BiasedMF','SVD'], 2000, 2019, 6, "Food Com")
plot_folder(prefix + "\\" + r"ML-1M", "ndcg", ['Pop','HPF','Bias','II','UU','BiasedMF','SVD'], 2000, 2003, 4, "ML-1M")
plot_folder(prefix + "\\" + r"ML-100k", "ndcg", ['Pop','HPF','Bias','II','UU','BiasedMF','SVD'], 1995, 1998, 8, "ML-100k")
plot_folder(prefix + "\\" + r"MovieTweetings", "ndcg", ['Pop','HPF','Bias','II','UU','BiasedMF','SVD'], 2013, 2022, 4, "MovieTweetings")