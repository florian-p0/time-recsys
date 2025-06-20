# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:08:45 2021

@author: tsche
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lenskit import batch, topn, util
from lenskit.algorithms import item_knn, user_knn, als
from lenskit.algorithms import basic, Recommender, funksvd
#from sklearn.model_selection import train_test_split
from lenskit.batch import predict
from lenskit.metrics.predict import rmse, mae
from lenskit.crossfold import partition_users, LastFrac
from utils import read_dataset, get_grid
from gridsearch import gs, gs_rmse, get_algo  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def evaluate_pred(name, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    preds = predict(fittable, test)
    preds['Algorithm'] = name
    # add the algorithm name for analyzability
    #recs['Algorithm'] = name
    return preds

def evaluate_rec(name, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 10)

    # add the algorithm name for analyzability
    recs['Algorithm'] = name
    return recs

def final_plot(results, metric, algos, start, end, steps, dataset):
    x = np.linspace(start, end, steps)
    for algo in algos:  
        plt.plot(x, results.loc[algo,metric])
    plt.legend(algos)
    plt.title('{} over time - {}'.format(metric, dataset))
    plt.xlabel('year')
    plt.ylabel(metric)
    plt.savefig(r"../Figures/{}_{}_10.png".format(dataset, metric))
    plt.show()

def main_rmse(dataset, frac=None, rs=42):
    names = ['HPF','Bias','II','UU','BiasedMF','SVD']
    data, start, end = read_dataset(dataset, frac=frac)
    grid = get_grid(dataset, 'rmse')
    if dataset == 'ML-100k':
        g = data.groupby(pd.Grouper(key='timestamp', freq='M'))
    else: 
        g = data.groupby(pd.Grouper(key='timestamp', freq='Y'))
    splits = [group for _,group in g]
    new_df = pd.DataFrame(columns = ['user', 'item', 'rating'])
    pred_results = np.array([['Algorithm','RMSE']])
    i = 0
    for df in splits:
        new_df = pd.merge(new_df, df, how='outer')
        new_df = new_df.groupby("user").filter(lambda grp: len(grp) > 2)
        if new_df.shape[0] > 80000: 
            i += 1
            print('set ', i)
            tp, tp2 = partition_users(new_df, 2, method=LastFrac(0.2, col='timestamp'), rng_spec=rs) # RNG seed not set, so different results each time
            for name in names: 
                if name == 'Pop':
                    best_para = 0
                else:
                    grid_algo = [int(s) for s in grid[name].split(',')]
                    best_para, _ = gs_rmse(name, grid_algo, tp.train, rs=rs)
                print(name + str(best_para))
                algo = get_algo(name, best_para)
                preds = evaluate_pred(name, algo, tp.train, tp.test)

                if preds.shape[0] > 3:
                     RMSE = rmse(preds['prediction'], preds['rating'])
                     pred_results = np.vstack((pred_results, [name, RMSE.astype(np.float64)]))
                     all_results_pred = pd.DataFrame(data=pred_results[1:,1:].astype(np.float64), index = pred_results[1:,0], columns = pred_results[0,1:])
                     all_results_pred.to_csv(r"..\Results\{dataset}_result_rmse2.csv".format(dataset=dataset))

            
    all_results_pred = pd.DataFrame(data=pred_results[1:,1:].astype(np.float64), index = pred_results[1:,0], columns = pred_results[0,1:])
    all_results_pred.to_csv(r"..\Results\{dataset}_result_rmse2.csv".format(dataset=dataset))

    final_plot(all_results_pred,'RMSE', names, start, end, i,dataset)
    #final_plot(all_results_pred,'MAE', names, start, end, i,dataset)
    
    return all_results_pred 

def main(dataset, frac=None, rs=42):
    names = ['HPF','Bias','II','UU','BiasedMF','SVD','Pop']
    data, start, end = read_dataset(dataset, frac=frac)
    grid = get_grid(dataset, 'rmse')
    if dataset == 'ML-100k':
        g = data.groupby(pd.Grouper(key='timestamp', freq='M'))
    elif dataset == 'amazon-video-games' or dataset == 'amazon-software':
        g = data.groupby(pd.Grouper(key='timestamp', freq='5Y'))
    elif dataset == 'food-com':
        g = data.groupby(pd.Grouper(key='timestamp', freq='3Y'))
    else: 
        g = data.groupby(pd.Grouper(key='timestamp', freq='Y'))
    splits = [group for _,group in g]
    new_df = pd.DataFrame(columns = ['user', 'item', 'rating'])
    all_results = pd.DataFrame(columns = ['ndcg', 'recall', 'precision'])
    i = 0
    for df in splits:
        all_recs = []
        new_df = pd.merge(new_df, df, how='outer')
        new_df = new_df.groupby("user").filter(lambda grp: len(grp) > 2)
        if new_df.shape[0] > 500: 
            i += 1
            print('set', i)
            tp, tp2 = partition_users(new_df, 2, method=LastFrac(0.2, col='timestamp'), rng_spec=rs)
            for name in names: 
                if name == 'Pop':
                    best_para = 0
                else:
                    grid_algo = [int(s) for s in grid[name].split(',')]
                    best_para, _ = gs(name, grid_algo, tp.train, rs=rs)
                print(name, str(best_para))
                algo = get_algo(name, best_para, rs=rs)
                recs = evaluate_rec(name, algo, tp.train, tp.test)
                all_recs.append(recs)
                
            all_recs = pd.concat(all_recs, ignore_index=True)
    
            rla = topn.RecListAnalysis()
            rla.add_metric(topn.ndcg, k=10)
            rla.add_metric(topn.recall, k=10)
            results = rla.compute(all_recs, tp.test)
            results = results.groupby('Algorithm').mean()
            all_results = all_results._append(results)

    all_results.to_csv(r"..\Results\{dataset}_result_10.csv".format(dataset=dataset))
    
    final_plot(all_results,'recall', names, start, end, i,dataset)
    final_plot(all_results,'ndcg', names, start, end, i,dataset)

    return all_results
