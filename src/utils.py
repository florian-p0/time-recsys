# -*- coding: utf-8 -*-
"""
Created on Sat Mai 10 2025
Modified in May 2025:
- Replaced pd.read_table with pd.read_csv
- Added Pathlib for cross-platform paths
- Added error handling for timestamp conversion
@author: Fiona Nlend
"""

import pandas as pd
from pathlib import Path

def get_grid(name, metric):
    grid_file = 'Grids.xls' if metric == 'ndcg' else 'Grids_rmse.xls'
    grids = pd.read_excel(grid_file)
    grids = grids.set_index('Algo')
    grid = grids[name]
    return grid

def read_dataset(name, frac=None):
    """ loading of different pre-downloaded datasets with optional sampling"""
    
    base_path = Path("..") / "Datasets"

    if name == 'ML-100k':
        path = base_path / "Movielens/ml-100k/u.data"
        data = pd.read_csv(path, sep='\t', names=['user', 'item', 'rating', 'timestamp'], header=None)
        start, end = 1995, 1998

    elif name == 'ML-1M':
        path = base_path / "Movielens/ml-1m/ratings.dat"
        data = pd.read_csv(path, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'], header=None)
        start, end = 2000, 2003

    elif name == 'ML-10M':
        path = base_path / "Movielens/ml-10M100K/ratings.dat"
        data = pd.read_csv(path, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'], header=None)
        start, end = 1996, 2009

    elif name == 'ML-100k-latest':
        path = base_path / "Movielens/ml-latest-small/ratings.csv"
        data = pd.read_csv(path, sep=',', names=['user', 'item', 'rating', 'timestamp'], header=0)
        start, end = 1995, 2017

    elif name == 'amazon-instantvideo':
        path = base_path / "Amazon/ratings_Amazon_Instant_Video.csv"
        data = pd.read_csv(path, names=['user', 'item', 'rating', 'timestamp'], header=0)
        start, end = 2007, 2014

    elif name == 'amazon-books':
        path = base_path / "Amazon/ratings_Books.csv"
        data = pd.read_csv(path, names=['user', 'item', 'rating', 'timestamp'], header=0)
        start, end = 1997, 2013

    elif name == 'amazon-toys':
        path = base_path / "Amazon/ratings_Toys_and_Games.csv"
        data = pd.read_csv(path, names=['user', 'item', 'rating', 'timestamp'], header=0)
        start, end = 2001, 2014

    elif name == 'amazon-electronics':
        path = base_path / "Amazon/ratings_Electronics.csv"
        data = pd.read_csv(path, names=['user', 'item', 'rating', 'timestamp'], header=0)
        start, end = 2000, 2014

    elif name == 'amazon-music':
        path = base_path / "Amazon/ratings_Digital_Music.csv"
        data = pd.read_csv(path, names=['item', 'user', 'rating', 'timestamp'], header=0)
        data = data[['user', 'item', 'rating', 'timestamp']]
        start, end = 1998, 2014

    elif name == 'netflix':
        path = base_path / "netflix/NetflixRatings.csv"
        data = pd.read_csv(path, names=['item', 'user', 'rating', 'timestamp'], header=None)
        data = data[['user', 'item', 'rating', 'timestamp']]
        start, end = 1998, 2005

    elif name == 'yelp':
        path = base_path / "yelp_training_set/yelp_training_set_review.json"
        data = pd.read_json(path, lines=True)
        data = data.rename(columns={"user_id": "user", "business_id": "item", "stars": "rating", "date": "timestamp"})
        data = data[['user', 'item', 'rating', 'timestamp']]
        start, end = 2006, 2013

    elif name == 'epinions':
        path = base_path / "epinions/rating_with_timestamp.txt"
        data = pd.read_csv(path, delim_whitespace=True,
                           names=['user', 'item', 'category', 'rating', 'helpfulness', 'timestamp'])
        data = data[['user', 'item', 'rating', 'timestamp']]
        start, end = 1999, 2011

    elif name == 'amazon-video-games':
        path = base_path / "Amazon/Video_Games.csv"
        data = pd.read_csv(path, names=['user', 'item', 'rating', 'timestamp'], header=0)
        start, end = 1998, 2024

    elif name == 'amazon-software':
        path = base_path / "Amazon/Software.csv"
        data = pd.read_csv(path, names=['user', 'item', 'rating', 'timestamp'], header=0)
        start, end = 1999, 2024

    elif name == 'food-com':
        path = base_path / "food.com/RAW_interactions.csv"
        data = pd.read_csv(path, names=['user', 'item', 'timestamp', 'rating', 'review'], header=0)
        start, end = 2000, 2019

    else:
        raise ValueError('Dataset not implemented')

    if name == 'amazon-software' or name == 'amazon-video-games':
        # For amazon-software, we need to handle the timestamps differently as they are in milliseconds
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', origin='unix', errors='coerce')
    elif name == 'food-com':
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    else:
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s', origin='1970-01-01', errors='coerce')

    # User- und Item-IDs neu codieren (0-basiert, int)
    if 'user' in data.columns and data['user'].dtype == 'object':
        data['user'] = data['user'].astype('category').cat.codes
    if 'item' in data.columns and data['item'].dtype == 'object':
        data['item'] = data['item'].astype('category').cat.codes

    data = data.groupby("user").filter(lambda grp: len(grp) > 2)

    if frac is not None:
        data = data.sample(frac=frac, random_state=42)

    return data, start, end
