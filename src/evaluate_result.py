import pandas as pd

def evaluate(*paths, num_algos = 7): # returns number of times the best algorithm has switched in a given result
    df = None
    algos = None
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
    df['algorithm'] = algos
    print(df)
    return

    df.columns = ['algorithm', 'ndcg', 'recall', 'precision', 'nrecs']
    data_dict = {}
    num_epochs = 0
    while len(df) > 0:
        i = num_epochs
        new_df = df.head(num_algos)
        df = df.tail(df.shape[0] - num_algos)
        new_df = new_df.sort_values(["ndcg"], ascending=False)
        data_dict[i] = new_df
        num_epochs+=1
    best_algo = ""
    swaps = 0
    for i in range(num_epochs):
        if i == 0:
            best_algo = data_dict[i]["algorithm"].iloc[0]
            continue
        new_best_algo = data_dict[i]["algorithm"].iloc[0]
        if best_algo != new_best_algo:
            best_algo = new_best_algo
            swaps += 1
    return swaps

            


print(evaluate(r"/home/florian/repos/time-recsys/Results/beer-advocate0_result_10.csv",
            r"/home/florian/repos/time-recsys/Results/beer-advocate1_result_10.csv",
            r"/home/florian/repos/time-recsys/Results/beer-advocate2_result_10.csv"))