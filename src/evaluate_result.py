import pandas as pd

def evaluate(path :str, num_algos = 7): # returns number of times the best algorithm has switched in a given result
    df = pd.read_csv(path)
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

            


print(evaluate(r"C:\Users\flerp\repos\time-recsys\Results\beer-advocate0_result_10.csv"))