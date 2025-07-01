import pandas as pd
import glob, os

def evaluate_files(*paths, num_algos = 7): # returns number of times the best algorithm has switched in a given result
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
    last_swap = None
    for i in range(num_epochs):
        if i == 0:
            best_algo = data_dict[i]["algorithm"].iloc[0]
            continue
        new_best_algo = data_dict[i]["algorithm"].iloc[0]
        if best_algo != new_best_algo:
            best_algo = new_best_algo
            swaps += 1
            last_swap = i
    return swaps, num_epochs, last_swap

            
def evaluate_folder(path):
    os.chdir(path)
    return(evaluate_files(*glob.glob("*.csv")))

#print(evaluate_folder(r"C:\Users\flerp\repos\time-recsys\Results\Amazon Video Games"))

prefix = r"C:\Users\flerp\repos\time-recsys\Results"
folders = [     r"Amazon Electronics",
                r"Amazon Instant Video",
                r"Amazon Software",
                r"Amazon Video Games",
                r"Beer Advocate",
                r"Food Com",
                r"ML-1M",
                r"ML-100k"          ]

for folder in folders:
    path = prefix + "\\" + folder

    print(folder)
    numbers = evaluate_folder(path)
    print("swaps: {}, last swap of {} epochs: {}".format(numbers[0],numbers[1],numbers[2],))