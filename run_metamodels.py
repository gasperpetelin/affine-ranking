from joblib import Parallel, delayed
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
from utils import *
import os

def append_pairwise_error(df):
    df["pairwise_error"] = 0
    counter = 0
    for algo1 in algorithms:
        for algo2 in algorithms:
            if algo1 != algo2:
                pair_error = ((df[f'{algo1}_pred'] < df[f'{algo2}_pred']) != (df[f'{algo1}_true'] < df[f'{algo2}_true'])).astype(int)
                df["pairwise_error"] += pair_error
                counter += 1
    df["pairwise_error"] /= counter
    return df

def transform_to_X_Y(data, features, algorithms):
    X = data[features]
    #X = X.replace([np.inf, -np.inf], np.nan)
    X = X.replace([np.inf, -np.inf, np.nan], 0)
    
    Y = data[algorithms]
    return X, Y

algorithms = ['GA', 'PSO', 'DE', 'CMAES', 'ES']
meta_columns = ['problem1', 'problem2', 'instance1', 'instance2', 'alpha', 'dim', 'algorithm_run']
meta_columns_no_run = ['problem1', 'problem2', 'instance1', 'instance2', 'alpha', 'dim']

runs = load_runs()
rruns = get_rank(runs, algorithms)
mean_rruns = rruns.groupby(meta_columns_no_run).mean().reset_index()
ela = load_ela()
features = [x for x in ela.columns if '.' in x]
data = rruns.merge(ela, on=['problem1', 'problem2', 'instance1', 'instance2', 'alpha', 'dim'], how='outer', suffixes=['_run', '_ela'])
feature_groups = set([x.split('.')[0] for x in features])

from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
import itertools
from tqdm import tqdm

models = [RandomForestRegressor(n_jobs=-1), DummyRegressor()]
problem_out_range = range(1, 25)
all_feature_groups = list(feature_groups) + ['all', 'all_norm', 'all_no_norm']

all_triplets = list(itertools.product(all_feature_groups, problem_out_range, models))
import random
random.shuffle(all_triplets)

for feature_group, problem_out, model in tqdm(all_triplets):
    #feature_subset = [x for x in features if x.startswith(feature_group)]
    if feature_group == "all":
        feature_subset = features
    elif feature_group == "all_norm": 
        feature_subset = [x for x in features if x.startswith('norm_')==True]
    elif feature_group == "all_no_norm":
        feature_subset = [x for x in features if x.startswith('norm_')==False]
    else:
        feature_subset = [x for x in features if x.startswith(feature_group)]
    directory = 'gecco'
    create_directory_if_not_exist(directory)

    file = f'{directory}/m_{model.__class__.__name__}__rp_{problem_out}__fg_{feature_group}.parquet'
    if os.path.isfile(file) == False:
        print(f'Training  {file}')
        train = data.query(f"problem1!={problem_out} and problem2 != {problem_out}")
        test = data.query(f"problem1!={problem_out} and problem2 == {problem_out}")

        X_train, Y_train = transform_to_X_Y(train, feature_subset, algorithms)
        model.fit(X_train, Y_train)

        X_test, Y_test = transform_to_X_Y(test, feature_subset, algorithms)
        prediction = model.predict(X_test)
        dfpred = pd.concat([pd.DataFrame(prediction, columns=algorithms).reset_index(), test[meta_columns_no_run].reset_index()], axis=1)

        joined_table = mean_rruns.merge(dfpred, on=meta_columns_no_run, suffixes=('_true', '_pred'))
        joined_table_with_error = append_pairwise_error(joined_table)
        joined_table_with_error
        joined_table_with_error.drop(columns=['index'], inplace=True)
        joined_table_with_error['metamodel'] = model.__class__.__name__
        joined_table_with_error['removed_function'] = problem_out
        joined_table_with_error['feature_group'] = feature_group
        joined_table_with_error['all_features'] = ','.join(feature_subset)
        joined_table_with_error.to_parquet(file)
    else:
        print(f"Skiping file {file}")


