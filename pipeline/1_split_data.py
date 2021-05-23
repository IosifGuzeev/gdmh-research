from pathlib import Path
import yaml
import argparse

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

np.random.seed(1024)


def get_std(dataset_name):
    if 'Boston' in dataset_name:
        return 0
    if 'simple_poly' in dataset_name:
        return 0.05
    if 'exp' in dataset_name:
        return 0.01
    if 'irrational' in dataset_name:
        return 0.01
    return 0


def main():
    parser = argparse.ArgumentParser(description='Split datasets on N train and test subsets')
    parser.add_argument('--input', type=str, help='Folder with datasets')
    parser.add_argument('--noise-stds', type=float, nargs='+', help='Name of the laws for data generation')
    parser.add_argument('--output', type=str, help="Output folder location")
    args = parser.parse_args()

    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
    subsets_count = params['split_data']['subsets_count']
    train_size = params['split_data']['train_size']

    output = Path(args.output)
    if not output.exists():
        output.mkdir()

    catalog = []
    for data_path in Path(args.input).glob("*.csv"):
        data = pd.read_csv(data_path)
        if 'kc_house' in str(data_path.name):
            kf = KFold(n_splits=int(subsets_count / 2), random_state=42, shuffle=True)
        else:
            kf = KFold(n_splits=int(subsets_count), random_state=42, shuffle=True)
        i = 0
        for train_index, test_index in kf.split(data):
            train, test = data.iloc[train_index], data.iloc[test_index]
            if 'kc_house' in str(data_path.name):
                y_train = train['Y'].values
                y_test = test['Y'].values
                train = train.drop(columns=['Y'])
                test = test.drop(columns=['Y'])
                scaler = MinMaxScaler().fit(train)
                columns = test.columns
                train = scaler.transform(train)
                test = scaler.transform(test)
                train = pd.DataFrame(data=train, columns=columns)
                train['Y'] = y_train
                test = pd.DataFrame(data=test, columns=columns)
                test['Y'] = y_test
            data_desc = {
                "initial_data": data_path,
                "train_path": output / f"{data_path.name[:-4]}_{i}_train.csv",
                "test_path": output / f"{data_path.name[:-4]}_{i}_test.csv",
            }
            i += 1
            noise_std = get_std(data_path.name)
            min_train, max_train = min(train['Y']), max(train['Y'])
            train['Y'] = (train['Y'] - min_train) / (max_train - min_train) + np.random.normal(0, noise_std)
            test['Y'] = (test['Y'] - min_train) / (max_train - min_train) + np.random.normal(0, noise_std)
            train.to_csv(data_desc['train_path'], index=False)
            test.to_csv(data_desc['test_path'], index=False)
            catalog.append(data_desc)
    pd.DataFrame(catalog).to_csv(output / "catalog.csv", index=False)


if __name__ == '__main__':
    main()
