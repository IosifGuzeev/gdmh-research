from pathlib import Path
import yaml
import argparse

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

np.random.seed(1024)


def main():
    parser = argparse.ArgumentParser(description='Split datasets on N train and test subsets')
    parser.add_argument('--input', type=str, help='Folder with datasets')
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
        for i in range(subsets_count):
            seed = np.random.randint(0, 10000)
            train, test = train_test_split(data, train_size=train_size, )
            data_desc = {
                "initial_data": data_path,
                "train_path": output / f"{data_path.name[:-4]}_{i}_train.csv",
                "test_path": output / f"{data_path.name[:-4]}_{i}_test.csv",
                "seed": seed
            }
            min_train, max_train = min(train['Y']), max(train['Y'])
            train['Y'] = (train['Y'] - min_train) / (max_train - min_train)
            test['Y'] = (test['Y'] - min_train) / (max_train - min_train)
            train.to_csv(data_desc['train_path'], index=False)
            test.to_csv(data_desc['test_path'], index=False)
            catalog.append(data_desc)
    pd.DataFrame(catalog).to_csv(output / "catalog.csv", index=False)


if __name__ == '__main__':
    main()
