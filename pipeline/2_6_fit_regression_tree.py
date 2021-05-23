from pathlib import Path
import argparse
import pickle

from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Load datasets from given links and fit tree regression models')
    parser.add_argument('--catalog', type=str, help='Path to catalog with datasets information')
    parser.add_argument('--output', type=str, help="Output folder location")
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog)

    output = Path(args.output)
    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)

    train_info = []
    for row in tqdm(catalog.itertuples(), desc="Models fitted:"):
        train_data = pd.read_csv(row.train_path)
        model = DecisionTreeRegressor(max_depth=3).fit(train_data.drop(columns=['Y']), train_data['Y'])
        train_info.append({
            "model_path": str(output) + '\\' + row.train_path.split('\\')[-1][:-4] + ".pkl"
        })
        with open(train_info[-1]['model_path'], 'wb') as f:
            pickle.dump(model, f)

    train_info = pd.DataFrame(train_info)
    train_info = pd.concat((catalog, train_info), axis=1)
    train_info.to_csv(output / "catalog.csv", index=False)


if __name__ == '__main__':
    main()
