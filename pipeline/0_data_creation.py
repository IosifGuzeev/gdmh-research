import numpy as np
import pandas as pd
import argparse
import yaml
from pathlib import Path


def get_laws(law_names):
    # TODO: Find better laws
    laws = {
        "simple_poly": lambda X: 2.0 * X[:, 0] ** 2 + 0.1 * X[:, 1] * X[:, 2] + 3.5 * X[:, 3] ** 2 * X[:, 4] ** 3 + X[:,5],
        "exp": lambda X: np.exp(2.1 * X[:, 0] + 0.4 * X[:, 1]) - np.exp(2.0 * X[:, 1] + X[:, 2]) + np.exp(3.4 * X[:, 1] * X[:, 2] ** 2 + 1),
        "irrational_poly": lambda X: (0.1 * X[:, 0] * np.exp(0.5 * X[:, 2]) + X[:, 1] + 0.3) ** 1.5 - (X[:, 2] * X[:, 1] * np.exp(3.5 * X[:, 6]) + X[:, 3] + 1.2 * np.exp(X[:, 7])) ** 0.8
    }
    return list((law, laws[law]) for law in law_names)


def main():
    parser = argparse.ArgumentParser(description='Load datasets from given links and generate data by given law')
    parser.add_argument('--links', type=str, nargs='+', help='Links to the data sources')
    parser.add_argument('--target-columns', type=str, nargs='+', help='Target column for given dataset')
    parser.add_argument('--laws', type=str, nargs='+', help='Name of the laws for data generation')
    parser.add_argument('--output', type=str, help="Output folder location")
    args = parser.parse_args()

    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)

    np.random.seed(params['base_seed'])

    output = Path(args.output)
    if not output.exists():
        output.mkdir()

    for link, target_column in zip(args.links, args.target_columns):
        data = pd.read_csv(link)
        data = data[(data['price'] < 1_250_000) & (data['price'] > 100_000)]
        data = data.drop(columns=["zipcode", "id", "date"])
        dataset_name = link.split("/")[-1]
        data['Y'] = data[target_column].copy()
        data = data.drop(columns=[target_column])

        data.to_csv(output / dataset_name, index=False)


if __name__ == '__main__':
    main()
