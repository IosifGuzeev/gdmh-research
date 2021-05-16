import numpy as np
import pandas as pd
import argparse
import yaml
from pathlib import Path


def get_laws(law_names):
    # TODO: Find better laws
    laws = {
        "simple_poly": lambda X: 2.0 * X[:, 0] ** 2 + 0.1 * X[:, 1] * X[:, 2] + 3.5 * X[:, 3] ** 2 * X[:, 4] ** 3 + X[:, 5],
        "exp": lambda X: 0.5 * np.exp(-5 * X[:, 0]) + 3.0 * np.exp(X[:, 0] * X[:, 1]) + np.exp(X[:, 2] * X[:, 3] ** 2)  - 2.0 * np.exp(X[:, 4] ** 3),
        "irrational_poly": lambda X: (X[:, 0] ** 0.3 + X[:, 1] ** 1.8) / (X[:, 1] ** 0.5 * X[:, 2] ** 3.4) - X[:, 0] ** 1.4 * X[:, 1] ** 1.6
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

    output = Path(args.output)
    if not output.exists():
        output.mkdir()

    for link, target_column in zip(args.links, args.target_columns):
        data = pd.read_csv(link)
        dataset_name = link.split("/")[-1]
        data['Y'] = data[target_column].copy()
        data = data.drop(columns=[target_column])
        data.to_csv(output / dataset_name, index=False)

    for law in get_laws(args.laws):
        data = np.random.uniform(0, 1, (params['generate_data']['sample_size'] + 2000, 10))
        y = law[1](data)
        arg_sorted = np.argsort(y)
        data = data[arg_sorted][2000:]
        y = y[arg_sorted][2000:]
        data = pd.DataFrame(data=data, columns=list(f"X_{i}" for i in range(data.shape[1])))
        data['Y'] = y
        data.to_csv(output / f"{law[0]}.csv", index=False)


if __name__ == '__main__':
    main()
