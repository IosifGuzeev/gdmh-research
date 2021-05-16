import argparse
from pathlib import Path

import pandas as pd


def add_suffix(s, suffix, exclude):
    if s not in exclude:
        return s + suffix
    else:
        return s


def main():
    parser = argparse.ArgumentParser(description='Build analysis for each model and dataset')
    parser.add_argument('--input', type=str, help='Path to metrics data')
    parser.add_argument('--output', type=str, help="Output folder with metrics")
    args = parser.parse_args()

    input = Path(args.input)
    metrics = pd.read_csv(input)

    output = Path(args.output)
    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)

    mean_data = (
        metrics[['train_r2_score', 'test_r2_score', 'train_mae_score', 'test_mae_score', 'model_name', 'dataset_name']]
            .groupby(['dataset_name', 'model_name'])
            .agg('mean')
            .reset_index()
            .rename(columns=lambda s: add_suffix(s, '_mean', ['dataset_name', 'model_name']))
    )
    std_data = (
        metrics[['train_r2_score', 'test_r2_score', 'train_mae_score', 'test_mae_score', 'model_name', 'dataset_name']]
            .groupby(['dataset_name', 'model_name'])
            .agg('std')
            .reset_index()
            .rename(columns=lambda s: add_suffix(s, '_std', ['dataset_name', 'model_name']))
    )
    result = mean_data.merge(std_data, on=['dataset_name', 'model_name'])

    ## TODO: Remove for linux
    result['model_name'] = result['model_name'].apply(lambda s: s.split('/')[-1])
    result.to_csv(output / 'analysis.csv', index=False)


if __name__ == '__main__':
    main()
