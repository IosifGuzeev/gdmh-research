import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_1samp


def main():
    parser = argparse.ArgumentParser(description='Build analysis for each model to say '
                                                 'if metric value is well represented')
    parser.add_argument('--input', type=str, help='Path to metrics data')
    parser.add_argument('--output', type=str, help="Output folder statistic analysis")
    args = parser.parse_args()

    input = Path(args.input)
    metrics = pd.read_csv(input)
    metrics['model_name'] = metrics['model_name'].apply(lambda s: s.split('/')[-1])
    output = Path(args.output)
    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)

    mean_data = (
        metrics[['train_r2_score', 'test_r2_score', 'train_mae_score', 'test_mae_score', 'model_name', 'dataset_name']]
            .groupby(['dataset_name', 'model_name'])
            .agg('mean')
            .reset_index()
    )
    result = []
    for _, row in mean_data.iterrows():
        for metric in ['train_r2_score', 'test_r2_score', 'train_mae_score', 'test_mae_score']:
            model_metric_values = metrics[
                (metrics['model_name'] == row['model_name']) &
                (metrics['dataset_name'] == row['dataset_name'])]
            tset, pval = ttest_1samp(model_metric_values[metric], row[metric])
            result.append({
                "model_name": row['model_name'],
                "dataset_name": row['dataset_name'],
                "metric_name": metric,
                "p_value": pval,
                "is_accepted (alpha is 0.05)": pval > 0.05
            })
    pd.DataFrame(result).to_csv(output / "statistics.csv", index=False)


if __name__ == '__main__':
    main()
