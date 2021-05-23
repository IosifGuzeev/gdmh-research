import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_1samp
from scipy.stats import kstest, shapiro, normaltest, anderson, anderson_ksamp


def main():
    parser = argparse.ArgumentParser(description='Build analysis for each model to say '
                                                 'if metric value is well represented')
    parser.add_argument('--metrics', type=str, help='Path to metrics data')
    parser.add_argument('--analysis', type=str, help='Path to analysis file')
    parser.add_argument('--output', type=str, help="Output folder statistic analysis")
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics)
    analysis = pd.read_csv(args.analysis)

    metrics['model_name'] = metrics['model_name'].apply(lambda s: s.split('/')[-1])
    output = Path(args.output)

    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)
    result = []
    for _, row in analysis.iterrows():
        for metric in ['train_r2_score', 'test_r2_score', 'train_mae_score', 'test_mae_score']:
            model_metric_values = metrics[
                (metrics['model_name'] == row['model_name']) &
                (metrics['dataset_name'] == row['dataset_name'])]
            _ , norm_pval = shapiro(model_metric_values[metric])
            is_norm = norm_pval > 0.01
            # if not is_norm:
            #     _, norm_pval = kstest(model_metric_values[metric], 'norm')
            #     is_norm = norm_pval > 0.01
            # if not is_norm:
            #     _, norm_pval = normaltest(model_metric_values[metric])
            #     is_norm = norm_pval > 0.01
            # if not is_norm:
            #     anderson_values = anderson(model_metric_values[metric])
            #     norm_pval = anderson_values.critical_values[-1]
            #     is_norm = anderson_values.statistic < norm_pval

            tset, pval = ttest_1samp(model_metric_values[metric], row[f"{metric}_mean"])
            result.append({
                "model_name": row['model_name'],
                "dataset_name": row['dataset_name'],
                "metric_name": metric,
                "shapiro_p_value": norm_pval,
                "is_norm": is_norm,
                "p_value": pval,
                "is_accepted (alpha is 0.05)": pval > 0.05
            })
    pd.DataFrame(result).to_csv(output / "statistics.csv", index=False)


if __name__ == '__main__':
    main()
