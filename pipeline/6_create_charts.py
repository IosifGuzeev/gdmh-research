import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from utility.charts import make_train_test_metrics_bar_chart, make_overfit_chart


def main():
    parser = argparse.ArgumentParser(description='Build charts based on given metrics')
    parser.add_argument('--input', type=str, help='Path to metrics data')
    parser.add_argument('--output', type=str, help="Output folder with charts")
    args = parser.parse_args()

    metrics = pd.read_csv(args.input)

    output = Path(args.output)
    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)

    # metrics[['train_r2_score', 'test_r2_score', 'train_mae_score', 'test_mae_score', 'model_name', 'dataset_name']]

    for _, group in metrics.groupby('dataset_name'):
        for metric_name, title, y_lim in [
            ('r2_score', None, (0.0, 1.0)),
            ('mae_score', None, (0.0, 0.2)),
        ]:
            dataset_name = group['dataset_name'].iloc[0][:-4]
            if 'irrational' in dataset_name:
                dataset_name = 'poly_expo_dataset'
            gmdh_model = group[group['model_name'].isin(['combi', 'mia', 'kandos_nn'])]
            base_models = group[~group['model_name'].isin(['combi', 'mia', 'kandos_nn'])]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            make_train_test_metrics_bar_chart(
                ax1,
                data=gmdh_model,
                target_metric=metric_name,
                y_lim=y_lim,
                title=f"Mean {metric_name} value for {dataset_name} dateset for GMDH models"
            )
            make_train_test_metrics_bar_chart(
                ax2,
                data=base_models,
                target_metric=metric_name,
                y_lim=y_lim,
                title=f"Mean {metric_name} value for {dataset_name} dateset for base models"
            )
            fig.suptitle(f"Mean {metric_name} value for {dataset_name} dataset")
            fig.savefig(str(output) + f'/{dataset_name}_{metric_name}.png')
            fig, ax = make_overfit_chart(group, metric_name)
            fig.savefig(str(output) + f'/{dataset_name}_overfit_{metric_name}.png')


if __name__ == '__main__':
    main()
