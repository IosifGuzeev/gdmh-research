import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def make_plot(data, target_metric, output_folder, title=None, y_lim=(0.0, 1.0)):
    figure(figsize=(8, 6), dpi=80)
    dataset_name = data['dataset_name'].iloc[0][:-4]
    plt.bar(
        data['model_name'],
        data[f'{target_metric}_mean'] + data[f'{target_metric}_std'],
        width=0.5,
        color='sandybrown'
    )
    plt.bar(
        data['model_name'],
        data[f'{target_metric}_mean'],
        width=0.5,
        color='chocolate'
    )
    plt.plot([0], [0], color='red')
    plt.legend([
        'Inferior deviation',
        'Upper deviation',
        'Mean value'
    ][::-1])
    for i in range(data.shape[0]):
        plt.plot(
            [-0.25 + 1.0 * i, 0.25 + 1.0 * i],
            [data[f'{target_metric}_mean'].iloc[i], data[f'{target_metric}_mean'].iloc[i]],
            color='red'
        )
    plt.bar(
        data['model_name'],
        data[f'{target_metric}_mean'],
        width=0.5,
        color='red',
        fill=False
    )
    plt.bar(
        data['model_name'],
        data[f'{target_metric}_mean'] - data[f'{target_metric}_std'],
        width=0.5,
        color='peru'
    )
    plt.bar(
        data['model_name'],
        data[f'{target_metric}_mean'] + data[f'{target_metric}_std'],
        width=0.5,
        color='red',
        fill=False
    )
    plt.bar(
        data['model_name'],
        data[f'{target_metric}_mean'] - data[f'{target_metric}_std'],
        width=0.5,
        color='red',
        fill=False
    )
    if title is None:
        plt.title(f"R2 score for {dataset_name} dataset on train subset")
    else:
        plt.title(title)
    plt.ylim(y_lim)
    plt.savefig(str(output_folder) + '/' + dataset_name + f'_{target_metric}.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Build charts based on given metrics')
    parser.add_argument('--input', type=str, help='Path to metrics data')
    parser.add_argument('--output', type=str, help="Output folder with charts")
    args = parser.parse_args()

    input = Path(args.input)
    metrics = pd.read_csv(input)

    output = Path(args.output)
    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)

    # metrics[['train_r2_score', 'test_r2_score', 'train_mae_score', 'test_mae_score', 'model_name', 'dataset_name']]

    for _, group in metrics.groupby('dataset_name'):
        for metric_name in ['train_r2_score', 'test_r2_score', 'train_mae_score', 'test_mae_score']:
            if 'mae' in metric_name:
                y_lim = (0.0, 0.2)
            else:
                y_lim = (0.0, 1.0)
            make_plot(
                data=group,
                target_metric=metric_name,
                output_folder=output,
                y_lim=y_lim
            )


if __name__ == '__main__':
    main()
