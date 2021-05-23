import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np


def make_bar_chart_with_std(data, target_metric, output_folder, title=None, y_lim=(0.0, 1.0)):
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


def make_train_test_metrics_bar_chart(ax, data, target_metric, title=None, y_lim=(0.0, 1.0), width=0.5):
    dataset_name = data['dataset_name'].iloc[0][:-4]
    train_positions = []
    test_positions = []
    labels_position = []
    for i in range(data.shape[0]):
        train_positions.append(- width / 2 + 3 * i * width)
        test_positions.append(width / 2 + 3 * i * width)
        labels_position.append(3 * i * width)
    ax.bar(train_positions, data[f'train_{target_metric}_mean'], width=width)
    ax.bar(test_positions, data[f'test_{target_metric}_mean'], width=width)
    ax.legend([
        'Train',
        'Test'
    ])
    ax.bar(labels_position, [0 for _ in range(len(labels_position))])
    ax.set_xticks(labels_position)
    ax.set_xticklabels(data['model_name'])
    ax.set_ylim(y_lim)
    if title is None:
        ax.set_title(f"Mean {target_metric} value for {dataset_name} dateset")
    else:
        ax.set_title(title)
    return ax


def make_overfit_chart(data, target_metric):
    dataset_name = data['dataset_name'].iloc[0][:-4]
    labels = np.array(data['model_name'].values)
    y = (data[f'test_{target_metric}_mean'] - data[f'train_{target_metric}_mean']).abs().values
    arg_sorted_y = np.argsort(y)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_title(f"Absolute difference of {target_metric} for train and test subset on {dataset_name} dateset")
    ax.barh(labels[arg_sorted_y], y[arg_sorted_y], color='darkorange')
    ax.legend([f'{target_metric}'])
    fig.tight_layout()
    return fig, ax
