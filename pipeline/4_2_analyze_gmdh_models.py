import pickle
import argparse
from pathlib import Path
import ast

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Calculates report about gmdh models behavior")
    parser.add_argument('--input', type=str, help='Path to gmdh train data')
    parser.add_argument('--output', type=str, help="Output of the location with analysis")
    args = parser.parse_args()

    input_data = pd.read_csv(args.input)
    input_data['total_nodes_count'] = input_data.filter(like='nodes_count', axis=1).fillna(0).sum(axis=1)
    input_data['layer_0_inputs'] = input_data['layer_0_input_variables'].apply(ast.literal_eval)
    result = []
    for (dataset_name, model_name), group in input_data.groupby(['dataset_name', 'model_name']):
        analysis = {
            'dataset_name': dataset_name,
            'model_name': model_name,
            'mean_complexity': group['total_nodes_count'].mean(),
            'mean_layers_count': group['layers_count'].mean()
        }
        analysis = {**analysis}
        result.append(analysis)

    output = Path(args.output)
    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result)[[
        'dataset_name', 'model_name', 'mean_complexity', 'mean_layers_count']].to_csv(output / 'analysis.csv', index=False)


if __name__ == '__main__':
    main()
