import pickle
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def analyze_nodes_inputs(layer):
    result = []
    for node in layer.nodes:
        result.append(tuple(node.columns))
    return result


def main():
    parser = argparse.ArgumentParser(description="Calculates train data for each GMDH model "
                                                 "and dataset it's attached to")
    parser.add_argument('--input', type=str, help='Path to catalog with models and datasets information')
    parser.add_argument('--output', type=str, help="Output file location")
    args = parser.parse_args()

    input_folder = Path(args.input)

    global_result = None

    for catalog_path in tqdm(input_folder.glob("**/catalog.csv"), desc='Catalogs processed: '):
        catalog = pd.read_csv(catalog_path)
        result = []
        for row in tqdm(catalog.itertuples(), desc='Models processed: '):
            if 'kandos' not in row.model_path and 'mia' not in row.model_path:
                continue
            with open(row.model_path, 'rb') as f:
                model = pickle.load(f)
            model_stats = {
                "model_name": row.model_path.split('\\')[-2],
                "dataset_name": row.initial_data.split('\\')[-1]
            }
            for i, layer in enumerate(model.layers):
                model_stats[f"layer_{i}_nodes_count"] = layer.output_dim
                model_stats[f"layer_{i}_fit_score"] = layer.fit_score
                model_stats[f"layer_{i}_input_variables"] = analyze_nodes_inputs(layer)
            result.append(model_stats)
            model_stats['layers_count'] = len(model.layers)
        result = pd.DataFrame(result)

        if global_result is None:
            global_result = result.copy()
        else:
            global_result = global_result.append(result.copy())
    global_result.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
