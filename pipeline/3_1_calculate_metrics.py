import pickle
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Calculates metric values for each model and subsets it's attached to")
    parser.add_argument('--input', type=str, help='Path to catalog with models and datasets information')
    parser.add_argument('--output', type=str, help="Output file location")
    args = parser.parse_args()

    input_folder = Path(args.input)

    global_result = None

    for catalog_path in tqdm(input_folder.glob("**/catalog.csv"), desc='Catalogs processed: '):
        catalog = pd.read_csv(catalog_path)
        result = []
        for row in tqdm(catalog.itertuples(), desc='Models processed: '):
            print(row.model_path)
            with open(row.model_path, 'rb') as f:
                model = pickle.load(f)
            train_data = pd.read_csv(row.train_path)
            test_data = pd.read_csv(row.test_path)
            prediction_train = model.predict(train_data.drop(columns=['Y']))
            prediction_test = model.predict(test_data.drop(columns=['Y']))
            train_r2_score = r2_score(train_data['Y'], prediction_train)
            test_r2_score = r2_score(test_data['Y'], prediction_test)
            train_mae_score = mean_absolute_error(train_data['Y'], prediction_train)
            test_mae_score = mean_absolute_error(test_data['Y'], prediction_test)
            result.append({
                "train_r2_score": train_r2_score,
                "test_r2_score": test_r2_score,
                "train_mae_score": train_mae_score,
                "test_mae_score": test_mae_score,
            })
        result = pd.DataFrame(result)
        result = pd.concat((catalog, result), axis=1)
        result['model_name'] = result['model_path'].apply(lambda s: s.split('\\')[-2])
        result['dataset_name'] = result['initial_data'].apply(lambda s: s.split('\\')[-1])

        if global_result is None:
            global_result = result.copy()
        else:
            global_result = global_result.append(result.copy())
    global_result.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
