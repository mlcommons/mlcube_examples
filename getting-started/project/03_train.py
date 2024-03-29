"""Train gradient boosting regressor on Boston housing dataset"""
import yaml
import argparse
from re import I
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


def train(dataset_file_path, n_estimators):
    df = pd.read_csv(dataset_file_path)

    data = df.drop(['PRICE'], axis=1)
    target = df[['PRICE']]
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size = 0.25)

    clf = GradientBoostingRegressor(n_estimators=n_estimators, verbose = 1)
    clf.fit(X_train, Y_train.values.ravel())

    train_predicted = clf.predict(X_train)
    train_expected = Y_train
    train_rmse = mean_squared_error(train_predicted, train_expected, squared=False)

    test_predicted = clf.predict(X_test)
    test_expected = Y_test
    test_rmse = mean_squared_error(test_predicted, test_expected, squared=False)

    print(f"\nTRAIN RMSE:\t{train_rmse}")
    print(f"TEST RMSE:\t{test_rmse}")

def main():

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--dataset_file_path', required=True,
                        help='Processed dataset file path')
    parser.add_argument('--parameters_file', required=False,
                        help='File containing hyperparameters')
    args = parser.parse_args()

    dataset_file_path = args.dataset_file_path
    parameters_file = args.parameters_file
    
    n_estimators = 100
    if parameters_file is not None:
        with open(parameters_file, 'r') as stream:
            parameters = yaml.safe_load(stream)
            n_estimators = int(parameters["n_estimators"])

    print(f"Using {n_estimators} estimators")
    train(dataset_file_path, n_estimators)


if __name__ == '__main__':
    main()
