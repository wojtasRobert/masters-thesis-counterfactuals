import numpy as np
from xgboost import XGBClassifier
from masters_utils import get_params_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
from sklearn.model_selection import train_test_split
from pprint import pprint
from threading import Thread
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score


class XGBWrapper:
    results = {}

    def __call__(self, x, *args, **kwargs):
        print('XGBWrapper called')
        print(x.values)
        tab = {}
        self.results = {}
        threads = []
        for idx, chunk in enumerate(np.array_split(x, 6)):
            thread = Thread(target=self.get_model_results, args=(chunk, idx))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        results = []
        for v in self.results.values():
            results = results + v

        print(results)
        return pd.DataFrame(np.array(results), columns=['y'])

    def __init__(self, x_train, y_train, x_val, y_val, param_names, int_params):
        print('init XGBWrapper')
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.param_names = param_names
        self.int_params = int_params

    def fit(self, params):
        params = self.params_to_dict(params)
        model = XGBClassifier(**params)
        model.fit(self.x_train, self.y_train, verbose=False)
        return model

    def predict(self, model):
        print('predict output')
        return model.predict(self.x_val)

    def params_to_dict(self, row):
        params_dict = dict(zip(self.param_names, row))
        for param in self.int_params:
            params_dict[param] = int(params_dict[param])
        return params_dict

    def get_model_accuracy(self, model):
        predictions = model.predict(self.x_val)
        return accuracy_score(self.y_val, predictions)

    def get_model_results(self, x, thread_number):
        self.results[thread_number] = []
        x_len = len(x) - 1
        counter = 0
        for row in x.values:
            model = self.fit(row)
            result = self.get_model_accuracy(model)
            self.results[thread_number].append(result)
            print('Thread no. ', thread_number, ' Iteration: ', counter, 'of', x_len, ': ', result)
            counter += 1


def get_prepared_adult_data():
    dataset = pd.read_csv('../works/data/adult.csv')
    dataset['income'] = dataset['income'].map({'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K': 1})
    dataset['workclass'] = dataset['workclass'].replace(['?'], 'Unknown')
    dataset['marital-status'] = dataset['marital-status'].replace(
        ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married')
    dataset['marital-status'] = dataset['marital-status'].replace(['Never-married', 'Divorced', 'Separated', 'Widowed'],
                                                                  'Single')
    dataset['marital-status'] = dataset['marital-status'].map({'Married': 0, 'Single': 1})
    dataset['marital-status'] = dataset['marital-status']
    dataset.drop(labels=['gender', 'workclass', 'education', 'occupation', 'relationship', 'race', 'native-country'],
                 axis=1, inplace=True)
    return dataset


if __name__ == '__main__':
    columns = ['n_estimators',
               'learning_rate',
               'max_depth',
               'min_child_weight']
    # 'reg_lambda'
    # 'gamma',
    int_columns = ['n_estimators', 'max_depth']

    # pierwszy zestaw danych
    # n_estimators = [1, 10, 50, 100, 200, 300, 500, 1000]
    # learning_rate = [0.000001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.8, 1]
    # max_depth = [1, 2, 3, 4, 5, 6, 7, 8]
    # min_child_weight = [0.0001, 0.01, 0.1, 1, 3, 10, 50, 100]
    # min_split_loss = [0, 0.0001, 0.01, 0.1, 1, 10, 50]
    # lam = [1, 1.5, 3]

    # drugi zestaw danych
    n_estimators = [1, 200, 1000]
    learning_rate = [0.000001, 0.3, 1]
    max_depth = [1, 4, 8]
    min_child_weight = [0.0001, 1, 100]
    # min_split_loss = [0, 0.0001, 0.01, 0.1, 1, 10, 50]
    # lam = [1, 1.5, 3]

    default_params = [200, 0.3, 6.0, 1, 0]

    numeric_columns = ['marital-status', 'age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss',
                       'hours-per-week']
    data = get_prepared_adult_data()
    X = data[numeric_columns]
    Y = data.income

    train_X, val_X, train_y, val_y = train_test_split(X, Y, test_size=0.2, random_state=0)

    wrapper = XGBWrapper(x_train=train_X, y_train=train_y, x_val=val_X, y_val=val_y,
                         param_names=columns, int_params=int_columns)

    input_data = get_params_dataset([n_estimators, learning_rate, max_depth, min_child_weight], columns)
    masker = shap.maskers.Independent(input_data)
    explainer = shap.Explainer(wrapper, masker, feature_names=columns)
    shap_values = explainer(input_data[43:44])

    shap_values.base_values = shap_values.base_values[0]
    shap_values.base_values = shap_values.base_values[0]
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.gcf().set_size_inches(18, 7)
    plt.show()

    # xgb_model = XGBClassifier(n_estimators=10, eval_metric='rmse')
    # xgb_model.fit(train_X, train_y, verbose=False, eval_set=[(val_X, val_y)])
    # evals = xgb_model.evals_result()
    # print(evals['validation_0']['rmse'][-1])
    # shap_values = explainer(train_X)
    # print(shap_values)
    # print(input_data)

    # p = wrapper.params_to_dict(default_params)
    # print(p)
    # model = XGBClassifier(**p)
    # output = wrapper(default_params)
    # print(output)
    # input_data = get_params_dataset([learning_rate, min_split_loss, max_depth, min_child_weight, lam], columns)

    # model = XGBClassifier(n_estimators=5)
    # params = dict(zip(columns, default_params))
    # pprint(params)
    # model.set_params(**params)

    # print(type(model))
    # model = XGBWrapperInherit()
    # print(callable(model))

    # xgb_wrapper = XGBWrapper(params=default_params, param_names=columns, x=X, y=Y)
    # xgb_wrapper.fit()
    # x = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    # import numpy as np
    #
    # x = np.array(x)
    # tab = []
    # tab = {}
    # for idx, chunk in enumerate(np.array_split(x, 2)):
    #     tab[idx] = chunk
