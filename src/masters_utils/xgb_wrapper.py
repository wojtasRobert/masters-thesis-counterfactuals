from xgboost import XGBClassifier
from masters_utils import get_params_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
from sklearn.model_selection import train_test_split
from pprint import pprint


class XGBWrapper:
    def __call__(self, x, *args, **kwargs):
        print('XGBWrapper called')
        model = self.fit(x)
        self.model = model
        return self.get_model_metric(model)

    def __init__(self, n_estimators, x_train, y_train, x_val, y_val, eval_metric, param_names):
        print('init XGBWrapper')
        self.n_estimators = n_estimators
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.eval_metric = eval_metric
        self.param_names = param_names

    def fit(self, params):
        print('fit model')
        params = self.params_to_dict(params)
        model = XGBClassifier(n_estimators=self.n_estimators, eval_metric=self.eval_metric, **params)
        model.fit(self.x_train, self.y_train, verbose=False, eval_set=[(self.x_val, self.y_val)])
        return model

    def predict(self, model, x):
        print('predict output')
        return model.predict(x)

    def params_to_dict(self, row):
        return dict(zip(self.param_names, row))

    def get_model_metric(self, model):
        result = model.evals_result()
        return result['validation_0'][self.eval_metric][-1]


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
    columns = ['learning_rate',
               'gamma',
               'max_depth',
               'min_child_weight',
               'reg_lambda']
    learning_rate = [0.3, 0.05, 0.1]
    min_split_loss = [0, 0.01, 0.05]
    max_depth = [6, 2, 4]
    min_child_weight = [1, 0.1, 0.5]
    lam = [1, 1.5, 3]
    default_params = [0.3, 0, 6, 1, 1]

    numeric_columns = ['marital-status', 'age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss',
                       'hours-per-week']
    data = get_prepared_adult_data()
    X = data[numeric_columns]
    Y = data.income

    train_X, val_X, train_y, val_y = train_test_split(X, Y, test_size=0.2, random_state=0)
    # xgb_model = XGBClassifier(n_estimators=10, eval_metric='rmse')
    # xgb_model.fit(train_X, train_y, verbose=False, eval_set=[(val_X, val_y)])
    # evals = xgb_model.evals_result()
    # print(evals['validation_0']['rmse'][-1])
    # shap_values = explainer(train_X)
    # print(shap_values)
    # print(input_data)

    # model = XGBClassifier(n_estimators=5)
    # params = dict(zip(columns, default_params))
    # pprint(params)
    # model.set_params(**params)

    wrapper = XGBWrapper(n_estimators=50, x_train=train_X, y_train=train_y, x_val=val_X, y_val=val_y,
                         eval_metric='rmse',
                         param_names=columns)
    # output = wrapper(default_params)
    # print(output)
    input_data = get_params_dataset([learning_rate, min_split_loss, max_depth, min_child_weight, lam], columns)
    masker = shap.maskers.Independent(input_data)
    explainer = shap.Explainer(wrapper, masker, feature_names=columns)
    shap_values = explainer(input_data)

    # print(type(model))
    # model = XGBWrapperInherit()
    # print(callable(model))

    # xgb_wrapper = XGBWrapper(params=default_params, param_names=columns, x=X, y=Y)
    # xgb_wrapper.fit()
