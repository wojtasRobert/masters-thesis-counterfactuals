# module masters_utils

from pprint import pprint

from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from itertools import product


def get_xgb_params_as_dataframe(model: XGBClassifier) -> pd.DataFrame:
    params = model.get_xgb_params()
    key_list = list(params.keys())
    val_list = [params[k] for k in key_list]
    result_df = pd.DataFrame(data=np.array([val_list]), columns=key_list)
    return result_df


def get_xgb_quality(model: XGBClassifier, test_y):
    pass


def get_params_dataset(params_lists, columns):
    return pd.DataFrame(data=[[*x] for x in list(product(*params_lists))], columns=columns)


if __name__ == '__main__':
    xgb = XGBClassifier(n_estimators=250)
    params_dataframe = get_xgb_params_as_dataframe(xgb)
    pprint(params_dataframe)