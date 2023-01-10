# %% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# DiCE
import dice_ml

# %%
%load_ext autoreload
%autoreload 2

# %%
data = pd.read_csv('src/works/data/adult.csv')
data.head(10)
# %%
data.shape
# %%
print(data.workclass.unique())
summary = data.describe()
summary = summary.transpose()
print(summary.head())
print(data.nunique())
print(data.workclass.value_counts())
data.workclass.hist()
# %%
data = data.drop('capital-gain', axis=1)
data = data.drop('capital-loss', axis=1)
data = data.drop('fnlwgt', axis=1)
data.head(10)
# %%
attr, counts = np.unique(data['workclass'], return_counts=True)
print(attr)
print(counts)
# %%
most_freq_attr = attr[np.argmax(counts, axis=0)]
print(most_freq_attr)
# %%
data['workclass'][data['workclass'] == '?'] = most_freq_attr
data.workclass.hist()
# %%
data.head(10)
# %%
attrib, counts = np.unique(data['occupation'], return_counts = True)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
data['occupation'][data['occupation'] == '?'] = most_freq_attrib 

attrib, counts = np.unique(data['native-country'], return_counts = True)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
data['native-country'][data['native-country'] == '?'] = most_freq_attrib 

data.head(10)
# %%
data['income'] = data['income'].map({'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K': 1})
data.head(10)

# %%
target = data['income']
train_dataset, test_dataset, y_train, y_test = train_test_split(data,
                                                                target,
                                                                test_size=0.2,
                                                                random_state=0,
                                                                stratify=target)
x_train = train_dataset.drop('income', axis=1)
x_test = test_dataset.drop('income', axis=1)
# %%
d = dice_ml.Data(dataframe=train_dataset, continuous_features=['age', 'hours-per-week'], outcome_name='income')

# %%
numerical = ['age', 'hours-per-week']
categorical = x_train.columns.difference(numerical)

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical)])
clf = Pipeline(steps=[('preprocessor', transformations), ('classifier', LogisticRegression(solver='sag',max_iter=10000))])
model = clf.fit(x_train, y_train)
# %%
m = dice_ml.Model(model=model, backend='sklearn') # inicjalizacja obiektu wyjaśniającego, podanie modelu ML i backendu z którego korzystamy
exp = dice_ml.Dice(d, m, method='random') 
# %%
e1 = exp.generate_counterfactuals(x_test[0:1], total_CFs=3, desired_class='opposite')
e1.visualize_as_dataframe(show_only_changes=True)

# %%
german_data = pd.read_csv('src/works/data/german_data_credit_cat.csv')
german_data.head(10)

#%%
german_data.isnull().sum()
#%%
german_data.dtypes
#%%
target = german_data['risk'] 
train_dataset, test_dataset, y_train, y_test = train_test_split(german_data,
                                                                target,
                                                                test_size=0.2,
                                                                random_state=0,
                                                                stratify=target)
x_train = train_dataset.drop('risk', axis=1)
x_test = test_dataset.drop('risk', axis=1)

#%%
numerical = ['monthly-duration', 'credit-amount', 'installment-rate', 'residence', 'age', 'number-of-credits-at-this-bank', 'number-of-liable-people']
d = dice_ml.Data(dataframe=train_dataset, continuous_features=numerical, outcome_name='risk')

#%%
categorical = x_train.columns.difference(numerical)

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical)])
clf = Pipeline(steps=[('preprocessor', transformations), ('classifier', LogisticRegression(solver='sag',max_iter=10000))])
model = clf.fit(x_train, y_train)

#%%
m = dice_ml.Model(model=model, backend='sklearn') # inicjalizacja obiektu wyjaśniającego, podanie modelu ML i backendu z którego korzystamy
exp = dice_ml.Dice(d, m, method='random') 

#%%
e2 = exp.generate_counterfactuals(x_test[0:1], total_CFs=3, desired_class='opposite')
e2.visualize_as_dataframe(show_only_changes=True)