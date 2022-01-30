# %% imports
# sklearn
import imp
from multiprocessing import Pipe
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# tensorflow
import tensorflow as tf

# DiCE
import dice_ml
from dice_ml.utils import helpers

# %%
%load_ext autoreload
%autoreload 2

# %%
dataset = helpers.load_adult_income_dataset()

# %%
dataset.head()

# %%
# description of transformed features
adult_info = helpers.get_adult_data_info()
adult_info

# %%
target = dataset["income"]
train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                target,
                                                                test_size=0.2,
                                                                random_state=0,
                                                                stratify=target)
x_train = train_dataset.drop('income', axis=1)
x_test = test_dataset.drop('income', axis=1)

# %%
d = dice_ml.Data(dataframe=train_dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')

# %%
numerical = ['age', 'hours_per_week']
categorical = x_train.columns.difference(numerical)

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical)])

clf = Pipeline(steps=[('preprocessor', transformations), ('classifier', RandomForestClassifier())])
model = clf.fit(x_train, y_train)

# %%
m = dice_ml.Model(model=model, backend='sklearn') # inicjalizacja obiektu wyjaśniającego, podanie modelu ML i backendu z którego korzystamy
exp = dice_ml.Dice(d, m, method='random') # użycie metody próbkowania losowego

# %%
e1 = exp.generate_counterfactuals(x_test[0:1], total_CFs=3, desired_class='opposite')
e1.visualize_as_dataframe(show_only_changes=True)

# %% 
e1.visualize_as_dataframe(show_only_changes=False)

# %%
# Zmiana tylko w wybranych atrybutach
e2 = exp.generate_counterfactuals(x_test[0:1], 
                                  total_CFs=3,
                                  desired_class='opposite',
                                  features_to_vary=['education', 'occupation'])
e2.visualize_as_dataframe(show_only_changes=True)

# %%
# Ustawienie ograniczeń na poszczególne atrybuty
e3 = exp.generate_counterfactuals(x_test[0:1], 
                                  total_CFs=3,
                                  desired_class='opposite',
                                  permitted_range={'age': [20, 30], 'education': ['Doctorate', 'Prof-school']})
e3.visualize_as_dataframe(show_only_changes=True)

# %%
# Lokalna ważność cech w zbiorze danych by DiCE
query_instance = x_test[0:1]
imp = exp.local_feature_importance(query_instance, total_CFs=100) # liczba CF - im więcej tym dokładniejszy wynik
print(imp.local_importance)

# %%
# Globalna ważność cech
query_instances = x_test[0:20] # im więcej punktów danych tym dokładniejszy wynik ważności globalnej
imp = exp.global_feature_importance(query_instances)
print(imp.summary_importance)


# %%
# Modele głębokie
# pominięcie warningów
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# nowy backend TF1
backend = 'TF'+tf.__version__[0]
ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
m = dice_ml.Model(model_path=ML_modelpath, backend=backend)

# %%
# Inicjalizacja DiCE'a
exp = dice_ml.Dice(d, m)

# %%
# query w formie słownika bądź dataframe
query_instance = {'age': 22,
                  'workclass': 'Private',
                  'education': 'HS-grad',
                  'marital_status': 'Single',
                  'occupation': 'Service',
                  'race': 'White',
                  'gender': 'Female',
                  'hours_per_week': 45,
                  'income': 0}

# %%
# tak jak wcześniej
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class='opposite')

# %%
# wizualizacja
dice_exp.visualize_as_dataframe(show_only_changes=True)

# %%
# MODELE PYTORCH
backend = 'PYT'
ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
m = dice_ml.Model(model_path=ML_modelpath, backend=backend)

# %%
exp = dice_ml.Dice(d, m)

# %%
# I jeszcze jeden... i jeszcze raz
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite")

# %%
dice_exp.visualize_as_dataframe(show_only_changes=True)

# %%
