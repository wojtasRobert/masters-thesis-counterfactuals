# %%
import dice_ml
from dice_ml.utils import helpers

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
# %%
%load_ext autoreload
%autoreload 2
# %%
dataset = helpers.load_adult_income_dataset()
# %%
dataset.head()
# %%
d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
# %%
target = dataset['income']
datasetX = dataset.drop('income', axis=1)
x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=target)
numerical = ['age', 'hours_per_week']
categorical = x_train.columns.difference(numerical)

#tworzenie pipeline'ów dla danych numerycznych i 
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categorical)
    ]
)
classifier = Pipeline(steps=[('preprocessor', transformations), ('classifier', RandomForestClassifier())])
model = classifier.fit(x_train, y_train)
# %%
backend = 'sklearn'
m = dice_ml.Model(model=model, backend=backend)
# %%
# Generowanie różnorodnych Counterfactuali
# %%
exp_random = dice_ml.Dice(d, m, method='random')
# %%
query_instances = x_train[4:6]
# %%
# generowanie CFs
dice_exp_random = exp_random.generate_counterfactuals(query_instances, total_CFs=2, desired_class='opposite', verbose=False)
# %%
dice_exp_random.visualize_as_dataframe(show_only_changes=True)
# %%
# Możemy zmieniać seed przy metodzie losowego generowania CF
dice_exp_random = exp_random.generate_counterfactuals(query_instances,
                                                      total_CFs=4,
                                                      desired_class="opposite",
                                                      random_seed=9)
# %%
dice_exp_random.visualize_as_dataframe(show_only_changes=True)
# %%
# Użycie k-wymiarowego drzewa do wyznaczania CFów polega na budowaniu drzew
# dla każdej klasy i odpytywanie drzewa konkretnej klasy w celu znalezienia 
# k najbliższych CFów ze zbioru danych. Idea znajdowania najbliższych punktów ze 
# zbioru danych ma zapewnić wykonalność CFów (feasibility)
# %%
exp_KD = dice_ml.Dice(d, m, method='kdtree')
# %%
dice_exp_KD = exp_KD.generate_counterfactuals(query_instances, total_CFs=4, desired_class='opposite')
# %%
dice_exp_KD.visualize_as_dataframe(show_only_changes=True)
# %%
# Dla tej metody można ustalić które atrybuty mają się zmieniać. Trzeba tylko pamiętać,
# że CFy pochodzą tylko ze zbioru treningowego. Nowe punkty generowane są przez metody 
# losowe i algorytm genetyczny
dice_exp_KD = exp_KD.generate_counterfactuals(
    query_instances, total_CFs=4, desired_class="opposite",
    features_to_vary=['age', 'workclass', 'education', 'occupation', 'hours_per_week'])
# %%
dice_exp_KD.visualize_as_dataframe(show_only_changes=True)
# %%
# Możliwa jest także manipulacja zasięgami ciągłych cech
dice_exp_KD = exp_KD.generate_counterfactuals(
    query_instances, total_CFs=5, desired_class="opposite",
    permitted_range={'age': [30, 50], 'hours_per_week': [40, 60]})
dice_exp_KD.visualize_as_dataframe(show_only_changes=True)
# %%
# %%
# %%
# %%
# %%