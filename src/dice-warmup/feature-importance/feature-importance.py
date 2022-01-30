# %%
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import dice_ml
from dice_ml import Dice
from dice_ml.utils import helpers
# %%
%load_ext autoreload
%autoreload 2
# %%
dataset = helpers.load_adult_income_dataset().sample(5000)
helpers.get_adult_data_info()
# %%
target = dataset["income"]

datasetX = dataset.drop("income", axis=1)
x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=target)
numerical = ["age", "hours_per_week"]
categorical = x_train.columns.difference(numerical)

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categorical)])

classifier = Pipeline(steps=[('preprocessor', transformations), ('classifier', RandomForestClassifier())])
model = classifier.fit(x_train, y_train)
# %%
d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
m = dice_ml.Model(model=model, backend='sklearn')
# %%
exp = Dice(d, m, method='random')
query_instance = x_train[1:2]
e1 = exp.generate_counterfactuals(query_instance, total_CFs=10, desired_range=None,
                                  desired_class='opposite', permitted_range=None, features_to_vary='all')
e1.visualize_as_dataframe(show_only_changes=True)
# %%
# wyliczanie LOKALNEJ ważności cech na podstawie wyznaczonych counterfactuali
imp = exp.local_feature_importance(query_instance, cf_examples_list=e1.cf_examples_list)
print(imp.local_importance)
# %%
imp = exp.local_feature_importance(query_instance, posthoc_sparsity_param=None)
print(imp.local_importance)
# %%
# wyliczanie GLOBALNEJ ważności cech na podstawie wyznaczonych counterfactuali
cobj = exp.global_feature_importance(x_train[0:10], total_CFs=10, posthoc_sparsity_param=None)
print(cobj.summary_importance)
# %%
json_str = cobj.to_json()
print(json_str)
# %%
imp_r = imp.from_json(json_str)
print([o.visualize_as_dataframe(show_only_changes=True) for o in imp_r.cf_examples_list])
print(imp_r.local_importance)
print(imp_r.summary_importance)
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
