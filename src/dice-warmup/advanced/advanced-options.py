# %%
# Skrypt prezentujący opcje zaawansowane przy wyznaczaniu CFów
# %%
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
from numpy.random import seed

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# %%
d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
# %%
# Załadowanie modelu uczenia maszynowego
# %%
# seeding 
seed(1)
tf.random.set_seed(1)
# %%
backend = 'TF'+tf.__version__[0]
# dostarczenie wytrenowanego modelu dla DiCE
ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
# %%
exp = dice_ml.Dice(d, m)
# %%
query_instance = {'age': 22,
                  'workclass': 'Private',
                  'education': 'HS-grad',
                  'marital_status': 'Single',
                  'occupation': 'Service',
                  'race': 'White',
                  'gender': 'Female',
                  'hours_per_week': 45}
# %%
# generowanie CF
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class='opposite')
# %%
dice_exp.visualize_as_dataframe(show_only_changes=True)
# %%
# MAD !!!
mads = d.get_mads(normalized=True)

# create feature weights
feature_weights = {}
for feature in mads:
    feature_weights[feature] = round(1/mads[feature], 2)
print(feature_weights)
# %%
feature_weights = {'age': 1, 'hours_per_week': 1}
# %%
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite",
                                        feature_weights=feature_weights)
# %%
# zmiana proximity_weight z domyślnej wartości 0.5 do 1.5
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite",
                                        proximity_weight=1.5, diversity_weight=1.0)
# %%
dice_exp.visualize_as_dataframe(show_only_changes=True)
# %%