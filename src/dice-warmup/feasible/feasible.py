# %%
import dice_ml
from dice_ml.utils import helpers

%load_ext autoreload
%autoreload 2
# %%
dataset = helpers.load_adult_income_dataset()
# %%
dataset.head()
# %%
adult_info = helpers.get_adult_data_info()
adult_info
# %%
d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'],
                 outcome_name='income', data_name='adult', test_size=0.1)
# %%
# Użycie wcześniej wytrenowanego modelu zapewniającego dużą celność. W zmiennej
# backendu trzeba określić metodę wyjaśnień, model i metoda wyjaśnień muszą korzystać
# z tej samej biblioteki
backend = {'model': 'pytorch_model.PyTorchModel',
           'explainer': 'feasible_base_vae.FeasibleBaseVAE'}
ML_modelpath = helpers.get_adult_income_modelpath(backend='PYT')
print(ML_modelpath)
ML_modelpath = ML_modelpath[:-4] + '_2nodes.pth'
m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
m.load_model()
print('ML Model', m.model)
# %%
# Generowanie CFów z wykorzystaniem metody VAE
# Inicjalizacja DiCE
exp = dice_ml.Dice(d, m, encoded_size=10, lr=1e-2,
                   batch_size=2048, validity_reg=42.0, margin=0.165, epochs=25,
                   wm1=1e-2, wm2=1e-2, wm3=1e-2)
exp.train(pre_trained=1)
# %%
# zapytanie, dla którego chcemy znaleźć wyjaśnienie (w postaci słownikowej)
query_instance = {'age': 41,
                  'workclass': 'Private',
                  'education': 'HS-grad',
                  'marital_status': 'Single',
                  'occupation': 'Service',
                  'race': 'White',
                  'gender': 'Female',
                  'hours_per_week': 45}
# %%
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=5, desired_class='opposite')
dice_exp.visualize_as_dataframe(show_only_changes=True)
# %%
backend = {'model': 'pytorch_model.PyTorchModel',
           'explainer': 'feasible_model_approx.FeasibleModelApprox'}
ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
ML_modelpath = ML_modelpath[:-4] + '_2nodes.pth'
m = dice_ml.Model(model_path=ML_modelpath, backend=backend)
m.load_model()
print('ML Model', m.model)
# %%
exp = dice_ml.Dice(d, m, encoded_size=10, lr=1e-2, batch_size=2048,
                   validity_reg=76.0, margin=0.344, epochs=25,
                   wm1=1e-2, wm2=1e-2, wm3=1e-2)
exp.train(1, [[0]], 1, 87, pre_trained=1)
# %%
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=5, desired_class="opposite")
dice_exp.visualize_as_dataframe(show_only_changes=True)
