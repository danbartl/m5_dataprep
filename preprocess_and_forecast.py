# Databricks notebook source
# %%
import gc
import os
# import pickle
import numpy as np 
import pandas as pd 
import category_encoders as ce
from category_encoders.ordinal import OrdinalEncoder
# import os
import copy
import time
# import numba
import lightgbm as lgb
#import click
#import tsforest 
#from tsforest.forecast import LightGBMForecaster
from tsforest.utils import make_time_range
from scipy.stats import trim_mean
# import plotly.graph_objects as go 
import matplotlib.pyplot as plt
#from tsforest.forecast import LightGBMForecaster
#from mahts import HTSDistributor
#import bayes_opt

# COMMAND ----------

import sys
#sys.path.append('/dbfs/Chronos/data/m5/')

# COMMAND ----------

#os.listdir("/dbfs/Chronos/data/m5/")

# COMMAND ----------
# %%
import os
# import pickle
import numpy as np 
import pandas as pd 
import category_encoders as ce
from category_encoders.ordinal import OrdinalEncoder
# %%

## Read data ##
#os.listdir("/dbfs/Chronos/data/m5")
#os.chdir("input/")
calendar = pd.read_csv('input/calendar.csv')
sell_prices = pd.read_csv('input/sell_prices.csv')
sales_train = pd.read_csv('input/sales_train_evaluation.csv')
sales_train["id"] = sales_train.id.map(lambda x: x.replace("_evaluation", ""))
# %%
# COMMAND ----------

sales_train["item_no"] = sales_train["item_id"].apply(lambda x: int(x[-3:]))
sales_train = sales_train[sales_train["item_no"].isin(range(1, 600,2))]
sales_train = sales_train.drop("item_no", axis = 1)

# COMMAND ----------

# ## Create Subset ##
# product = sales_train[['id','item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
# product["item_no"] = product["item_id"].apply(lambda x: x[-3:])
# product["item_no"] = product["item_no"].apply(lambda x: int(x)) ## Insgesamt 827 unique item_no
# product = product[product.cat_id.isin(['HOBBIES','FOODS'])]

# product_subset = product[product["item_no"] < 80] 
# product_subset.drop("item_no",axis=1,inplace=True)
# keys = list(product_subset.columns.values)
# i1 = sales_train.set_index(keys).index
# i2 = product_subset.set_index(keys).index
# sales_train = sales_train[i1.isin(i2)]

# COMMAND ----------
# %%
## Hierarchical Encoding ##
hierarchy_raw = (sales_train.loc[:, ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]]
             .drop_duplicates())
encoders = dict()
hierarchy = hierarchy_raw.copy()

id_encoder = OrdinalEncoder()
id_encoder.fit(hierarchy.loc[:, ["id"]])
hierarchy["ts_id"]  = id_encoder.transform(hierarchy.loc[:, ["id"]])
encoders["id"] = id_encoder

item_encoder = OrdinalEncoder()
item_encoder.fit(hierarchy.loc[:, ["item_id"]])
hierarchy.loc[:, "item_id"]  = item_encoder.transform(hierarchy.loc[:, ["item_id"]])
encoders["item"] = item_encoder

dept_encoder = OrdinalEncoder()
dept_encoder.fit(hierarchy.loc[:, ["dept_id"]])
hierarchy.loc[:, "dept_id"]  = dept_encoder.transform(hierarchy.loc[:, ["dept_id"]])
encoders["dept"] = dept_encoder

cat_encoder = OrdinalEncoder()
cat_encoder.fit(hierarchy.loc[:, ["cat_id"]])
hierarchy.loc[:, "cat_id"]   = cat_encoder.transform(hierarchy.loc[:, ["cat_id"]])
encoders["cat"] = cat_encoder

store_encoder = OrdinalEncoder()
store_encoder.fit(hierarchy.loc[:, ["store_id"]])
hierarchy.loc[:, "store_id"] = store_encoder.transform(hierarchy.loc[:, ["store_id"]])
encoders["store"] = store_encoder

state_encoder = OrdinalEncoder()
state_encoder.fit(hierarchy.loc[:, ["state_id"]])
hierarchy.loc[:, "state_id"] = state_encoder.transform(hierarchy.loc[:, ["state_id"]])
encoders["state"] = state_encoder

# %%
## Encode Calendar Events ##
event_name_1_encoder = OrdinalEncoder()
event_name_1_encoder.fit(calendar.loc[:, ["event_name_1"]])
calendar.loc[:, "event_name_1"] = event_name_1_encoder.transform(calendar.loc[:, ["event_name_1"]])

event_type_1_encoder = OrdinalEncoder()
event_type_1_encoder.fit(calendar.loc[:, ["event_type_1"]])
calendar.loc[:, "event_type_1"] = event_type_1_encoder.transform(calendar.loc[:, ["event_type_1"]])

event_name_2_encoder = OrdinalEncoder()
event_name_2_encoder.fit(calendar.loc[:, ["event_name_2"]])
calendar.loc[:, "event_name_2"] = event_name_2_encoder.transform(calendar.loc[:, ["event_name_2"]])

event_type_2_encoder = OrdinalEncoder()
event_type_2_encoder.fit(calendar.loc[:, ["event_type_2"]])
calendar.loc[:, "event_type_2"] = event_type_2_encoder.transform(calendar.loc[:, ["event_type_2"]])

# COMMAND ----------

## Enocde categorical features ##
sales_train["ts_id"] = id_encoder.transform(sales_train.loc[:, ["id"]])
sales_train.loc[:, "item_id"]  = item_encoder.transform(sales_train.loc[:, ["item_id"]])
sales_train.loc[:, "dept_id"]  = dept_encoder.transform(sales_train.loc[:, ["dept_id"]])
sales_train.loc[:, "cat_id"]   = cat_encoder.transform(sales_train.loc[:, ["cat_id"]])
sales_train.loc[:, "store_id"] = store_encoder.transform(sales_train.loc[:, ["store_id"]])
sales_train.loc[:, "state_id"] = state_encoder.transform(sales_train.loc[:, ["state_id"]])

## Encode features in sell_prices ##
sell_prices.loc[:, "store_id"] = store_encoder.transform(sell_prices.loc[:, ["store_id"]])
sell_prices.loc[:, "item_id"]  = item_encoder.transform(sell_prices.loc[:, ["item_id"]]) 

# COMMAND ----------

actual_data = pd.melt(sales_train, 
               id_vars=["ts_id","item_id","dept_id","cat_id","store_id","state_id"],
               value_vars=[f"d_{i}" for i in range(1913,1942)],
               var_name="d",
               value_name="q")
actual_data = actual_data.merge(calendar[["d","date"]], on = "d").drop("d", axis = 1)
actual_data["date"] = pd.to_datetime(actual_data["date"])

# COMMAND ----------

## Melt and Merge Datasets ##
data = pd.melt(sales_train, 
               id_vars=["ts_id","item_id","dept_id","cat_id","store_id","state_id"],
               value_vars=[f"d_{i}" for i in range(1,1913)],
               var_name="d",
               value_name="q")

calendar_columns = ["date", "wm_yr_wk", "d", "snap_CA", "snap_TX", "snap_WI",
                    "event_name_1", "event_type_1", "event_name_2", "event_type_2"]

data = pd.merge(data, 
                calendar.loc[:, calendar_columns],
                how="left",
                on="d")

data = pd.merge(data, sell_prices,
                on=["store_id", "item_id", "wm_yr_wk"],
                how="left")

data.sort_values(["item_id","store_id","date"], inplace=True, ignore_index=True)
# %%
# state_encoder.mapping[0]["mapping"]

# COMMAND ----------

## Encode snap-features ##
data["snap"] = 0

idx_snap_ca = data.query("state_id==1 & snap_CA==1").index
data.loc[idx_snap_ca, "snap"] = 1

idx_snap_tx = data.query("state_id==2 & snap_TX==1").index
data.loc[idx_snap_tx, "snap"] = 2

idx_snap_wi = data.query("state_id==3 & snap_WI==1").index
data.loc[idx_snap_wi, "snap"] = 3

data.drop(["snap_CA", "snap_TX", "snap_WI"], axis=1, inplace=True)

# COMMAND ----------


def reduce_mem_usage(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
#data = reduce_mem_usage(data)

# COMMAND ----------

def remove_starting_zeros(dataframe):
    idxmin = dataframe.query("q > 0").index.min()
    return dataframe.loc[idxmin:, :]

# In[39]:
data = (data
        .groupby(["item_id","store_id"])
        .apply(remove_starting_zeros)
        .reset_index(drop=True)
       )

# COMMAND ----------

def find_out_of_stock(df):
    df = df.copy()

    df["no_stock_days"] = 0
    zero_mask = (df.q == 0)
    transition_mask = (zero_mask != zero_mask.shift(1)) #TRUE, falls es von 0 zu !0 (oder umgekehrt) übergeht
    zero_sequences = transition_mask.cumsum()[zero_mask] ## Aufsummieren aller TRUE (1) 
    zero_seqs_count = zero_sequences.map(zero_sequences.value_counts()).to_frame() ## Zähle wie häufig die jeweiligen kumulierten Werte auftreten
    df.loc[zero_seqs_count.index, "no_stock_days"] = zero_seqs_count.q.values
    return df

## no_stock_data gibt an wie lange die Periode ist, für die das jeweilige Item im jeweiligen Store nicht verkauft wird. 
## Ist q für ein Item in eime Store für 4 Tage lang 0 (= es wird vier Tage lang nicht verkauft), so ist no_stock_data für 
## alle vier Zeitpunkte gleich 4 (siehe z.B. data.tail(20)) 
data = data.groupby(["item_id","store_id"]).apply(find_out_of_stock)
# %%
# COMMAND ----------

data.reset_index(drop=True, inplace=True)
data.drop(["d", "wm_yr_wk"], axis=1, inplace=True)
data.rename({"date":"ds"}, axis=1, inplace=True)
data = data.rename({"q":"y"}, axis=1)
data["ds"] = pd.to_datetime(data["ds"])


# COMMAND ----------

## Create evaluation dataframe ##
calendar_columns = ["date", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI",
                    "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
calendar["date"] = pd.to_datetime(calendar["date"])

eval_dataframe = (pd.concat([make_time_range("2016-04-24", "2016-05-22", "D").assign(**row)
                             for _,row in hierarchy.iterrows()], ignore_index=True)
                  .merge(calendar.loc[:, calendar_columns],
                         how="left", left_on="ds", right_on="date")
                  .merge(sell_prices, how="left")
                  .drop(["id","date","wm_yr_wk"], axis=1)
                 )

eval_dataframe["snap"] = 0

idx_snap_ca = eval_dataframe.query("state_id==1 & snap_CA==1").index
eval_dataframe.loc[idx_snap_ca, "snap"] = 1

idx_snap_tx = eval_dataframe.query("state_id==2 & snap_TX==1").index
eval_dataframe.loc[idx_snap_tx, "snap"] = 2

idx_snap_wi = eval_dataframe.query("state_id==3 & snap_WI==1").index
eval_dataframe.loc[idx_snap_wi, "snap"] = 3

eval_dataframe.drop(["snap_CA", "snap_TX", "snap_WI"], axis=1, inplace=True)

eval_dataframe["no_stock_days"] = None
eval_dataframe["ds"] = pd.to_datetime(eval_dataframe["ds"])


# %%
# COMMAND ----------

## Set model parameter ##
def compute_czeros(x):
  return np.sum(np.cumprod((x==0)[::-1]))/x.shape[0]

def compute_sfreq(x):
  return np.sum(x!=0)/x.shape[0]

approach=1

model_params = {
  'objective':'tweedie',
  'tweedie_variance_power': 1.1,
  'metric':'None',
  'max_bin': 127,
  'bin_construct_sample_cnt':20000000,
  'num_leaves': 2**10-1,
  'min_data_in_leaf': 2**10-1,
  'learning_rate': 0.05,
  'feature_fraction':0.8,
  'bagging_fraction':0.8,
  'bagging_freq':1,
  'lambda_l2':0.1,
  'boost_from_average': False,
}

time_features = [
  "year",
  "month",
  #"year_week",
  #"year_day",
  "week_day",
  "month_progress",
  #"week_day_cos",
  #"week_day_sin",
  #"year_day_cos",
  #"year_day_sin",
  "year_week_cos",
  "year_week_sin",
  #"month_cos",
  #"month_sin"
]

exclude_features = [
  "ts_id",
  "no_stock_days",
  "sales",
]

categorical_features = {
  "store_id": "default",  ## Wird automatisch mit "default" hinzugefügt, da in ts_uid_columns
  "item_id": "default", ## Wird automatisch mit "default" hinzugefügt, da in ts_uid_columns
  "state_id": "default",
  "dept_id": "default",
  "cat_id": "default",
  "event_name_1": "default",}
if approach == 1:
  categorical_features["item_id"] = "default"
elif approach == 2:
  categorical_features["item_id"] = ("y", ce.GLMMEncoder, None)
else:
  print("Invalid input.")

model_kwargs = {
  "model_params":model_params,
  "time_features":['week_day', 'month_progress', 'year_week_cos', 'year_week_sin'],
  "window_functions":{
    "mean":   (None, [1,7,28], [7,14,28]),
      "std":    (None, [1,7,28], [7,14,28]),
    "kurt":   (None, [1,7,28], [7,28]),
    "czeros": (compute_czeros, [1,], [7,14])
    },
  "exclude_features":exclude_features,
  "categorical_features":categorical_features,
  "ts_uid_columns":["item_id","store_id"],
  #"ts_uid_columns": ["store_id"]
}

lagged_features_to_dropna = list()
if "lags" in model_kwargs.keys():
    lag_features = [f"lag{lag}" for lag in model_kwargs["lags"]]
    lagged_features_to_dropna.extend(lag_features)
if "window_functions" in model_kwargs.keys():
    rw_features = list()
    for window_func,window_def in model_kwargs["window_functions"].items():
        _,window_shifts,window_sizes = window_def
        if window_func in ["mean","median","std","min","max"]:
            rw_features.extend([f"{window_func}{window_size}_shift{window_shift}"
                                for window_size in window_sizes
                                for window_shift in window_shifts])
    lagged_features_to_dropna.extend(rw_features)

# SEEDS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 
#          43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
# NUM_ITER_RANGE = (500,701)

# %%
import datatable as dt
data = dt.Frame(data)

# %%
data.to_jay("daten.jay")

# %%
# COMMAND ----------

## Create modell with model parameters and set features ##
tic = time.time()
model_level12_base = LightGBMForecaster(**model_kwargs) ## Erzeugen eines Objekts LightGBMForecaster, das von ForecasterBase erbt
model_level12_base.prepare_features(train_data=data) ## Methode aus ForecasterBase
model_level12_base.train_features.dropna(subset=lagged_features_to_dropna, axis=0, inplace=True) ## Methode aus ForecasterBase
# model_level12_base.train_features = reduce_mem_usage(model_level12_base.train_features) ## Methode aus ForecasterBase
gc.collect()
tac = time.time()
print(f"Elapsed time: {(tac-tic)/ 60.} [min]")

# COMMAND ----------

## Fit the model ## 
# test = list()
#for i,seed in enumerate(SEEDS[1]):   
#num_iterations = np.random.randint(*NUM_ITER_RANGE)
num_iterations = 600
seed = 3
#print("#"*100)
# print(f" model {i+1}/{len(SEEDS)} - seed: {seed} - num_iterations: {num_iterations} ".center(100, "#"))
#print("#"*100)

model_level12 = copy.deepcopy(model_level12_base)
model_params["seed"] = seed
model_params["num_iterations"] = num_iterations
model_level12.set_params(model_params)

print("Fitting the model")
tic = time.time()
model_level12.fit()
tac = time.time()
print(f"Elapsed time: {(tac-tic)/60.} [min]")

# COMMAND ----------

## Predict values ## 
# thresh_value
predict_data = model_level12.train_features.query("no_stock_days >= 28").loc[:, model_level12.input_features]
predictions = model_level12.model.model.predict(predict_data) ## Aufrufen der predict methode von lightgbm.train
print(predictions)
thresh_value = trim_mean(predictions, proportiontocut=0.05)
print(thresh_value)
def bias_corr_func(x, tv=thresh_value):
    x[x < tv] = 0
    return x

print("Predicting")
tic = time.time()
forecast = model_level12.predict(eval_dataframe, recursive=True, bias_corr_func=bias_corr_func) ## Erzeugen einer rekursiven Vorhersage 
## Rekursiv beduetet hier, dass die lag-feature in Periode t abhängig von den in Periode t-1 berechneten lag-featuren ermittlet werden
print(forecast)
tac = time.time()
# test.append(forecast)
print(f"Elapsed time: {(tac-tic)/60.} [min]")


# COMMAND ----------

forecast_test = forecast.merge(actual_data[["ts_id","item_id", "store_id", "date"]], left_on = ["ds","item_id", "store_id"], right_on = ["date","item_id", "store_id"])

# COMMAND ----------

## Feature importance ##
# item_id ist GLMM-Encoded und daher nicht mehr so relevant ##
lgb.plot_importance(model_level12.model.model, importance_type="split", figsize=(15,10))


# COMMAND ----------

## Forecast ohne trend correction und hierarchical reconciliation aggregiert (insgesamt) ##
data_agg = data.groupby(["ds"])["y"].sum().reset_index()
data_act = actual_data.groupby(["date"])["q"].sum().reset_index()
forecast_agg = forecast.groupby(["ds"])["y_pred"].sum().reset_index()

plt.figure(figsize=(20,7))
plt.plot_date(data_agg.ds, data_agg.y, "o-", label="historic", color = "lightblue")
plt.plot_date(data_act.date, data_act.q, "o-", label="original", color = "navy")
plt.plot_date(forecast_agg.ds, forecast_agg.y_pred, "o-", label="einfacher forecast", color = "orange")
plt.grid()
plt.legend(loc="best")
plt.show()

# COMMAND ----------

# Forecast ohne trend correction und hierarchical reconciliation aggregiert (auf store_id) ##
data_agg_store = data.groupby(["ds", "store_id"])["y"].sum().reset_index()
data_act_store = actual_data.groupby(["date", "store_id"])["q"].sum().reset_index()
forecast_agg_store = forecast.groupby(["ds", "store_id"])["y_pred"].sum().reset_index()

for store_id in range(1,10):
  data_agg_store_i = data_agg_store[data_agg_store["store_id"] == store_id]
  data_act_store_i = data_act_store[data_act_store["store_id"] == store_id]
  forecast_agg_store_i = forecast_agg_store[forecast_agg_store["store_id"] == store_id]

  plt.figure(figsize=(20,7))
  plt.plot_date(data_agg_store_i.ds, data_agg_store_i.y, "o-", label="original", color = "lightblue")
  plt.plot_date(data_act_store_i.date, data_act_store_i.q, "o-", label="original", color = "navy")
  plt.plot_date(forecast_agg_store_i.ds, forecast_agg_store_i.y_pred, "o-", label="einfacher forecast", color = "orange")
  plt.grid()
  plt.legend(loc="best")
  plt.show()

# COMMAND ----------

## Forecast ohne trend correction und hierarchical reconciliation für item_id = 4# 
# data_item = data[data["item_id"] == 4]
# data_act_item = actual_data[actual_data["item_id"] == 4]
# forecast_item = forecast[forecast["item_id"] == 4]

for i in range(1,10): 
  data_item = data[data["ts_id"] == i]
  data_act_item = actual_data[actual_data["ts_id"] == i]
  forecast_item = forecast_test[forecast_test["ts_id"] == i]
  plt.figure(figsize=(20,7))
  plt.plot_date(data_item.ds, data_item.y, "o-", label="original", color = "lightblue")
  plt.plot_date(data_act_item.date, data_act_item.q, "o-", label="original", color = "navy")
  plt.plot_date(forecast_item.ds, forecast_item.y_pred, "o-", label="einfacher forecast", color = "orange")
  plt.grid()
  plt.legend(loc="best")
  plt.show()

# COMMAND ----------

## Set parameters for trend correction model ##
kwargs_list = [
    # round 0
    ({"primary_bandwidths": np.arange(41, 47),
     "middle_bandwidth": 33,
     "final_bandwidth": 15,
     "alpha": 4,
      "drop_last_n": 0},
     {"primary_bandwidths": np.arange(112, 119),
     "middle_bandwidth": 38,
     "final_bandwidth": 33,
     "alpha": 2}),
    
    ({"primary_bandwidths": np.arange(42, 46),
     "middle_bandwidth": 19,
     "final_bandwidth": 18,
     "alpha": 4,
      "drop_last_n": 0},
     {"primary_bandwidths": np.arange(112, 119),
     "middle_bandwidth": 30,
     "final_bandwidth": 33,
     "alpha": 1}),
    
    ({"primary_bandwidths": np.arange(42, 46),
     "middle_bandwidth": 35,
     "final_bandwidth": 15,
     "alpha": 4,
      "drop_last_n": 0},
     {"primary_bandwidths": np.arange(112, 119),
     "middle_bandwidth": 42,
     "final_bandwidth": 33,
     "alpha": 10}),
    
    # round 2
    ({"primary_bandwidths": np.arange(20, 55),
     "middle_bandwidth": 55,
     "final_bandwidth": 38,
     "alpha": 0,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(112, 119),
     "middle_bandwidth": 43,
     "final_bandwidth": 33,
     "alpha": 0}),
    
    ({"primary_bandwidths": np.arange(21, 54),
     "middle_bandwidth": 55,
     "final_bandwidth": 41,
     "alpha": 1,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(112, 119),
     "middle_bandwidth": 41,
     "final_bandwidth": 33,
     "alpha": 0}),
    
    ({"primary_bandwidths": np.arange(24, 56),
     "middle_bandwidth": 54,
     "final_bandwidth": 41,
     "alpha": 1,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(112, 119),
     "middle_bandwidth": 31,
     "final_bandwidth": 33,
     "alpha": 10}),
        
    # round 3
    ({"primary_bandwidths": np.arange(23, 48),
     "middle_bandwidth": 46,
     "final_bandwidth": 14,
     "alpha": 8,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(103, 119),
     "middle_bandwidth": 52,
     "final_bandwidth": 31,
     "alpha": 10}),
    
    ({"primary_bandwidths": np.arange(29, 41),
     "middle_bandwidth": 54,
     "final_bandwidth": 18,
     "alpha": 5,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(111, 116),
     "middle_bandwidth": 88,
     "final_bandwidth": 37,
     "alpha": 10}),
    
    ({"primary_bandwidths": np.arange(29, 43),
     "middle_bandwidth": 50,
     "final_bandwidth": 14,
     "alpha": 7,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(112, 115),
     "middle_bandwidth": 70,
     "final_bandwidth": 37,
     "alpha": 10}),
    
    # round 4
    ({"primary_bandwidths": np.arange(16, 30),
     "middle_bandwidth": 39,
     "final_bandwidth": 29,
     "alpha": 5,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(107, 116),
     "middle_bandwidth": 156,
     "final_bandwidth": 36,
     "alpha": 0}),
    
    ({"primary_bandwidths": np.arange(15, 33),
     "middle_bandwidth": 43,
     "final_bandwidth": 25,
     "alpha": 6,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(110, 114),
     "middle_bandwidth": 154,
     "final_bandwidth": 39,
     "alpha": 0}),
    
    ({"primary_bandwidths": np.arange(16, 30),
     "middle_bandwidth": 39,
     "final_bandwidth": 29,
     "alpha": 5,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(106, 118),
     "middle_bandwidth": 155,
     "final_bandwidth": 31,
     "alpha": 0}),
        
    # round 5
    ({"primary_bandwidths": np.arange(16, 18),
     "middle_bandwidth": 31,
     "final_bandwidth": 38,
     "alpha": 9,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(112, 114),
     "middle_bandwidth": 62,
     "final_bandwidth": 36,
     "alpha": 9}),
        
    ({"primary_bandwidths": np.arange(16, 18),
     "middle_bandwidth": 31,
     "final_bandwidth": 40,
     "alpha": 4,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(111, 113),
     "middle_bandwidth": 65,
     "final_bandwidth": 37,
     "alpha": 10}),
    
    ({"primary_bandwidths": np.arange(16, 18),
     "middle_bandwidth": 32,
     "final_bandwidth": 41,
     "alpha": 8,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(109, 114),
     "middle_bandwidth": 38,
     "final_bandwidth": 39,
     "alpha": 9}),
    
    # round 6      
    ({"primary_bandwidths": np.arange(37, 39),
     "middle_bandwidth": 28,
     "final_bandwidth": 18,
     "alpha": 2,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(108, 119),
     "middle_bandwidth": 110,
     "final_bandwidth": 34,
     "alpha": 0}),
        
    ({"primary_bandwidths": np.arange(26, 40),
     "middle_bandwidth": 34,
     "final_bandwidth": 16,
     "alpha": 2,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(108, 119),
     "middle_bandwidth": 92,
     "final_bandwidth": 35,
     "alpha": 0}),
    
    ({"primary_bandwidths": np.arange(28, 40),
     "middle_bandwidth": 44,
     "final_bandwidth": 16,
     "alpha": 0,
      "drop_last_n": 1},
     {"primary_bandwidths": np.arange(108, 119),
     "middle_bandwidth": 85,
     "final_bandwidth": 34,
     "alpha": 2}),
    
    # round 7    
    ({"primary_bandwidths": np.arange(41, 45),
     "middle_bandwidth": 44,
     "final_bandwidth": 15,
     "alpha": 10,
      "drop_last_n": 0},
     {"primary_bandwidths": np.arange(112, 119),
     "middle_bandwidth": 28,
     "final_bandwidth": 33,
     "alpha": 1}),
    
    ({"primary_bandwidths": np.arange(41, 45),
     "middle_bandwidth": 48,
     "final_bandwidth": 17,
     "alpha": 9,
      "drop_last_n": 0},
     {"primary_bandwidths": np.arange(112, 119),
     "middle_bandwidth": 124,
     "final_bandwidth": 33,
     "alpha": 3}),
    
    ({"primary_bandwidths": np.arange(17, 47),
     "middle_bandwidth": 47,
     "final_bandwidth": 16,
     "alpha": 8,
      "drop_last_n": 0},
     {"primary_bandwidths": np.arange(112, 117),
     "middle_bandwidth": 106,
     "final_bandwidth": 36,
     "alpha": 2})
]

# COMMAND ----------



# COMMAND ----------

## Trend Correction auf store_id level ##
from trend import apply_robust_trend_correction
forecast_trend = apply_robust_trend_correction(data, forecast, level=3, kwargs_list=kwargs_list)

# COMMAND ----------

data.groupby(["store_id","ds"])["y"].agg("sum")

# COMMAND ----------

## Forecast mit trend correction (insgesamt) ##
forecast_trend_agg = forecast_trend.groupby(["ds"])["y_pred"].sum().reset_index()
plt.figure(figsize=(20,7))
plt.plot_date(data_agg[data_agg["ds"]>= "2015-01-01"].ds, data_agg[data_agg["ds"]>= "2015-01-01"].y, "o-", label="historic",color = "lightblue")
plt.plot_date(data_act.date, data_act.q, "o-", label="real",color = "blue")
plt.plot_date(forecast_agg.ds, forecast_agg.y_pred, "o-", label="einfacher forecast", color = "orange")
plt.plot_date(forecast_trend_agg.ds, forecast_trend_agg.y_pred, "o-", label="trend corrected forecast", color = "green")
plt.grid()
plt.legend(loc="best")
plt.show()

# COMMAND ----------

## Forecast ohne trend correction und hierarchical reconciliation aggregiert (auf store_id) ##
forecast_trend_agg_store = forecast_trend.groupby(["ds", "store_id"])["y_pred"].sum().reset_index()
data_agg_store = data.groupby(["ds", "store_id"])["y"].sum().reset_index()
data_act_store = actual_data.groupby(["date", "store_id"])["q"].sum().reset_index()
forecast_agg_store = forecast.groupby(["ds", "store_id"])["y_pred"].sum().reset_index()



for store_id in range(1,10):
  data_agg_store_i = data_agg_store[(data_agg_store["store_id"] == store_id) & (data_agg_store["ds"] >= "2016-02-01")]
  forecast_agg_store_i = forecast_agg_store[forecast_agg_store["store_id"] == store_id]
  forecast_trend_agg_store_i = forecast_trend_agg_store[forecast_trend_agg_store["store_id"] == store_id]
  data_act_store_i = data_act_store[data_act_store["store_id"] == store_id]

  plt.figure(figsize=(25,8))
  plt.plot_date(data_agg_store_i.ds, data_agg_store_i.y, "o-", label="historic", color = "lightblue")
  plt.plot_date(forecast_agg_store_i.ds, forecast_agg_store_i.y_pred, "o-", label="einfacher forecast", color = "orange")
  plt.plot_date(data_act_store_i.date, data_act_store_i.q, "o-", label="real", color = "navy")
  plt.plot_date(forecast_trend_agg_store_i.ds, forecast_trend_agg_store_i.y_pred, "o-", label="trend corrected forecast", color = "green")
  plt.grid()
  plt.legend(loc="best")
  plt.show()

# COMMAND ----------

# Forecast mit trend_correction auf store_id und mit hierarchical reconciliation auf item_id #
# from tqdm import tqdm

## Einlesen und "dekodieren" der Hierarchie ##
hierarchy_dict = {"root":hierarchy_raw.store_id.unique()}

for store_id in hierarchy_raw.store_id.unique():
    hierarchy_dict[store_id] = hierarchy_raw.query("store_id == @store_id").id.unique()

hts = HTSDistributor(hierarchy_dict)
forecast_trend_hts = forecast_trend.copy()
forecast_trend_hts["store_id"] = encoders["store"].inverse_transform(forecast_trend_hts.store_id)

## Wie viel wird pro Tag in allen stores verkauft ##
forecast_level1_hts = forecast_trend_hts.groupby("ds")["y_pred"].sum().reset_index().set_index("ds").rename({"y_pred":"root"}, axis=1)
## Wie viel wird pro Tag pro Store verkauft ##
forecast_trend_hts2 = forecast_trend_hts.pivot(index="ds", columns="store_id", values="y_pred")
## Wie viel wird pro Tag von jedem Item evrkauft ##
forecast_hts = forecast.merge(hierarchy.loc[:, ["id","item_id","store_id"]], how="left")
forecast_level12_hts = forecast_hts.pivot(index="ds", columns="id", values="y_pred")
## Zusammeführen (eigtl mergen auf ds) ##
forecast_concat = pd.concat([forecast_level1_hts, forecast_trend_hts2, forecast_level12_hts], axis=1)

## Berechnen der Anteile der Spalten auf Bottomebene (Items / Store_ids) an der Top Ebenen Spalte ##
fcst_reconc = hts.compute_forecast_proportions(forecast_concat)
fcst_reconc.set_index(forecast_concat.index, inplace=True)
fcst_reconc = fcst_reconc.loc[:, hts.bottom_nodes].transpose()

predict_start = pd.to_datetime("2016-04-24")
predict_end = pd.to_datetime("2016-06-19")

forecast_reconc = (
    fcst_reconc
    .reset_index()
    .rename({"index":"id"}, axis=1)
    .melt(id_vars="id", 
          value_vars=[predict_start+pd.DateOffset(days=i) for i in range(29)],
          value_name="y_pred"))
forecast_reconc["id_encoded"]= encoders["id"].transform(forecast_reconc.id)

# COMMAND ----------

for i in forecast_test["item_id"].sample(15): 
  data_item = data[data["ts_id"] == i].query("ds>='2016-02-01'")
  data_act_item = actual_data[actual_data["ts_id"] == i]
  forecast_item = forecast_test[forecast_test["ts_id"] == i]
  forecast_reconc_item = forecast_reconc[forecast_reconc["id_encoded"] == i]
  
  plt.figure(figsize=(20,7))
  plt.plot_date(data_item.ds, data_item.y, "o-", label="historic", color = "lightblue")
  plt.plot_date(data_act_item.date, data_act_item.q, "o-", label="real", color = "navy")
  plt.plot_date(forecast_item.ds, forecast_item.y_pred, "o-", label="einfacher forecast", color = "orange")
  plt.plot_date(forecast_reconc_item.ds, forecast_reconc_item.y_pred, "o-", label="reconc forecast", color = "green")
  plt.grid()
  plt.legend(loc="best")
  plt.show()

# COMMAND ----------

from typing import Union
from tqdm.notebook import tqdm_notebook as tqdm

class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f'lv{i + 1}_scale', np.array(scale))
            setattr(self, f'lv{i + 1}_train_df', train_y)
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            print("LV-weight", lv_weight)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())
            print("LV-weight Sum", lv_weight.sum())


    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        #print("Valid_y:", valid_y)
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        print("Scale:", scale)
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores.sum())

        return np.mean(all_scores)

# COMMAND ----------

weight

# COMMAND ----------

train_df

# COMMAND ----------

day_to_week = evaluator.calendar.set_index('d')['wm_yr_wk'].to_dict()
weight_df = evaluator.train_df[['item_id', 'store_id'] + evaluator.weight_columns].set_index(['item_id', 'store_id'])
weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

# COMMAND ----------

weight_df

# COMMAND ----------

weight_df = weight_df.merge(evaluator.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
weight_df['value'] = weight_df['value'] * weight_df['sell_price']

# COMMAND ----------

weight_df

# COMMAND ----------

weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
weight_df

# COMMAND ----------

weight_df = weight_df.loc[zip(evaluator.train_df.item_id, evaluator.train_df.store_id), :].reset_index(drop=True)

# COMMAND ----------

weight_df

# COMMAND ----------

weight_df = pd.concat([evaluator.train_df[evaluator.id_columns], weight_df], axis=1, sort=False)

# COMMAND ----------

weight_df

# COMMAND ----------

weight_df

# COMMAND ----------

weight_df.groupby("store_id")[weight_columns].sum().sum(axis = 1)

# COMMAND ----------

getattr(evaluator, f'lv{3}_weight')

# COMMAND ----------

train_y = train_df.groupby("store_id")[train_df.loc[:, train_df.columns.str.startswith('d_')].columns.tolist()].sum()

# COMMAND ----------

for i, group_id in enumerate(evaluator.group_ids):
  print(group_id, i)

# COMMAND ----------

 valid_y = getattr(evaluator, f'lv{3}_valid_df')

# COMMAND ----------

getattr(evaluator, f'lv{3}_scale')

# COMMAND ----------

valid_y

# COMMAND ----------

valid_preds.groupby('store_id')[evaluator.valid_target_columns].sum()

# COMMAND ----------

((valid_y - valid_preds.groupby('store_id')[evaluator.valid_target_columns].sum())**2).mean(axis = 1)

# COMMAND ----------

test = ((valid_y - valid_preds.groupby('store_id')[evaluator.valid_target_columns].sum())**2).reset_index(drop = True)

# COMMAND ----------

test.iloc[0].mean()

# COMMAND ----------

((valid_y - valid_preds.groupby('store_id')[evaluator.valid_target_columns].sum())**2)

# COMMAND ----------

lv_scores = evaluator.rmsse(valid_preds.groupby('store_id')[evaluator.valid_target_columns].sum(), 3)

# COMMAND ----------

lv_scores

# COMMAND ----------

lv_scores

# COMMAND ----------

valid_preds = pd.concat([evaluator.valid_df[evaluator.id_columns], forecast_right_format.reset_index(drop = True)], axis=1, sort=False)


# COMMAND ----------

valid_preds

# COMMAND ----------

evaluator = WRMSSEEvaluator(train_df =  new, valid_df = new2, calendar =  calendar, prices = sell_prices)
evaluator.score(forecast_right_format.reset_index(drop = True))

# COMMAND ----------

evaluator.rmsse(new2)

# COMMAND ----------

## Prepare data to calculate WRMSSE  ##
valid_df = sales_train.loc[:,"d_1913": "d_1941"]
train_df = sales_train.drop(sales_train.loc[:,"d_1913": "d_1941"], axis = 1)
train_df = train_df.drop({"ts_id"}, axis = 1)

# [Formatieren für WRMSSE Berechnung]
help_train = train_df[["id", "store_id", "item_id"]]
calendar_merge = calendar[["date", "d"]].rename({"date": "ds"}, axis = 1)
calendar_merge["ds"] = pd.to_datetime(calendar_merge["ds"])

# COMMAND ----------

# Calculate WRMSSE for forecast without trend correction and reconciliation ##
forecast_id = help_train.reset_index().merge(forecast, on = ["store_id", "item_id"]).set_index('index')
forecast_help = forecast_id.reset_index().merge(calendar_merge, on ="ds").set_index("index")
forecast_right_format = forecast_help[["d", "item_id","store_id","id", "y_pred"]].pivot(columns =  "d", values = "y_pred")
new = train_df.reset_index(drop = True)
new2 = valid_df.reset_index(drop = True)

# COMMAND ----------

#valid_df: new2, calendar: pd.DataFrame, prices: pd.DataFrame):
train_y = new.loc[:, new.columns.str.startswith('d_')]
train_target_columns = train_y.columns.tolist()
weight_columns = train_y.iloc[:, -28:].columns.tolist()

# COMMAND ----------

train_df['all_id'] = 0  # for lv1 aggregation

id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

if not all([c in valid_df.columns for c in id_columns]):
    valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

# COMMAND ----------

day_to_week = calendar.set_index('d')['wm_yr_wk'].to_dict()
weight_df = train_df[['item_id', 'store_id'] + weight_columns].set_index(['item_id', 'store_id'])
weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

# COMMAND ----------

evaluator = WRMSSEEvaluator(train_df =  new, valid_df = new2, calendar =  calendar, prices = sell_prices)

# COMMAND ----------

evaluator = WRMSSEEvaluator(train_df =  new, valid_df = new2, calendar =  calendar, prices = sell_prices)
evaluator.get_weight_df()
evaluator.score(forecast_right_format.reset_index(drop = True))

# COMMAND ----------

## Calculate WRMSSE for forecast with trend correction and reconciliation ##
forecast_reconc_wmrsse = train_df[["id", "store_id", "item_id"]].reset_index().merge(forecast_reconc, on = "id").set_index("index")
forecast_reconc_wmrsse = forecast_reconc_wmrsse[["ds","item_id","store_id","id", "y_pred"]].reset_index().merge(calendar_merge, on = "ds").set_index("index")

# Pivot table #
forecast_reconc_wmrsse = forecast_reconc_wmrsse[["d", "item_id","store_id","id", "y_pred"]].pivot(columns =  "d", values = "y_pred")
evaluator.score(forecast_reconc_wmrsse.reset_index(drop = True))


# COMMAND ----------

train_df[["id", "store_id", "item_id"]]

