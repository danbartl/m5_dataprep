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
