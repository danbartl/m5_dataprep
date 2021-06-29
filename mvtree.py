# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#from IPython import get_ipython

# %%
##  In diesem Skript wird die Verwendung des TSfreshFeatureExtractor zum Extrahieren und rekursiven Vorhersagen von Zeitreihen beispielhaft erklärt ##
#from warnings import warn
#import os
import pandas as pd

from sklearn.pipeline import Pipeline
from mvtree_class import *
from lightgbm.sklearn import LGBMRegressor
import datatable as dt
from sklearn.model_selection import TimeSeriesSplit
# %%

#data= dt.fread("daten.jay")
data= pd.read_pickle("daten.csv")
# %%
#data = data.to_pandas()
# %%
#data['ds'] = data['ds'].apply(lambda x : x[0:10])
data['ds'] = pd.to_datetime(data['ds'],format="%Y-%m-%d")
# %%
data.drop(["combine"],axis=1,inplace=True)
# %%
## Currently just dummy, not really used (only last split)
data["Period"] = data["ds"]


y = data[["y","ts_id","Period"]]
X = data.drop(["ds"],axis = 1,inplace=False)

#train_data, test_data = TimeSeriesSplit(data,test_size=20)
# %%

timeframe = X["Period"].drop_duplicates().sort_values().reset_index().drop("index",axis=1)
timeframe = timeframe["Period"]
tscv = TimeSeriesSplit(n_splits=2, test_size=7)
for train_index, test_index in tscv.split(timeframe):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[X["Period"].isin(timeframe[train_index])],X[X["Period"].isin(timeframe[test_index])]
    y_train, y_test = y[y["Period"].isin(timeframe[train_index])],y[y["Period"].isin(timeframe[test_index])]


# %%
## Festlegen der Fetaure, die der Feature Extractor berechnen soll (für verfügbare Feature siehe Klasse MVTreeFeatureExtractor)
def compute_czeros(x):
  return np.sum(np.cumprod((x==0)[::-1]))/x.shape[0]

model_kwargs = {
  "time_features":['month',"year_month_sin","year_month_cos"],
  "window_functions":{
    "mean":   (None, [1], [7,14,28]),
    #"median":   (None, [1], [6]),
    "kurt":   (None, [1], [7,14]),
    "czeros": (compute_czeros, [1], [7]), 
    "std":    (None, [1], [8]),
     },
  "exclude_features": ["ts_id"],
  #"categorical_features" : ["dept_id","cat_id","store_id","state_id","event_name_1","event_type_1","event_name_2","event_type_2"],
  "ts_values": "y",
  "ts_uid": ["ts_id"], ## Must be a list
  "ts_date": "Period"
}

# %%
## Erzeugen einer sklearn Pipeline, in der zunächst die Feature extrahiert werden und auf den neuen, durch die berechneten Feature ergänzten Daten ein Modell gefittet wird ## 

regressor = Pipeline(
    [('tsfresh3',MVTreeFeatureExtractor(**model_kwargs))
    ]
)
# %%
#outcome_cut.drop(["combine"])
# %%
outcome = regressor.fit_transform(X=X_train)
# %%
from extract_windowlength import find_maxlag

maxlag = find_maxlag(model_kwargs)

outcome_cut = outcome.groupby("ts_id").apply(lambda x: x[maxlag:])
y_cut= y_train.groupby("ts_id").apply(lambda x: x[maxlag:])
# %%
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
# %%
#regressor.transform(X,y)
#regressor.predict(train_data,test_data["Number of airline passengers"])
regressor2 = Pipeline(
    [('lgbm',LGBMRegressor(**model_params))
    ]
)

regressor2.fit(X=outcome_cut,y=y_cut["y"],
lgbm__categorical_feature=["month","dept_id","cat_id","store_id","state_id","event_name_1","event_type_1","event_name_2","event_type_2","snap"])

# %%
predict_data = pd.concat([X_train, X_test]).sort_values(["ts_id","Period"])
predict_data["yhat"] = None
# %%

startstamp = X_train["Period"].drop_duplicates().sort_values().tail(maxlag).min()
test_stamps = X_test.Period.drop_duplicates().sort_values()
base_df = predict_data.query("Period >= @startstamp")
# %%

for timestamp in test_stamps:
  pred_sub = base_df.query("Period <= @timestamp")
  pred_sub.loc[pred_sub["Period"] == timestamp, "y"] = None 
  pred_sub.drop("yhat",axis=1,inplace=True) 
  pred_sub2 = regressor.fit_transform(pred_sub)
  values = regressor2.predict(pred_sub2.groupby("ts_id").tail(1))
  base_df.loc[base_df["Period"] == timestamp,"yhat"] = values

# %%
base_df.set_index(['ts_id',"Period"], inplace=True)
predict_data.set_index(['ts_id',"Period"], inplace=True)
predict_data["yhat"] = None
predict_data.update(base_df)

#forecast_values = pred_sub2.groupby("ts_id").tail(1).loc[:, ["ts_id", "Period"]]

#gs.fit(X=train_data,
#y=find_cutoff(train_data["Number of airline passengers"],model_kwargs)
#)
# %%
#from sktime.performance_metrics.forecasting import median_absolute_error
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from sktime.performance_metrics.forecasting import mean_squared_error
yresult = predict_data.loc[predict_data["yhat"].notnull(),["yhat","y"]]
print(mean_absolute_percentage_error(yresult["y"],yresult["yhat"]))
print(mean_squared_error(yresult["y"],yresult["yhat"],square_root=True))
print(yresult["y"].sum())
print(yresult["yhat"].sum())
#regressor = make_pipeline(
 #   TSFreshFeatureExtractor3(**model_kwargs)
#)

# %%
