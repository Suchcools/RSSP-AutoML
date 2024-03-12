import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import ZeroCount
from xgboost import XGBRegressor

# Average CV score on the training set was:-0.16760240880668126
exported_pipeline = make_pipeline(
    ZeroCount(),
    SelectPercentile(score_func=f_regression, percentile=73),
    XGBRegressor(learning_rate=0.1, max_depth=2, min_child_weight=4, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.7500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
