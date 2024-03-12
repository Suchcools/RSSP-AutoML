import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import OneHotEncoder, ZeroCount

# Average CV score on the training set was:-0.18475958690883712
exported_pipeline = make_pipeline(
    ZeroCount(),
    OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10),
    RandomForestRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=3, min_samples_split=5, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
