import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import OneHotEncoder, ZeroCount

# Average CV score on the training set was:-0.15448784788928638
exported_pipeline = make_pipeline(
    ZeroCount(),
    ZeroCount(),
    OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10),
    GradientBoostingRegressor(alpha=0.75, learning_rate=0.1, loss="huber", max_depth=3, max_features=0.35000000000000003, min_samples_leaf=16, min_samples_split=3, n_estimators=100, subsample=1.0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
