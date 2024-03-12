import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import OneHotEncoder, StackingEstimator, ZeroCount

# Average CV score on the training set was:-0.16904083218350746
exported_pipeline = make_pipeline(
    ZeroCount(),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.8, learning_rate=0.001, loss="huber", max_depth=9, max_features=1.0, min_samples_leaf=9, min_samples_split=16, n_estimators=100, subsample=0.3)),
    OneHotEncoder(minimum_fraction=0.2, sparse=False, threshold=10),
    RandomForestRegressor(bootstrap=False, max_features=0.2, min_samples_leaf=2, min_samples_split=12, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)