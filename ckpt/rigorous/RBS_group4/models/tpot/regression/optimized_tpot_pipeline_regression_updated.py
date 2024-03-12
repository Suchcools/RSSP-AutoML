import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from tpot.builtins import StackingEstimator, ZeroCount

# Average CV score on the training set was:-0.18967003715351333
exported_pipeline = make_pipeline(
    ZeroCount(),
    Normalizer(norm="l1"),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.45, min_samples_leaf=7, min_samples_split=6, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
