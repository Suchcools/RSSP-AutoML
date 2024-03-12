import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, Normalizer
from tpot.builtins import StackingEstimator, ZeroCount

# Average CV score on the training set was:-0.1574977667747152
exported_pipeline = make_pipeline(
    ZeroCount(),
    SelectPercentile(score_func=f_regression, percentile=37),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.6500000000000001, min_samples_leaf=17, min_samples_split=19, n_estimators=100)),
    MinMaxScaler(),
    Normalizer(norm="l1"),
    ElasticNetCV(l1_ratio=0.65, tol=0.001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
