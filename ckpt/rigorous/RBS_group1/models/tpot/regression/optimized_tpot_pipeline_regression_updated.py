import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator

# Average CV score on the training set was:-0.1700203566433446
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.30000000000000004, tol=0.0001)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.2, min_samples_leaf=5, min_samples_split=6, n_estimators=100)),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    RandomForestRegressor(bootstrap=False, max_features=0.35000000000000003, min_samples_leaf=8, min_samples_split=18, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
