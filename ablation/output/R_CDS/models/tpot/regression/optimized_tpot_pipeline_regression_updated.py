import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import OneHotEncoder, StackingEstimator, ZeroCount
from xgboost import XGBRegressor

# Average CV score on the training set was:-0.20886240713781684
exported_pipeline = make_pipeline(
    ZeroCount(),
    OneHotEncoder(minimum_fraction=0.1, sparse=False, threshold=10),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=2, min_child_weight=7, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.15000000000000002)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=LinearSVR(C=5.0, dual=True, epsilon=0.01, loss="epsilon_insensitive", tol=0.001)),
    LinearSVR(C=0.01, dual=True, epsilon=0.1, loss="squared_epsilon_insensitive", tol=0.01)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
