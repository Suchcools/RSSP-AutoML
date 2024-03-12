import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import OneHotEncoder, ZeroCount

# Average CV score on the training set was:-0.1861912058729816
exported_pipeline = make_pipeline(
    ZeroCount(),
    OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10),
    VarianceThreshold(threshold=0.1),
    ElasticNetCV(l1_ratio=0.2, tol=1e-05)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
