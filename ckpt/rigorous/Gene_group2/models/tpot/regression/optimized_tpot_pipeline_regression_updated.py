import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Average CV score on the training set was:-0.19319753174333248
exported_pipeline = make_pipeline(
    Normalizer(norm="l1"),
    RandomForestRegressor(bootstrap=True, max_features=0.7500000000000001, min_samples_leaf=7, min_samples_split=8, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
