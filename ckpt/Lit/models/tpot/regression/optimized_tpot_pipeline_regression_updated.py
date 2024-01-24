import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Average CV score on the training set was:-0.06820257228033583
exported_pipeline = GradientBoostingRegressor(alpha=0.9, learning_rate=0.1, loss="huber", max_depth=5, max_features=0.9500000000000001, min_samples_leaf=2, min_samples_split=6, n_estimators=100, subsample=0.5)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
