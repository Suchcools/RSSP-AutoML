import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import OneHotEncoder, StackingEstimator, ZeroCount

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-0.20971537393362386
exported_pipeline = make_pipeline(
    ZeroCount(),
    OneHotEncoder(minimum_fraction=0.1, sparse=False, threshold=10),
    StackingEstimator(estimator=LinearSVR(C=5.0, dual=True, epsilon=0.01, loss="epsilon_insensitive", tol=0.01)),
    LinearSVR(C=0.01, dual=True, epsilon=0.1, loss="squared_epsilon_insensitive", tol=0.01)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)