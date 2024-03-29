import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, Normalizer
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-0.16123356839001252
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.01, loss="exponential", n_estimators=100)),
    SelectPercentile(score_func=f_regression, percentile=37),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.6500000000000001, min_samples_leaf=19, min_samples_split=19, n_estimators=100)),
    MinMaxScaler(),
    Normalizer(norm="l1"),
    ElasticNetCV(l1_ratio=0.4, tol=0.001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
