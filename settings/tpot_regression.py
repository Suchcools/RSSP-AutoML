def run_best_tpot(training_features, training_target, testing_features): 
	import numpy as np
	import pandas as pd
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.linear_model import LassoLarsCV
	from sklearn.model_selection import train_test_split
	from sklearn.pipeline import make_pipeline, make_union
	from sklearn.preprocessing import PolynomialFeatures
	from tpot.builtins import StackingEstimator
	
	# Average CV score on the training set was:-0.17476615181466668
	exported_pipeline = make_pipeline(
	    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
	    StackingEstimator(estimator=LassoLarsCV(normalize=False)),
	    RandomForestRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=13, min_samples_split=14, n_estimators=100)
	)
	
	exported_pipeline.fit(training_features, training_target)
	results = exported_pipeline.predict(testing_features)

	return exported_pipeline, results