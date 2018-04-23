from getFeatures import DataSetGenerator
from graphing import plot_CI_performance
from crossValidation import random_forest_hyperparameter_selection, ttest
from crossValidation import performance_CI
import utils
import time
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

features = {
	'ff_raw': None,
	'ff_pca': None,
	'ff_info': None,
	'mfcc_raw': ['mfcc'],
	'mfcc_pca': ['mfcc'],
	'mfcc_info': ['mfcc'],
	'hp_raw': ['mfcc', 'zcr', 'spectral_contrast', 'spectral_bandwith', 'spectral_centroid'],
	'hp_pca': ['mfcc', 'zcr', 'spectral_contrast', 'spectral_bandwith', 'spectral_centroid'],
	'hp_info': ['mfcc', 'zcr', 'spectral_contrast', 'spectral_bandwith', 'spectral_centroid']
}


hyparams = {
	'ff_pca': {'max_depth': 7, 'max_features': 4, 'n_estimators':62 },
	'ff_info': {'max_depth': 1, 'max_features': 1, 'n_estimators': 37},
	'mfcc_raw': {'max_depth':52, 'max_features':96, 'n_estimators': 82},
	'mfcc_pca': {'max_depth': 12, 'max_features':13, 'n_estimators': 32 },
	'mfcc_info': {'max_depth':52, 'max_features':1, 'n_estimators': 62},
	'hp_raw': {'max_depth': 12, 'max_features': 121, 'n_estimators': 62},
	'hp_pca': {'max_depth': 6, 'max_features': 1, 'n_estimators': 72},
	'hp_info': {'max_depth': 6, 'max_features': 1, 'n_estimators': 12},
}

genre1 = "Folk"

genre2 = "Experimental"



def full_feature_raw_data_forest(iters, dsg):
	## Explore various results on the FULL set of librosa features.
	# split into training and test sets
	data = dsg.create_X_y_split(genre1=genre1, genre2=genre2)

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)


	# set up a random classifier w/ those parameters

def full_feature_PCA_data_forest(iters, dsg):
	## Explore various results on full set of librosa features, applying PCA


	# split into training and test sets
	data = dsg.create_X_y_split(genre1=genre1, genre2=genre2, usePCA=True)


	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Full Feature PCA", best_params, score, iters)

def full_feature_best_info_gain_forest(iters, dsg):
	## Explore results on full feature set; subsetted by top info gaining features

	# split into training and test sets
	X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1=genre1, genre2=genre2)

	# create dataset that uses only most info-gaining features
	data = dsg.create_info_gain_subset(X_train, y_train, X_test, y_test)

	# do random forest hyperparameter selection
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Full Feature Info Gain", best_params, score, iters)


def full_feature_tests(dsg, iters):
	# Load full data set (librosa features)
	dsg.set_lib_feature_sets(features['ff_raw'])

	full_feature_PCA_data_forest(iters, dsg)
	full_feature_best_info_gain_forest(iters, dsg)



def mfcc_feature_raw_data_forest(iters, dsg):
	# split into training and test sets
	data = dsg.create_X_y_split(genre1=genre1, genre2=genre2)

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("MFCC Raw", best_params, score, iters)

def mfcc_feature_PCA_data_forest(iters, dsg):
	# split into training and test sets
	data = dsg.create_X_y_split(genre1=genre1, genre2=genre2, usePCA=True)


	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("MFCC PCA", best_params, score, iters)

def mfcc_feature_best_info_gain_forest(iters, dsg):
	# split into training and test sets
	X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1=genre1, genre2=genre2)

	# create dataset that uses only most info-gaining features
	data = dsg.create_info_gain_subset(X_train, y_train, X_test, y_test)

	# do random forest hyperparameter selection
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("MFCC Info Gain", best_params, score, iters)


def mfcc_feature_tests(dsg, iters):
	# Load full data set (librosa features)
	dsg.set_lib_feature_sets(features['mfcc_raw'])

	mfcc_feature_raw_data_forest(iters, dsg)
	mfcc_feature_PCA_data_forest(iters, dsg)
	mfcc_feature_best_info_gain_forest(iters, dsg)



def chroma_feature_raw_data_forest(iters):
	# Load full data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1=genre1, genre2=genre2, libFeatureSets=['chroma_cens', 'chroma_cqt', 'chroma_stft'])

	# split into training and test sets
	data = dsg.create_X_y_split(genre1=genre1, genre2=genre2)

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Chroma Raw", best_params, score, iters)


## chroma features w/ PCA
## chroma features w/ info gain

def hand_picked_raw_data_forest(dsg, iters):
	# split into training and test sets
	data = dsg.create_X_y_split(genre1=genre1, genre2=genre2)

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Hand Picked Raw", best_params, score, iters)

def hand_picked_PCA_data_forest(dsg, iters):
	# split into training and test sets
	data = dsg.create_X_y_split(genre1=genre1, genre2=genre2, usePCA=True)

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Hand Picked PCA", best_params, score, iters)

def hand_picked_info_data_forest(dsg, iters):
	# split into training and test sets
	X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1=genre1, genre2=genre2)


	data = dsg.create_info_gain_subset(X_train, y_train, X_test, y_test)

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Hand Picked Info", best_params, score, iters)


def hand_picked_tests(dsg, iters):
	# Load data set (librosa features)
	dsg.set_lib_feature_sets(features['hp_raw'])

	hand_picked_raw_data_forest(dsg, iters)
	hand_picked_PCA_data_forest(dsg, iters)
	hand_picked_info_data_forest(dsg, iters)




def results_to_file(exp_name, best_params, score, iters,features=0):
	with open("new_results.txt", "a") as result_file:
		result_file.write(exp_name+"\n")
		result_file.write("Number of iterations of random search: {}\n".format(iters))
		if features:
			result_file.write("Number of features: {}\n".format(features) )
		result_file.write("Hyperparameters: {}\n".format(str(best_params)))
		result_file.write("Accuracy: {}%\n\n".format(score))

# iters = 100
# with open("results.txt", "w") as result_file:
# 	result_file.write("")



# start = time.time()
# full_feature_tests(iters)
# mfcc_feature_tests(iters)
# hand_picked_raw_data_forest(iters)
# hand_picked_mfcc_PCA_data_forest(iters)

# end = time.time()

# print("Time elasped: {}".format(end - start))


def gen_feature_subset_twitter_plot(genre1, genre2):

	training_results = []
	test_results = []
	baseline_results = []


	dsg = DataSetGenerator(subset="small", genre1=genre1, genre2=genre2, libFeatureSets=None)

	# ## Get baseline results
	# X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1=genre1, genre2=genre2)
	# baseline = DecisionTreeClassifier(criterion='entropy', max_depth=3)
	# baseline.fit(X_train, y_train)
	# baseline_results += [performance_CI(baseline, X_test, y_test)]
	# baseline_results += [performance_CI(baseline, X_test, y_test)]


	## Get MFCC results
	dsg = DataSetGenerator(subset="small", genre1=genre1, genre2=genre2, libFeatureSets=['mfcc'])
	X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1=genre1, genre2=genre2)
	mfcc_forest = RandomForestClassifier(criterion='entropy', max_depth=52, max_features=96, n_estimators=82)
	mfcc_forest.fit(X_train, y_train)

	mfcc_train_result = mfcc_forest.score(X_train, y_train)
	mfcc_train_results = (mfcc_train_result, mfcc_train_result, mfcc_train_result)

	training_results += [mfcc_train_results]

	test_results += [performance_CI(mfcc_forest, X_test, y_test)]

	baseline = DecisionTreeClassifier(criterion='entropy', max_depth=3)
	baseline.fit(X_train, y_train)
	baseline_results += [performance_CI(baseline, X_test, y_test)]

	## get hand-picked results
	hand_picked = ['mfcc', 'zcr', 'spectral_contrast', 'spectral_bandwith', 'spectral_centroid']
	dsg = DataSetGenerator(subset="small", genre1=genre1, genre2=genre2, libFeatureSets=hand_picked)
	X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1=genre1, genre2=genre2)
	_, d = X_train.shape
	print("Number of hand picked features is: ", d)

	hand_forest = RandomForestClassifier(criterion='entropy', max_depth=12, max_features=121, n_estimators=62)
	hand_forest.fit(X_train, y_train)

	hand_train_result = hand_forest.score(X_train, y_train)
	hand_train_results = (hand_train_result, hand_train_result, hand_train_result)
	training_results += [hand_train_results]

	test_results += [performance_CI(hand_forest, X_test, y_test)]

	baseline = DecisionTreeClassifier(criterion='entropy', max_depth=3)
	baseline.fit(X_train, y_train)
	baseline_results += [performance_CI(baseline, X_test, y_test)]

	clusters=["MFCC Features", "Hand-Picked Features"]
	bars=["Training", "Test"]
	# bars = ["Test"]

	plot_CI_performance(clusters, bars, baseline_results, training_results, test_results)
	# plot_CI_performance(clusters, bars, baseline_results, test_results)


def hyperparameter_optimization_test(dsg):
	iters = 100
	with open("new_results.txt", "w") as result_file:
		result_file.write("")

	start = time.time()
	#full_feature_tests(iters)
	mfcc_feature_tests(dsg, iters)
	#hand_picked_tests(100)

	end = time.time()

	print("Time elasped: {}".format(end - start))

def ttest_pca_info(dsg, model):
	dsg.set_lib_feature_sets(features[model])

	X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1=genre1, genre2=genre2, usePCA=True)
	print(X_train.shape)
	X_train_info, y_train_info, X_test_info, y_test_info = dsg.create_info_gain_subset(*dsg.create_X_y_split(genre1=genre1, genre2=genre2))

	pca = RandomForestClassifier(criterion='entropy', **hyparams[model.split("_")[0]+'_pca'])
	info = RandomForestClassifier(criterion='entropy', **hyparams[model.split("_")[0]+'_info'])

	print("Training pca...")
	pca.fit(X_train, y_train)
	print("Done Training pca...")
	print("Training info...")
	info.fit(X_train_info, y_train_info)
	print("Done Training info...")

	print("Performing t-test...")
	result = ttest(pca, X_test, y_test, info, X_test_info, y_test_info)

	return result

def ttest_feature_sets(dsg, model1, model2):
	dsg.set_lib_feature_sets(features[model1])
	X1_train, y1_train, X1_test, y1_test = dsg.create_X_y_split(genre1=genre1, genre2=genre2)


	dsg.set_lib_feature_sets(features[model2])
	X2_train, y2_train, X2_test, y2_test = dsg.create_X_y_split(genre1=genre1, genre2=genre2)

	clf1 = RandomForestClassifier(criterion='entropy', **hyparams[model1])
	clf2 = RandomForestClassifier(criterion='entropy', **hyparams[model2])

	print("Training "+model1 + "...")
	clf1.fit(X1_train, y1_train)
	print("Done Training " + model1 + "..." )
	print("Training "+model2 +"...")
	clf2.fit(X2_train, y2_train)
	print("Done Training "+model2+"...")

	print("Performing t-test...")
	result = ttest(clf1, X1_test, y1_test, clf2, X2_test, y2_test)
	return result



def main():
	dsg = DataSetGenerator(subset="small", genre1=genre1, genre2=genre2)
	hyperparameter_optimization_test(dsg)




if __name__ == '__main__':
	main()

