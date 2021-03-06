from getFeatures import DataSetGenerator
from graphing import plot_CI_performance
from crossValidation import random_forest_hyperparameter_selection
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



def full_feature_raw_data_forest(dsg, iters):
	## Explore various results on the FULL set of librosa features.
	# split into training and test sets
	data = dsg.create_X_y_split(genre1=genre1, genre2=genre2)

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)


	# set up a random classifier w/ those parameters

def full_feature_PCA_data_forest(dsg, iters):
	## Explore various results on full set of librosa features, applying PCA


	# split into training and test sets
	data = dsg.create_X_y_split(genre1=genre1, genre2=genre2, usePCA=True)


	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Full Feature PCA", best_params, score, iters)

def full_feature_best_info_gain_forest(dsg, iters, num_feat=None):
	## Explore results on full feature set; subsetted by top info gaining features

	# split into training and test sets
	X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1=genre1, genre2=genre2)

	# create dataset that uses only most info-gaining features
	data = dsg.create_info_gain_subset(X_train, y_train, X_test, y_test, num_feat)

	# do random forest hyperparameter selection
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Full Feature Info Gain", best_params, score, iters)


def full_feature_tests(dsg, iters):
	# Load full data set (librosa features)
	dsg.set_lib_feature_sets(features['ff_raw'])

	full_feature_PCA_data_forest(dsg,iters)
	full_feature_best_info_gain_forest(dsg,iters)



def mfcc_feature_raw_data_forest(dsg, iters):
	# split into training and test sets
	data = dsg.create_X_y_split(genre1=genre1, genre2=genre2)

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("MFCC Raw", best_params, score, iters)

def mfcc_feature_PCA_data_forest(dsg, iters):
	# split into training and test sets
	data = dsg.create_X_y_split(genre1=genre1, genre2=genre2, usePCA=True)


	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("MFCC PCA", best_params, score, iters)

def mfcc_feature_best_info_gain_forest(dsg,iters):
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

	mfcc_feature_raw_data_forest(dsg,iters)
	mfcc_feature_PCA_data_forest(dsg,iters)
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

	full_feature_tests(dsg, iters)
	mfcc_feature_tests(dsg, iters)
	hand_picked_tests(dsg, iters)

	end = time.time()

	print("Time elasped: {}".format(end - start))

def main():
        dsg = DataSetGenerator(subset="small", genre1=genre1, genre2=genre2)


        #hand_picked_raw_data_forest(dsg,100)
        #full_feature_PCA_data_forest(dsg,100)
        #full_feature_best_info_gain_forest(dsg,100)
        #full_feature_best_info_gain_forest(dsg, 100, 1)
        full_feature_best_info_gain_forest(dsg, 100, 2)


if __name__ == '__main__':
	main()


