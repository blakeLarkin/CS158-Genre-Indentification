from getFeatures import DataSetGenerator
import graphing
from crossValidation import random_forest_hyperparameter_selection, ttest
import utils
import time
import sys

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

def full_feature_raw_data_forest(iters, dsg):
	## Explore various results on the FULL set of librosa features.
	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)


	# set up a random classifier w/ those parameters

def full_feature_PCA_data_forest(iters, dsg):
	## Explore various results on full set of librosa features, applying PCA


	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental", usePCA=True)
	

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Full Feature PCA", best_params, score, iters)

def full_feature_best_info_gain_forest(iters, dsg):
	## Explore results on full feature set; subsetted by top info gaining features

	# split into training and test sets
	X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	# create dataset that uses only most info-gaining features
	data = dsg.create_info_gain_subset(X_train, y_train, X_test, y_test)

	# do random forest hyperparameter selection
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Full Feature Info Gain", best_params, score, iters)


def full_feature_tests(iters):
	# Load full data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=None)

	#full_feature_PCA_data_forest(iters, dsg)
	full_feature_best_info_gain_forest(iters, dsg)



def mfcc_feature_raw_data_forest(iters, dsg):
	## Explore various results on just MFCC features
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=['mfcc'])

	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("MFCC Raw", best_params, score, iters)

def mfcc_feature_PCA_data_forest(iters, dsg):

	# Load full data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=['mfcc'])

	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental", usePCA=True)
	

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("MFCC PCA", best_params, score, iters)

def mfcc_feature_best_info_gain_forest(iters, dsg):

	# split into training and test sets
	X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	# create dataset that uses only most info-gaining features
	data = dsg.create_info_gain_subset(X_train, y_train, X_test, y_test)

	# do random forest hyperparameter selection
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("MFCC Info Gain", best_params, score, iters)


def mfcc_feature_tests(iters):
	# Load full data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=['mfcc'])

	mfcc_feature_raw_data_forest(iters, dsg)
	mfcc_feature_PCA_data_forest(iters, dsg)
	mfcc_feature_best_info_gain_forest(iters, dsg)



def chroma_feature_raw_data_forest(iters):
	# Load full data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=['chroma_cens', 'chroma_cqt', 'chroma_stft'])
	
	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Chroma Raw", best_params, score, iters)


## chroma features w/ PCA
## chroma features w/ info gain

def hand_picked_raw_data_forest(dsg, iters):
	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Hand Picked Raw", best_params, score, iters)

def hand_picked_PCA_data_forest(dsg, iters):
	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental", usePCA=True)

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Hand Picked PCA", best_params, score, iters)

def hand_picked_info_data_forest(dsg, iters):
	# split into training and test sets
	X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")


	data = dsg.create_info_gain_subset(X_train, y_train, X_test, y_test)

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Hand Picked Info", best_params, score, iters)


def hand_picked_tests(iters):
	# Load data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=['mfcc', 'zcr', 'spectral_contrast', 'spectral_bandwith', 'spectral_centroid'])
	
	#hand_picked_raw_data_forest(dsg, iters)
	hand_picked_PCA_data_forest(dsg, iters)
	hand_picked_info_data_forest(dsg, iters)




def results_to_file(exp_name, best_params, score, iters,features=0):
	with open("results.txt", "a") as result_file:
		result_file.write(exp_name+"\n")
		result_file.write("Number of iterations of random search: {}\n".format(iters))
		if features:
			result_file.write("Number of features: {}\n".format(features) )
		result_file.write("Hyperparameters: {}\n".format(str(best_params)))
		result_file.write("Accuracy: {}%\n\n".format(score))


def hyperparameter_optimization_test():
	iters = 100
	with open("results.txt", "w") as result_file:
		result_file.write("")

	start = time.time()
	#full_feature_tests(iters)
	#mfcc_feature_tests(iters)
	hand_picked_tests(100)

	end = time.time()

	print("Time elasped: {}".format(end - start))

def ttest_pca_info(dsg, model):
	dsg.set_lib_feature_sets(features[model])

	X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental", usePCA=True)
	print(X_train.shape)
	X_train_info, y_train_info, X_test_info, y_test_info = dsg.create_info_gain_subset(*dsg.create_X_y_split(genre1="Rock", genre2="Instrumental"))

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
	X1_train, y1_train, X1_test, y1_test = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	dsg.set_lib_feature_sets(features[model2])
	X2_train, y2_train, X2_test, y2_test = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

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
	dsg = DataSetGenerator('small')
	#print(ttest_feature_sets(dsg, 'mfcc_raw', 'hp_raw'))
	print(ttest_pca_info(dsg, 'mfcc_raw'))

	

if __name__ == '__main__':
	main()

	


## try spectral centroid; spectral contrast w/ mfcc and chroma cqt
## TRY ZERO CROSSING RATE ; way of identifying percussive sounds