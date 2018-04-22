from getFeatures import DataSetGenerator
import graphing
from crossValidation import random_forest_hyperparameter_selection
import utils
import time
import sys


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

	#mfcc_feature_raw_data_forest(iters, dsg)
	#mfcc_feature_PCA_data_forest(iters, dsg)
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

def hand_picked_raw_data_forest(iters):

	# Load data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=['mfcc', 'zcr', 'spectral_contrast', 'spectral_bandwith', 'spectral_centroid'])
	
	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Hand Picked Raw", best_params, score, iters)

def hand_picked_mfcc_PCA_data_forest(iters):

	# Load data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=['chroma_cqt', 'mfcc', 'zcr', 'spectral_contrast'])
	
	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental", usePCA=True)

	# do random forest hyperparameter selection -> print best-performing results
	best_params, score = random_forest_hyperparameter_selection(data, iters)
	results_to_file("Hand Picked MFCC PCA", best_params, score, iters)




def results_to_file(exp_name, best_params, score, iters,features=0):
	with open("results.txt", "a") as result_file:
		result_file.write(exp_name+"\n")
		result_file.write("Number of iterations of random search: {}\n".format(iters))
		if features:
			result_file.write("Number of features: {}\n".format(features) )
		result_file.write("Hyperparameters: {}\n".format(str(best_params)))
		result_file.write("Accuracy: {}%\n\n".format(score))


iters = 100
with open("results.txt", "w") as result_file:
	result_file.write("")



start = time.time()
full_feature_tests(iters)
mfcc_feature_tests(iters)
hand_picked_raw_data_forest(iters)
hand_picked_mfcc_PCA_data_forest(iters)

end = time.time()

print("Time elasped: {}".format(end - start))

## try spectral centroid; spectral contrast w/ mfcc and chroma cqt
## TRY ZERO CROSSING RATE ; way of identifying percussive sounds