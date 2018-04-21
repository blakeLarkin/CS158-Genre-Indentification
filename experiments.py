from getFeatures import DataSetGenerator
import graphing
from crossValidation import random_forest_hyperparameter_selection
import utils


def full_feature_raw_data_forest(iters):
	## Explore various results on the FULL set of librosa features.

	# Load full data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=None)
	
	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	# do random forest hyperparameter selection -> print best-performing results
	grid_searcher = random_forest_hyperparameter_selection(data, iters)

	# set up a random classifier w/ those parameters

def full_feature_PCA_data_forest(iters):
	## Explore various results on full set of librosa features, applying PCA
	

	# Load full data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=None)

	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental", usePCA=True)
	

	# do random forest hyperparameter selection -> print best-performing results
	grid_searcher = random_forest_hyperparameter_selection(data, iters)

def full_feature_best_info_gain_forest(iters, num_feat):
	## Explore results on full feature set; subsetted by top info gaining features

	# Load full data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=None)

	# split into training and test sets
	X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	# create dataset that uses only most info-gaining features
	data = dsg.create_info_gain_subset(X_train, y_train, X_test, y_test, num_feat)

	# do random forest hyperparameter selection
	grid_searcher = random_forest_hyperparameter_selection(data, iters)

def mfcc_feature_raw_data_forest(iters):
	## Explore various results on just MFCC features
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=['mfcc'])

	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	# do random forest hyperparameter selection -> print best-performing results
	grid_searcher = random_forest_hyperparameter_selection(data, iters)

def mfcc_feature_PCA_data_forest(iters):

	# Load full data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=['mfcc'])

	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental", usePCA=True)
	

	# do random forest hyperparameter selection -> print best-performing results
	grid_searcher = random_forest_hyperparameter_selection(data, iters)

def mfcc_feature_best_info_gain_forest(iters, num_feat):

	# Load full data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=['mfcc'])

	# split into training and test sets
	X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	# create dataset that uses only most info-gaining features
	data = dsg.create_info_gain_subset(X_train, y_train, X_test, y_test, num_feat)

	# do random forest hyperparameter selection
	grid_searcher = random_forest_hyperparameter_selection(data, iters)

def chroma_feature_raw_data_forest(iters):
	# Load full data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=['chroma_cens', 'chroma_cqt', 'chroma_stft'])
	
	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	# do random forest hyperparameter selection -> print best-performing results
	grid_searcher = random_forest_hyperparameter_selection(data, iters)

## chroma features w/ PCA
## chroma features w/ info gain

def hand_picked_raw_data_forest(iters):

	# Load data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=['mfcc', 'zcr', 'spectral_contrast', 'spectral_bandwith', 'spectral_centroid'])
	
	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental")

	# do random forest hyperparameter selection -> print best-performing results
	grid_searcher = random_forest_hyperparameter_selection(data, iters)

def hand_picked_mfcc_PCA_data_forest(iters):

	# Load data set (librosa features)
	dsg = DataSetGenerator(subset="small", genre1="Rock", genre2="Instrumental", libFeatureSets=['chroma_cqt', 'mfcc', 'zcr', 'spectral_contrast'])
	
	# split into training and test sets
	data = dsg.create_X_y_split(genre1="Rock", genre2="Instrumental", usePCA=True)

	# do random forest hyperparameter selection -> print best-performing results
	grid_searcher = random_forest_hyperparameter_selection(data, iters)

# info gain

full_feature_PCA_data_forest(1)

## try spectral centroid; spectral contrast w/ mfcc and chroma cqt
## TRY ZERO CROSSING RATE ; way of identifying percussive sounds