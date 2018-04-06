import numpy as np
import pandas as pd

import utils

class DataSetGenerator(object):
    def __init__(self, subset, data_dir="", genre1="Experimental", genre2="Pop", featureSets=['mfcc']):
        """
        Initalize data set generator with the name of the subset we want,
        and load in entire features and tracks datasets. 

        :param str subset: Which subset of the tracks we want to use, 'small', 'medium', or 'large'
        :param str data_dir: Parent directory that stores fma dataset csv's, default=''
        :param str genre1: First genre to include in X and y, default='Experimental'
        :param str genre2: Second genre to include in X and y, default='Pop'
        :param [str] featureSets: sets of high level librosa features to include in X, default=['mfcc']
                                    set to None for all librosa features
        """
        if subset not in ['small', 'medium', 'large']:
            raise ValueError("size is not 'small', 'medium', or 'large'.")
        else:
            self.subset = subset
            self.tracks = utils.load(data_dir+'tracks.csv')
            self.features = utils.load(data_dir+'features.csv')
            self.genre1 = genre1
            self.genre2 = genre2
            self.featureSets = featureSets

    def __getSubTracksAndFeatures(self, tracks, subclass, goal, features):
        """
        Given a starting list of tracks and features, creates subset of tracks and features that follow a 
        dataset constraint as well as the genre and feature constraints of the generator

        :param sequence tracks: a sequence of starting list of tracks to create subset from\
        :type sequence: Series or DataFrame
        :param str subclass: This is the subcategorization of the track's set, either 'split' or 'subset'
        :param str goal: This is the desired value of the subclass to keep in the subset
        :param sequence features: a sequence object of starting list of features to create subset from
        """

        # TODO: This line needs to somehow take into account that subset sizes are ordered 
        #       i.e. any track labelled as part of the small subset, needs to be included in medium and big
        indices = tracks.index[tracks['set', subclass] == goal] # grab the track_ids of all songs in the desired subset.
        subTracks = tracks.loc[indices] # These are subsets of the original tracks
        subFeatures = features.loc[indices] # and features datasets

        genre1 = tracks.index[tracks['track', 'genre_top'] == self.genre1] # collect tracks of genre1
        genre2 = tracks.index[tracks['track', 'genre_top'] == self.genre2] #  collect tracks of genre2
        
        outTracks = subTracks.loc[indices & (genre1 | genre2)] # get small tracks of wanted genres

        if self.featureSets is not None: # get desired features of small tracks of wanted genres
            outFeatures = subFeatures.loc[indices & (genre1 | genre2), self.featureSets]
        else: # use all features
            outFeatures = subFeatures.loc[indices & (genre1 | genre2)]

        return outTracks, outFeatures


    def create_X_y(self, genre1="Experimental", genre2="Pop"):
        """
        Create ndarrays from the subsets of features and tracks datasets that we want to look at.
        """
        #
        self.genre1 = genre1
        self.genre2 = genre2

        genreTracks, genreFeatures = self.__getSubTracksAndFeatures(self.tracks, 'subset', self.subset, self.features) # get desired tracks and features

        X = genreFeatures.as_matrix() # convert features to input matrix
        y = self.__output_classes_from_string_labels(genreTracks['track', 'genre_top']) # create 1v1 output categorization

        return X,y


    def create_X_y_split(self, genre1="Experimental", genre2="Pop"):
        """
        Creates ndarrays from the subsets of features and tracks datasets we want to look at, separating into training, validation, and testing sets
    
        :param str genre1: The first genre we'd like data for
        :param str genre2: The second genre we'd like data for 
        :return X_train, y_train, X_validation, y_validation, X_test, y_test
        """

        self.genre1 = genre1
        self.genre2 = genre2

        indices = self.tracks.index[self.tracks['set', 'subset'] == self.subset] # grab the track_ids of all songs in the 'self.subset' subset.
        tracks = self.tracks.loc[indices]     # These are subsets of the original tracks
        features = self.features.loc[indices] # and features datasets

        splitXy = []

        for split in ['training', 'validation', 'test']:
            subTracks, subFeatures = self.__getSubTracksAndFeatures(tracks, 'split', split, features) # get training items of small set
            splitXy.append(subFeatures.as_matrix()) # append next X
            splitXy.append(self.__output_classes_from_string_labels(subTracks['track', 'genre_top'])) # append next y

        return splitXy

    def __output_classes_from_string_labels(self, sequence):
        """
        Take an iterable of output labels such as a numpy array or pandas series,
        identify all unique labels with numbers, store maps
        from labels to numbers and vice versa. Output numerical labels as ndarray

        :param sequence: a sequence of output labels (strings)
        :type sequence: Series or DataFrame
        :return and nd array of length: len(sequence) of numerical output classes
        """

        labels = sequence.unique()
        enumerated = enumerate(labels)
        self.labels_to_classes = {label: i for i,label in enumerated}
        self.classes_to_labels = {i: label for i,label in enumerated}
        classes = np.zeros(len(sequence))
        for i,label in enumerate(sequence):
            classes[i] = self.labels_to_classes[label]
        return classes


