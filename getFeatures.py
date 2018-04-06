import numpy as np
import pandas as pd
from fixtures import DEF_LIB_SETS, DEF_ECHO_SETS
import sklearn as skl
import sklearn.decomposition, sklearn.preprocessing
import utils

class DataSetGenerator(object):
    def __init__(self, subset, data_dir="", genre1="Experimental", genre2="Pop", libFeatureSets=DEF_LIB_SETS, echoFeatureSets=DEF_ECHO_SETS):
        """
        Initalize data set generator with the name of the subset we want,
        and load in entire features and tracks datasets. 

        :param str subset: Which subset of the tracks we want to use, 'small', 'medium', or 'large'
        :param str data_dir: Parent directory that stores fma dataset csv's, default=''
        :param str genre1: First genre to include in X and y, default='Experimental'
        :param str genre2: Second genre to include in X and y, default='Pop'
        :param [str] libFeatureSets: sets of high level librosa features to include in X, default=['mfcc']
                                    set to None for all librosa features
        :param [str] echoFeatureSets: sets of high level echonest features to include in X, default=['mfcc']
                                    set to None for all echonest features
        """
        if subset not in ['small', 'medium', 'large']:
            raise ValueError("size is not 'small', 'medium', or 'large'.")
        else:
            self.subset = subset
            self.tracks = utils.load(data_dir+'tracks.csv')
            self.libFeatures = utils.load(data_dir+'features.csv')
            self.echoFeatures = utils.load(data_dir+'echonest.csv')
            self.genre1 = genre1
            self.genre2 = genre2
            self.libFeatureSets = libFeatureSets
            # currently there are issues with echo features, please avoid use
            self.echoFeatureSets = echoFeatureSets

    def __getSubTracksAndFeatures(self, tracks, subclass, goal, libFeatures, echoFeatures):
        """
        Given a starting list of tracks and features, creates subset of tracks and features that follow a 
        dataset constraint as well as the genre and feature constraints of the generator

        :param sequence tracks: a sequence of starting list of tracks to create subset from\
        :type sequence: Series or DataFrame
        :param str subclass: This is the subcategorization of the track's set, either 'split' or 'subset'
        :param str goal: This is the desired value of the subclass to keep in the subset
        :param sequence libFeatures: a sequence object of starting list of librosa features to create subset from
        :param sequence echoFeatures: a sequence object of starting list of echonest features to create subset from
        """
        indices = tracks.index[tracks['set', subclass] == goal] # grab the track_ids of all songs in the desired subset.
        subTracks = tracks.loc[indices] # These are subsets of the original tracks
        subLibFeatures = libFeatures.loc[indices] # and librosa features
        subEchoFeatures = echoFeatures.loc[indices] # and echonest feature datasets

        genre1 = tracks.index[tracks['track', 'genre_top'] == self.genre1] # collect tracks of genre1
        genre2 = tracks.index[tracks['track', 'genre_top'] == self.genre2] #  collect tracks of genre2
        
        outTracks = subTracks.loc[indices & (genre1 | genre2)] # get small tracks of wanted genres

        if self.libFeatureSets is not None: # get desired librosa features of small tracks of wanted genres
            libFeatures = subLibFeatures.loc[indices & (genre1 | genre2), self.libFeatureSets]
        else: # use all features
            libFeatures = subLibFeatures.loc[indices & (genre1 | genre2)]

        # currently there are issues with echoFeatures, please avoid use
        echoFeatures = subEchoFeatures.loc[indices & (genre1 | genre2), ['echonest']]
        if self.echoFeatureSets is not None: # get desired echonest features of small tracks of wanted genres
            echoFeatures = echoFeatures.loc[indices & (genre1 | genre2), self.echoFeatureSets]

        outFeatures = pd.concat([libFeatures, echoFeatures]) if len(self.echoFeatureSets) != 0 else libFeatures

        return outTracks, outFeatures


    def create_X_y(self):
        """
        Create ndarrays from the subsets of features and tracks datasets that we want to look at.
        """
        genreTracks, genreFeatures = self.__getSubTracksAndFeatures(self.tracks, 'subset', self.subset, self.libFeatures, self.echoFeatures) # get desired tracks and features

        X = genreFeatures.as_matrix() # convert features to input matrix
        y = self.__output_classes_from_string_labels(genreTracks['track', 'genre_top']) # create 1v1 output categorization

        return X,y


    def create_X_y_split(self):
        """
        Creates ndarrays from the subsets of features and tracks datasets we want to look at, separating into training, validation, and testing sets
        """
        indices = self.tracks.index[self.tracks['set', 'subset'] == self.subset] # grab the track_ids of all songs in the 'self.subset' subset.
        tracks = self.tracks.loc[indices]     # These are subsets of the original tracks
        libFeatures = self.libFeatures.loc[indices] # and librosa features
        echoFeatures = self.echoFeatures.loc[indices] # and echonest features datasets

        splitXy = []

        for split in ['training', 'validation', 'test']:
            subTracks, subFeatures = self.__getSubTracksAndFeatures(tracks, 'split', split, libFeatures, echoFeatures) # get training items of small set
            splitXy.append(subFeatures.as_matrix()) # append next X
            splitXy.append(self.__output_classes_from_string_labels(subTracks['track', 'genre_top'])) # append next y

        return splitXy

    def create_Viz_Data(self, genre1 = None, genre2 = None):
        if (genre1 is not None) and (genre2 is not None):
            self.genre1 = genre1
            self.genre2 = genre2

        genreTracks, genreFeatures = self.__getSubTracksAndFeatures(self.tracks, 'subset', self.subset, self.libFeatures, self.echoFeatures)

        X = skl.decomposition.PCA(n_components=2).fit_transform(genreFeatures)
        y = genreTracks['track', 'genre_top']
        y = skl.preprocessing.LabelEncoder().fit_transform(y)

        return X, y

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


