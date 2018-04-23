import collections
import numpy as np
import pandas as pd
from fixtures import DEF_LIB_SETS, DEF_ECHO_SETS, DEF_CACHE_CAP
import sklearn as skl
import sklearn.decomposition, sklearn.preprocessing, sklearn.feature_selection
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
            #self.echoFeatures = utils.load(data_dir+'echonest.csv')
            self.genre1 = genre1
            self.genre2 = genre2
            self.libFeatureSets = libFeatureSets
            # currently there are issues with echo features, please avoid use
            self.echoFeatureSets = echoFeatureSets
            self.cache = LRUCache(DEF_CACHE_CAP)
            self.num_PCA_components = None
            self.num_ig_features = None

    def getSubTracksAndFeatures(self, tracks, subclass, goal, libFeatures, echoFeatures, allGenres=False):
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

        # TODO: This line needs to somehow take into account that subset sizes are ordered 
        #       i.e. any track labelled as part of the small subset, needs to be included in medium and big
        # this was done with a <= before, but that is pretty handwavy and doesn't work for subclass = 'split'
        indices = tracks.index[tracks['set', subclass] == goal] # grab the track_ids of all songs in the desired subset.
        subTracks = tracks.loc[indices] # These are subsets of the original tracks
        subLibFeatures = libFeatures.loc[indices] # and librosa features
        subEchoFeatures = echoFeatures.loc[indices] # and echonest feature datasets

        genre1 = tracks.index[tracks['track', 'genre_top'] == self.genre1] # collect tracks of genre1
        genre2 = tracks.index[tracks['track', 'genre_top'] == self.genre2] #  collect tracks of genre2

        # Check for features and tracks in cache
        query_key = cache_hash([subclass, goal, self.genre1, self.genre2])
        query_result = self.cache.get(query_key)
        if not allGenres:
            if query_result != -1:
                # Successful cache query
                return query_result
            else:
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

                # Add new outTracks and Features to cache
                self.cache.set(query_key, [outTracks, outFeatures])
                return outTracks, outFeatures
        else: #I don't get the cache so I'm avoiding it here
            outTracks = subTracks.loc[indices] # get small tracks of wanted genres

            if self.libFeatureSets is not None: # get desired librosa features of small tracks of wanted genres
                libFeatures = subLibFeatures.loc[indices, self.libFeatureSets]
            else: # use all features
                libFeatures = subLibFeatures.loc[indices]

            # currently there are issues with echoFeatures, please avoid use
            echoFeatures = subEchoFeatures.loc[indices, ['echonest']]
            if self.echoFeatureSets is not None: # get desired echonest features of small tracks of wanted genres
                echoFeatures = echoFeatures.loc[indices, self.echoFeatureSets]

            # currently there are issues with echoFeatures, please avoid use
            echoFeatures = subEchoFeatures.loc[indices & (genre1 | genre2), ['echonest']]
            if self.echoFeatureSets is not None: # get desired echonest features of small tracks of wanted genres
                echoFeatures = echoFeatures.loc[indices & (genre1 | genre2), self.echoFeatureSets]

            outFeatures = pd.concat([libFeatures, echoFeatures]) if len(self.echoFeatureSets) != 0 else libFeatures

            # Add new outTracks and Features to cache
            self.cache.set(query_key, [outTracks, outFeatures])
            return outTracks, outFeatures


    def create_X_y(self, genre1="Experimental", genre2="Pop", usePCA=False, l=None, allGenres=False):
        """
        Create ndarrays from the subsets of features and tracks datasets that we want to look at.
        """

        # update genres if necessary
        if (genre1 is not None) and (genre2 is not None):
            self.genre1 = genre1
            self.genre2 = genre2

        indices = self.tracks.index[self.tracks['set', 'subset'] == self.subset] # grab the track_ids of all songs in the 'self.subset' subset.
        tracks = self.tracks.loc[indices]     # These are subsets of the original tracks
        libFeatures = self.libFeatures.loc[indices] # and librosa features
        echoFeatures = self.echoFeatures.loc[indices] # and echonest features datasets    

        genreTracks, genreFeatures = self.getSubTracksAndFeatures(tracks, 'subset', self.subset, libFeatures, echoFeatures, allGenres=allGenres) # get desired tracks and features
        print(genreTracks.shape)
        print(genreTracks['track','genre_top'].unique())

        if usePCA:
            if l is None:
                PCA, X = utils.preserveVarPCA(genreFeatures)
                l = PCA.n_components_
                print("The number of PCA components that preserves 95% variance is: ", l)
            else:
                X = skl.decomposition.PCA(n_components=l).fit_transform(genreFeatures)
            y = genreTracks['track', 'genre_top']
            y = skl.preprocessing.LabelEncoder().fit_transform(y)
        else:
            X = genreFeatures.as_matrix() # convert features to input matrix
            y = self.__output_classes_from_string_labels(genreTracks['track', 'genre_top']) # create 1v1 output categorization

        return X,y


    def create_X_y_split(self, genre1="Experimental", genre2="Pop", usePCA=False, l=None):
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
        libFeatures = self.libFeatures.loc[indices] # and librosa features
        echoFeatures = self.echoFeatures.loc[indices] # and echonest features datasets

        splitXy = {}

        for split in ['training', 'validation', 'test']:
            splitXy[split] = tuple(self.getSubTracksAndFeatures(tracks, 'split', split, libFeatures, echoFeatures)) # get training items of small set
            
            # splitXy.append(subFeatures.as_matrix()) # append next X
            # splitXy.append(self.__output_classes_from_string_labels(subTracks['track', 'genre_top'])) # append next y

        sub_features = pd.concat([splitXy['training'][1], splitXy['validation'][1]])
        sub_tracks = pd.concat([splitXy['training'][0], splitXy['validation'][0]])

        test_sub_features = splitXy['test'][1]
        test_sub_tracks = splitXy['test'][0]

        if usePCA:
            if l is None:
                PCA, X_train = utils.preserveVarPCA(sub_features)
                l = PCA.n_components_
                with open('results.txt', 'a') as result_file:
                    result_file.write("Num components PCA preserving 95% variance: {}\n".format(l))
            else:
                PCA = skl.decomposition.PCA(n_components=l)
                # fits to training data and transforms
                X_train = PCA.fit_transform(sub_features)
            self.num_PCA_components = l
            self.num_ig_features = l

            # use values from training data to create X
            X_test = PCA.transform(test_sub_features)

            y_train = sub_tracks['track', 'genre_top']
            y_train = skl.preprocessing.LabelEncoder().fit_transform(y_train)
            
            y_test = test_sub_tracks['track', 'genre_top']
            y_test = skl.preprocessing.LabelEncoder().fit_transform(y_test)
        else:
            X_train = sub_features.as_matrix() # convert features to input matrix
            y_train = self.__output_classes_from_string_labels(sub_tracks['track', 'genre_top']) # create 1v1 output categorization

            X_test = test_sub_features.as_matrix() # convert features to input matrix
            y_test = self.__output_classes_from_string_labels(test_sub_tracks['track', 'genre_top']) # create 1v1 output categorization
            self.num_PCA_components = None

        return X_train, y_train, X_test, y_test

    def create_Viz_Data(self, genre1 = None, genre2 = None):
        if (genre1 is not None) and (genre2 is not None):
            self.genre1 = genre1
            self.genre2 = genre2

        genreTracks, genreFeatures = self.getSubTracksAndFeatures(self.tracks, 'subset', self.subset, self.libFeatures, self.echoFeatures)

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

    def create_info_gain_subset(self, X_train, y_train, X_test, y_test, num_feat=None) :
        """
        Take in a training and test set and returns versions of those sets
        that only include the num_feat most information-gaining features.
        :param X_train: The full matrix of training examples
        :param y_train: The labels for those training examples
        :param X_test: The full matrix of text examples
        :param y_test: The labels for the test examples
        :param num_feat: The number of features in the outputted training and test sets 
        :return Training and test sets and their labels, now with num_feat features
        """
        _, d = X_train.shape

        # Use number of features from PCA
        if num_feat is None:

            # Calculate number of PCA components if we haven't done it yet 
            if self.num_ig_features is None:
                self.create_X_y_split(self.genre1, self.genre2, usePCA=True)
            num_feat = self.num_ig_features

        if num_feat > d:
            print("A subset of features was not created due to input num_feat param.\n Full feature set used.")
            return X_train, y_train, X_test, y_test

        info_gains = skl.feature_selection.mutual_info_classif(X_train, y_train)

        ranking = np.argsort(info_gains)[-num_feat::]

        X_sub_train = np.take(X_train, ranking, axis=1)
        X_sub_test = np.take(X_test, ranking, axis=1)

        return X_sub_train, y_train, X_sub_test, y_test


####################################
###                              ###
### Helper Classes and Functions ###
###                              ###
####################################


class LRUCache(object):
    """
    Simple LRU Cache implementation taken from https://www.kunxi.org/blog/2014/05/lru-cache-in-python/
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def get(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

def cache_hash(list_of_words):
    return "-".join(list_of_words)







