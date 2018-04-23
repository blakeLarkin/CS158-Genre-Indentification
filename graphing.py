import numpy as np
import matplotlib.pyplot as plt
from crossValidation import gen_depth_vs_accuracy
from crossValidation import performance_CI
from sklearn.dummy import DummyClassifier
from getFeatures import DataSetGenerator 
from fixtures import TOP_GENRES
from utils import calcMedoids

def add_dsg(dsg):
    def decorator(function):
        def wrapper(*args, **kwargs):
            function(*args, **kwargs)

        return wrapper
    return decorator


def create_multi_line_graph(x_min, x_max, step, ys, title="", x_label="", y_label="", legend=None, **kwargs):
    """
    Create a line graph with multiple curves, axis labels, and an appropriate legend.
    :param [float] x
    :param [[float]] ys: Outputs/y-values for each of the sets that will be plotted
    :param str title: Title of graph
    :param str x_label: X axis label
    :param str y_label: y-axis label
    :param [str] legend: legend labels, in same order of outputs 
    """
    x = np.arange(x_min, x_max+1, step, dtype=np.float64)
    for y in ys: 
        plt.plot(x, y)

    if legend is not None :
        plt.legend(legend, loc=0)

    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_CI_performance(metrics, classifiers, *args):
    """
    Make a results plot.

    Parameters
    --------------------
        metrics      -- list of strings, metrics
        classifiers  -- list of strings, classifiers
        args         -- variable length argument
                          results for baseline
                          results for classifier 1
                          results for classifier 2
                          ...
                        each results is a tuple (score, lower, upper)
    """

    num_metrics = len(metrics)
    num_classifiers = len(args) - 1

    ind = np.arange(num_metrics)  # the x locations for the groups
    width = 0.7 / num_classifiers # the width of the bars

    fig, ax = plt.subplots()

    # loop through classifiers
    rects_list = []
    for i in range(num_classifiers):
        results = args[i+1] # skip baseline
        means = [it[0] for it in results]
        errs = [(it[0] - it[1], it[2] - it[0]) for it in results]
        rects = ax.bar(ind + i * width, means, width, label=classifiers[i])
        ax.errorbar(ind + i * width, means, yerr=np.array(errs).T, fmt="None", ecolor='k')
        rects_list.append(rects)

    # baseline
    results = args[0]
    for i in range(num_metrics) :
        mean = results[i][0]
        err_low = results[i][1]
        err_high = results[i][2]
        xlim = (ind[i] - 0.8 * width, ind[i] + num_classifiers * width - 0.2 * width)
        plt.plot(xlim, [mean, mean], color='k', linestyle='-', linewidth=2)
        plt.plot(xlim, [err_low, err_low], color='k', linestyle='--', linewidth=2)
        plt.plot(xlim, [err_high, err_high], color='k', linestyle='--', linewidth=2)

    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_xticks(ind + width / num_classifiers)
    ax.set_xticklabels(metrics)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar displaying its height"""
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%.3f' % height, ha='center', va='bottom')

    for rects in rects_list:
        autolabel(rects)

    plt.show()


def generate_confidence_interval_graph(dsg,classifiers_str, classifiers, genre1, genre2):
    X_train, y_train, X_test, y_test = dsg.create_X_y_split(genre1, genre2)

    results = {}
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]

    # Train classifiers
    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)

    for classifier in classifiers:
        classifier.fit(X_train, y_train)


    # Predict and calculate confidence intervals
    dummy_results = []

    for metric in metric_list:
        # Calculate dummy performance
        dummy_results.append(performance_CI(dummy, X_test, y_test, metric=metric))

        # Calculate performance of other classifiers
        for index, classifier_name in enumerate(classifiers_str):
            result = performance_CI(classifiers[index], X_test, y_test, metric=metric)
            if classifier_name not in results:
                results[classifier_name] = []
            results[classifier_name].append(result)

    # plot other results
    plot_CI_performance(metric_list, classifiers_str, dummy_results, *[results[classifier] for classifier in classifiers_str])


def gen_depth_vs_acc_plot(dsg, genre_prs, min_depth=2, max_depth=5, step=1):

    train_scores = []
    test_scores = []

    train_legend = []
    test_legend = []

    #dsg = DataSetGenerator('small', libFeatureSets=feature_sets, data_dir=data_dir)

    for pair in genre_prs:
        train_score, test_score = gen_depth_vs_accuracy(list(dsg.create_X_y_split(pair[0], pair[1])), max_depth_min=min_depth, max_depth_max=max_depth, step=step)
        train_scores.append(train_score)
        test_scores.append(test_score)

        pair_string = pair[0] + " and " +  pair[1]
        train_legend.append(pair_string + ": Training" )
        test_legend.append(pair_string + ": Test")


    x_label="Maximum Decision Tree Depths"
    y_label = "Accuracy"
    title="Binary Genre Classification: Decision Trees"
    create_multi_line_graph(min_depth, max_depth, step, train_scores+test_scores, title=title, x_label=x_label, y_label = y_label, legend=train_legend+test_legend, ylim=(0,1))


def metric_vs_hyperparameter_plot(title, x_label, y_label, genre_prs, dsg, score_args, score_kwargs):
    train_scores = []
    test_scores = []

    train_legend = []
    test_legend = []

    sorted_x = sorted(score_args[2])
    x_min = sorted_x[0]
    x_max = sorted_x[-1]

    for pair in genre_prs:
        train_score, test_score = metric_vs_hyperparameters(dsg.create_X_y_split(pair[0], pair[1]), *score_args, **score_kwargs)
        train_scores.append(train_score)
        test_scores.append(test_score)

        pair_string = pair[0] + " and " +  pair[1]
        train_legend.append(pair_string + ": Training" )
        test_legend.append(pair_string + ": Test")

    create_multi_line_graph(x_min, x_max, 1, x_label, y_label, train_scores+test_scores, )


def genrePCA(dsg, genres = TOP_GENRES):
  '''
  pass in a string list of genre names to see all combinations, default is all genres
  '''
  # create DataSetGenerator
  #dsg = DataSetGenerator('small', echoFeatureSets=[])
  # go through each genre combination
  for i in range(len(genres)):
    for j in range(i + 1, len(genres)):
      X, y = dsg.create_Viz_Data(genres[i], genres[j])
      plt.scatter(X[:,0], X[:,1], c=y, cmap='RdBu', alpha=0.5)
      plt.title('PCA Comparison of %s vs. %s' % (genres[i], genres[j]))
      plt.show()


def allGenrePCA(dsg):
    # get 2 component version of examples
    X, y = dsg.create_X_y(usePCA=True, l=2, allGenres=True)

    # find mediods
    medoidsX, medoidsy = calcMedoids(X, y)
    # break up dimensions to plot
    medoidPCA1 = [medoid[0] for medoid in medoidsX]
    medoidPCA2 = [medoid[1] for medoid in medoidsX]

    plt.scatter(X[:,0], X[:,1], c=y, cmap='Set1', alpha=0.1)
    plt.scatter(medoidPCA1, medoidPCA2, marker='P', c=medoidsy, cmap='Set1', alpha=1, edgecolors='k')
    plt.title('PCA Comparison of All Eight Genres')

    # TODO: figure out legend

    plt.show()
# data_dir = ../fma_metadata
    
