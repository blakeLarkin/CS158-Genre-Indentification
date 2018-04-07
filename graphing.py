import numpy as np
import matplotlib.pyplot as plt
from crossValidation import gen_depth_vs_accuracy

def create_multi_line_graph(x_min, x_max, step, ys, title="", x_label="", y_label="", legend=None):
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

	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()

def gen_depth_vs_acc_plot(genre_prs, min_depth=2, max_depth=5, step=1, feature_sets=None, data_dir=""):

	scores = []
	legend = []
	for pair in genre_prs:

		scores.append(gen_depth_vs_accuracy(genre1=pair[0], genre2=pair[1], max_depth_min=min_depth, max_depth_max=max_depth, step=step, feature_sets=feature_sets, data_dir="" ))
		legend.append(pair[0] + " and " +  pair[1] )

	print(scores)
	print(scores[0][0])

	x_label="Maximum Decision Tree Depths"
	y_label = "Accuracy"
	title="Binary Genre Classification: Decision Trees"
	create_multi_line_graph(min_depth, max_depth, step, scores, title=title, x_label=x_label, y_label = y_label, legend=legend )



# data_dir = ../fma_metadata