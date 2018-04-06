import numpy as np
import matplotlib.pyplot as plt

def create_multi_line_graph(x, ys, title="", x_label="", y_label="", legend=None):
	"""
	Create a line graph with multiple curves, axis labels, and an appropriate legend.
	:param [float] x
	:param [[float]] ys: Outputs/y-values for each of the sets that will be plotted
	:param str title: Title of graph
	:param str x_label: X axis label
	:param str y_label: y-axis label
	:param [str] legend: legend labels, in same order of outputs 
	"""
	for y in ys: 
		plt.plot(x, y)

	if legend is not None :
		plt.legend(legend, loc=0)

	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()