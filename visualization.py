import matplotlib.pyplot as plt
from getFeatures import DataSetGenerator
from fixtures import TOP_GENRES

def genrePCA(genres = TOP_GENRES):
  '''
  pass in a string list of genre names to see all combinations, default is all genres
  '''
  # create DataSetGenerator
  dsg = DataSetGenerator('small', echoFeatureSets=[])
  # go through each genre combination
  for i in range(len(genres)):
    for j in range(i + 1, len(genres)):
      X, y = dsg.create_Viz_Data(genres[i], genres[j])
      plt.scatter(X[:,0], X[:,1], c=y, cmap='RdBu', alpha=0.5)
      plt.title('PCA Comparison of %s vs. %s' % (genres[i], genres[j]))
      plt.show()

