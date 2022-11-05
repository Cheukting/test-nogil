"""
=======================================================================
Plot the decision surface of decision trees trained on the iris dataset
=======================================================================

Plot the decision surface of a decision tree trained on pairs
of features of the iris dataset.

See :ref:`decision tree <tree>` for more information on the estimator.

For each pair of iris features, the decision tree learns decision
boundaries made of combinations of simple thresholding rules inferred from
the training samples.

We also show the tree structure of a model built on all of the features.
"""

# First load the copy of the Iris dataset shipped with scikit-learn:
from sklearn.datasets import load_iris

import time

iris = load_iris()

import numpy as np

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay


# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

train_time = []

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Train
    t0 = time.time()
    clf = DecisionTreeClassifier().fit(X, y)
    print(f"Trining takes {time.time()-t0}s")
    train_time.append(time.time()-t0)

print(f"On average: {sum(train_time)/len(train_time)*1000}ms")
