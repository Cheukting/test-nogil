"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

import time

print("==================")
print("Experiment: plot_digits_classification.py\n")

digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Learn the digits on the train subset
time_list = []
for _ in range(50):
    clf = svm.SVC(gamma=0.001)
    t0 = time.time()
    clf.fit(X_train, y_train)
    time_list.append(time.time()-t0)
print(f"Fitting takes {sum(time_list)/len(time_list)}s on average")
print("==================\n")
