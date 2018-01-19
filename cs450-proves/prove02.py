# PART 1
import inline as inline
import matplotlib
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print("Printing Data (Called Features)")
print(iris.data)

print("\nName of the features")
print(iris.feature_names)

# Show the target values (in numeric format) of each instance
print("\nPrinting Targets")
print(iris.target)

# Show the actual target names that correspond to each number
print("\nPrinting TargetNames")
print(iris.target_names)

# PART 2
Xdata = iris.data
ytarget = iris.target

# Using train_test_split (30% test, 70% train)
data_train, data_test, targets_train, targets_test = train_test_split(Xdata, ytarget, test_size=0.3)
print(data_train.shape, targets_train.shape)
print(data_test.shape, targets_test.shape)

# PART 3
classifier = GaussianNB()
model = classifier.fit(data_train, targets_train)

# PART 4
targets_predicted = classifier.predict(data_test)
print("\nPrinting GaussianNB Classifier")
print(targets_predicted)
print(targets_test)
comparison = metrics.accuracy_score(targets_test, targets_predicted)
print("{:.3f}%".format(comparison * 100))


# PART 5
class HardCodedModel:
    def __init__(self):
        pass

    def predict(self, data):
        targets = []

        for row in data:
            targets.append(self.predict_one(row))

        return targets


class HardCodeClassifier:
    # Professor's Code
    def __init__(self):
        pass

    # Professor's Code
    def fit(self, data, targets):
        return HardCodedModel();

    # MINE CODE
    # def fit(self, data, target):
    #     hc = HardCodeClassifier()
    #     return hc
    #
    # def predict(self, data_test):
    #     array = np.array([])
    #     for item in data_test:
    #         array = np.append(array,0)
    #     return array


# PROFESSOR CODE
classifierMine = HardCodeClassifier()
model = classifier.fit(data_train, targets_train)
targets_predicted = model.predict(data_test)

# Compute the amount we got correct
correct = (targets_test == targets_predicted).sum()
total = len(targets_test)
percent = correct / total * 100

# Display result
print("Correct: {}/{} or {:.2f}%".format(correct, total, percent))

# Send the model back
#return model

# MINE CODE
# modelMine = classifierMine.fit(data_train, targets_train)
# targets_predictedMine = modelMine.predict(data_test)
# print("\nPrinting Mine HardCodeClassifier")
# print(targets_predictedMine)
# print(targets_test)
# comparison = metrics.accuracy_score(targets_test, targets_predictedMine)
# print("{:.3f}%".format(comparison * 100))



# Implementing KNeighbors
k_range = range(1,50)
scores = []
for k in k_range:
    classifier = KNeighborsClassifier(n_neighbors=k)
    model = classifier.fit(data_train, targets_train)
    predictions = model.predict(data_test)
    scores.append(metrics.accuracy_score(targets_test, predictions))
    print();
    print("Printing KNeighbors K= {} ".format(k))
    comparison = metrics.accuracy_score(targets_test, predictions)
    print("{:.3f}%".format(comparison * 100))

# Generating graphs to compare the best K number to use
plt.plot(k_range, scores)
plt.xlabel('Value of K for KN')
plt.ylabel('Testing Accuracy')
plt.show()


