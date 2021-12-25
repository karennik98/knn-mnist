import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

def knn(X_train, y_train, test_point, k=5):
    distances = []  # Contains list of tuples (distance, label

    for data_point, label in zip(X_train, y_train):
        distances.append((euclidean(test_point, data_point), label))
    #     for i in range(X_train.shape[0]):
    #         data_point = X_train[i]
    #         label = y_train[i]

    sorted_distances = sorted(distances, key=lambda x: x[0])
    k_nearest_neighbors = np.array(sorted_distances[:k])
    freq = np.unique(k_nearest_neighbors[:, 1], return_counts=True)
    labels, counts = freq
    majority_vote = labels[counts.argmax()]
    return majority_vote

def euclidean(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

def calculate_accuracy(X_test, y_test, X_train, y_train, k=5):
    predictions = []

    for test_point in X_test:
        pred_label = knn(X_train, y_train, test_point, k)
        predictions.append(pred_label)

    predictions = np.array(predictions)

    accuracy = (predictions == y_test).sum() / y_test.shape[0]
    return accuracy

dataset = pd.read_csv("./train_data.csv")
dataset.head()

# Plotting images
a = np.random.random((10,10))
plt.figure()
plt.imshow(a, cmap='gray')
plt.show()

data = dataset.values[:5000]
data.shape

X, y = data[:,1:], data[:, 0]

# Plotting digit
im = X[4997].reshape((28,28))
plt.figure()
plt.imshow(im, cmap='gray')
plt.show()

split = int(X.shape[0] * 0.80)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(y_train)
print(y_test)

euclidean(np.array([1,2,3]), np.array([4,5,6]))

calculate_accuracy(X_test, y_test, X_train, y_train, k=5)

test_df = pd.read_csv("./test_data.csv")
test_df.head()

test_data = test_df.values
test_images = test_data[:10]
test_images.shape

for test in test_images:
    im = test.reshape((28, 28))
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.show()
    print("Label:", knn(X_train, y_train, test))

