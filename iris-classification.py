from sklearn import datasets
from collections import Counter
import numpy as np

#Setting up data
iris = datasets.load_iris()
x = iris.data
y = iris.target

prediction_label = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

#Used to calculate accuracy
def cross_Validation_NN(data, kN):
  predictions = []
  for i in range(len(data)):
    target = data[i]
    train = data
    predictions.append(nearest_Neighbors(target, train, kN))
  return predictions

#K Nearest Neighbors Algorithm
def nearest_Neighbors(target, data, kN):
  bestDist = []
  for i in range(len(data)):
    if np.all(data == target):
      bestDist.append(float('inf'))
      continue
    bestDist.append(np.linalg.norm(data[i] - target))
  bestDist = np.array(bestDist)
  bestDist = bestDist.argsort()[:kN]
  prediction = Counter(y[bestDist]).most_common(1)[0][0]
  return prediction


def main(sL, sW, pL, pW, kN):
    pred = cross_Validation_NN(x, kN)
    result = 100 * (np.sum(pred == y) / len(y))
    userData = [sL, sW, pL, pW]
    print('------------------------------')
    print('[SAMPLE] :', userData)
    print("[NUMBER OF NEIGHBORS] :", kN)
    print("[ACCURACY] : %.2f" % result, '%')
    print('[PREDICTION] :', prediction_label[nearest_Neighbors(userData, x, kN)])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-sl", "--sepal_length", help="Enter Length of Sepal", type=float)
    parser.add_argument("-sw", "--sepal_width", help="Enter Width of Sepal", type=float)
    parser.add_argument("-pl", "--petal_length", help="Enter Length of Petal", type=float)
    parser.add_argument("-pw", "--petal_width", help="Enter Width of Petal", type=float)
    parser.add_argument("-n", "--n_neighbors", help="Enter N Neighbors", type=int)

    args = parser.parse_args()
    main(args.sepal_length, args.sepal_width, args.petal_length, args.petal_width, args.n_neighbors)


