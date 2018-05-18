from sklearn.datasets import fetch_mldata
from sklearn import linear_model
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

mnist = fetch_mldata('MNIST original')

X_train = mnist.data[:60000] / 255.0
Y_train = mnist.target[:60000]

X_test = mnist.data[60000:] / 255.0
Y_test = mnist.target[60000:]

Y_train[Y_train > 1.0] = 0.0
Y_test[Y_test > 1.0] = 0.0

clf = linear_model.LogisticRegression()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_pred))

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average="weighted")
recall = recall_score(Y_test, Y_pred, average="weighted")

print(accuracy, precision, recall)
