from data.dataset import load_data_mat, random_split_train_val
from classifier.softmax import Softmax

import numpy as np

random_seed_train = np.random.randint(0, 100)
random_seed_test = np.random.randint(0, 100)
X_train, y_train = load_data_mat("data/train_32x32.mat", seed=random_seed_train, max_samples=20000)

X_test, y_test = load_data_mat("data/test_32x32.mat", seed=random_seed_test, max_samples=2000)

classifier = Softmax(learning_rate=3, regul_lambda=10 ** 9, batch_size=500)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

def accuracy_score(true_values, predictions):
    return sum([int(true_values[i] == predictions[i]) for i in range(len(true_values))]) / len(predictions)

acc = accuracy_score(y_test, predictions)
print(f"Gained accuracy: {acc}")