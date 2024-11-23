import numpy as np
from generate_dataset import generate_binary_problem
from logistic_regression import LogisticRegressionEP34
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
N = 1000
C = np.array([[0, 8], [0, 8]])  # Centers for two classes
X, y = generate_binary_problem(C, N)

# Split dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=102)

# Initialize and train the logistic regression model
model = LogisticRegressionEP34()
model.fit(X_train, y_train, iterations=10000, batch_size=None, show_step=1000, show_line=True)

# Evaluate the model
predictions = model.predict(X_test) >= 0.5  # Threshold for binary classification
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")