import numpy as np
from time import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Load Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Set up parameters
num_iterations = 20
accuracies = []

# Measure total execution time
start_time = time()
random_state_offset = np.random.randint(0, 1000)
for i in range(num_iterations):
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state_offset*i)

    # Normalize features to [0, 1] range
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the Scikit-Learn logistic regression model
    model = LogisticRegression()  # Using default parameters
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    accuracies.append(accuracy)

execution_time = time() - start_time

# Calculate mean accuracy and standard deviation
mean_accuracy = np.mean(accuracies) * 100
std_dev_accuracy = np.std(accuracies) * 100

# Print results
print(f"Mean Test Accuracy over {num_iterations} runs: {mean_accuracy:.2f}%")
print(f"Standard Deviation of Accuracy: {std_dev_accuracy:.2f}%")
print(f"Total Execution Time: {execution_time:.2f} seconds")
