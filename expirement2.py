import os
import platform
import numpy as np
import psutil
import subprocess
from time import time
from logistic_regression import LogisticRegressionEP34
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
    # Step 1: Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state_offset*i)
    
    # Step 2: Normalize features to [0, 1] range
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 3: Initialize and train logistic regression model
    model = LogisticRegressionEP34()
    model.fit(X_train, y_train, iterations=10000, batch_size=64)

    # Step 4: Evaluate the model
    predictions = model.predict(X_test) >= 0.5  # Binary classification threshold
    accuracy = np.mean(predictions == y_test)
    accuracies.append(accuracy)

execution_time = time() - start_time

# Calculate mean accuracy and standard deviation
mean_accuracy = np.mean(accuracies) * 100
std_dev_accuracy = np.std(accuracies) * 100

# Print results
print("\nExperiment 2: Breast Cancer Dataset")
print(f"\tMean Test Accuracy over {num_iterations} runs: {mean_accuracy:.2f}%")
print(f"\tStandard Deviation of Accuracy: {std_dev_accuracy:.2f}%")
print(f"\tTotal Execution Time: {execution_time:.2f} seconds")

cpu_info = subprocess.check_output("lscpu", shell=True).strip().decode()
cpu_model = [cpu_str for cpu_str in cpu_info.split('\n') if 'Model name:' in cpu_str]
if len(cpu_model) > 0:
    cpu_model = cpu_model[0].split(':')[1].strip()
else:
    cpu_model = "Unknown"

# Print out the cpu and available memory
print("\nSystem Information:")
print(f"\tSystem: {platform.system()} {platform.release()}")
print(f"\tCPU Model: {platform.processor()} ({cpu_model})")
print(f"\tCPU Cores: {os.cpu_count()}")
print(f"\tTotal Memory: {psutil.virtual_memory().total / 1024**3:.2f} GB")
print(f"\tAvailable Memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
