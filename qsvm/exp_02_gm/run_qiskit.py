import json
import time
import os  # <--- Added os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
# from qiskit_aer import AerSimulator  <--- REMOVE THIS LINE (It causes the crash and isn't used)
from data_loader import load_and_process_data

# 1. Setup
X_train, X_test, y_train, y_test, config = load_and_process_data()

# <--- ADD THIS: Ensure directory exists
os.makedirs(config['output_folder'], exist_ok=True)

print(f"--- Qiskit QSVC Start ---")

# 2. Define Feature Map
feature_map = zz_feature_map(
    feature_dimension=config["n_qubits"], 
    reps=config["reps"], 
    entanglement=config["entanglement"]
)

# 3. Create Kernel
# Note: In Qiskit 1.0+, this implicitly uses the StatevectorSampler if no backend is provided.
kernel = FidelityQuantumKernel(feature_map=feature_map)

# 4. Train QSVM
svc = SVC(kernel=kernel.evaluate)
start_time = time.time()
print("Computing Kernel and Fitting...")
svc.fit(X_train, y_train)
train_time = time.time() - start_time

# 5. Evaluate
score = svc.score(X_test, y_test)
print(f"Qiskit Accuracy: {score:.2f}")
print(f"Time Taken: {train_time:.2f}s")

start_time = time.time()

# 6. Visualization
h = 0.2
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

print("Predicting for plot (this may take a moment)...")
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='x', label='Test')
plt.title(f"Qiskit QSVC (Acc: {score:.2f})")
plt.legend()
plt.savefig(f"{config['output_folder']}/qiskit_plot.png")
print("Plot saved.")

train_time = time.time() - start_time
print(f"Time Taken to plot decision boundary: {train_time:.2f}s")
