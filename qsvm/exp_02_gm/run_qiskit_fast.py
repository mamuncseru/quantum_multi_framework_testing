import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from data_loader import load_and_process_data

# 1. Setup
X_train, X_test, y_train, y_test, config = load_and_process_data()
os.makedirs(config['output_folder'], exist_ok=True)

print(f"--- Qiskit Fast Mode (Statevector) ---")

# 2. Define Feature Map
# We use the standard ZZFeatureMap
feature_map = zz_feature_map(
    feature_dimension=config["n_qubits"], 
    reps=config["reps"], 
    entanglement=config["entanglement"]
)

# 3. The Fast Kernel Function
def statevector_kernel(X_A, X_B):
    # Step A: Transform all data points into Statevectors
    # This is the only "Quantum" part of the calculation
    print(f"Generating statevectors for {len(X_A)} samples...")
    
    # assign_parameters creates a bound circuit for each x
    # Statevector(circ).data gives the raw complex numpy array
    states_A = np.array([
        Statevector(feature_map.assign_parameters(x)).data 
        for x in X_A
    ])
    
    if X_A is X_B:
        states_B = states_A
    else:
        print(f"Generating statevectors for {len(X_B)} samples...")
        states_B = np.array([
            Statevector(feature_map.assign_parameters(x)).data 
            for x in X_B
        ])
        
    # Step B: Compute Overlaps using Linear Algebra
    # Matrix A shape: (N_samples_A, 2^N_qubits)
    # Matrix B shape: (N_samples_B, 2^N_qubits)
    # We want the dot product of every row in A with every row in B
    print("Computing Kernel Matrix via Linear Algebra...")
    
    # Conjugate Transpose of B
    # Result = | A . B^H |^2
    kernel_matrix = np.abs(np.dot(states_A, states_B.conj().T))**2
    
    return kernel_matrix

# 4. Train
# Pass the custom kernel function to SVC
svc = SVC(kernel=statevector_kernel)

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
