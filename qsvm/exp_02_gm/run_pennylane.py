import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.svm import SVC
from data_loader import load_and_process_data

# 1. Setup
X_train, X_test, y_train, y_test, config = load_and_process_data()
os.makedirs(config['output_folder'], exist_ok=True)

print(f"--- PennyLane QSVC Start ---")

# 2. Define Device
dev = qml.device("default.qubit", wires=config["n_qubits"])

# 3. Define the Raw Quantum Circuit
# This returns the full probability distribution (Vector)
def zz_feature_map(x, wires):
    n_qubits = len(wires)
    # Loop for repetitions
    for _ in range(config["reps"]):
        # 1. Hadamards
        for i in range(n_qubits):
            qml.Hadamard(wires=wires[i])
        
        # 2. RZ(2x) Data Encoding
        for i in range(n_qubits):
            qml.RZ(2 * x[i], wires=wires[i])
            
        # 3. ZZ Interaction: exp(i * (pi-xi)(pi-xj) * Z_i Z_j)
        # For linear entanglement (0-1, 1-2, etc.)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])
            # Calculate phase: 2 * (pi - xi) * (pi - xj)
            phi = 2 * (np.pi - x[i]) * (np.pi - x[i+1])
            qml.RZ(phi, wires=wires[i+1])
            qml.CNOT(wires=[wires[i], wires[i+1]])

@qml.qnode(dev)
def raw_kernel_circuit(x1, x2):
    # Apply feature map for x1
    zz_feature_map(x1, wires=range(config["n_qubits"]))
    
    # Apply inverse feature map for x2
    qml.adjoint(zz_feature_map)(x2, wires=range(config["n_qubits"]))
    
    return qml.probs(wires=range(config["n_qubits"]))

# 4. Define the Scalar Helper
# We need a function that returns ONLY the first element (the overlap)
def scalar_kernel_circuit(x1, x2):
    # [0] extracts the probability of the all-zero state |00..0>
    return raw_kernel_circuit(x1, x2)[0]

# 5. Define the Matrix Builder
# This loops over all pairs in A and B to build the Gram matrix
def kernel_matrix_function(A, B):
    # We pass the scalar helper to kernel_matrix
    return np.array(qml.kernels.kernel_matrix(A, B, scalar_kernel_circuit))

# 6. Train
# Pass the matrix builder to SVC
svc = SVC(kernel=kernel_matrix_function)

start_time = time.time()
print("Computing Kernel and Fitting...")
svc.fit(X_train, y_train)
train_time = time.time() - start_time

# 7. Evaluate
score = svc.score(X_test, y_test)
print(f"PennyLane Accuracy: {score:.2f}")
print(f"Time Taken: {train_time:.2f}s")

start_time = time.time()

# 8. Plotting
h = 0.2
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

print("Predicting for plot...")
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='x', label='Test')
plt.legend()
plt.title(f"PennyLane QSVC (Acc: {score:.2f})")
plt.savefig(f"{config['output_folder']}/pennylane_plot.png")
print("Plot saved.")

train_time = time.time() - start_time
print(f"Time Taken to plot decision boundary: {train_time:.2f}s")
