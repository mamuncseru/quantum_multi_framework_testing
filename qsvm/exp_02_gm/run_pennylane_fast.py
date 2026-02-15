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

print(f"--- PennyLane Fast Mode (Statevector) ---")

# 2. Define Device
# 'default.qubit' is a statevector simulator perfect for this
dev = qml.device("default.qubit", wires=config["n_qubits"])

# 3. Define Manual ZZ Feature Map (Matches Qiskit)
def zz_feature_map(x, wires):
    n_qubits = len(wires)
    for _ in range(config["reps"]):
        # Hadamards
        for i in range(n_qubits):
            qml.Hadamard(wires=wires[i])
        # RZ(2x)
        for i in range(n_qubits):
            qml.RZ(2 * x[i], wires=wires[i])
        # ZZ Interaction
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])
            phi = 2 * (np.pi - x[i]) * (np.pi - x[i+1])
            qml.RZ(phi, wires=wires[i+1])
            qml.CNOT(wires=[wires[i], wires[i+1]])

# 4. Define QNode returning STATE
@qml.qnode(dev)
def get_state_circuit(x):
    zz_feature_map(x, wires=range(config["n_qubits"]))
    # Return the full complex state vector
    return qml.state()

# 5. The Fast Kernel Function
def statevector_kernel(X_A, X_B):
    print(f"Generating statevectors for {len(X_A)} samples...")
    # Convert input to float to avoid PennyLane casting issues
    X_A = np.array(X_A, dtype=float)
    
    # Get all states for A
    # We use a simple loop, but it's fast because we only run N circuits (not N^2)
    states_A = np.array([get_state_circuit(x) for x in X_A])
    
    if X_A is X_B:
        states_B = states_A
    else:
        print(f"Generating statevectors for {len(X_B)} samples...")
        X_B = np.array(X_B, dtype=float)
        states_B = np.array([get_state_circuit(x) for x in X_B])
    
    print("Computing Kernel Matrix via Linear Algebra...")
    # Compute |<psi_a | psi_b>|^2 for all pairs at once
    # Matrix multiplication: (N_a, 2^n) @ (2^n, N_b) -> (N_a, N_b)
    kernel_matrix = np.abs(np.dot(states_A, states_B.conj().T))**2
    
    return kernel_matrix

# 6. Train
svc = SVC(kernel=statevector_kernel)

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
