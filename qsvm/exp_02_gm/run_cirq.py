import cirq
import numpy as np
import matplotlib.pyplot as plt
import time
import sympy
import os  # <--- Added os
from sklearn.svm import SVC
from data_loader import load_and_process_data

# 1. Setup
X_train, X_test, y_train, y_test, config = load_and_process_data()

# <--- ADD THIS: Ensure directory exists
os.makedirs(config['output_folder'], exist_ok=True)

print(f"--- Cirq QSVC Start ---")


# 2. Build Circuit & Simulator
qubits = [cirq.GridQubit(0, i) for i in range(config["n_qubits"])]
simulator = cirq.Simulator()

def create_ansatz(qubits, data_params, zz_params):
    """
    data_params: Symbols for linear terms RZ(2x)
    zz_params: Symbols for interaction terms RZ(2(pi-x)(pi-y))
    """
    circuit = cirq.Circuit()
    for _ in range(config["reps"]):
        # 1. Hadamards
        circuit.append(cirq.H.on_each(qubits))
        
        # 2. RZ(2x)
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.rz(data_params[i]).on(qubit))
            
        # 3. ZZ Interactions
        for i in range(len(qubits) - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
            circuit.append(cirq.rz(zz_params[i]).on(qubits[i+1]))
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
    return circuit

def cirq_kernel(X_A, X_B):
    n_a = len(X_A)
    n_b = len(X_B)
    
    # Define Symbols
    # We need separate symbols for the linear parts (x) and the ZZ parts (phi)
    # Because Cirq sweep prefers simple symbol-to-float mapping
    sym_names_data = [f'd{i}' for i in range(config["n_qubits"])]
    sym_names_zz = [f'zz{i}' for i in range(config["n_qubits"]-1)]
    
    syms_data = [sympy.Symbol(s) for s in sym_names_data]
    syms_zz = [sympy.Symbol(s) for s in sym_names_zz]
    
    circuit = create_ansatz(qubits, syms_data, syms_zz)
    
    # Helper to convert a raw data point 'x' into the specific angles needed
    # for the ansatz
    def get_resolver_values(x):
        # 1. Linear RZ angle = 2 * x
        linear_vals = {f'd{i}': 2 * x[i] for i in range(len(x))}
        
        # 2. ZZ angle = 2 * (pi - xi)(pi - xj)
        zz_vals = {}
        for i in range(len(x) - 1):
            val = 2 * (np.pi - x[i]) * (np.pi - x[i+1])
            zz_vals[f'zz{i}'] = val
        
        # Merge dicts
        return {**linear_vals, **zz_vals}

    # 1. Pre-calculate states for X_A
    print(f"Calculating states for set A ({n_a} samples)...")
    resolvers_a = [get_resolver_values(x) for x in X_A]
    results_a = simulator.simulate_sweep(circuit, params=resolvers_a)
    states_a = np.array([r.final_state_vector for r in results_a])

    # 2. Pre-calculate states for X_B
    if X_A is X_B:
        states_b = states_a
    else:
        print(f"Calculating states for set B ({n_b} samples)...")
        resolvers_b = [get_resolver_values(x) for x in X_B]
        results_b = simulator.simulate_sweep(circuit, params=resolvers_b)
        states_b = np.array([r.final_state_vector for r in results_b])

    # 3. Compute overlaps
    overlap_matrix = np.abs(np.dot(states_a, states_b.conj().T))**2
    return overlap_matrix

# 3. Train
svc = SVC(kernel=cirq_kernel)
start_time = time.time()
print("Fitting SVC with Cirq Kernel...")
svc.fit(X_train, y_train)
train_time = time.time() - start_time

# 4. Evaluate
score = svc.score(X_test, y_test)
print(f"Cirq Accuracy: {score:.2f}")
print(f"Time Taken: {train_time:.2f}s")


start_time = time.time()

# 5. Plotting
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
plt.title(f"Cirq QSVC (Acc: {score:.2f})")
plt.savefig(f"{config['output_folder']}/cirq_plot.png")
print("Plot saved.")

train_time = time.time() - start_time
print(f"Time Taken to plot decision boundary: {train_time:.2f}s")
