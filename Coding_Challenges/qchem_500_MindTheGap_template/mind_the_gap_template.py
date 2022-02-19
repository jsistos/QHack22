import sys
import pennylane as qml
from pennylane import numpy as np
from pennylane import hf


def ground_state_VQE(H):
    """Perform VQE to find the ground state of the H2 Hamiltonian.
    Args:
        - H (qml.Hamiltonian): The Hydrogen (H2) Hamiltonian
    Returns:
        - (float): The ground state energy
        - (np.ndarray): The ground state calculated through your optimization routine
    """

    # QHACK #
    n_qubits=len(H.wires)
    dev=qml.device("default.qubit",wires=n_qubits)
    b=[]
    b.append(1)
    b.append(1)
    
    for i in range (n_qubits-2):
        b.append(0)
    
    hf_state=np.array(b)
    
    def circuit(param, wires):
        qml.BasisState(hf_state, wires=wires)
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    
    
    @qml.qnode(dev)
    def cost_fn(param):
        circuit(param, wires=range(n_qubits))
        return qml.expval(H)
    
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)
    energy = [cost_fn(theta)]

    angle = [theta]

    max_iterations = 100
    conv_tol = 1e-06

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)

        energy.append(cost_fn(theta))
        angle.append(theta)

        conv = np.abs(energy[-1] - prev_energy)

        if conv <= conv_tol:
            break

    p=angle[-1]
    
    dev1=qml.device("default.qubit",wires=n_qubits)
    @qml.qnode(dev1)
    def cost_fn1(param):
        circuit(param, wires=range(n_qubits))
        return qml.state()
    q=cost_fn1(p).real

    return energy[-1], q
    


def create_H1(ground_state, beta, H):
    
    """Create the H1 matrix, then use `qml.Hermitian(matrix)` to return an observable-form of H1.
    Args:
        - ground_state (np.ndarray): from the ground state VQE calculation
        - beta (float): the prefactor for the ground state projector term
        - H (qml.Hamiltonian): the result of hf.generate_hamiltonian(mol)()
    Returns:
        - (qml.Observable): The result of qml.Hermitian(H1_matrix)
    """
    for i in range (len(ground_state)):
        ground_state[i]=ground_state[i] * 15
    
    q=np.kron(ground_state,ground_state).reshape(16,16)
    
    n_qubits=len(H.wires)
    Hmat = qml.utils.sparse_hamiltonian(H).real
    p=Hmat.toarray()
    
    # QHACK #
    h1_matrix =q+p
    return qml.Hermitian(h1_matrix,wires=range(n_qubits))
    

    # QHACK #


def excited_state_VQE(H1):
    """Perform VQE using the "excited state" Hamiltonian.
    Args:
        - H1 (qml.Observable): result of create_H1
    Returns:
        - (float): The excited state energy
    """

    # QHACK #
    n_qubits=len(H.wires)
    dev=qml.device("default.qubit",wires=n_qubits)
    b=[]
    b.append(1)
    b.append(0)
    b.append(1)
    
    for i in range (n_qubits-3):
        b.append(0)
    
    hf_state=np.array(b)
    
    def circuit(param, wires):
        qml.BasisState(hf_state, wires=wires)
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    
    
    @qml.qnode(dev)
    def cost_fn(param):
        circuit(param, wires=range(n_qubits))
        return qml.expval(H1)
    
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)
    energy = [cost_fn(theta)]

    angle = [theta]

    max_iterations = 100
    conv_tol = 1e-06
    min_loss=100

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)

        loss=cost_fn(theta)
        angle.append(theta)

        conv = np.abs(loss - prev_energy)

        if loss < min_loss:
            best_params_0 = theta
            min_loss = loss

        if conv <= conv_tol:
            break

            

    return cost_fn(theta)


if __name__ == "__main__":
    coord = float(sys.stdin.read())
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, -coord], [0.0, 0.0, coord]], requires_grad=False)
    mol = hf.Molecule(symbols, geometry)

    H = hf.generate_hamiltonian(mol)()
    E0, ground_state = ground_state_VQE(H)

    beta = 15.0
    H1 = create_H1(ground_state, beta, H)
    E1 = excited_state_VQE(H1)

    answer = [np.real(E0), E1]
    print(*answer, sep=",")
