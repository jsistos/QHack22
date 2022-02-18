import sys
import pennylane as qml
from pennylane import numpy as np


def second_renyi_entropy(rho):
    """Computes the second Renyi entropy of a given density matrix."""
    # DO NOT MODIFY anything in this code block
    rho_diag_2 = np.diagonal(rho) ** 2.0
    return -np.real(np.log(np.sum(rho_diag_2)))


def compute_entanglement(theta):
    """Computes the second Renyi entropy of circuits with and without a tardigrade present.

    Args:
        - theta (float): the angle that defines the state psi_ABT

    Returns:
        - (float): The entanglement entropy of qubit B with no tardigrade
        initially present
        - (float): The entanglement entropy of qubit B where the tardigrade
        was initially present
    """

    dev = qml.device("default.qubit", wires=3)

    # QHACK #

    """
    In Pennylane101 300, we built an entangled state of the form cosA|00> + sinA|11>.
    Using the same technique, we can add a PauliX to the circuit to make |e> = cosA|10> + sinA|01>.

    Finally, we can entangle this with qubit 0 by using a Hadamard gate and controlling
    every step to build |e> on qubit 0 as well, resulting in 0.5|0>|00> + 0.5|1>|e>. The last step is
    a PauliX gate on 0 to make the appropriate state.
    """
    @qml.qnode(dev)
    def tardigrade_entangler():

        #Hadamard gate allows for superposition
        qml.Hadamard(0)

        #Steps to make |e> are
        #RY, X(1), X(2), CNOT(1,2)
        #Each gate is controlled with qubit 0
        qml.CRY(theta, wires=[0,1])
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[0,2])
        qml.Toffoli(wires=[0,1,2])

        #X Gate finishes the desired state
        qml.PauliX(0)

        return qml.density_matrix([1])

    @qml.qnode(dev)
    def bell_entangler():
        qml.Hadamard(0)
        qml.PauliX(1)
        qml.CNOT(wires=[0,1])

        return qml.density_matrix([1])

    return [second_renyi_entropy(bell_entangler()), second_renyi_entropy(tardigrade_entangler())]
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    theta = np.array(sys.stdin.read(), dtype=float)

    S2_without_tardigrade, S2_with_tardigrade = compute_entanglement(theta)
    print(*[S2_without_tardigrade, S2_with_tardigrade], sep=",")