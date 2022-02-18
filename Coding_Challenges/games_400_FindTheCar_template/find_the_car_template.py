#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


dev = qml.device("default.qubit", wires=[0, 1, "sol"], shots=1)


def find_the_car(oracle):
    """Function which, given an oracle, returns which door that the car is behind.

    Args:
        - oracle (function): function that will act as an oracle. The first two qubits (0,1)
        will refer to the door and the third ("sol") to the answer.

    Returns:
        - (int): 0, 1, 2, or 3. The door that the car is behind.
    """

    @qml.qnode(dev)
    def circuit1():
        # QHACK #
        #Prepare |0+> on first two qubits
        qml.Hadamard(1)

        #Prepare |-> on third qubit
        qml.PauliX('sol')
        qml.Hadamard('sol')

        #Apply oracle
        oracle()

        #Undoing superposition in last two qubits
        qml.Hadamard(1)
        qml.Hadamard('sol')
        # QHACK #
        return qml.sample()

    @qml.qnode(dev)
    def circuit2():
        # QHACK #
        #Prepare |+0> on first two qubits
        qml.Hadamard(0)

        #Prepare |-> on third qubit
        qml.PauliX('sol')
        qml.Hadamard('sol')

        #Apply oracle
        oracle()

        #Undoing superposition in two qubits
        qml.Hadamard(0)
        qml.Hadamard('sol')
        # QHACK #
        return qml.sample()

    sol1 = circuit1()
    sol2 = circuit2()

    # QHACK #

    # process sol1 and sol2 to determine which door the car is behind.
    in_00_01 = sol1[1]
    in_00_10 = sol2[0]

    #Is in both (00, 01) and (00, 10)
    if in_00_01 and in_00_10:
        return 0
    #Is in (00, 01) but is not in (00, 10)
    elif in_00_01:
        return 1
    #Is not in (00, 01) but is in (00, 10)
    elif in_00_10:
        return 2
    #Is not in (00, 01) and not in (00, 10)
    else:
        return 3
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    def oracle():
        if numbers[0] == 1:
            qml.PauliX(wires=0)
        if numbers[1] == 1:
            qml.PauliX(wires=1)
        qml.Toffoli(wires=[0, 1, "sol"])
        if numbers[0] == 1:
            qml.PauliX(wires=0)
        if numbers[1] == 1:
            qml.PauliX(wires=1)

    output = find_the_car(oracle)
    print(f"{output}")
