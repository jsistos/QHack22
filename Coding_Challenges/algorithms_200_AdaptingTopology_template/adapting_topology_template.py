#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml

graph = {
    0: [1],
    1: [0, 2, 3, 4],
    2: [1],
    3: [1],
    4: [1, 5, 7, 8],
    5: [4, 6],
    6: [5, 7],
    7: [4, 6],
    8: [4],
}


def n_swaps(cnot):
    """Count the minimum number of swaps needed to create the equivalent CNOT.

    Args:
        - cnot (qml.Operation): A CNOT gate that needs to be implemented on the hardware
        You can find out the wires on which an operator works by asking for the 'wires' attribute: 'cnot.wires'

    Returns:
        - (int): minimum number of swaps
    """

    # QHACK #
    previous = []
    current = []
    nex_t = []
    final_count = -2
    found_node = False
    def update_next(current, nex_t):
        nex_t.clear()
        for i in range(len(current)):
            for j, hw_node in enumerate(graph[current[i]]):
                nex_t.append(hw_node)
    def update_current(previous, current, nex_t, final_count):
        final_count = final_count + 1
        previous.extend(current)
        current.clear()
        for hw_node in nex_t:
            if previous.count(hw_node) == 0 and current.count(hw_node) == 0:
                current.append(hw_node)
        return final_count
    
    current.append(cnot.wires[0])
    while found_node == False:
        update_next(current, nex_t)
        final_count = update_current(previous, current, nex_t, final_count)
        if previous.count(cnot.wires[1]) > 0:
            found_node = True
    return 2 * final_count
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = n_swaps(qml.CNOT(wires=[int(i) for i in inputs]))
    print(f"{output}")
