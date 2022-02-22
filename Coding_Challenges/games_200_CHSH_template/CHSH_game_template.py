#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #

    init_state = [alpha, 0, 0, beta]
    qml.AmplitudeEmbedding(init_state, wires=[0,1], normalize=True)
    # QHACK #

@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives x=0
        - theta_B1 (float): angle that Bob chooses when he receives x=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    # QHACK #
    def rotate_basis(angle, wire):
        qml.RY(-2*angle, wires=wire)

    if x == 0:
        rotate_basis(theta_A0, 0)
    else:
        rotate_basis(theta_A1, 0)

    if y == 0:
        rotate_basis(theta_B0, 1)
    else:
        rotate_basis(theta_B1, 1)

    # QHACK #

    return qml.probs(wires=[0, 1])
    

def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """

    # QHACK #
    # x = 0, y = 0 is won trough 00 or 11
    probs = chsh_circuit(params[0], params[1], params[2], params[3], 0 , 0, alpha, beta)
    win_00 = probs[0] + probs[3]

    # x = 0, y = 1 is won either through 00 or 11
    probs = chsh_circuit(params[0], params[1], params[2], params[3], 0 , 1, alpha, beta)
    win_01 = probs[0] + probs[3]

    # x = 0, y = 1 is won either through 00 or 11
    probs = chsh_circuit(params[0], params[1], params[2], params[3], 1 , 0, alpha, beta)
    win_10 = probs[0] + probs[3]

    # x = 1, y = 1, is won either through 01 0r 10
    probs = chsh_circuit(params[0], params[1], params[2], params[3], 1 , 1, alpha, beta)
    win_11 = probs[1] + probs[2]

    #Each case has 1/4 probability of being chose.
    prob = (win_00 + win_01 + win_10 + win_11)/4.
    return prob
    # QHACK #
    

def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """

    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""
        return (1 - winning_prob(params, alpha, beta))**2

    # QHACK #

    #Initialize parameters, choose an optimization method and number of steps
    pi = np.pi
    #init_params = np.array([pi/2, pi/2, pi/2, pi/2], requires_grad = True)
    opt = qml.GradientDescentOptimizer(0.3)
    steps = 250

    # QHACK #
    
    # set the initial parameter values
    params = np.array([pi/4, -pi/4, pi/4, 0], requires_grad = True)

    for i in range(steps):
        # update the circuit parameters 
        # QHACK #
        params = opt.step(cost, params)

        # QHACK #

    return winning_prob(params, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")