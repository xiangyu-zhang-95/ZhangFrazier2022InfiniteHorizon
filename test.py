import numpy as np
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

from fluid_model import fluid_model

def test_runnable():
    np.random.seed(1014)

    num_actions = 2
    num_states = 3
    T = 5
    gamma = 1.
    r = np.array([[k, k * k] for k in range(num_states)])
    init_occupation = np.array([1.0 / num_states for _ in range(num_states)])

    P0 = np.random.uniform(0, 1, (num_states, num_states))
    P0 = P0 / P0.sum(axis=1, keepdims=True)

    P1 = np.random.uniform(0, 1, (num_states, num_states))
    P1 = P1 / P1.sum(axis=1, keepdims=True)

    budgets = [1/3] * T

    params = {
        "num_actions": num_actions, "num_states": num_states, 
        "T": T, "gamma": gamma, "r": r, "init_occupation": init_occupation, 
        "P0": P0, "P1": P1, "budgets": budgets
    }

    fluid = fluid_model(params)
    fluid.solve()
    assert(np.isclose(fluid.model.objVal, 9.33411865392877))

def test_bayesian_bandit():
    T = 15
    gamma = 1.
    budgets = [1 / 3] * T

    states = [(i, j) for i in range(T) for j in range(T)]
    states_to_idx = {v: i for i, v in enumerate(states)}

    num_actions = 2
    num_states = len(states)

    # r, init_occupation, P0, P1
    init_occupation = defaultdict(float)
    init_occupation[(0, 0)] = 1
    init_occupation = np.array([init_occupation[states[idx]] for idx in range(num_states)])

    # r, P0, P1
    def reward(s, a):
        if a == 0:
            return 0
        x, y = s
        return (x + 1) / (x + y + 2)

    r = np.array([[reward(s, 0), reward(s, 1)] for s in states])

    # P0, P1
    def transition(s, a, sprime):
        if a == 0:
            return s == sprime
        if a == 1:
            tmp_a, tmp_b = s[0], s[1]
            tmp_c, tmp_d = sprime[0], sprime[1]
            if tmp_a == tmp_c and tmp_b + 1 == tmp_d:
                return 1 - (tmp_a + 1.)/(tmp_a + tmp_b + 2)
            if tmp_a + 1 == tmp_c and tmp_b == tmp_d:
                return (tmp_a + 1.)/(tmp_a + tmp_b + 2)
            return 0

    P0 = np.array([[transition(s, 0, sprime) for sprime in states] for s in states])
    P1 = np.array([[transition(s, 1, sprime) for sprime in states] for s in states])


    params = {
        "num_actions": num_actions, "num_states": num_states, 
        "T": T, "gamma": gamma, "r": r, "init_occupation": init_occupation, 
        "P0": P0, "P1": P1, "budgets": budgets
    }

    fluid = fluid_model(params)
    fluid.solve()
    assert(np.isclose(fluid.model.objVal, 3.516196286))

def test_isfeasible():
    num_actions = 2
    num_states = 4
    T = 5000
    gamma = 0.99999999

    # r, init_occupation, P0, P1, budgets
    r = np.array([
        [-1, -1],
        [0, 0],
        [0, 0],
        [1, 1],
    ])

    init_occupation = np.array([1/6, 1/3, 1/2, 0])
    # init_occupation = np.array([1/6, 1/3, 1/4, 1/4])
    P1 = np.array([
        [1/2, 1/2,   0, 0  ],
        [  0, 1/2, 1/2, 0  ],
        [  0,   0, 1/2, 1/2],
        [1/2,   0,   0, 1/2],
    ])

    P0 = np.array([
        [1/2,   0,   0, 1/2],
        [1/2, 1/2,   0, 0  ],
        [  0, 1/2, 1/2, 0  ],
        [  0,   0, 1/2, 1/2],
    ])

    budgets = [2/3] * 5000

    params = {
        "num_actions": num_actions, "num_states": num_states, 
        "T": T, "gamma": gamma, "r": r, "init_occupation": init_occupation, 
        "P0": P0, "P1": P1, "budgets": budgets
    }

    fluid = fluid_model(params)

    def test_output(a, b):
        return a[0] == b[0] and np.isclose(a[1], b[1])

    assert(
        test_output(fluid.is_feasible([False, False, True, False], 2), [True,  1  ]) and
        test_output(fluid.is_feasible([False, True,  True, False], 1), [True,  1/2]) and
        test_output(fluid.is_feasible([True,  True,  True, False], 0), [True, -1/2]) and
        test_output(fluid.is_feasible([True,  True,  True,  True], 3), [True, -1  ])
    )