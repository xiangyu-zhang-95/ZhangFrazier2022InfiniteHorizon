import numpy as np

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

from fluid_model import fluid_model

def test_1():
    np.random.seed(1014)

    num_actions = 2
    num_states = 3
    T = 5
    gamma = 1
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
