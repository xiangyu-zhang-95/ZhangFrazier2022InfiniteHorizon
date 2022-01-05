import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

class fluid_model():
    def __init__(self, params):
        self.params = params
        num_actions, num_states, T, gamma, r, init_occupation, P0, P1, budgets \
        = params["num_actions"], params["num_states"], params["T"], params["gamma"], \
        params["r"], params["init_occupation"], params["P0"], params["P1"], params["budgets"]
        
        assert(isinstance(num_states, int))
        assert(isinstance(num_actions, int))
        assert(num_actions == 2)

        assert(r.shape == (num_states, num_actions))
        assert(init_occupation.shape == (num_states, ))
        assert(0 <= gamma <= 1)
        assert(np.isclose(1, np.sum(init_occupation)))

        assert(P0.shape == (num_states, num_states))
        assert(np.allclose(P0.sum(axis=1, keepdims=True), 1))
        assert(P1.shape == (num_states, num_states))
        assert(np.allclose(P1.sum(axis=1, keepdims=True), 1))

        assert(len(budgets) == T)
        
    def solve(self):
        params = self.params
        num_actions, num_states, T, gamma, r, init_occupation, P0, P1, budgets =\
            params["num_actions"], params["num_states"], params["T"], params["gamma"], \
            params["r"], params["init_occupation"], params["P0"], params["P1"], \
            params["budgets"]

        m = gp.Model("bandit")
        
        
        occupation0 = []
        occupation1 = []
        for t in range(T):
            names = [f"x_{t}({s}, 0)" for s in range(num_states)]
            occupation0.append(pd.Series(m.addVars(names), index=names))

            names = [f"x_{t}({s}, 1)" for s in range(num_states)]
            occupation1.append(pd.Series(m.addVars(names), index=names))

        # initial constraints
        m.addConstrs(occupation0[0][s] + occupation1[0][s] == init_occupation[s] for s in range(num_states))

        # resource constraints
        m.addConstrs(occupation1[t] @ np.ones((num_states, )) == budgets[t] for t in range(T))

        # flow balance
        for t in range(T - 1):
            next_occ = occupation1[t] @ P1 + occupation0[t] @ P0# - occupation0[t + 1] - occupation1[t + 1]
            m.addConstrs(occupation0[t + 1][s] + occupation1[t + 1][s] == next_occ[s] for s in range(num_states))
        
        obj = np.array([gamma ** t for t in range(T)]) @ occupation0 @ r[:, 0] + \
                np.array([gamma ** t for t in range(T)]) @ occupation1 @ r[:, 1]

        m.setObjective(obj, GRB.MAXIMIZE)

        m.optimize()
        self.model = m