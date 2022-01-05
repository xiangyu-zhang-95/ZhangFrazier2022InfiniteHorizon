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
        assert(isinstance(T, int))
        assert(isinstance(gamma, float))
        assert(isinstance(r, np.ndarray))
        assert(isinstance(init_occupation, np.ndarray))
        assert(isinstance(P0, np.ndarray))
        assert(isinstance(P1, np.ndarray))
        assert(isinstance(budgets, list))
        assert(num_actions == 2)

        assert(r.shape == (num_states, num_actions))
        assert(init_occupation.shape == (num_states, ))
        assert(0 <= gamma <= 1)
        assert(np.isclose(1, np.sum(init_occupation)))

        assert(P0.shape == (num_states, num_states))
        assert(np.all(P0.sum(axis=1, keepdims=True) <= 1))
        assert(P1.shape == (num_states, num_states))
        assert(np.all(P1.sum(axis=1, keepdims=True) <= 1))

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
    
    def simulate(self, priority):
        params = self.params
        num_actions, num_states, T, gamma, r, init_occupation, P0, P1, budgets =\
            params["num_actions"], params["num_states"], params["T"], params["gamma"], \
            params["r"], params["init_occupation"], params["P0"], params["P1"], \
            params["budgets"]
        
        assert(isinstance(priority, list))
        assert(len(priority) == len(set(priority)))
        assert(sum(priority) == num_states * (num_states - 1) // 2)

        def get_pull(states, priority, budget):
            pull = [0] * len(states)
            for i in priority:
                if np.isclose(budget, 0):
                    break

                pull[i] = min(budget, states[i])
                budget -= pull[i]
            return pull

        
        rewards = 0
        states = init_occupation
        for t in range(T):
            pull = get_pull(states, priority, budgets[t])
            idle = states - pull
            rewards += (idle @ r[:, 0] + pull @ r[:, 1]) * gamma**t
            states = pull @ P1 + idle @ P0
            assert(states.shape == (num_states, ))
        return rewards
    
    def is_feasible(self, active, tie_idx):
        params = self.params
        num_actions, num_states, T, gamma, r, init_occupation, P0, P1, budgets =\
            params["num_actions"], params["num_states"], params["T"], params["gamma"], \
            params["r"], params["init_occupation"], params["P0"], params["P1"], \
            params["budgets"]

        assert(isinstance(active, list))
        assert(isinstance(tie_idx, int))
        assert(0 <= tie_idx <= num_states - 1)

        # V = [0] * num_states
        # lambd = 0

        # V(s) = max_a {r(s, a) - lambd * a + gamma * V[s' | s, a]}
        # left = V = left_matrix @ V
        left_matrix = np.identity(num_states)

        # right_0 = r[:, 0] + gamma * P0 @ V
        # right_1 = r[:, 1] - lambd + gamma * P1 @ Vrig
        right_0_adding = r[:, 0]
        right_0_lambda = np.array([0] * num_states)
        right_0_matrix = gamma * P0

        right_1_adding = r[:, 1]
        right_1_lambda = np.array([-1] * num_states)
        right_1_matrix = gamma * P1


        right_adding = [right_1_adding[idx] if active[idx] else right_0_adding[idx] for idx in range(num_states)]
        right_lambda = [right_1_lambda[idx] if active[idx] else right_0_lambda[idx] for idx in range(num_states)]
        right_matrix = [right_1_matrix[idx] if active[idx] else right_0_matrix[idx] for idx in range(num_states)]

        A = np.zeros((num_states + 1, num_states + 1))
        b = np.zeros((num_states + 1,))
        A[0: num_states, 0: num_states] = left_matrix - right_matrix
        A[0: num_states, -1] = -np.array(right_lambda)
        b[0: num_states] = right_adding

        A[-1, 0: num_states] = right_0_matrix[tie_idx] - right_1_matrix[tie_idx]
        A[-1, -1] = right_0_lambda[tie_idx] - right_1_lambda[tie_idx]
        b[-1] = right_1_adding[tie_idx] - right_0_adding[tie_idx]

        sol = np.linalg.solve(A, b)

        v, lambd = sol[0: num_states], sol[-1]

        active_reward = right_1_adding + right_1_lambda * lambd + right_1_matrix @ v

        inactive_reward = right_0_adding + right_0_lambda * lambd + right_0_matrix @ v

        def is_greater_than(a, b):
            return a >= b or np.isclose(a, b)

        for idx, status in enumerate(active):
            if status == True:
                assert(is_greater_than(active_reward[idx], inactive_reward[idx]))
            else:
                assert(is_greater_than(inactive_reward[idx], active_reward[idx]))
        
        return True, lambd
        