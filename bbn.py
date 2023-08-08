import numpy as np
import itertools
import math
from pgmax import fgraph, fgroup, infer, vgroup

def log_potential(arr):
    for i in range(arr.size):
        if arr[i] == 0:
            arr[i] = np.NINF
        else:
            arr[i] = math.log(arr[i])
    return arr


class Node:
    def __init__(self, name, states_list, parents_list, prob_list, evidence_index=-1):
        self.name = name
        self.states_list = states_list
        self.parents_list = parents_list
        self.prob_list = prob_list

        parent_states = map(lambda node: node.states_list, parents_list)
        self.all_states = list([states_list, *parent_states])
        
        self.variable = vgroup.NDVarArray(
            num_states=len(states_list),
            shape=(1,)
        )

        parent_variables = map(lambda node: node.variable[0], parents_list)
        self.all_variables = list([self.variable[0], *parent_variables])
        
        self.factor = fgroup.EnumFactorGroup(
            variables_for_factors=[self.all_variables],
            factor_configs=np.array(list(itertools.product(*self.all_states))),
            log_potentials=log_potential(np.array(self.prob_list)),
        )

        self.evidence = np.zeros_like(states_list)
        if evidence_index >= 0:
            self.evidence[evidence_index] = 1000
        
class Graph:
    def __init__(self, nodes_list):
        self.nodes_list = nodes_list
        
        vars = map(lambda node: node.variable, nodes_list)
        self.fg = fgraph.FactorGraph(variable_groups=list(vars))
        

        factors = map(lambda node: node.factor, nodes_list)
        self.fg.add_factors(list(factors))
        

    def run(self):
        evidence_array = np.array([])
        for node in self.nodes_list:
            evidence_array = np.append(evidence_array, node.evidence)
        evidence = infer.Evidence(self.fg.fg_state, value=evidence_array)

        bp_state = infer.BPState(
            log_potentials=infer.LogPotentials(fg_state=self.fg.fg_state),
            ftov_msgs=infer.FToVMessages(fg_state=self.fg.fg_state),
            evidence=evidence,
        )
        
        bp = infer.build_inferer(bp_state, backend="bp")
        bp_arrays = bp.run(bp.init(), num_iters=100, damping=0.5, temperature=1.0)
        beliefs = bp.get_beliefs(bp_arrays)
        marginals = infer.get_marginals(beliefs)

        for node in self.nodes_list:
            node.marginal = np.array(marginals[node.variable][0]).round(decimals=3)