"""Example creating a ParameterSweep"""
from fish2eod import ParameterSet, ParameterSweep

p1 = [0, 1, 2, 3]  # some sequence of parameters for first paramer
p2 = [100, 200, "q", -3.14]  # some sequence of parameters for second parameter
p3 = ["a", "b"]

"""
Define a ParameterSet named example_set with two parameters named

parameter_1 and parameter_2

Additionally instruct fish2eod to not rebuild the mesh for each new parameter
"""
set_1 = ParameterSet("example_set", parameter_1=p1, parameter_2=p2, rebuild_mesh=False)

"""
Define a ParameterSet named example_set_2 with one parameters named parameter_3

Additionally instruct fish2eod to rebuild the mesh for each new parameter
"""
set_2 = ParameterSet("example_set_2", parameter_3=p3, rebuild_mesh=True)

"""
Crate the parameter sweep
"""
ps = ParameterSweep(set_1, set_2)
