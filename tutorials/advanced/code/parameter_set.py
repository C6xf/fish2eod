"""Example creating a ParameterSet"""
from fish2eod import ParameterSet

p1 = [0, 1, 2, 3]  # some sequence of parameters for first paramer
p2 = [100, 200, "q", -3.14]  # some sequence of parameters for second parameter

"""
Define a ParameterSet named example_set with two parameters named

parameter_1 and parameter_2

Additionally instruct fish2eod to rebuild the mesh for each new parameter
"""
ParameterSet("example_set", parameter_1=p1, parameter_2=p2, rebuild_mesh=True)
