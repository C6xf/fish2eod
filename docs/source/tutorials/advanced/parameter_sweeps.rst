Parameter Sweeps
================

If you've used *Comsol* before you'll be familiar with the notion of a parameter sweep: specify one model and then run
it several times for many combinations of parameters.

We start the parameter sweep by defining a `.ParameterSet`

Parameter Sets
--------------

Parameter sets define a set of parameters which are swept together. For example if you wanted to move the fish it's
``x`` and ``y`` cooridinates would have to sweep together. When parameters are swept together we refer this is a serial
sweep.

In this example we'll create a ``ParameterSet`` named ``example_set`` which which will have 2 parameters
(``parameter_1`` and ``parameter_2``), parameter names don't need to follow any pattern and could have been named
``alice`` and ``bob``.

The ``rebuild_mesh`` flag specifies whether or not these parameters require the mesh to be recomputed. If these
parameters affect something structural (e.g. coordinates of an object) then the mesh must be rebuilt for the model to
update. However if something "physics-esque" is being changed (e.g. conductivity) then the mesh does not need to be
rebuilt. Since meshing is an expensive operation minimizing re-meshing is in your best interest.

.. note::
   There are almost no restrictions on parameters. We can see here that p2 is heterogenous and incoherent. As long as
   your downstream code can handle the parameter value then fish2eod will accept it. The only requirement is that all
   parameters in a `.ParameterSet` have the same number of steps (same length)

.. plot:: tutorials/advanced/code/parameter_set.py
   :include-source:

This example creates 4 steps of parameters

- (0, 100)
- (1, 200)
- (2, "q")
- (3, -3.14)

Parameter Sweeps
----------------

A `.ParameterSweep` takes one or more `.ParameterSets` and arranges them optimally. When specifying multiple sets they
are swept combinatorially (i.e. the cartesian product of the sets, or every combination of the 2 or more). The
`.ParameterSweep` additionally sorts the parameters to minimize remeshing.

.. plot:: tutorials/advanced/code/parameter_sweep.py
   :include-source:

This example creates 8 steps of parameters

- (0, 100, "a")
- (1, 200, "a")
- (2, "q", "a")
- (3, -3.14, "a")
- (0, 100, "b")
- (1, 200, "b")
- (2, "q", "b")
- (3, -3.14, "b")

You'll notice that set_1 is iterated before set_2 meaning that only 2 remeshes need to occur instead of 8

Running Parameter Sweeps
------------------------

Creating a `.ParameterSweep` is the precursor to running the sweep.

Sweeps are run using an `.IterativeSolver`. In this example we will move an object around beside the fish

.. plot:: tutorials/advanced/code/iterative_solver.py
   :include-source:

Importantly with an `.IterativeSolver` some parameters may need to be specified but not part of the sweep
(fish coordinates in this example). Any fixed parameters may be specified in the `.IterativeSolver` by adding them
as parameters as you would for a `.Model` constructor

Special Parameter Sets
----------------------

To make life easy for basic parameter sets we provied classes for specifing sweeps over eod phase (`.EODPhase`) and
fish coordinates (`.FishPosition`). Please read their documentations for use case.