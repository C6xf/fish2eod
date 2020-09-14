Basics
======

This tutorial assumes that you've already read the :doc:`Basic tutorial<basic_tutorals>`. We will use a common model of
a fish with an object near the head. The details are not important as all `~.Model` descendants will behave similarly.

The `~.Model.solution` object is actually a plotting data structure. All plotting functions will require this along with
some configuration settings. We'll cover the two basic plots in this section: solutions and outlines.

Plots are broken into 2D and 1D plots. 2D plots describe a function over the space: for example the EOD, the model
conductance as a function of position, or the domains. 1D plots describe a function on the boundaries - the simplest on
is the outline which is outline = 1 if an edge describes the geometry outline and outline = 0 otherwise.

The `~.solution` object has several fields corresponding to different properties they are

.. _valid_properties:

* 2D properties
   * solution
   * active (only present if image computed)
   * domain
   * sigma
* 1D properties
   * boundary
   * outline
* Misc properties
   * SkinStructure
   * ElectricImage

Plot Solution
--------------

The ``solution`` field contains the "most resent" simulation result. I.e. the voltage from the EOD (note however that if
the field perturbation is computed to compute the electric image them ``solution`` is the field perturbation and an
additional variable ``active`` is created which contains the "raw EOD".

.. plot:: tutorials/plotting/code/basic_solution_plot.py
   :include-source:

Here where we specify the "solution" variable we could have used any valid 2D function.

Plot Outline
------------

Technically the outline can be computed from a 1D plot - however we've provided a helper function to do so. All thats
needed is to call the `~.plot_outline` function.

.. plot:: tutorials/plotting/code/basic_outline_plot.py
   :include-source:

Masking
-------

In many cases some information is included in the plot that you don't want. For example inside the fish is usually
"not interesting" so this can be masked out.

.. plot:: tutorials/plotting/code/basic_mask.py
   :include-source:

Additionally it may be more convenient to set a certain domain to be ommitted rather than have a long list of included
domains. For this reason we provide an ``include`` flag to set whether the mask is inclusive or exclusive.

Sweeps
------

Plotting during sweeps is almost the same as for regular plotting. You just need to specify the sweep parameter values

Note that because of the way matplotlib works we cannot pass properties as ``prop_0 = 0`` but we need to specify the
``properties`` argument. This is necessary to provide the flexibility of passing arbitrary matplotlib settings.

Not that setting the properties argument does not influence any other settings and this can be used in concert with all
other flags and settings.

.. plot:: tutorials/plotting/code/basic_sweep_plot.py
   :include-source: