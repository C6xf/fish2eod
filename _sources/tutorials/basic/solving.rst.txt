Solving
=======

This tutorial assumes that you've already read the :doc:`Parameterization tutorial<parametric>`.

Solving models while the most computationally expensive part, is also the easiest. There are a large number of
configurable options for the solver. We will show the basic properties here (things like tank size) and remit the
complex properties to the :doc:`advanced topics<adv>`

Setting Model Properties
------------------------

Setting Pre-baked parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Properties are set in a more dynamics manner. All properties have defaults, as such only the ones you want to change
need to be added

.. plot:: tutorials/basic/code/base_fish_model_set_properties_init.py
   :include-source:

Setting EOD Phase
~~~~~~~~~~~~~~~~~

Depending on the application you may want to study the effect of the temporal properties of the electric field.

Phase is set like other parameters see :doc:`documentation <parametric>`) where the ``phase`` variable is added to the
parameters dictionary.

.. note::

   Importantly ``phase`` is not added to any method argument list as it's handled internally.

.. plot:: tutorials/basic/code/base_fish_model_phase.py
   :include-source:

In later sections we'll explain the ``model.solve`` **HERE**

Setting Species
~~~~~~~~~~~~~~~

Later taters

Solving Simple Models
---------------------

After compilation (`.Model.compile` method) we need to solve it with the `~.Model.solve` method.

.. plot:: tutorials/basic/code/base_fish_model_solve.py
   :include-source:

This will not compute any electric images. For this see :ref:`advanced topics<adv>` for detailed descriptions.
Additionally here we plot the result