Paramaterizing Models
=====================

This tutorial assumes that you've already read the :doc:`domains tutorial<domains>`.

Generally speaking you will never make one model. You may be interested in multiple object conductances or object
locations. Instead of making 10 models for each object you can simply parameterize them.

Parameterizing Functions
------------------------

There are 3 parameterizable methods in the `.BaseFishModel`

#. `.add_geometry`
#. `.add_voltage_sources`
#. `.add_current_sources`

These methods have the same default signature

.. code-block:: python

   def method_name(self, **kwargs):
    IMPLEMENTATION-HERE

If we want to add parameters we can put them anywhere between ``self`` and ``**kwargs``

For example here we're going to make a model with a movable object by defining its x, and y position parametrically


.. plot:: tutorials/basic/code/parametric_object.py
   :include-source:

As you can see we've added two parameters to the model ``object_x`` and ``object_y``. We set parameters with a parameter
dictionary with all necessary parameters included in it. The `.BaseFishModel` has two required parameters: ``fish_x``
and ``fish_y``. These two parameters are lists of points which define the x and y coordinates of the spine. For this
model we've added two additional parameters and therefore they're also included in the dictionary. You can have as many
or as few as you'd like. For example the radius could also be parameterized.

.. note::

   While this example only showed how to work with geometry. The process for `.add_voltage_sources` or
   `.add_current_sources` is identical. Parameter names are placed between ``self`` and ``**kwargs``.

   As an additional note: the same parameter can be used in more than one place just by adding the same name. You do not
   need to redefine parameters.

Fish Position Parameters
------------------------

The only requirement for the points is that they are in rostro-caudal order and the midline does not self-intersect.

To highlight the flexibility we will create a 45cm fish with it's head at (-30, -30), an exponential body, and rotated
by 45°.

.. plot:: tutorials/basic/code/base_fish_model_fish_position.py
   :include-source:

While this code may look complicated, the majority of the code is purely used to compute the coordinates for the fish
body. All of the ``compute_coordinates`` function is doing that role.