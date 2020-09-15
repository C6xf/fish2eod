Geometry
========

.. warning::

   It's very important to have read the previous section on what a model is and how they're defined.
   Please see the :doc:`model tutorial<models>` for this explanation.

In fish2eod *geometry* is the physical content of a model - it's the tank, the fish, electrodes; any physical region.
These regions form boundaries and domains which are used to specify the physics on the model (for an intro on this see
our :ref:`finite element introduction<fem-intro>`).

Geometry Basics
***************
There are 3 primitive types of geometry in fish2eod: circles, rectangles and polygons which in principle can be used
to create an arbitrarily complex environment.

The natural unit for all created geometries is *cm* i.e. a radius=1 implies a 1cm radius. For full details on the
geometry api please see the `Primitives Module<.primitives>`

Circles
~~~~~~~

Circles are defined from a center point and a radius. Here we will create a circle with a 1cm radius centered at the
point (0,5).

.. plot:: tutorials/basic/code/circle.py
   :include-source:

For full circle documentation please see the `.Circle` class

Rectangles
~~~~~~~~~~

Rectangles are defined relative to the bottom left corner with a given width (horizontal) and height (vertical) for
convenience if the height is omitted then the height is assumed to equal the width (i.e. a square).

Here we'll create a 5x3 rectangle with its bottom left corner situated at the point (-2,3). And a 2x2 square starting
from the same point.

For convenience if defining the rectangle from its center is more convenient, for example a background centered at (0,0)
this can also be done with the `.from_center` method as shown below.

For full circle documentation please see the `.Rectangle` class

.. plot:: tutorials/basic/code/rectangle.py
   :include-source:

Polygons
~~~~~~~~

When combining circles and rectangles is insufficient to replicate your geometry you can use a polygon to replicate
complex geometry.

.. note::

   In general it is inadvisable to use polygons unless absolutely necessary as they are more computationally expensive
   than their square/circle counterparts (by several orders of magnitude: think seconds instead of milliseconds.) For
   example an 'L' shape would be better created by two perpendicular rectangles than a polygon, whereas a curve must be
   created by a polygon

For illustration we will create a star using pre-defined coordinates

.. plot:: tutorials/basic/code/polygon.py
   :include-source:

.. warning::

   Polygons must **not** be closed i.e. the last point implicitly connects to the first point.

For full polygon documentation please see the `.Polygon` class

Integrating Geometries into Models
**********************************

Making geometries starts with creating a subclass of the parent model (`.BaseFishModel`).

BaseFishModel
~~~~~~~~~~~~~
Adding geometry is quite simple and there's only one thing to know: Adding a domain or objects to a model is handled
by the `.add_geometry` method.

Here we'll present a fish model with a 2cm radius metal rod at (5cm, 5cm) - we will  call this domain the "rod_domain"

.. note::

   Please ignore the ``fish_x`` portion of the code this will be explained in the
   :doc:`parameters tutorial<parametric>`.

.. plot:: tutorials/basic/code/make_geometry_base_fish.py
   :include-source:

You may notice a few lines of code that you haven't seen before. ``model.compile()`` compiles the model i.e. integrating
everything you've specified in your description. and ``model.draw_geometry()`` does exactly what it sounds like - it draws
the outline of each object with a unique color for each domain.

Additionally the argument ``sigma=1e6`` means that this domain has a conductance of 1e6 S/cm - any positive number (valid
conductance) can be used here.

For a full documentation on these methods please see the `~.Model.compile` and the `~.Model.plot_geometry` methods.