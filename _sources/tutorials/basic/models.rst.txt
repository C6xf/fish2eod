Models
======

In fish2eod a model is the physical description of the scientific setup. It defines what the objects are and where they
are located. This could be a tank of water or a fish or an electrode. Additionally it defines their physical properties.

For users familiar with *Comsol*, a model is analagous to a *mph* file where the geometry and physics are defined.

An aside on classes
-------------------

In programming a *class* is a collection of methods (functions) and data. In fish2eod a model is created as a class. The
reason for this is something called inheritance. A sub-class i.e. a child class *inherits* all its data and behaviour
for its parent class. In doing so if we only want to tweak a small part of its behaviour we only need to update the new
behaviour and everything else will behave like the parent.

Consider the following example where we define a class for a ``Cat``. It knows how to walk, drink and meow.

.. code-block:: python

   class Cat:
    def walk():
        print("I'm walking")
    def drink():
        print("I'm drinking")
    def meow():
        print("I'm meowing")

Now if we were to use the cat and ask it to walk, drink, or run we'd get a readout of what it's doing.

Now suppose that we also wanted to define a dainty cat (as they often are). It doesn't meow but complains about the
service that it's getting. However it still walks and drinks the same. We could define our ``DaintyCat`` in much the same
way but we'd have to re-implement everything from walking and drinking. Instead of this we can subclass our ``Cat`` and
only update the meowing.

.. code-block:: python

    class DaintyCat(Cat):
        def meow():
            print("This water isn't cold enough for me")

Now we have a ``DaintyCat`` that behaves in exactly the same way as a normal ``Cat`` - it just complains when it meows. It
still knows how to walk and drink.

What does this have to do finite element modeling, and has Aaron lost his mind? Well when you define a model you want
the freedom to inject whatever code you want wherever. On the other hand you don't want to implement your own backend
for computing domains, or boundaries, or meshes, or the electric field physics. Subclassing a model allows you to
inherit all of that behaviour - while being able to overwrite some behaviour like :doc:`defining geometry<geometry>`
or :ref:`boundary conditions<bc_intro>`.

Subclassing Models
------------------

fish2eod comes with several classes of models. However for most users only one will be of interest.

BaseFishModel
*************

The `.BaseFishModel` is the base model for any fish based model. It provides a fish, a ground electrode and a tank of
water - all with editable properties. Additionally it pre-populates the physics with the details of all of the objects
including the EOD.

.. code-block:: python

    from fish2eod import BaseFishModel

    class OurSubClass(BaseFishModel):
        pass

This above block of code creates a model that does absolutely nothing except defines a fish, a ground and a tank as in
`Shifman, Longtin, and Lewis (2015) <https://www.nature.com/articles/srep15780>`_

Additionally there are other three other methods that may be overwritten to define the geometry, voltage and current
sources. These are the `add_geometry<.BaseFishModel.add_geometry>`,
`add_voltage_sources<.BaseFishModel.add_voltage_sources>`, and the
`add_current_sources<.BaseFishModel.add_current_sources>` methods For a full explanation on how this works see
:doc:`next section<geometry>` and :ref:`here<bc_intro>`.

.. code-block:: python

    from fish2eod import BaseFishModel

    class OurSubClass(QESModel):
        def add_geometry(self, **kwargs):
            GEOMETRY-IMPLEMENTATION-HERE

        def add_current_sources(self, **kwargs):
            CURRENT-SOURCE-IMPLEMENTATION-HERE

        def add_voltage_sources(self, **kwargs):
            VOLTAGE-SOURCE-IMPLEMENTATION-HERE