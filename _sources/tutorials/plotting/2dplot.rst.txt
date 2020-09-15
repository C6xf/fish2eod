2D Plots
========

A 2D plot is a plot of any variable present in the :ref:`2D property list<valid_properties>`. Generally speaking these
are variables that "make sense to talk about on a surface" We give the basic introduction to these in the
:doc:`basic plot tutorial<plotting_basics>`. In this section we'll provide detailed operations

Importantly there are 4 types of color scales. A ``None`` colorscale sets the range of the colorbar "Standard" i.e.
between max and mix. A ``full`` colorbar centers the colorbar at zero and extends the color range to the min and the
max of the data (i.e. the "color slope" for positive nubers is different for negative). A ``equal`` colorscale set the
max and min to the larges magnitude value i.e. the intermediate color represents a "true 0". Lastly is an array of two
numbers is set then the colorbar will scale between the two numbers.


.. plot:: tutorials/plotting/code/2d_details.py
   :include-source: