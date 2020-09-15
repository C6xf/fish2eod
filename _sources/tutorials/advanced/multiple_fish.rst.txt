Multiple Fish
=============

Despite being in the "advanced section" specifying multiple fish is trivial. All you need to do is to specify a list
of x and y coordinates and fish2eod will handle all the mechanics of creating the fish

.. plot:: tutorials/advanced/code/two_fish.py
   :include-source:

You'll notice that all model code is identical to code you've seen before. The only exception is that ``fish_x`` and
``fish_y`` parameters are lists of lists.

This is not limited to two fish or constant eod phase. In this example we'll make 4 fish with 4 different eod phase

.. plot:: tutorials/advanced/code/multi_fish.py
   :include-source:

Different EOD phases are specified in the same way: by specifying a list of phases to match each fish. (n.b. if a single
EOD phase is specified then it is assumed to be shared across all fish.