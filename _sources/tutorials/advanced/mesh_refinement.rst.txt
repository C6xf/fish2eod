Mesh Refinement
===============

Sometimes the meshes generated will be too large (not enough spatial resolution) leading to low quality solutions.
To solve this: fish2eod provides domain level mesh refinement highlighted here.

.. plot:: tutorials/advanced/code/mesh_refinement.py
   :include-source:

The parameters ``refine_domains`` and ``n_refine`` specify the which domains to refine and how many times the refinement
should be done. Each refinement operation breaks the mesh elements into smaller pieces so a high ``n_refine`` provides
a higher resolution mesh.

The ``refine_domains`` and ``n_refine`` parameters can be directly specified to `.Model.compile` method or passed as
additional parameters to the `.IterativeSolver` during parameter sweeps.