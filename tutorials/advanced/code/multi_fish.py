from fish2eod import BaseFishModel, plotting

"""
Specify two fish

One fish with spine (0,0) -> (20, 0)
One fish with spine (0,5) -> (20, 5)
One fish with spine (-20,-10) -> (-20, 20)
One fish with spine (30,30) -> (30, 10)
"""
fish_x = [[0, 20], [0, 20], [-20, -20], [30, 30]]

fish_y = [[0, 0], [5, 5], [-10, 20], [30, 10]]
phase = [0.2, 0.4, 0.8, 0.5]
model = BaseFishModel()
model.compile(fish_x=fish_x, fish_y=fish_y, eod_phase=phase)
model.solve(fish_x=fish_x, fish_y=fish_y, eod_phase=phase)

plotting.mesh_plot_2d(model.solution, "solution")
plotting.plot_outline(model.solution)
