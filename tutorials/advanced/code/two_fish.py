from fish2eod import BaseFishModel, plotting

"""
Specify two fish

One fish with spine (0,0) -> (20, 0)
One fish with spine (0,5) -> (20, 5)
"""
fish_x = [[0, 20], [0, 20]]

fish_y = [[0, 0], [5, 5]]

model = BaseFishModel()
model.compile(fish_x=fish_x, fish_y=fish_y)
model.solve(fish_x=fish_x, fish_y=fish_y)

plotting.mesh_plot_2d(model.solution, "solution")
plotting.plot_outline(model.solution)
