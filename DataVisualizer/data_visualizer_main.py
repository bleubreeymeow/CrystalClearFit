"""
With the following command, the viewer is started. The path to the data folder and the debug mode can be set.
Feel free to put as many data files as you want in the data folder. The viewer will automatically detect them and
display them in the dropdown menu.
"""
import viewer

data_path = "../ROIPointFitter/data/"
debug = True

viewer.run_viewer(data_path, debug)
