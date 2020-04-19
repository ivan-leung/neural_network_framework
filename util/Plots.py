import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)

class Plot():
    """ Abstract Plot class """
    def __init__(self):
        self.mat = None

    def construct_subplot(self, sp):
        """ constructs a figure """
        raise NotImplementedError("Implement function 'construct_figure'")

    def plot(self, fig_num=1, xlabel=None, ylabel=None, title=None):
        """ shows plot """
        fig = plt.figure(fig_num)
        sp = fig.add_subplot(111)
        if xlabel is not None: sp.set_xlabel(xlabel)
        if ylabel is not None: sp.set_ylabel(ylabel)
        if title is not None: sp.set_title(title)
        self.construct_subplot(sp)
        fig.show()
        raw_input("Press Enter to Continue")

class MultiPlot():
    """ MutiPlot Class plots multiple figures, given Plot instances """
    def __init__(self, *args):
        self.plot_objs = args

    def plot(self, fig_num=1, ncol=None):
        if ncol is None: ncol = len(self.plot_objs)
        fig_ids = self._get_fig_ids(ncol)
        fig = plt.figure(fig_num)
        for i in range(len(self.plot_objs)):
            sp = fig.add_subplot(*fig_ids[i])
            self.plot_objs[i].construct_subplot(sp)
        fig.subplots_adjust(hspace=0.8)
        fig.show()
        raw_input("Press Enter to Continue")

    def _get_fig_ids(self, ncol):
        t_rows = math.ceil(len(self.plot_objs)/float(ncol))
        return [(t_rows, ncol, i+1) for i in range(len(self.plot_objs))]

class Bar(Plot):
    def __init__(self, labels, y, color="r", width=0.5):
        Plot.__init__(self)
        self.y = y
        self.x = labels
        self.c = color
        self.w = width
    
    def construct_subplot(self, sp):
        pos = [p for p in range(len(self.y))]
        plt.bar(range(len(self.y)), self.y, self.w, color=self.c, \
                align="center", linewidth=0)
        plt.xticks(range(len(self.y)), self.x, rotation=90, fontsize=8)
        sp.set_xlim(-self.w, len(self.y)+self.w)


class Scatter2D(Plot):
    def __init__(self, data_mat):
        Plot.__init__(self)
        self.x = np.squeeze(data_mat[:,0])
        self.y = np.squeeze(data_mat[:,1])

    def construct_subplot(self, sp):
        plt.scatter(self.x, self.y, color="b")

class Scatter2DColored(Plot):
    def __init__(self, data_mat, color_list, c_map=None, xindex=0, yindex=1):
        Plot.__init__(self)
        self.x = np.squeeze(data_mat[:,xindex])
        self.y = np.squeeze(data_mat[:,yindex])
        self.c_list = color_list
        self.c_set = sorted(list(set(color_list)))
        self.c_map = self._mk_c_map(c_map)

    def construct_subplot(self, sp):
        for chosen in self.c_set:
            xc = [e for e,c in zip(self.x,self.c_list) if c == chosen]
            yc = [e for e,c in zip(self.y,self.c_list) if c == chosen]
            plt.scatter(xc, yc, color=self.c_map[chosen])

    def _mk_c_map(self, cycle=None):
        if cycle is None:
            cycle = ["r","b","g","k"]
        c_map = {}
        for c_val in self.c_set:
            c_map[c_val] = cycle[c_val % len(cycle)]
        return c_map

class Scatter2DColoredPrediction(Scatter2DColored):
    def __init__(self, data_mat, color_list, pred_fn, c_map=None):
        Scatter2DColored.__init__(self, color_list, c_map)
        self.pred_fn = pred_fn
    
    def construct_subplot(self, sp, margin=0.1, step=100):
        EPS = 0.01
        for chosen in self.c_set:
            xc = [e for e,c in zip(self.x,self.c_list) if c == chosen]
            yc = [e for e,c in zip(self.y,self.c_list) if c == chosen]
            plt.scatter(xc, yc, color=self.c_map[chosen], zorder=2)
        x_range = max(self.x)-min(self.x)
        x_range = (min(self.x)-margin*x_range-EPS, max(self.x)+margin*x_range+EPS)
        y_range = max(self.y)-min(self.y)
        y_range = (min(self.y)-margin*y_range-1, max(self.y)+margin*y_range+1)
        x_rast = np.linspace(x_range[0],x_range[1],step)
        y_rast = np.linspace(y_range[0],y_range[1],step)
        xx, yy = np.meshgrid(x_rast, y_rast)
        xpts = self._grid_to_pts(xx)
        ypts = self._grid_to_pts(yy)
        c_rast = [self.pred_fn([xval,yval]) for xval,yval in zip(xpts, ypts)]
        c_rast_map = self._mk_c_map(["k","y","r","b"])

        for chosen in self.c_set:
            xc = [e for e,c in zip(xpts, c_rast) if c == chosen]
            yc = [e for e,c in zip(ypts, c_rast) if c == chosen] 
            plt.scatter(xc, yc, color=c_rast_map[chosen], zorder=1)

    def _grid_to_pts(self, mgrid):
        pts = []
        for p_list in mgrid:
            pts += p_list.tolist()
        return pts

class Lines(Plot):
    def __init__(self, color_list=None, data_mat=None):
        Plot.__init__(self)
        if data_mat is not None:
            self.data_list = [data_mat]
        else:
            self.data_list = []
        self.color_list = color_list
        if self.color_list is None:
            self.color_list = ['dodgerblue', 'firebrick', 'teal',\
                               'steelblue', 'crimson', 'seagreen']

    def add_data(self, data_tup):
        self.data_list.append(data_tup)

    def construct_subplot(self, sp):
        for i,data_tup in enumerate(self.data_list):
            x = np.squeeze(data_tup[0])
            y = np.squeeze(data_tup[1])
            c = self.color_list[i % len(self.color_list)]
            plt.plot(x,y, color=c)

class XYLine(Plot):
    def __init__(self, data_mat, color="b"):
        Plot.__init__(self)
        self.x = np.squeeze(data_mat[:,0])
        self.y = np.squeeze(data_mat[:,1])
        self.c = color

    def construct_subplot(self, sp):
        plt.plot(self.x, self.y, color=self.c)

        


        


