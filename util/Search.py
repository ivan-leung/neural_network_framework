#!/usr/bin/env python
import sys
import random
import itertools
from operator import mul
import scipy.stats
import numpy as np




class SearchGrid():
    def __init__(self, ranges, n):
        self.ranges = ranges
        self.n = n
        self.pts = [self._draw_pt() for _ in xrange(n)]
        self.vls = [None for _ in xrange(n)]
    def sample_grid(self, fn, *args):
        for i in xrange(self.n):
            if self.vls[i] is None:
                self.vls[i] = fn(self.pts[i], *args)
    def __str__(self):
        s = "Range: " + str(self.ranges) + "\n"
        for pt, vl in zip(self.pts, self.vls):
            pt_str = ["{0:6.4f}".format(c) for c in pt]
            if vl is None:
                s += "Pt: " + str(pt_str) + " Val: None "
                print "NONE FOUND"
            else:
                s += "Pt: " + str(pt_str) + " Val: {0:6.4f} ".format(vl)
        return s
    def has_pt(self, query_pt):
        if query_pt in self.pts:
            return True
        return False
    def pt_in_range(self, query_pt):
        for i,c in enumerate(query_pt):
            if c < self.ranges[i][1] and c >= self.ranges[i][0]:
                continue
            return False
        return True
    def add_pt(self, pt):
        self.pts.append(pt)
        self.vls.append(None)
    def add_pt_vl(self, pt, vl):
        self.pts.append(pt)
        self.vls.append(vl)
    def _draw_pt(self):
        return [random.uniform(r[0],r[1]) for r in self.ranges]

class SearchGridManager():    
    def __init__(self, ranges, fn, eval_fn, select_fn):
        self.ranges = ranges
        self.fn = fn
        self.e_fn = eval_fn
        self.s_fn = select_fn
        self.grids = [SearchGrid(self.ranges,0)]

    def __str__(self):
        s = ""
        for g in self.grids:
            s += str(g) + "\n"
        return s

    def search(self, n_split, partition, n_sample, *args):
        for i in xrange(n_split):
            self.search_grid(partition, n_sample, *args)

    def search_grid(self, partition, n_sample, *args):
        g = self.select_grid()
        self.sample_subgrid(g, partition, n_sample, *args)

    def get_samples(self):
        pts = []
        vls = []
        for g in self.grids:
            pts += g.pts
            vls += g.vls
        return zip(pts, vls)

    def select_grid(self):
        evaled = [(self.e_fn(g.vls),i) for i,g in enumerate(self.grids) if g.n > 0]
        if len(evaled) == 0: return self.grids[0]
        for i,e in enumerate(evaled):
            print "Grid:" + str(self.grids[i].ranges) + " Evaled:" + str(e[0])
        return self.grids[evaled.index(self.s_fn(evaled))]

    def sample_subgrid(self, g, partition, n_sample, *args):
        assert g in self.grids
        self.subdivide_grid(g, partition, n_sample)
        grid_len = len(self.grids)
        subgrid_index = range(grid_len - reduce(mul,partition,1), grid_len)
        for i in subgrid_index:
            print "grid space", self.grids[i].ranges
            self.grids[i].sample_grid(self.fn, *args)

    def subdivide_grid(self, g, partition, n_sample):
        subgrids = self._mk_subgrids(g, partition, n_sample)
        self.grids.remove(g)
        self.grids += subgrids

    def _mk_subgrids(self, g, partitions, n_sample):
        p_ranges = [xrange(p) for p in partitions]
        subgrids = []
        spans = [r[1]-r[0] for r in g.ranges]
        off = [r[0] for r in g.ranges]
        for positions in itertools.product(*p_ranges):
            s_pos = [off[i] + float(p)/partitions[i]*spans[i] \
                     for i,p in enumerate(positions)]
            e_pos = [off[i] + float(p+1)/partitions[i]*spans[i] \
                     for i,p in enumerate(positions)]
            subgrid = SearchGrid(zip(s_pos, e_pos), n_sample)
            for pt, vl in zip(g.pts, g.vls):
                if subgrid.pt_in_range(pt):
                    subgrid.add_pt_vl(pt, vl)
            subgrids.append(subgrid)
        return subgrids

       





        
                
        
    
    
    
    
