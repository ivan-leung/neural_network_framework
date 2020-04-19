#!/usr/bin/env python
import sys
import os.path
import glob
import numpy as np
import util

WB = "Data/v9.xlsx"
WS = "dense"
OBJ_MAP = {1:"User Emotion", 2:"Quantitative Performance", 3:"Qualitative Performance"}

def layer_contribution(W):
    return np.divide(np.square(W).T, np.sum(np.square(W), axis=1).T)

def plot_model(m, hdrs, fig_num):
    wts = m["wts"]
    L = np.sum(np.abs(wts[0][0]), axis=0)
    impact_arr = sorted(zip(L,hdrs), reverse=True)


    x_val = [hdr for imp, hdr in impact_arr]
    y_val = [imp for imp, hdr in impact_arr]
    barplot = util.Bar(x_val, y_val, color="firebrick", width = 0.5)
    obj = OBJ_MAP[m["params"].obj]
    barplot.plot(title=obj, xlabel = "VDC features")
    return impact_arr

def output_feats(feats, r, obj, num=0):
    fname = None
    while True:
        fname = "Features/" + str(obj) + "_" + str(num) + ".pkl"
        if os.path.isfile(fname):
            #print "File", fname, "already exists"
            num += 1
        else:
            break
    result = {"features": feats, "r": r}
    util.write_pickle(fname, [result])

def main(argv):
    models = [util.read_pickle(f)[0] for f in glob.glob("Models/*.pkl")]
    if len(argv) > 1:
        produce_output = bool(int(argv[1]))
    else:
        produce_output = False
    hdrs = util.DataFrame((WB, WS)).get_col_names()
    for i,m in enumerate(models):
        sorted_feats = plot_model(m, hdrs, i)
        if produce_output:
            output_feats(sorted_feats, m["params"].reg, m["params"].obj)

if __name__ == "__main__":
    main(sys.argv)
    
    
    
