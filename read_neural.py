#!/usr/bin/env python
import sys
import glob
from operator import itemgetter
import numpy as np
import util

"""
Reads neural network training results from directory DIR,
and prints out the 10 models with lowest test error from each of the objectives.

In addition, it writes a bash program (the name is specified by OUT_FILE) that
contains commands to train the best model (all data is used. note that this is NOT
cross-validation) in each objective, so that the weights assigned to each input feature
could be subsequently inspected (see documentation for plot_neural.py)
"""

OBJ_MAP = {1:"Emotion", 2:"quanEval", 3:"qualEval"}
CODE_MAP = {"Emotion":1, "quanEval":2, "qualEval":3}
# name of bash program that contians the command to train the neural net
# with parameters that yield the lowest test error
OUT_FILE = "min_error.sh"
# directory to read results from
DIR = "RESULTS"


def add_error(f, e_arr):
    lambda_1 = f["params"].l1
    n = f["params"].max_iter
    k = f["params"].k
    units = f["params"].units
    alp = f["params"].alp
    alp_dec = f["params"].adec
    #if k < 10 or n < 5000: return
    for elem in f["summary"]:
        reg, err = elem
        if err >= 0.35: continue
        if reg < 0: continue
        e_arr.append([lambda_1, reg, err, n, units, k, alp, alp_dec])

def report_error(e_map):
    for obj_str in e_map:
        print obj_str
        print "    U     |  L1  |  L2   | Error |   n   |  k   |  a"
        for arr in e_map[obj_str]:
            print arr[4].ljust(10) +"|", \
                  "{0:3.2f} | {1:4.3f} | {2:4.3f} | {3:5.0f}"\
                  .format(arr[0],arr[1],arr[2], arr[3], arr[4]), "|", arr[5], "|", arr[6]

def sort_error(error_map):
    for obj_str in error_map:
        arrs = error_map[obj_str]
        arrs.sort(key=itemgetter(2,4,0,1))
        error_map[obj_str] = arrs

def get_min_error_cmd(error_map, rep):
    s = "for i in `seq 1 {}`;\ndo\n".format(rep)
    for obj_str in error_map:
        c = CODE_MAP[obj_str]
        l, r, err, n, u, k, a, q = error_map[obj_str][0]
        s += "    ./neural.py -k 0 -u {} -l {} -r {} -n {} -a {} -q {} -o {}\n".format(\
              u, l, r, n, a, q, c)
    s += "done"
    return s 

def output_cmd(s):
    with open(OUT_FILE, "wb") as f:
        f.write(s)

def main(argv):
    try:
        rep = int(argv[1])
    except:
        rep = 3
    f_names = [f for f in glob.glob(DIR + "/*.pkl")]
    f_dicts = [util.read_pickle(f)[0] for f in f_names]
    obj_set = set([f["params"].obj for f in f_dicts])
    error_map = {}
    for fn, fd in zip(f_names, f_dicts):
        obj = fd["params"].obj
        obj_str = OBJ_MAP[obj]
        if obj_str not in error_map:
            error_map[obj_str] = []
        add_error(fd, error_map[obj_str])
    sort_error(error_map)
    report_error(error_map)
    cmd = get_min_error_cmd(error_map, rep)
    output_cmd(cmd)
        
if __name__ == "__main__":
    main(sys.argv)
