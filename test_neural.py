#!/usr/bin/env python
import sys
import glob
import os.path
from operator import itemgetter
import numpy as np
import util
import neural

# directory of test results to read from
DIR = "Subsets"

WB = "Data/v9.xlsx"
WS_X = "dense"
WS_Y = "y"
OBJ_MAP = {1:"Emotion", 2:"quanEval", 3:"qualEval"}
Y_MAP = {1:[4,6], 2:[4,6], 3:[4,6]}

def sort_feats(model, n_feats):
    wts = model["wts"]
    obj = model["params"].obj
    reg = model["params"].reg
    L = np.sum(np.square(wts[0][0]), axis=0)
    L = L[0:min(n_feats,L.shape[0])]
    impact_arr = sorted(zip(L, np.arange(L.shape[0])), reverse=True)
    return impact_arr

def read_args(args):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-o", "--obj", dest="obj", type="int", default=1)
    parser.add_option("-f", "--features", dest="feats", type="int", default=5, help="sets # of top features to train on")
    parser.add_option("-l", dest="el1", type="float", default=1, help="sets L1 regularization constant")
    parser.add_option("-d", dest="rep", type="int", default=1)
    parser.add_option("--report", dest="report", type="int", default=0)
    parser.add_option("-r", dest="reg", type="float", default=0.8, help="sets L2 regularization constant")
    parser.add_option("-q", dest="adec", type="float", default=15)
    parser.add_option("-n", dest="max_iter", type="int", default=1000)
    parser.add_option("-t", dest="treg", type="int", default=1)
    parser.add_option("--rounds", dest="rounds", type="int", default=2)
    parser.add_option("-k", dest="k", type="int", default=10, help="sets # of fold CV")
    parser.add_option("-a", dest="a", type="float", default=4e-2, help="sets starting learning rate")
    parser.add_option("-u", dest="units", type="str", default="10,6", help="sets # units in each neural layer")
    parser.add_option("-v", dest="v", type="int", default=10)
    ops, others = parser.parse_args(args)
    if len(others) > 0:
        print "Received unknown arguements: {}".format(','.join(others))
        exit(1)
    return ops

def format_data(feats, obj_code):
    obj_str = OBJ_MAP[obj_code]
    df_x = util.DataFrame((WB, WS_X), none_as=-2)
    feats = [df_x.get_col_names()[index] for wt,index in feats]
    df_x = df_x[:,feats]
    neural.center_mean_var(df_x)  
    df_y = util.DataFrame((WB, WS_Y))
    df_y = df_y[:,obj_str]
    for i in range(df_y.nrows):
        df_y[i,0] = neural.get_bucket(df_y.get_val(i,0), Y_MAP[obj_code])
    valid_y = neural.get_non_null_indexes(df_y)
    df_x = df_x[valid_y,:]
    df_y = df_y[valid_y,:]
    return (df_x, df_y)


def output_results(results, obj, directory="Subsets"):
    suf = 0
    feat_len = len(results[1])
    while True:
        out_name = directory + "/" + str(obj) + "w_{}_features_{}.pkl".format(feat_len,suf)

        print out_name
        if os.path.isfile(out_name):
            suf += 1
        else:
            break
    samples, sorted_feats, units, max_iter = results
    output = {"obj":obj, "samples":samples, "feats":sorted_feats, "units":units, "n":max_iter}
    util.write_pickle(out_name, [output])

def select_r2(summary):
    return min(summary)[1]["reg2"]

def select_r1(summary):
    return min(summary)[1]["reg1"]


def add_error(fd, arr):
    print "features", fd["feats"]
    for pt, vl in fd["samples"]:
        
        arr.append([vl, pt[0], pt[1], fd["feats"]])

def sort_error(e_map):
    for obj_str in e_map:
        arrs = e_map[obj_str]
        arrs.sort()
        e_map[obj_str] = arrs

def report_error(e_map, tot=10):
    for obj_str in e_map:        
        count = 0
        print obj_str
        print " Error |  L1  |  L2  |    Features"
        for arr in e_map[obj_str]:
            print "{0:4.3f} | {1:8.7f} | {2:8.7f} | {3}".format(arr[0],arr[1],arr[2],arr[3])
            count += 1
            if count > tot: break

def print_results(directory="Subsets"):
    f_names = [f for f in glob.glob(directory + "/*.pkl")]
    f_dicts = [util.read_pickle(f)[0] for f in f_names]
    obj_set = set([f["obj"] for f in f_dicts])
    error_map = {}
    for fn, fd in zip(f_names, f_dicts):
        obj = fd["obj"]
        obj_str = OBJ_MAP[obj]
        if obj_str not in error_map:
            error_map[obj_str] = []
        add_error(fd, error_map[obj_str])
    sort_error(error_map)
    report_error(error_map)

def train_one(lam, xmat, ymat, units, max_iter, k, alp, v, a_dec_fn):
    l1, l2 = lam
    while True:
        if alp < 1e-7:
            print "alpha is too small. exiting"
            return None
        try:
            cv_result = util.cross_validation(xmat, ymat, units,\
                            k, max_iter, alp, l2, v=v, l1=l1)
            errors, losses = cv_result
            mean_err = sum(errors)/len(errors)
            return mean_err
        except ValueError as e:
            alp = a_dec_fn(alp)

def test_model(model, num_feats, reg, obj, rep, ops):
    if ops.units == "0":
        units = [3]
    else:
        units = [int(u) for u in ops.units.split(",")] + [3]
    print units
    v = ops.v
    k = ops.k
    a_dec_fn = neural.get_a_dec_fn(ops.adec)
    max_iter = ops.max_iter
    alp = ops.a
    l1 = ops.el1
    

    print "CV on obj {} with L2 reg {}".format(OBJ_MAP[obj], model["params"].reg)
    sorted_feats = sort_feats(model, num_feats)   
    df_x, df_y = format_data(sorted_feats, obj)
    feat_str_list = ", ".join(df_x.get_col_names())
    print "Features", ", ".join(df_x.get_col_names())
    summary = []
    logs = {}
    def get_cv_error(cv_errs):
        return util.t_confint(cv_errs, 0.8)[0]

    searcher = util.SearchGridManager([(0,l1),(0,reg)], train_one, get_cv_error, min)
    searcher.search(4,[3,3],3, df_x.get_entries(), df_y.get_entries(), units, max_iter, k, alp, v, a_dec_fn)
    min_error = 1
    min_pt = None
    return (searcher.get_samples(), feat_str_list, units, max_iter)
    
    




def main(argv):
    ops = read_args(argv[1:])

    if ops.report != 0:
        print_results(DIR)
        return

    obj, num_feats, reg, rep = ops.obj, ops.feats, ops.reg, ops.rep
    models = [util.read_pickle(f)[0] for f in glob.glob("Models/*.pkl")\
              if f.split("/")[1].startswith(OBJ_MAP[obj])]
    for model in models:
        model_results = test_model(model, num_feats, reg, obj, rep, ops)
        output_results(model_results, obj)

    
if __name__ == "__main__":
    main(sys.argv)
    
