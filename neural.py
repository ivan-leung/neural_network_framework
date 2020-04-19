#!/usr/bin/env python
import sys
import numpy as np
import os.path
import util
#from matplotlib import interactive
#interactive(True)

WB = "Data/v9.xlsx"
WS_X = "dense"
WS_Y = "y"
Y_MAP = {"Emotion":[4,6], "quanEval":[4,6], "qualEval":[4,6]}



def main(argv):

    ops = read_args(argv[1:])
    df_x, df_y, obj = format_data(ops.obj) 
    units = format_units(ops.units)

 
    print "OBJECTIVE: optimize", obj
    r_dec_fn = get_r_dec_fn(ops.rdec)
    a_dec_fn = get_a_dec_fn(ops.adec)


    if ops.k == 0:
        wts, err = train_one(df_x.get_entries(), df_y.get_entries(),\
                units, ops.max_iter, ops.alp, ops.reg, ops.v, a_dec_fn, ops.l1)
        result = {"wts":wts, "params":ops, "error": err}
    else:
        summary = []
        logs = {}
        search_reg(summary, logs, df_x.get_entries(), df_y.get_entries(), ops.reg,\
                   units, ops.treg, r_dec_fn, ops.alp, a_dec_fn, ops.k, \
                   ops.max_iter, ops.v, ops.donum, ops.l1)
        if len(summary) == 0:
            print "No viable regularization found"
            return
        result = {"summary":summary, "logs":logs, "params":ops}


    out_fname = get_output_name(units, ops.reg, ops.treg, r_dec_fn, obj,\
                                ops.l1, ops.k)
    util.write_pickle(out_fname, [result])
 
def get_output_name(units, reg, treg, r_dec_fn, obj, lambda_1, k):

    unit_str = obj + '_' +'_'.join([str(u) for u in units])
    if k > 0:
        max_reg = reg
        min_reg = max_reg
        for i in range(treg-1):
            min_reg = r_dec_fn(min_reg)
        output_name = unit_str + "_{0:4.3f}_{1:4.3f}_{2:3.2f}.pkl"\
                      .format(max_reg, min_reg, lambda_1)
        output_name = "Results/" + output_name
    else:
        cur_num = 0
        while True:
            output_name = "Models/"+ obj + "_{}.pkl".format(cur_num) 
            if os.path.isfile(output_name):
                cur_num += 1
            else:
                break
    return output_name
    
def train_one(xmat, ymat, units, max_iter, alp, min_reg, v, a_dec_fn, el1):
     while True:
        if alp < 1e-7:
            print "train_best failed because alpha is too small. Exiting"
            return None
        try:
            net = util.NeuralNetwork(xmat, ymat, units=units)
            net.train(max_iter, alp, min_reg, v=v, l1=el1)    
            preds = net.predict(xmat)
            corr = [p == a for p, a in zip(preds, np.squeeze(ymat.tolist()))]   
            error = 1-sum(corr)/float(len(corr))     
            return (net.wts, error)
        except ValueError:
            alp = a_dec_fn(alp)
        

def search_reg(summary, logs, xmat, ymat, reg, units,\
               t_reg, r_dec_fn, alp, a_dec_fn, k, max_iter, v, donum, el1):
    """
    Searches for t_reg regularization values, beginning from reg.
    Decrements the current regularization value using dec_fn.
    """
    for i in range(t_reg):
        print " "*3, "L1 Reg: {0:3.2f}; L2 Reg: {1:5.4f}".format(el1, reg)
        mean_err = None
        for d in range(donum):
            print " "*3, "Trial", d
            this_err, cv_result = search_alp(xmat, ymat, reg, units, \
                                  alp, a_dec_fn, k, max_iter, v, el1)
            if mean_err is None or this_err < mean_err:
                mean_err = this_err
        if mean_err is None:
            continue
        summary.append((reg, mean_err))
        logs[reg] = cv_result
        reg = r_dec_fn(reg)


def search_alp(xmat, ymat, reg, units, alp, a_dec_fn, k, max_iter, v, el1):
    """
    Attempts to perform k-fold CV, starting with learning rate alpha.
    If divergence is detected, re-compute CV with new alpha that is
    calculated with a_dec_fn.
    """
    while True:
        if alp < 1e-7:
            print "cur_alp", alp
            print "Failed to converge"
            return (None, None)
            break
        try:
            cv_result = util.cross_validation(xmat, ymat, units,\
                        k, max_iter, alp, reg, v=v, l1=el1)
            errors, losses = cv_result
            mean_err = sum(errors)/len(errors)
            return mean_err, cv_result  
        except ValueError as e:
            alp = a_dec_fn(alp)

"""
Decreasing function helper functions
"""
def get_r_dec_fn(dec):
    return get_dec_fn(dec, 0.01)

def get_a_dec_fn(dec):
    return get_dec_fn(dec, 0.001)

def get_dec_fn(dec, linconst):
    def lin(s, c):
        return s-c
    def geo(s, c):
        return s*c
    if dec <= 10:
        return lambda s: lin(s, linconst*dec)
    else:
        return lambda s: geo(s, max(0.1, 1-(dec-10)*0.1))

"""
Data formatting helper functions
"""
def read_args(args):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-o", "--obj", dest="obj", type="int", default=1,\
                      help="Sets objective to optimize")   
    parser.add_option("-u", "--units", dest="units", type="str", default="0",\
                      help="Sets neural network units") 
    parser.add_option("-p", "--rdec", dest="rdec", type="float", default=5,\
                      help="Sets lambda decreasing function") 
    parser.add_option("-r", dest="reg", type="float", default=0.7,\
                      help="Sets L2 reg constant lambda_2")
    parser.add_option("-l", dest="l1", type="float", default=1.0,\
                      help="Sets L1 reg constant lambda_1")
    parser.add_option("-t", dest="treg", type="int", default=1,\
                      help="Sets total regularization searches")

    parser.add_option("-a", dest="alp", type="float", default=0.01,\
                      help="Sets learning rate alpha") 
    parser.add_option("-q", "--adec", dest="adec", type="int", default=12,\
                      help="Sets alpha decreasing function") 
    parser.add_option("-k", dest="k", type="int", default=10,\
                      help="Sets k for k-fold CV") 
    parser.add_option("-n", dest="max_iter", type="int", default=1000,\
                      help="Sets max iterations for gradient descent")
    parser.add_option("-v", dest="v", type="int", default=10,\
                      help="Sets verbosity coefficient")
    parser.add_option("-d", dest="donum", type="int", default=1,\
                      help="Sets number of random trials") 


    ops, otherjunk = parser.parse_args(args)
    if len(otherjunk) > 0:
        print "Received unknown arguments: {}".format(','.join(otherjunk))
        exit(1)
    return ops 

def format_units(u_str):
    if u_str == "0": return [3]
    return [int(u) for u in u_str.split(',')] + [3]

def format_data(obj_code):
    obj_map = {1:"Emotion", 2:"quanEval", 3:"qualEval"}
    metric = obj_map[obj_code]
    df_x = util.DataFrame((WB, WS_X), none_as=-2, val_type='float')
    center_mean_var(df_x)
    df_y = util.DataFrame((WB, WS_Y)) 
    df_y = df_y[:,metric]
    bucketize_mat(df_y) 
    valid_y = get_non_null_indexes(df_y)
    df_x = df_x[valid_y,:]
    df_y = df_y[valid_y,:]
    return (df_x, df_y, metric)

def center_mean_var(df):
    """ centers data to mean 0 and var 1 """
    x = df.get_entries(cpy=False)
    x = x.astype(np.float64)
    x = np.divide(x - np.mean(x, axis=0), np.std(x, axis=0))
    for r in range(df.nrows):
        for c in range(df.ncols):
            df[r,c] = x[r,c]

def get_non_null_indexes(df):
    """ returns non-null indexes of the first column of index. """
    return [i for i in range(df.nrows) if df.get_val(i,0) != df.none_val]

def fill_nulls(df, null_val=-1):
    for i in range(df.nrows):
        for j in range(df.ncols):
            if df.get_val(i,j) is None:
                df[i,j] = null_val
    
def bucketize_mat(df):
    col_names = df.get_col_names()
    if col_names.ndim == 0: 
        bucketize_col(df, col_names, Y_MAP[str(col_names)])
        return
    for hdr in col_names:
        if hdr in Y_MAP:
            bucketize_col(df, hdr, Y_MAP[hdr])

def bucketize_col(df, hdr, bounds):
    col_index = df.get_col_index(hdr)
    for i in range(df.nrows):
        df[i, col_index] = get_bucket(df.get_val(i,col_index), bounds)

def get_bucket(val, bounds):
    if val is None: return None
    for i in range(len(bounds)):
        if val < bounds[i]: return i
    return len(bounds)

    
if __name__ == "__main__":
    main(sys.argv)
