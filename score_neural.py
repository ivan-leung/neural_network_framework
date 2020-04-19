#!/usr/bin/env python
import sys
import util
import neural
import test_neural as tn

# directory of results to print from
DIR = "ScoredSet"
WB = "Data/v9.xlsx"
WS_X = "Scored"
WS_Y = "y"
OBJ_MAP = {1:"Emotion", 2:"quanEval", 3:"qualEval"}

def format_data(obj_code):
    obj_str = OBJ_MAP[obj_code]
    df_x = util.DataFrame((WB, WS_X), none_as=-2)
    neural.center_mean_var(df_x)
    df_y = util.DataFrame((WB, WS_Y))
    df_y = df_y[:,obj_str]
    for i in range(df_y.nrows):
        df_y[i,0] = neural.get_bucket(df_y.get_val(i,0), tn.Y_MAP[obj_code])
    valid_y = neural.get_non_null_indexes(df_y)
    df_x = df_x[valid_y,:]
    df_y = df_y[valid_y,:]
    return (df_x, df_y)

def get_cv_error(cv_errs):
    return util.t_confint(cv_errs, 0.8)[0]

def main(argv):

    ops = tn.read_args(argv[1:])
    if ops.report != 0:
        tn.print_results(DIR)
        return
    df_x, df_y = format_data(ops.obj)
    if ops.units == "0":
        units = [3]
    else:
        units = [int(u) for u in ops.units.split(",")] + [3]
    v, k, max_iter, alp, l1, reg = ops.v, ops.k, ops.max_iter, ops.a, ops.el1, ops.reg
    a_dec_fn = neural.get_a_dec_fn(ops.adec)
    searcher = util.SearchGridManager([(0,l1),(0,reg)], tn.train_one,\
                                      get_cv_error, min) 
    searcher.search(4,[3,3],3, df_x.get_entries(), df_y.get_entries(), \
                    units, max_iter, k, alp, v, a_dec_fn)
    r = (searcher.get_samples(), ["NA"], units, max_iter)
    tn.output_results(r, ops.obj, "ScoredSet")
    
if __name__ == "__main__":
    main(sys.argv)
