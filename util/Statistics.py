#!/usr/bin/env python
from ErrorHandle import validate_input
import copy
import numpy as np
import scipy.stats 

def t_confint(x, p):
    """ Given list or array of values, return the 100p%  t confindence interval"""
    x = np.array(x)
    se = scipy.stats.sem(x)
    return scipy.stats.t.interval(p, x.size-1, np.mean(x), se)

def col_mean(arr, wt=None):
    """Return the column mean of array, with weights wt if given.
    None entries are ignored.
    """
    arr = np.array(arr)
    validate_input(len(arr.shape) == 2, "function admits an array of 2 dims")
    return [mean(arr[:,i]) for i in range(arr.shape[1])]

def mean(vec, wt=None):
    """Return the mean of vec, with weights wt if given.
    None entries are ignored.
    """
    if wt is not None:
        validate_input(len(vec)==len(wt),"vec and wt must have the same length")
        validate_input(all(isinstance(x, (int,float,long)) for x in wt), "wt must be numeric")

    indexes = [i for i,val in enumerate(vec) if val is not None]
    vec = [vec[i] for i in indexes]
    if wt is not None:
        wt = [wt[i] for i in indexes]
    if len(vec) == 0:
        return None
    return np.average(vec, weights=wt)
    

def percentilize(vec):
    """Return a vector of percentile for given list of values.
        Ignores None elements in computation.
        Returns None if there are no non-None elements.
    """
    vec_sorted = sorted(copy.deepcopy([elem for elem in vec if elem is not None]))
    vec_len = float(len(vec_sorted))
    if vec_len == 0:
        return None
    p_map = {None:None}
    for i,val in enumerate(vec_sorted):
        p_map[val] = (i+1)/vec_len
    return [p_map[val] for val in vec]

def percentile(p, vec, p_vec=None):
    """Return the smallest value in vec with percentile >= p.
        Input of p_vec, the percentilzed vector, will save computation time
        if percentile function is called multiple times on the same vector.
    """
    if p_vec is None:
        p_vec = percentilize(vec)
    if len(p_vec) == 0:
        return None
    p_vec_cpy = copy.deepcopy(p_vec)
    p_vec = set(p_vec)
    p_vec.discard(None)
    p_vec = sorted(list(p_vec))
    if isinstance(p, (int, float)):
        index = binary_search(p, p_vec, False)
        real_index = p_vec_cpy.index(p_vec[index])
        return vec[real_index]
    elif isinstance(p, list):
        index_list = [binary_search(pt, p_vec, False) for pt in p]
        real_index_list = [p_vec_cpy.index(p_vec[index]) for index in index_list]
        return [vec[real_index] for real_index in real_index_list]

def percentile_range(l_bound, u_bound, vec, p_vec=None):
    """Return the [lower, upper) value interval that falls within
        the percentile interval [l_bound, u_bound)
    """
    if p_vec is None:
        p_vec = percentilize(vec)
    l_index, l_val = percentile(l_bound, vec, p_vec)
    u_index, u_val = percentile(u_bound, vec, p_vec)
    return (l_val, u_val)    

def bucketize(vec, buckets):
    """Bucketize a vector according to specified buckets

    Key Arguments:
    vec -- a list of values. May contain None's.
    buckets -- int of number of buckets or vector of bounds.
        If buckets is int, elements that partitioned into the bucket that their
        percentile rank falls into, i.e. floor(percentile(val)*buckets).
        If buckets is list, values of buckets are int/float that mark the bounds
        of each bucket. value x is in bucket i if buckets[i-1] <= x < buckets[i].
        One additional overflow bucket for all values > upper_bound is included
    Returns:
    tuple (assignment, bucket_map):
    assignment -- list of len(vec) whose elements indicate bucket number
                  the matching element in vec is assigned to.
    bucket_map -- dict with bucket number as key and 2-tuple of boundary
    """
    validate_input(isinstance(vec,list) and isinstance(buckets,(int,list)),\
                   "function takes a list of values, and either \
                    integer bucket number or list of boundaries")

    assignments = [None for _ in range(len(vec))]
    if isinstance(buckets, int):
        buckets = [i/float(buckets) for i in range(1,buckets)]
        buckets = percentile(buckets, vec, percentilize(vec))
         
    for i,val in enumerate(vec):
        if val is None:
            continue
        
        index = binary_search(val, buckets, True)
        if index is not None:
            assignments[i] = index+1
        else:
            index = binary_search(val, buckets, False)
            assignments[i] = index
    bucket_map = {0:(None,buckets[0]), len(buckets):(buckets[-1],None)}
    for i in range(1,len(buckets)):
        bucket_map[i] = (buckets[i-1],buckets[i])
    return (assignments, bucket_map)
        

    
def binary_search(val, vec, exact=True):
    """Binary search.
    if exact, returns index of val in vec, and None otherwise.
    if not exact, returns smallest index of vec where the element >= val.
    vec must not have any Null values, and each element is unique.
    """ 
    validate_input(isinstance(val,(float,int)),"val must be numeric")
    validate_input(isinstance(vec,list),"vec msut be a list")
    return binary_search_helper(val, vec, 0, exact)

def binary_search_helper(val, vec, offset, exact):
    if len(vec) == 1:
        if val == vec[0]:
            return offset
        elif not exact:
            if vec[0] > val:
                return offset
            elif vec[0] < val:
                return offset+1
        else:
            return None
    mid_pt = len(vec)/2
    if vec[mid_pt] == val:
        return mid_pt + offset
    elif vec[mid_pt] < val:
        return binary_search_helper(val, vec[mid_pt:], offset+mid_pt, exact)
    elif vec[mid_pt] > val:
        return binary_search_helper(val, vec[:mid_pt], offset, exact)
'''
x = [1,1,1,2,2,5,0,5,None,7,10] 
print "======== BEGIN ========="
print x
print percentilize(x)
print percentile([0,0.1, 0.2, 0.3, 0.4, 0.9,1], x)
print bucketize(x, 3)
print "_______ end _______"
'''

