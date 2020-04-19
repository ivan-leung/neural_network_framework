import sys
import inspect

def usage(msg):
    parent =  inspect.stack()[1][3]
    print "\nUsage of {}:\n{}\n".format(parent, msg)

def array_str(arr):
    s = ''
    for r in arr:
        s += str(r) + '\n'
    return s

def printstr(obj, num_fig=5):
    s = ""
    if isinstance(obj, list):
        for i,elem in enumerate(obj):
            if isinstance(elem, list):
                s += str(i) + ": " + printstr(elem) + "\n "
            else:
                s += printstr(elem) + ", "
        s = "[" + s[0:len(s)-2] + "]"
    elif isinstance(obj, tuple):
        for i,elem in enumerate(obj):
            s += printstr(elem) + ", "
        s = "(" + s[0:len(s)-2] + ")"
    elif isinstance(obj, (int)):
        s += str(obj)
    elif isinstance(obj, (float,long)):
        s += str(round(obj,num_fig))
    else:
        s += str(obj)
    return s
    


     
