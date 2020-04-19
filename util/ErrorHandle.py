#!/usr/local/bin/python
import inspect

def report_error_source(fn):
    print "\nERROR found at function '{}'".format(fn)

def validate_input(condition, msg, verbose=False):
    fn = inspect.stack()[1][3]
    if condition and verbose:
        print "{} input is valid".format(fn)
    if not condition:
        report_error_source(fn)
        print "\tInvalid input:\n\t{}\n".format(msg)
        exit(1)
