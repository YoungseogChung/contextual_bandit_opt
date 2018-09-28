import numpy as np

def ConstructKey(input_list):
    """
    Takes an input list of numpy arrays
    Reshape each array into 1D
    Append each array into a 1D list
    Construct a tuple with the 1D list
    """
    out_list = []
    for i in input_list:
        out_list = out_list + (i.flatten()).tolist()

    return tuple(out_list)

class ContextManager():
    def __init__(self):
        self.opt_data = {}

    def add(self, ctx, (opt_val, opt_pt)):
        self.opt_data.update({ConstructKey(ctx): (opt_val, opt_pt)})

    def print_stored(self):
        for context, val in self.opt_data.iteritems():
            opt_val, opt_pt = val
            print '---------------------------------------------'
            print 'Context:', context
            print 'Opt Value:', val[0]
            print 'Opt Pt:', val[1]
