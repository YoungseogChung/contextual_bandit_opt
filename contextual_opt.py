"""
Functions to run the Bayesian Optimization for fixed contexts.
"""

import sys
import numpy as np

from function_context_wrapper import FunctionContextWrapper
from synthetic_contextual_fcn import hartmann6

sys.path.append('/home/ian/Documents/projects/dragonfly')
from dragonfly import maximise_function

def optimize_fixed_context(fixed_func, max_capital, domain_bds):
    # TODO: I think that this flow should be improved if/when class for
    # managing a function's context is created.
    opt_val, opt_pt, _ = maximise_function(fixed_func, max_capital, domain_bounds=domain_bds)
    return (opt_val, opt_pt)

def optimize_over_contexts(func_wrapper, ctxs, max_capital, domain_bds):
    """
    Finds optimum for each of the given contexts.
    func_wrapper: Wrapped function to be optimized.
    ctxs: List of contexts.
    Returns: ContextManager loaded with results.
        """
    # ctx_manager = ContextManager() # TODO: implement this
    for ctx in ctxs:
        fixed_func = func_wrapper.get_contexed_func(ctx)
        opt_val, opt_pt = optimize_fixed_context(fixed_func, max_capital, domain_bds)
        # ctx_manager.add(ctx, (opt_val, opt_pt))
        print opt_val, opt_pt
    # return ctx_manager

if __name__ == '__main__':
    contexts = [[np.random.rand(4, 1),
                 np.random.rand(4, 6),
                 np.random.rand(4, 6)] for _ in xrange(3)]
    domains = [[0, 1] for _ in xrange(6)]
    full_func = FunctionContextWrapper(hartmann6)
    optimize_over_contexts(full_func, contexts, 5, domains)
