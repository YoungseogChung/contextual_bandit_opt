"""
Functions to run the Bayesian Optimization for fixed contexts.
"""

import sys
import numpy as np

from function_context_wrapper import FunctionContextWrapper
from synthetic_contextual_fcn import hartmann6

from context_mgr import ContextManager

sys.path.append('/home/ian/Documents/projects/dragonfly')
# sys.path.append('/home/master/programs/dragonfly')
from dragonfly import maximise_function

def optimize_fixed_context(fixed_func, max_capital, domain_bds):
    # TODO: I think that this flow should be improved if/when class for
    # managing a function's context is created.
    opt_val, opt_pt, _ = maximise_function(fixed_func, max_capital,
                                           domain_bounds=domain_bds)
    return (opt_val, opt_pt)

def optimize_over_contexts(func_wrapper, ctxs, max_capital, domain_bds,
                           ctx_manager):
    """
    Finds optimum for each of the given contexts.
    func_wrapper: Wrapped function to be optimized.
    ctxs: List of contexts.
    Returns: ContextManager loaded with results.
        """
    for ctx in ctxs:
        fixed_func = func_wrapper.get_contexed_func(ctx)
        opt_val, opt_pt = optimize_fixed_context(fixed_func, max_capital,
                                                 domain_bds)
        ctx_manager.add(ctx, (opt_val, opt_pt))
    return ctx_manager

if __name__ == '__main__':
    contexts = [[np.random.rand(4),
                 np.random.rand(4, 6),
                 np.random.rand(4, 6)] for _ in xrange(3)]
    ctx_manager = ContextManager()
    domains = [[0, 1] for _ in xrange(6)]
    full_func = FunctionContextWrapper(hartmann6)
    tmp = full_func.get_contexed_func(contexts[0])
    print tmp(np.random.rand(1, 6))
    print hartmann6(np.random.rand(1, 6), contexts[0][0], contexts[0][1], contexts[0][2])
    optimize_over_contexts(full_func, contexts, 5, domains, ctx_manager)
    ctx_manager.print_stored()
