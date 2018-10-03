"""
Instead of essentially running separate, independent BO problems for each
context, it would be more beneficial potentially to assume some amount of
smoothness between the contexts. Ideas taken from Krause, Ong, et. al. 2011.
"""

import sys
import numpy as np
from scipydirect import minimize

from contextual_gp import ContextualGP

sys.path.append('../dragonfly_loop')
from function_context_wrapper import FunctionContextWrapper
from synthetic_contextual_fcn import hartmann6
from context_mgr import ContextManager

def create_controller(func_wrapper, contexts, kernel, capital, param_dim,
                      total_dim, bound, warm_up_pts=5):
    cgp = ContextualGP(total_dim, bound, kernel, 0) # Say noise is 0 for now.
    unrolled_ctxs = [_unroll_context(ctx) for ctx in contexts]
    best_pts = [None for _ in xrange(len(contexts))]
    best_vals = [-float('inf') for _ in xrange(len(contexts))]
    param_bds = [[0, bound] for _ in xrange(param_dim)]
    # Add warm up points for each of the contexts.
    for i, ctx in enumerate(contexts):
        f = func_wrapper.get_contexed_func(ctx)
        for _ in xrange(warm_up_pts):
            rand_pt = np.random.rand(param_dim)
            response = f(rand_pt)
            cgp.add_observation(unrolled_ctxs[i], rand_pt, response)
    # Find best points for each of the contexts.
    for _ in xrange(capital):
        for i, ctx in enumerate(contexts):
            nxt = _find_next_pt(cgp, unrolled_ctxs[i], param_bds)
            val = func_wrapper.get_contexed_func(ctx)(nxt)
            if best_vals[i] < val:
                best_vals[i] = val
                best_pts[i] = nxt
            cgp.add_observation(unrolled_ctxs[i], nxt, val)
    # Add best points to the context manager
    mgr = ContextManager()
    for i, ctx in enumerate(contexts):
        mgr.add(ctx, (best_vals[i], best_pts[i]))
    return mgr

def contextual_kernel(pt1, pt2):
    """Kernel that is linear in context and square-exp in params."""
    ctx1, ctx2 = pt1[0], pt2[0]
    param1, param2 = pt1[1], pt2[1]
    to_return = np.dot(ctx1, ctx2)
    to_return += np.exp(-1/2 * np.norm(param1 - param2))
    return to_return

def _find_next_pt(cgp, context, bounds):
    opt_f = lambda p: -1 * cgp.get_ucb_val(context, p)
    return minimize(opt_f, bounds=bounds).x

def _unroll_context(context):
    return np.append([], [c.ravel() for c in context])

if __name__ == '__main__':
    f_wrapper = FunctionContextWrapper(hartmann6)
    contexts = [[np.random.rand(4),
                 np.random.rand(4, 6),
                 np.random.rand(4, 6)] for _ in xrange(3)]
    mgr = create_controller(f_wrapper, contexts, contextual_kernel, 10,
                            6, 34, 1)
    mgr.print_stored()
