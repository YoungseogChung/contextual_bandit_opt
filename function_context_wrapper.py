"""
Class to wrap a function, making it easy to fix contexts.
"""

class FunctionContextWrapper(object):

    def __init__(self, func):
        """Constructor: the function passed in must be in the form
        func(X, context_1, context_2, ...)
        """
        self.func = func

    def get_contexed_func(self, context):
        """Returns a new function where our original function is conditioned
        on the context.
        context: List of other context variables to be loaded in.
        """
        return lambda x: self.func(x, *context)


if __name__ == '__main__':
    def eg_func(x, ctx1, ctx2, ctx3, ctx4):
        print 'Input:', x
        print 'Contexts:', ctx1, ctx2, ctx3, ctx4
    wrapped = FunctionContextWrapper(eg_func)
    wrapped.get_contexed_func([2, 3, 4, 5])(1)
