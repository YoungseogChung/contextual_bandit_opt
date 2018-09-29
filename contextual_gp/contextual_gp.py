"""
Gaussian process that takes context into account.
Implementation as per Krause, Ong, et. al. 2011.
"""

import numpy as np

class ContextualGP(object):
    """
    GP that takes context into account.
    """
    def __init__(self, dim, bound, joint_kernel, noise):
        self.dim = dim # This is dimension of context plus params.
        self.bound = bound # Assume space is compact&convex on [0, bound]^dim
        # kernel should be in the form k((context1, param1), (context2, param2))
        self.kernel = joint_kernel
        self.noise = noise
        self.prev_params = [] # Stored as tuples (context, input)
        self.prev_responses = None # ndarray of shape (n, 1)
        self.prev_cov = None
        self.t = 0

    def add_observation(self, context, param, response):
        """Add a single observation."""
        self.prev_params.append((context, param))
        if self.prev_responses is None:
            self.prev_responses = np.array([[response]])
        else:
            self.prev_responses = np.append(self.prev_responses, [[response]],
                                            axis=0)
        self.t += 1
        # Set to None because now invalid. Lazy update done in get_ucb_val.
        self.prev_cov = None

    def get_ucb_val(self, context, param):
        """Get upper confidence bound for a (context, param) point."""
        # Get terms used in the expressions.
        prev_cov = self._get_prev_cov()
        interaction = self._get_interaction_term(context, param)
        new_cov = self._get_new_cov(context, param)
        beta = self._get_beta_coef()

        intermediate = np.dot(interaction, prev_cov)
        mu = np.dot(intermediate, self.prev_responses)
        var = new_cov - np.dot(intermediate, np.transpose(interaction))
        return float(mu + beta * var)

    def _get_prev_cov(self):
        """Lazy calculation of prev cov term."""
        if self.prev_cov is None:
            self.prev_cov = self._update_prev_cov()
        return self.prev_cov

    def _update_prev_cov(self):
        """Update (K(X, X) - etaI)^-1 matrix after we have seen a new obs."""
        cov = np.ndarray(shape=(self.t, self.t), dtype=float, order='F')
        for i in xrange(self.t):
            for j in xrange(self.t):
                cov[i, j] = self.kernel(self.prev_params[i],
                                        self.prev_params[j])
        cov -= self.noise * np.eye(self.t)
        # TODO: Taking inverse here is really bad, do Cholesky or something.
        return np.linalg.inv(cov)

    def _get_interaction_term(self, context, param):
        """Get K(X^*, X)."""
        # TODO: Right now this function and the next only takes one new pt.
        term = np.ndarray(shape=(1, self.t), dtype=float, order='F')
        new_pt = (context, param)
        for j in xrange(self.t):
            term[0, j] = self.kernel(new_pt, self.prev_params[j])
        return term

    def _get_new_cov(self, context, param):
        """Get K(X^*, X^*)"""
        pt = (context, param)
        return self.kernel(pt, pt)

    def _get_beta_coef(self):
        """Get beta_t based on theorem 1 part 2 in Krause, et. al"""
        t2 = self.t ** 2
        beta = 2 * np.log(t2 * 2 * np.pi ** 2 / 0.3)
        beta += 2 * self.dim * np.log(t2 * self.dim * self.bound)
        return beta

def tmp_kern(pt1, pt2):
    # TODO: Remove this when we have a real kernel, this is just a linear
    # kernel for now.
    v1 = np.append(pt1[0], pt1[1])
    v2 = np.append(pt2[0], pt2[1])
    return np.dot(v1, v2)

if __name__ == '__main__':
    cgp = ContextualGP(5, 1, tmp_kern, 0)
    for _ in xrange(5):
        cgp.add_observation([np.random.random() for _ in xrange(2)],
                            [np.random.random() for _ in xrange(3)],
                            np.random.random())
    print cgp.get_ucb_val([np.random.random() for _ in xrange(2)],
                          [np.random.random() for _ in xrange(3)])
