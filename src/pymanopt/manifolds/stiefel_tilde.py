from __future__ import division

import numpy as np
import numpy.linalg as la

from pymanopt.manifolds.manifold import Manifold

if not hasattr(__builtins__, "xrange"):
    xrange = range


class Stiefel_tilde(Manifold):
    """
    Factory class for the manifold of matrices with orthogonal columns. Initiation requires the dimensions
    n, p to be specified. Optional argument k allows the user to optimize over
    the product of k Stiefels.

    Elements are represented as n x p matrices (if k == 1), and as k x n x p
    matrices if k > 1.
    """

    def __init__(self, height, width):
        # Check that n is greater than or equal to p
        if height < width or width < 1:
            raise ValueError("Need n >= p >= 1. Values supplied were n = %d "
                             "and p = %d." % (height, width))
        # Set the dimensions of the Stiefel
        self._n = height
        self._p = width

        # Set dimension
        self._dim =  (self._n * self._p -
                               0.5 * self._p * (self._p + 1))

    @property
    def dim(self):
        return self._dim

    def __str__(self):
        return "Manifold of matrices with orthognal columns St_tilde(%d, %d)" % (self._n, self._p)

    
    def sym_tilde(self,M):
        return 0.5*(M+M.T) - np.diag(np.diag(M))


    def proj_perp(self,X,Z):
        p=X.shape[1]
        O = np.ones((p,p))
        D = np.sqrt(np.diag(np.diag((X.T@X))))
        D2 = D**2
        E = O@D2+D2@O
        F = 1./E
        L = self.sym_tilde(2*D@((la.inv(D)@X.T@Z)*F))
        Pi = X@L
        return Pi

    
    def proj(self, X, U):
        return U - self.proj_perp(X,U)


    # Retract to the manifold using a modified qr decomposition of X + G.
    def retr(self, X, G):


        # Calculate 'thin' qr decomposition of X + G
        Q1,R1 = la.qr(X+G)
        d = np.diag(np.diag(R1))
        Q = Q1@d
        return Q

    def norm(self, X, G):
        # Norm on the tangent space of the Stiefel is simply the Euclidean
        # norm.
        return np.linalg.norm(G)

    # Generate random point in the manifold using a modified qr decomposition of random normally distributed
    # matrix.
    def rand(self):
        X = np.random.randn(self._n, self._p)
        q, r = np.linalg.qr(X)
        return q@np.diag(np.diag(r))


    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def log(self, X, Y):
        raise NotImplementedError


    def pairmean(self, X, Y):
        raise NotImplementedError
