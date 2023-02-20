import numpy as np
import numpy.linalg as la

from pymanopt.manifolds.manifold import RiemannianSubmanifold

class Stiefel_tilde(RiemannianSubmanifold):
    r"""The Stiefel_tilde manifold.

    The Stiefel_tilde manifold :math:`\St(n, p)` is the manifold of ``n x
    p`` matrices with orthognal columns.
    A point :math:`\vmX \in \widetilde{St}(n, p)` therefore satisfies the condition
    :math:`\transp{\vmX}\vmX is diagonal`.
    Points on the manifold are represented as arrays of shape ``(n, p)`` 

    The metric is the usual Euclidean metric on :math:`\R^{n \times p}` which
    turns :math:`\widetilde{St}(n, p)` into a Riemannian submanifold.

    Args:
        n: The number of rows.
        p: The number of columns.

    Note:
        The computations of the projections and retraction can be found in [FMLBM2023]_.

    """

    def __init__(self, n: int, p: int):
        self._n = n
        self._p = p

        # Check that n is greater than or equal to p
        if n < p or p < 1:
            raise ValueError(
                f"Need n >= p >= 1. Values supplied were n = {n} and p = {p}"
            )
        name = f"Stiefel tilde manifold St_tilde({n},{p})"
        dimension = int((n * p - p * (p + 1) / 2))
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return np.sqrt(self._p)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def sym_tilde(self,M):
        return 0.5*(M+M.T) - np.diag(np.diag(M))


    def projection_to_normal(self,X,Z):
        p=X.shape[1]
        O = np.ones((p,p))
        D = np.sqrt(np.diag(np.diag((X.T@X))))
        D2 = D**2
        E = O@D2+D2@O
        F = 1./E
        L = self.sym_tilde(2*D@((la.inv(D)@X.T@Z)*F))
        Pi = X@L
        return Pi

    
    def projection(self, point, vector):
        return vector - self.proj_perp(point,vector)

    to_tangent_space = projection

    def retraction(self, point, tangent_vector):
        Q1,R1 = la.qr(point+tangent_vector)
        d = np.diag(np.diag(R1))
        Q = Q1@d
        return Q

    def norm(self, point, tangent_vector):
        return np.linalg.norm(tangent_vector)

    def random_point(self):
        q, r = np.linalg.qr(np.random.randn(self._n, self._p))
        return q@np.diag(np.diag(r))


    def random_tangent_vector(self, point):
        vector = np.random.normal(size=point.shape)
        vector = self.projection(point, vector)
        return vector / np.linalg.norm(vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def zero_vector(self, point):
        return np.zeros((self._n, self._p))
