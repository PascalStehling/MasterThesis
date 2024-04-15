import numpy as np
from numpy.random import randint
from scipy.linalg import circulant


class PolynomialTensor(object):

    def __init__(self, poly_mat: np.ndarray, modulus: int):
        self.poly_mat = poly_mat
        self.modulus = modulus

    @staticmethod
    def _create(poly_mat: np.ndarray, modulus: int) -> "PolynomialTensor":
        return PolynomialTensor(poly_mat, modulus)

    @property
    def shape(self):
        return self.poly_mat.shape[:-1], self.poly_mat.shape[-1]

    def transpose(self) -> "PolynomialTensor":
        return self._create(np.transpose(self.poly_mat, (1, 0, 2)),
                            self.modulus) if self.poly_mat.ndim == 3 else self

    @property
    def T(self) -> "PolynomialTensor":
        return self.transpose()

    @staticmethod
    def random_polynomial_matrix(poly_len: int,
                                 modulus: int,
                                 matrix_shape: tuple = (),
                                 min_val: int = 0,
                                 max_value: int = None) -> "PolynomialTensor":
        """
        Generate an random polynomial of shape in the ring R^{rows x cols}_{modulus} with values
        evenly retrieved from min_val (default 0) to max_val (default modulus)
        """
        if max_value is None:
            max_value = modulus
        assert all([s > 0 for s in matrix_shape]) and poly_len > 0
        return PolynomialTensor(
            randint(min_val, max_value, (*matrix_shape, poly_len)) % modulus,
            modulus)

    def __repr__(self) -> str:
        return repr(self.poly_mat)

    def __add__(self, other: "PolynomialTensor") -> "PolynomialTensor":
        assert self.shape == other.shape and self.modulus == other.modulus
        return self._create((self.poly_mat + other.poly_mat) % self.modulus,
                            self.modulus)

    def __sub__(self, other: "PolynomialTensor") -> "PolynomialTensor":
        assert self.shape == other.shape and self.modulus == other.modulus
        return self._create((self.poly_mat - other.poly_mat) % self.modulus,
                            self.modulus)

    def __mul__(self, other: float) -> "PolynomialTensor":
        assert isinstance(other, float)
        return self._create(self.poly_mat * other, self.modulus)

    def __rmul__(self, other: float) -> "PolynomialTensor":
        return self.__mul__(other)

    @staticmethod
    def _generate_multiplication_matrix(polys: np.ndarray) -> np.ndarray:
        assert polys.ndim > 1, "Not Implemented Error"
        if polys.ndim == 2:
            rows = 1
            cols = polys.shape[0]
        else:
            rows, cols = polys.shape[0:2]
        blocks = [
            circulant(vec) * ((np.tri(polys.shape[-1]) * 2) - 1)
            for vec in polys.reshape(-1, (polys.shape[-1]))
        ]
        return np.vstack(
            [np.hstack(blocks[i * cols:(i + 1) * cols]) for i in range(rows)])

    @property
    def mul_matrix(self) -> np.ndarray:
        return self._generate_multiplication_matrix(self.poly_mat)

    def __matmul__(self, other: "PolynomialTensor") -> "PolynomialTensor":
        assert self.shape[1] == other.shape[1] and self.modulus == other.modulus
        assert self.modulus == other.modulus
        end_shape = (1, self.shape[1]) if self.poly_mat.ndim == 2 else (
            self.poly_mat.shape[0], self.shape[1])
        return self._create(
            (self.mul_matrix @ other.poly_mat.flatten()).reshape(end_shape) %
            self.modulus, self.modulus)


class Polynomial(PolynomialTensor):

    @staticmethod
    def random_polynomial(poly_len: int,
                          modulus: int,
                          min_val: int = 0,
                          max_value: int = None) -> "Polynomial":
        pt = super(Polynomial, Polynomial).random_polynomial_matrix(
            poly_len, modulus, (1, ), min_val, max_value)
        return Polynomial(pt.poly_mat, pt.modulus)

    @staticmethod
    def _create(poly_mat: np.ndarray, modulus: int) -> "Polynomial":
        return Polynomial(poly_mat, modulus)

    def __repr__(self) -> str:
        return str(np.polynomial.Polynomial(self.poly_mat[0]))
