from typing import Union
import numpy as np
from numpy.random import randint
from scipy.linalg import circulant


class PolynomialTensor(object):

    def __init__(self, poly_mat: np.ndarray, modulus: int):
        if isinstance(poly_mat, list):
            self.poly_mat = np.asarray(poly_mat)
        else:
            self.poly_mat = poly_mat

        if self.poly_mat.ndim == 1:
            self.poly_mat = np.expand_dims(self.poly_mat, 0)
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
            max_value = modulus - 1
        assert all([s > 0 for s in matrix_shape]) and poly_len > 0
        return PolynomialTensor(
            randint(min_val, max_value + 1,
                    (*matrix_shape, poly_len)) % modulus, modulus)

    def __repr__(self) -> str:
        return repr(self.poly_mat)

    def __add__(self, other: "PolynomialTensor") -> "PolynomialTensor":
        assert self.modulus == other.modulus and (self.shape == other.shape
                                                  or self.shape[0][0] == 1
                                                  or other.shape[0][0] == 1)
        return self._create((self.poly_mat + other.poly_mat), self.modulus)

    def __sub__(self, other: "PolynomialTensor") -> "PolynomialTensor":
        assert self.modulus == other.modulus and (self.shape == other.shape
                                                  or self.shape[0][0] == 1
                                                  or other.shape[0][0] == 1)
        return self._create((self.poly_mat - other.poly_mat), self.modulus)

    def __mul__(
            self, other: Union[float, int,
                               "PolynomialTensor"]) -> "PolynomialTensor":
        assert isinstance(
            other, (float, int, PolynomialTensor)
        ), f"Wrong type, {type(other)} instead of int or float or PolynomialTensor"

        if isinstance(other, PolynomialTensor):
            assert self.shape == other.shape and self.modulus == other.modulus
            return self._create((self.poly_mat * other.poly_mat), self.modulus)

        return self._create((self.poly_mat * other), self.modulus)

    def __rmul__(self, other: float | int) -> "PolynomialTensor":
        return self.__mul__(other)

    @staticmethod
    def _generate_multiplication_matrix(polys: np.ndarray,
                                        axis=0) -> np.ndarray:
        if polys.ndim == 1:
            rows, cols = (1, 1)
        elif polys.ndim == 2:
            rows, cols = (polys.shape[0], 1)
        elif polys.ndim == 3:
            rows, cols = polys.shape[:2]
        else:
            raise NotImplementedError()

        blocks = [
            circulant(vec) * ((np.tri(polys.shape[-1]) * 2) - 1)
            for vec in polys.reshape(-1, (polys.shape[-1]))
        ]
        if axis == 1:
            return np.vstack([
                np.hstack(blocks[i * cols:(i + 1) * cols]) for i in range(rows)
            ])
        return np.vstack(
            [np.hstack(blocks[i * rows:(i + 1) * rows]) for i in range(cols)])

    def mul_matrix(self, axis=0) -> np.ndarray:
        return self._generate_multiplication_matrix(self.poly_mat, axis)

    def __matmul__(self, other: "PolynomialTensor") -> "PolynomialTensor":
        assert self.shape[1] == other.shape[1] and self.modulus == other.modulus
        assert self.modulus == other.modulus
        end_shape = (1, self.shape[1]) if self.poly_mat.ndim == 2 else (
            self.poly_mat.shape[0], self.shape[1])

        # If its bigger then tow, the flatten function is the wrong one to use
        if other.poly_mat.ndim > 2:
            raise NotImplementedError()

        if other.poly_mat.shape[0] == 1:
            return self._create(
                (self.mul_matrix() @ other.poly_mat.flatten().repeat(
                    self.poly_mat.shape[0], 0)).reshape(end_shape),
                self.modulus)
        return self._create(
            (self.mul_matrix() @ other.poly_mat.flatten()).reshape(end_shape),
            self.modulus)

    def __mod__(self, other: int) -> "PolynomialTensor":
        assert isinstance(other, int)
        return self._create(self.poly_mat % other, self.modulus)

    def change_modulus(self, mod: int) -> "PolynomialTensor":
        return self._create(self.poly_mat % mod, mod)

    def __pow__(self, other: int) -> "PolynomialTensor":
        original_shape = self.poly_mat.shape
        polynomials = [
            PolynomialTensor([poly], self.modulus)
            for poly in self.poly_mat.reshape((-1, original_shape[-1]))
        ]

        def _multi(poly: PolynomialTensor, amount: int) -> PolynomialTensor:
            if amount == 1:
                return poly
            return poly @ _multi(poly, amount - 1)

        return self._create(
            np.asarray([(_multi(poly, other)).poly_mat[0]
                        for poly in polynomials]).reshape(original_shape) %
            self.modulus, self.modulus)


class Polynomial(PolynomialTensor):

    @staticmethod
    def random_polynomial(poly_len: int,
                          modulus: int,
                          min_val: int = 0,
                          max_value: int = None) -> "Polynomial":
        while True:
            pt = super(Polynomial, Polynomial).random_polynomial_matrix(
                poly_len, modulus, (1, ), min_val, max_value)
            if (pt.poly_mat == 0).all():
                continue
            return Polynomial(pt.poly_mat, pt.modulus)

    @staticmethod
    def _create(poly_mat: np.ndarray, modulus: int) -> "Polynomial":
        return Polynomial(poly_mat, modulus)

    def __repr__(self) -> str:
        return str(np.polynomial.Polynomial(self.poly_mat[0]))


def poly_round(poly: PolynomialTensor) -> Polynomial:
    return Polynomial(np.round(poly.poly_mat), poly.modulus)


if __name__ == "__main__":
    # Some tests
    s = PolynomialTensor(np.asarray([[3, 3, 3], [98, 3, 0]]), 100)
    A = PolynomialTensor(
        np.asarray([[[53, 83, 66], [27, 29, 34]], [[16, 25, 87], [48, 96,
                                                                  0]]]), 100)
    e = PolynomialTensor(np.asarray([[97, 99, 99], [98, 98, 1]]), 100)

    b = (A @ s + e) % 100
    assert (np.array([[53., 32., 24.], [14., 12.,
                                        73.]]) == b.poly_mat).all(), b

    m = np.asarray([np.random.randint(0, 2, 3)])
    m_p = PolynomialTensor(m * (100 // 2), 100)

    r = PolynomialTensor.random_polynomial_matrix(3,
                                                  100, (2, ),
                                                  min_val=0,
                                                  max_value=1)
    e1 = PolynomialTensor.random_polynomial_matrix(3,
                                                   100, (2, ),
                                                   min_val=-3,
                                                   max_value=3)
    e2 = PolynomialTensor.random_polynomial_matrix(3,
                                                   100, (1, ),
                                                   min_val=-3,
                                                   max_value=3)

    u = (A.T @ r) + e1
    v = b.T @ r + e2 + m_p
    m_d = np.rint((v - (s.T @ u)).poly_mat * (1 / (100 // 2))) % 2

    s = Polynomial(np.asarray([[1, 0, 1]]), 100)
    A = Polynomial(np.asarray([[28, 56, 1]]), 100)
    e = Polynomial(np.asarray([[1, 99, 2]]), 100)

    b = (A @ s + e) % 100

    assert (b.poly_mat == np.asarray([[73, 54, 31]])).all(), b

    message = Polynomial(np.asarray([[1, 1, 0]]), 100)
    r = Polynomial(np.asarray([[0, 1, 1]]), 100)
    e1 = Polynomial(np.asarray([[98, 0, 98]]), 100)
    e2 = Polynomial(np.asarray([[1, 0, 0]]), 100)
    q_half = np.floor(100 / 2)

    u = A @ r + e1
    v = b @ r + e2 + message * q_half

    decrypt = np.round((1 / q_half * (v - s @ u)).poly_mat) % 2

    assert all((message.poly_mat == decrypt)[0])

    s = PolynomialTensor(np.asarray([[3, 3, 3], [98, 3, 0]]), 100)
    assert (np.array([[91., 9., 27.], [4., 88., 9.]]) == (s**2).poly_mat).all()

    i1 = np.array([1, 2, 3])
    o1 = np.array([[1, -3, -2],
                   [2, 1, -3],
                   [3, 2, 1]]) # yapf: disable
    assert np.array_equal(PolynomialTensor._generate_multiplication_matrix(i1),
                          o1)
    i2 = np.array([[1, 2, 3]])
    o2 = np.array([[1, -3, -2],
                   [2,  1, -3],
                   [3,  2,  1]]) # yapf: disable
    assert np.array_equal(PolynomialTensor._generate_multiplication_matrix(i2),
                          o2)
    i3 = np.array([[1, 2, 3], [4, 5, 6]])
    o3 = np.array([[ 1, -3, -2,  4, -6, -5],
                   [ 2,  1, -3,  5,  4, -6],
                   [ 3,  2,  1,  6,  5,  4]]) # yapf: disable
    assert np.array_equal(PolynomialTensor._generate_multiplication_matrix(i3),
                          o3)
    i4 = np.array([[[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [0, 1, 2]]]) # yapf: disable
    o4 = np.array([[ 1., -3., -2., 4., -6., -5.],
                   [ 2.,  1., -3., 5.,  4., -6.],
                   [ 3.,  2.,  1., 6.,  5.,  4.],
                   [ 7., -9., -8., 0., -2., -1.],
                   [ 8.,  7., -9., 1.,  0., -2.],
                   [ 9.,  8.,  7., 2.,  1.,  0.]],) # yapf: disable
    assert np.array_equal(PolynomialTensor._generate_multiplication_matrix(i4),
                          o4)

    a = PolynomialTensor(np.asarray([[3, 3, 3], [98, 3, 0]]), 100)
    b = PolynomialTensor(np.asarray([[1, 1, 1]]), 100)
    assert ((a @ b).poly_mat == np.array([[92., 104., 110.]])).all()
