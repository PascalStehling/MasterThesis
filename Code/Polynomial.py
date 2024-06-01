from functools import reduce
from itertools import chain
from operator import add, mul, sub, truediv
from random import randint
from typing import Callable, Sized
from xmlrpc.client import boolean
import numpy as np


def matmul(a, b):
    zip_b = list(zip(*b))
    return [[
        reduce(add, (ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b)))
        for col_b in zip_b
    ] for row_a in a]


def generate_multiplication_matrix(
        poly: list[int | float]) -> list[list[int | float]]:
    poly_len = len(poly)
    return [[
        poly[i - j] if j <= i else -poly[poly_len - (j - i)]
        for j in range(poly_len)
    ] for i in range(poly_len)]


class RingPoly:

    def __init__(self, poly: list[int | float]):
        self.poly = poly

    @staticmethod
    def random_ring_poly(poly_len: int, min_val: int,
                         max_val: int) -> "RingPoly":
        poly = [randint(min_val, max_val) for _ in range(poly_len)]
        if all([p == 0 for p in poly]):
            return RingPoly.random_ring_poly(poly_len, min_val, max_val)
        return RingPoly(poly)

    def _calc(self, other, operator) -> "RingPoly":
        if isinstance(other, RingPoly):
            assert len(self.poly) == len(other.poly)
            return RingPoly(
                [operator(a, b) for a, b in zip(self.poly, other.poly)])
        if isinstance(other, (int, float)):
            return RingPoly([operator(v, other) for v in self.poly])

        raise NotImplementedError(f"{operator} Not implemented for type: {type(other)}")

    def __mul__(self, other) -> "RingPoly":
        if isinstance(other, RingPoly):
            assert len(self.poly) == len(other.poly)
            mul_matrix = generate_multiplication_matrix(self.poly)
            return RingPoly(
                list(chain(*matmul(mul_matrix, [[v] for v in other.poly]))))
        elif isinstance(other, (int, float)):
            return RingPoly([v * other for v in self.poly])
        else:
            raise NotImplementedError()
        
    def __rmul__(self, other):
        return self * other

    def __add__(self, other) -> "RingPoly":
        return self._calc(other, add)

    def __sub__(self, other) -> "RingPoly":
        return self._calc(other, sub)

    def __truediv__(self, other) -> "RingPoly":
        return self._calc(other, truediv)

    def __eq__(self, other) -> boolean:
        if isinstance(other, RingPoly):
            return self.poly == other.poly
        elif isinstance(other, list):
            return self.poly == other
        else:
            raise NotImplementedError(
                f"Equal not implemented for type: {type(other)}")

    def __repr__(self):
        return f"RingPoly({str(np.polynomial.Polynomial(self.poly))})"

    def __len__(self):
        return len(self.poly)

    def __mod__(self, other: int) -> "RingPoly":
        assert isinstance(other, int)
        return RingPoly([v % other for v in self.poly])

    def __round__(self) -> "RingPoly":
        return RingPoly([round(v) for v in self.poly])


def recursive_length(l) -> list:
    if isinstance(l, Sized):
        if isinstance(l, RingPoly):
            return [len(l)]
        return [len(l)] + recursive_length(l[0])
    return []


class PolynomialMatrix(object):

    def __init__(self, poly_mat: list[RingPoly] | RingPoly, modulus: int):
        self.modulus = modulus
        if isinstance(poly_mat, RingPoly):
            self.poly_mat = [[poly_mat]]
        else:
            self.poly_mat = poly_mat
        shape = tuple(recursive_length(self.poly_mat))
        if len(shape) == 2:
            self.poly_mat = [self.poly_mat]
            shape = tuple(recursive_length(self.poly_mat))
        elif len(shape) > 3:
            raise NotImplementedError()
        self.shape = shape[:-1], shape[-1]

    @staticmethod
    def _create(poly_mat: list[RingPoly] | RingPoly,
                modulus: int) -> "PolynomialMatrix":
        return PolynomialMatrix(poly_mat, modulus)

    def transpose(self) -> "PolynomialMatrix":
        return PolynomialMatrix(
            [[self.poly_mat[j][i] for j in range(self.shape[0][0])]
             for i in range(self.shape[0][1])], self.modulus)

    @property
    def T(self) -> "PolynomialMatrix":
        return self.transpose()

    @staticmethod
    def random_polynomial_matrix(poly_len: int,
                                 modulus: int,
                                 matrix_shape: tuple = (),
                                 min_val: int = 0,
                                 max_val: int = None) -> "PolynomialMatrix":
        """
        Generate an random polynomial of shape in the ring R^{rows x cols}_{modulus} with values
        evenly retrieved from min_val (default 0) to max_val (default modulus)
        """
        assert len(matrix_shape) == 2
        if max_val is None:
            max_val = modulus - 1
        assert all([s > 0 for s in matrix_shape]) and poly_len > 0
        return PolynomialMatrix([[
            RingPoly.random_ring_poly(poly_len, min_val, max_val)
            for _ in range(matrix_shape[1])
        ] for _ in range(matrix_shape[0])], modulus)


    def __repr__(self) -> str:
        return f"PolyMatrix({repr(self.poly_mat)})"

    def _calc_matrix(self: "PolynomialMatrix", other: "PolynomialMatrix",
                     operation: Callable) -> "PolynomialMatrix":
        if not isinstance(other, PolynomialMatrix):
            raise NotImplementedError()
        # print(self.shape == other.shape)
        # print(self.shape, other.shape)
        assert self.modulus == other.modulus and (self.shape == other.shape
                                                  or self.shape[0] == (1, 1)
                                                  or other.shape[0] == (1, 1))
        if self.shape[0] == (1, 1):
            return PolynomialMatrix(
                [[operation(self.poly_mat[0][0], v) for v in row]
                 for row in other.poly_mat], self.modulus)
        if other.shape[0] == (1, 1):
            return PolynomialMatrix(
                [[operation(other.poly_mat[0][0], v) for v in row]
                 for row in self.poly_mat], self.modulus)

        return PolynomialMatrix([[
            operation(self.poly_mat[i][j], other.poly_mat[i][j])
            for j in range(self.shape[0][1])
        ] for i in range(self.shape[0][0])], self.modulus)

    def _calc_scalar(self: "PolynomialMatrix", other: int | float,
                     operation: Callable) -> "PolynomialMatrix":

        return PolynomialMatrix([[
            operation(self.poly_mat[i][j], other)
            for j in range(self.shape[0][1])
        ] for i in range(self.shape[0][0])], self.modulus)

    def __add__(self, other) -> "PolynomialMatrix":
        if isinstance(other, PolynomialMatrix):
            return self._calc_matrix(other, add)
        raise NotImplementedError()

    def __sub__(self, other) -> "PolynomialMatrix":
        if isinstance(other, (int, float)):
            return self._calc_scalar(other, sub)
        if isinstance(other, PolynomialMatrix):
            return self._calc_matrix(other, sub)
        raise NotImplementedError()

    def __mul__(self, other) -> "PolynomialMatrix":
        if isinstance(other, (int, float, RingPoly)):
            return self._calc_scalar(other, mul)
        if isinstance(other, PolynomialMatrix):
            return self._calc_matrix(other, mul)
        raise NotImplementedError()

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other) -> "PolynomialMatrix":
        if isinstance(other, (int, float)):
            return self._calc_scalar(other, truediv)
        raise NotImplementedError()

    def __matmul__(self, other: "PolynomialMatrix") -> "PolynomialMatrix":
        assert self.shape[1] == other.shape[1], "Different Ring Polynomials"
        assert self.modulus == other.modulus, "Different Modulus"
        assert self.shape[0][1] == other.shape[0][
            0], f"Wrong shape: {self.shape[0]} -- {other.shape[0]}"
        mul = matmul(self.poly_mat, other.poly_mat)
        if other.shape[0] == (1, 1) or self.shape[0] == (1, 1):
            return PolynomialMatrix(reduce(add, chain(*mul)), self.modulus)
        return PolynomialMatrix(mul, self.modulus)

    def __mod__(self, other: int) -> "PolynomialMatrix":
        assert isinstance(other, int)
        return PolynomialMatrix(
            [[self.poly_mat[i][j] % other for j in range(self.shape[0][1])]
             for i in range(self.shape[0][0])], self.modulus)

    def __round__(self) -> "PolynomialMatrix":
        return PolynomialMatrix(
            [[round(self.poly_mat[i][j]) for j in range(self.shape[0][1])]
             for i in range(self.shape[0][0])], self.modulus)

    def change_modulus(self, mod: int) -> "PolynomialMatrix":
        return PolynomialMatrix((self % mod).poly_mat, mod)

    # def __pow__(self, other: int) -> "PolynomialMatrix":
    #     original_shape = self.poly_mat.shape
    #     polynomials = [
    #         PolynomialMatrix([poly], self.modulus)
    #         for poly in self.poly_mat.reshape((-1, original_shape[-1]))
    #     ]

    #     def _multi(poly: PolynomialMatrix, amount: int) -> PolynomialMatrix:
    #         if amount == 1:
    #             return poly
    #         return poly @ _multi(poly, amount - 1)

    #     return self._create(
    #         np.asarray([(_multi(poly, other)).poly_mat[0]
    #                     for poly in polynomials]).reshape(original_shape) %
    #         self.modulus, self.modulus)

    def __eq__(self, other):
        if not isinstance(other, PolynomialMatrix):
            raise NotImplementedError()

        return self.poly_mat == other.poly_mat
    
    def __iter__(self):
        yield from chain(*self.poly_mat)


class Polynomial(PolynomialMatrix):

    @staticmethod
    def random_polynomial(poly_len: int,
                          modulus: int,
                          min_val: int = 0,
                          max_value: int = None) -> "Polynomial":
        pt = super(Polynomial, Polynomial).random_polynomial_matrix(
            poly_len, modulus, (1, 1), min_val, max_value)
        return Polynomial(pt.poly_mat, pt.modulus)

    @staticmethod
    def _create(poly_mat: np.ndarray, modulus: int) -> "Polynomial":
        return Polynomial(poly_mat, modulus)

    def __repr__(self) -> str:
        return f"Polynomial({str(np.polynomial.Polynomial(self.poly_mat[0][0]))})"


if __name__ == "__main__":
    a = [[1, 2], [3, 4]]
    b = [[1], [-1]]
    assert matmul(a, b) == [[-1], [-1]]

    i1 = [1, 2, 3, 4]
    o1 = [[1, -4, -3, -2],
          [2, 1, -4, -3],
          [3, 2, 1, -4],
          [4, 3, 2, 1]] # yapf: disable
    assert generate_multiplication_matrix(i1) == o1

    a = RingPoly([98, 3, 0])
    b = RingPoly([1, 1, 1])
    assert a * b == [95, 101, 101]
    assert a * 2 == [196, 6, 0]
    assert a + b == [99, 4, 1]
    assert a + 5 == [103, 8, 5]

    a = PolynomialMatrix([RingPoly([3, 3, 3]), RingPoly([98, 3, 0])], 100)
    assert a.shape == ((1, 2), 3)
    assert a.T.shape == ((2, 1), 3)
    assert a.T.poly_mat == [[RingPoly([3, 3, 3])], [RingPoly([98, 3, 0])]]
    b = PolynomialMatrix([RingPoly([1, 1, 1])], 100)
    assert b.shape == ((1, 1), 3)

    assert a + b == PolynomialMatrix(
        [RingPoly([4, 4, 4]), RingPoly([99, 4, 1])], 100), a + b
    assert a + b == b + a

    assert (a.T @ b) == PolynomialMatrix(RingPoly([92, 104, 110]), 100)

    s = PolynomialMatrix(([[RingPoly([3, 3, 3])], [RingPoly([98, 3, 0])]]),
                         100)
    A = PolynomialMatrix([[
        RingPoly([53, 83, 66]), RingPoly([27, 29, 34])
    ], [RingPoly([16, 25, 87]), RingPoly([48, 96, 0])]], 100)
    e = PolynomialMatrix([[RingPoly([97, 99, 99])], [RingPoly([98, 98, 1])]],
                         100)

    b = (A @ s + e) % 100
    assert b.poly_mat == [[RingPoly([53., 32., 24.])],
                          [RingPoly([14., 12., 73.])]], b.poly_mat

    assert PolynomialMatrix(RingPoly([3, 0, 1]), 100) * 50 == PolynomialMatrix(
        RingPoly([150, 0, 50]), 100)
    m = RingPoly.random_ring_poly(3, 0, 1)
    m_p = PolynomialMatrix(m, 100) * (100 // 2)

    r = PolynomialMatrix.random_polynomial_matrix(3,
                                                  100, (2, 1),
                                                  min_val=0,
                                                  max_val=1)
    e1 = PolynomialMatrix.random_polynomial_matrix(3,
                                                   100, (2, 1),
                                                   min_val=-3,
                                                   max_val=3)
    e2 = PolynomialMatrix.random_polynomial_matrix(3,
                                                   100, (1, 1),
                                                   min_val=-3,
                                                   max_val=3)

    u = (A.T @ r) + e1
    v = b.T @ r + e2 + m_p
    m_d = round((v - (s.T @ u)) / (100 // 2)) % 2
    assert PolynomialMatrix(m, 100) == m_d
