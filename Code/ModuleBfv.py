from dataclasses import dataclass
from functools import reduce
from operator import add

import numpy as np
from Polynomial import Polynomial, PolynomialTensor


@dataclass
class BfvConfig:
    mat_size: tuple[int, int]
    poly_len: int
    modulus: int
    p: int

    def __post_init__(self):
        self.index_distribution = [(i, j) for i in range(self.mat_size)
                                   for j in range(i, self.mat_size) if j > i]


@dataclass
class BfvSecretKey:
    sk: Polynomial


@dataclass
class BfvPublicKey:
    A: Polynomial
    b: Polynomial


@dataclass
class BfvRlk:
    ra: PolynomialTensor
    rb: PolynomialTensor


class BfvEncrypted:

    def __init__(self, config: BfvConfig, rlks: list[BfvRlk], u: Polynomial,
                 v: Polynomial):
        self.config = config
        self.u = u
        self.v = v
        self.rlks = rlks

    def __add__(self, other: "BfvEncrypted") -> "BfvEncrypted":
        assert self.config == other.config
        return BfvEncrypted(self.config,
                            self.rlks,
                            u=(self.u + other.u) % self.config.modulus,
                            v=(self.v + other.v) % self.config.modulus)

    def __mul__(self, other: "BfvEncrypted") -> "BfvEncrypted":
        assert self.config == other.config

        def round_with_poly(poly: PolynomialTensor) -> PolynomialTensor:
            return PolynomialTensor(
                (np.round(poly.poly_mat)) % self.config.modulus,
                self.config.modulus)

        def tp(arr: np.ndarray) -> PolynomialTensor:
            return PolynomialTensor(arr, self.config.modulus)

        d0 = round_with_poly((2 / self.config.modulus) * (self.v @ other.v))

        d1 = round_with_poly(
            (2 / self.config.modulus) *
            (tp([(tp(self.u.poly_mat[i]) @ tp(other.v.poly_mat)).poly_mat[0]
                 for i in range(self.config.mat_size)]) +
             tp([(tp(other.u.poly_mat[i]) @ tp(self.v.poly_mat)).poly_mat[0]
                 for i in range(self.config.mat_size)])))

        d2 = round_with_poly(
            (2 / self.config.modulus) *
            (tp([(tp(self.u.poly_mat[i]) @ tp(other.u.poly_mat[i])).poly_mat[0]
                 for i in range(self.config.mat_size)])))
        # print("d2:", d2)

        d3 = ([
            round_with_poly(
                (2 / self.config.modulus) *
                ((tp(self.u.poly_mat[i]) @ tp(other.u.poly_mat[j])) +
                 (tp(self.u.poly_mat[j]) @ tp(other.u.poly_mat[i]))))
            for i, j in self.config.index_distribution
        ])
        # print("d3:", d3)

        assert len(d3) == len(self.rlks) - 1

        ip = 1 / self.config.p
        big_mod = self.config.modulus * self.config.p

        v_relin: PolynomialTensor = reduce(
            add, [
                round_with_poly(ip * (rlk.rb @ d3i.change_modulus(big_mod)))
                for d3i, rlk in zip(d3, self.rlks[1:])
            ],
            round_with_poly(ip *
                            (d2.change_modulus(big_mod) @ self.rlks[0].rb)))

        u_relin: PolynomialTensor = reduce(
            add, [
                round_with_poly(ip * (rlk.ra @ d3i.change_modulus(big_mod)))
                for d3i, rlk in zip(d3, self.rlks[1:])
            ],
            round_with_poly(ip *
                            (self.rlks[0].ra @ d2.change_modulus(big_mod))))

        return BfvEncrypted(
            self.config,
            self.rlks,
            u=(d1 + u_relin.change_modulus(self.config.modulus)) %
            self.config.modulus,
            v=(d0 + v_relin.change_modulus(self.config.modulus)) %
            self.config.modulus)


class BFV:

    @staticmethod
    def keygen(conf: BfvConfig) -> tuple[BfvSecretKey, BfvPublicKey, BfvRlk]:
        s = PolynomialTensor.random_polynomial_matrix(
            conf.poly_len,
            conf.modulus,
            (conf.mat_size, ),
            min_val=0,
            max_value=1,
        )
        e = PolynomialTensor.random_polynomial_matrix(conf.poly_len,
                                                      conf.modulus,
                                                      (conf.mat_size, ),
                                                      min_val=0,
                                                      max_value=1)
        A = PolynomialTensor.random_polynomial_matrix(
            conf.poly_len,
            conf.modulus, (conf.mat_size, conf.mat_size),
            min_val=0,
            max_value=conf.modulus)

        # s = PolynomialTensor([[1, 0, 0, 1]], conf.modulus)
        # e = PolynomialTensor([[0, 1, 1, 1]], conf.modulus)
        # A = PolynomialTensor([[723, 177, 914, 847]], conf.modulus)

        b = (-1 * (A @ s + e)) % conf.modulus

        rs = PolynomialTensor(s.poly_mat, conf.p * conf.modulus)

        def _calculate_relin(second_secret: PolynomialTensor) -> BfvRlk:
            re = PolynomialTensor.random_polynomial_matrix(
                conf.poly_len, conf.p * conf.modulus, (conf.mat_size, ), 0, 1)
            ra = PolynomialTensor.random_polynomial_matrix(
                conf.poly_len, conf.p * conf.modulus,
                (conf.mat_size, conf.mat_size))

            # re = Polynomial([[1, 0, 0, 1]], conf.p * conf.modulus)
            # ra = Polynomial([[
            #     837051908812184, 471019529082268, 242724841222094,
            #     606619688219965
            # ]], conf.p * conf.modulus)

            rb = (-1 * (ra @ rs + re) + conf.p *
                  (second_secret)) % (conf.p * conf.modulus)

            return BfvRlk(ra=ra, rb=rb)

        relins = [_calculate_relin(rs @ rs)] + [
            _calculate_relin(
                PolynomialTensor(rs.poly_mat[i], rs.modulus)
                @ PolynomialTensor(rs.poly_mat[j], rs.modulus))
            for i, j in conf.index_distribution
        ]

        return (BfvSecretKey(s), BfvPublicKey(A=A, b=b), relins)

    @staticmethod
    def encrypt(conf: BfvConfig, pk: BfvPublicKey, rlks: list[BfvRlk],
                message: list) -> BfvEncrypted:
        assert isinstance(message, list)
        e1 = PolynomialTensor.random_polynomial_matrix(conf.poly_len,
                                                       conf.modulus, (1, ), 0,
                                                       1)
        e2 = PolynomialTensor.random_polynomial_matrix(conf.poly_len,
                                                       conf.modulus,
                                                       (conf.mat_size, ), 0, 1)
        r = PolynomialTensor.random_polynomial_matrix(conf.poly_len,
                                                      conf.modulus,
                                                      (conf.mat_size, ), 0, 2)
        # e1 = PolynomialTensor([[0, 1, 0, 0]], conf.modulus)
        # e2 = PolynomialTensor([[1, 1, 0, 1]], conf.modulus)
        # r = PolynomialTensor([[0, 2, 0, 1]], conf.modulus)

        dm = Polynomial(
            np.asarray([message]) * (conf.modulus // 2), conf.modulus)
        v = (pk.b @ r + e1 + dm) % conf.modulus
        u = (pk.A @ r + e2) % conf.modulus

        return BfvEncrypted(conf, rlks, u=u, v=v)

    @staticmethod
    def decrypt(sk: BfvSecretKey, m_enc: BfvEncrypted) -> list:
        return (np.round(
            ((2 / m_enc.config.modulus) *
             ((m_enc.v + m_enc.u @ sk.sk) % m_enc.config.modulus)).poly_mat) %
                2).tolist()[0]


if __name__ == "__main__":
    conf = BfvConfig(2, 4, 5000, 10000**3)
    sk, pk, rlk = BFV.keygen(conf)

    m1 = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0, 1)
    # m1 = Polynomial([[1, 1, 0, 0]], conf.modulus)
    m_e1 = BFV.encrypt(conf, pk, rlk, m1.poly_mat[0].tolist())

    print(f"Message: {m1.poly_mat[0]}")
    print(f"+: {(m1+m1).poly_mat[0] % 2} == {BFV.decrypt(sk, m_e1+m_e1)}")
    print(f"*: {(m1 @ m1).poly_mat[0] % 2} == {BFV.decrypt(sk, m_e1*m_e1)}")

    mul = m_e1*m_e1
    print((mul.v + mul.u @ sk.sk) % mul.config.modulus)
    print((((m_e1.v + m_e1.u @ sk.sk) % m_e1.config.modulus) @ ((m_e1.v + m_e1.u @ sk.sk) % m_e1.config.modulus))% mul.config.modulus )
    # assert ((m1 + m1).poly_mat[0] % 2 == BFV.decrypt(sk, m_e1 + m_e1)).all()
    # assert ((m1 @ m1).poly_mat[0] % 2 == BFV.decrypt(sk, m_e1 * m_e1)).all()

    conf = BfvConfig(1, 4, 5000, 10000**3)

    op_count = []
    for j in range(50):

        # Single Test Start
        sk, pk, rlks = BFV.keygen(conf)
        m1 = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0, 1)
        m_e1 = BFV.encrypt(conf, pk, rlks, m1.poly_mat[0].tolist())
        for i in range(10000):
            m2 = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0,
                                              1)
            m_e2 = BFV.encrypt(conf, pk, rlks, m2.poly_mat[0].tolist())
            m_e1 = m_e1 * m_e2
            m1 = (m1 @ m2) % 2
            # assert BFV.decrypt(sk, m_e1) == m1.poly_mat[0].tolist(), f"{i}: {m1} -- {BFV.decrypt(sk, m_e1)}"
            if BFV.decrypt(sk, m_e1) != m1.poly_mat[0].tolist():
                op_count.append(i)
                break

    print("Average Operations:", sum(op_count) / len(op_count))
