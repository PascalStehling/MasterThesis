from dataclasses import dataclass
from functools import reduce
from itertools import chain
from operator import add, mul
from statistics import mean
from typing import Callable

from Polynomial import Polynomial, PolynomialMatrix, RingPoly

from tqdm import tqdm


# https://tches.iacr.org/index.php/TCHES/article/view/11261/10803
@dataclass
class BfvConfig:
    mat_size: int
    poly_len: int
    q_mod: int
    p_mod: int

    def __post_init__(self):
        self.index_distribution = [(i, j) for i in range(self.mat_size)
                                   for j in range(i, self.mat_size) if j > i]


@dataclass
class BfvSecretKey:
    sk: PolynomialMatrix


@dataclass
class BfvPublicKey:
    A: PolynomialMatrix
    b: PolynomialMatrix


@dataclass
class BfvRlk:
    ra: PolynomialMatrix
    rb: PolynomialMatrix


def inverse_poly(poly: PolynomialMatrix):
    return PolynomialMatrix(list(reversed(poly.poly_mat)), poly.modulus)


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
                            u=(self.u + other.u) % self.config.q_mod,
                            v=(self.v + other.v) % self.config.q_mod)

    def __mul__(self, other: "BfvEncrypted") -> "BfvEncrypted":
        assert self.config == other.config

        def tq_round(inp):
            return (inp * 2) // self.config.q_mod

        c0 = (self.v * other.v)
        c1 = (self.v * other.u + other.v * self.u).T

        c0 = tq_round(c0) % self.config.q_mod
        c1 = tq_round(c1) % self.config.q_mod

        big_mod = self.config.p_mod * self.config.q_mod
        c2s = [(
            tq_round((self.u * other.u.poly_mat[r][0])) % self.config.q_mod
        ).change_modulus(big_mod)
            for r in range(self.config.mat_size)]

        assert len(self.rlks) == len(c2s)
        v_relin = [
            ((relin.rb.T @ c) // self.config.p_mod).change_modulus(
                self.config.q_mod) for relin, c in zip(self.rlks, c2s)
        ]
        v = (c0 + reduce(add, v_relin)) % self.config.q_mod

        u_relin = [
            ((relin.ra @ c) // self.config.p_mod).change_modulus(
                self.config.q_mod) for relin, c in zip(self.rlks, c2s)
        ]
        u = (c1.T + reduce(add, u_relin)) % self.config.q_mod

        return BfvEncrypted(self.config, self.rlks, u=u, v=v)


class BFV:

    @staticmethod
    def keygen(
            conf: BfvConfig
    ) -> tuple[BfvSecretKey, BfvPublicKey, list[BfvRlk]]:
        s = PolynomialMatrix.random_polynomial_matrix(
            conf.poly_len,
            conf.q_mod,
            (conf.mat_size, 1),
            min_val=0,
            max_val=1,
        )
        e = PolynomialMatrix.random_polynomial_matrix(conf.poly_len,
                                                      conf.q_mod,
                                                      (conf.mat_size, 1),
                                                      min_val=0,
                                                      max_val=1)
        A = PolynomialMatrix.random_polynomial_matrix(
            conf.poly_len,
            conf.q_mod, (conf.mat_size, conf.mat_size),
            min_val=0,
            max_val=conf.q_mod)

        b = (-1 * (A @ s + e)) % conf.q_mod
        big_mod = conf.p_mod * conf.q_mod
        rs = PolynomialMatrix(s.poly_mat, big_mod)

        def _calculate_relin(second_secret: PolynomialMatrix) -> BfvRlk:
            re = PolynomialMatrix.random_polynomial_matrix(
                conf.poly_len, big_mod, (conf.mat_size, 1), 0, 1)
            ra = PolynomialMatrix.random_polynomial_matrix(
                conf.poly_len, big_mod, (conf.mat_size, conf.mat_size))

            rb = (-1 * (ra.T @ rs + re) + conf.p_mod * second_secret) % (big_mod)

            return BfvRlk(ra=ra, rb=rb)

        relins = [
            _calculate_relin(rs * rs.poly_mat[r][0])
            for r in range(conf.mat_size)
        ]

        return (BfvSecretKey(s), BfvPublicKey(A=A, b=b), relins)

    @staticmethod
    def encrypt(conf: BfvConfig, pk: BfvPublicKey, rlks: list[BfvRlk],
                message: list[int]) -> BfvEncrypted:
        if isinstance(message, list):
            message = RingPoly(message)
        elif not isinstance(message, RingPoly):
            raise TypeError("Message needs to be list or RingPoly")
        e1 = PolynomialMatrix.random_polynomial_matrix(conf.poly_len,
                                                       conf.q_mod, (1, 1), 0,
                                                       1)
        e2 = PolynomialMatrix.random_polynomial_matrix(conf.poly_len,
                                                       conf.q_mod,
                                                       (conf.mat_size, 1), 0,
                                                       1)
        r = PolynomialMatrix.random_polynomial_matrix(conf.poly_len,
                                                      conf.q_mod,
                                                      (conf.mat_size, 1), 0, 2)

        dm = PolynomialMatrix(message, conf.q_mod) * (conf.q_mod // 2)
        v = (pk.b.T @ r + e1 + dm) % conf.q_mod
        u = (pk.A.T @ r + e2) % conf.q_mod

        return BfvEncrypted(conf, rlks, u=u, v=v)

    @staticmethod
    def decrypt(sk: BfvSecretKey, m_enc: BfvEncrypted) -> list:
        return (
            round(
                ((((m_enc.v + m_enc.u.T @ sk.sk) %
                 m_enc.config.q_mod) * 2) / m_enc.config.q_mod)
            ) % 2
        ).poly_mat[0][0]  # yapf: disable


def get_error(m_enc: BfvEncrypted, m_tar: RingPoly, sk: BfvSecretKey):
    dec = ((
        ((((m_enc.v + m_enc.u.T @ sk.sk) %
           m_enc.config.q_mod) * 2) / m_enc.config.q_mod)
    ) % 2).poly_mat[0][0].poly

    return mean([abs(d-t) if t == 1 else (d if d < 0.5 else 2-d) for d, t in zip(dec, m_tar.poly)])


def depth_test(conf: BfvConfig, op:Callable, max_depht=200):
    op_count = []
    errors = []
    for _ in tqdm(range(100)):

        error_dev = []
        # Single Test Start
        sk, pk, rlks = BFV.keygen(conf)
        m1 = RingPoly.random_ring_poly(conf.poly_len, 0, 1)
        m_e1 = BFV.encrypt(conf, pk, rlks, m1)
        for i in range(max_depht):
            m2 = RingPoly.random_ring_poly(conf.poly_len, 0, 1)
            m_e2 = BFV.encrypt(conf, pk, rlks, m2)
            m_e1 = op(m_e1, m_e2)
            m1 = op(m1, m2) % 2
            error_dev.append(get_error(m_e1, m1, sk))

            if BFV.decrypt(sk, m_e1) != m1:
                break
        op_count.append(i)
        errors.append(error_dev)

    print(f"Average {op.__name__} Operations:", sum(op_count) / len(op_count))

if __name__ == "__main__":
    print("*"*10+"\nRunning Plain-LWE\n"+"-"*10)
    conf = BfvConfig(15, 1, 2**32, 2**128)
    print(conf)
    sk, pk, rlk = BFV.keygen(conf)

    m1 = RingPoly([1])
    m_e1 = BFV.encrypt(conf, pk, rlk, m1.poly)
    print(f"+: {(m1+m1) % 2} == {BFV.decrypt(sk, m_e1+m_e1)}")
    print(f"*: {(m1 * m1) % 2} == {BFV.decrypt(sk, m_e1*m_e1)}")
    assert (m1 + m1) % 2 == BFV.decrypt(sk, m_e1 + m_e1)
    assert (m1 * m1) % 2 == BFV.decrypt(sk, m_e1 * m_e1)
    depth_test(conf, add)
    depth_test(conf, mul)

    print("")
    print("*"*10+"\nRunning Ring-LWE\n"+"-"*10)
    conf = BfvConfig(1, 4, 2**32, 2**128)
    print(conf)
    sk, pk, rlk = BFV.keygen(conf)

    m1 = RingPoly([1, 1, 0, 0])
    m_e1 = BFV.encrypt(conf, pk, rlk, m1.poly)


    print(f"+: {(m1+m1) % 2} == {BFV.decrypt(sk, m_e1+m_e1)}")
    print(f"*: {(m1 * m1) % 2} == {BFV.decrypt(sk, m_e1*m_e1)}")
    assert (m1 + m1) % 2 == BFV.decrypt(sk, m_e1 + m_e1)
    assert (m1 * m1) % 2 == BFV.decrypt(sk, m_e1 * m_e1)
    depth_test(conf, add)
    depth_test(conf, mul)

    print("")
    print("*"*10+"\nRunning Module-LWE\n"+"-"*10)
    conf = BfvConfig(3, 4, 2**32, 2**128)
    print(conf)
    sk, pk, rlk = BFV.keygen(conf)

    m1 = RingPoly([1, 1, 0, 0])
    m_e1 = BFV.encrypt(conf, pk, rlk, m1.poly)

    print(f"+: {(m1+m1) % 2} == {BFV.decrypt(sk, m_e1+m_e1)}")
    print(f"*: {(m1 * m1) % 2} == {BFV.decrypt(sk, m_e1*m_e1)}")
    assert (m1 + m1) % 2 == BFV.decrypt(sk, m_e1 + m_e1)
    assert (m1 * m1) % 2 == BFV.decrypt(sk, m_e1 * m_e1)
    depth_test(conf, add)
    depth_test(conf, mul)
