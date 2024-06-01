from dataclasses import dataclass
from functools import reduce
from itertools import chain
from operator import add

from Polynomial import Polynomial, PolynomialMatrix, RingPoly

from tqdm import tqdm


# https://tches.iacr.org/index.php/TCHES/article/view/11261/10803
@dataclass
class BfvConfig:
    mat_size: int
    poly_len: int
    modulus: int
    p: int

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
                            u=(self.u + other.u) % self.config.modulus,
                            v=(self.v + other.v) % self.config.modulus)

    def __mul__(self, other: "BfvEncrypted") -> "BfvEncrypted":
        assert self.config == other.config

        c0 = (self.v * other.v)
        c1 = (self.v*other.u+other.v*self.u).T # yapf: disable
        c2 = self.u @ other.u.T  # yapf: disable

        tq = 2 / self.config.modulus
        c0 = round(c0 * tq) % self.config.modulus
        c1 = round(c1 * tq) % self.config.modulus
        c2 = round(c2 * tq) % self.config.modulus

        c20 = round((self.u * other.u) * tq) % self.config.modulus  # s^2
        c21 = round((self.u * inverse_poly(other.u)) *
                    tq) % self.config.modulus  # s1s2 - vector

        big_mod = self.config.p * self.config.modulus
        c20 = c20.change_modulus(big_mod)
        c21 = c21.change_modulus(big_mod)

        c20_relin = self.rlks[0]
        c21_relin = self.rlks[1]
        v = (c0
            + round((c20_relin.rb.T @ c20) / self.config.p).change_modulus(self.config.modulus)
            + round((c21_relin.rb.T @ c21) / self.config.p).change_modulus(self.config.modulus)
        ) % self.config.modulus # yapf: disable
        u = (c1.T
            + round((c20_relin.ra @ c20) / self.config.p).change_modulus(self.config.modulus)
            + round((c21_relin.ra @ c21) / self.config.p).change_modulus(self.config.modulus)
        ) % self.config.modulus # yapf: disable


        return BfvEncrypted(self.config, self.rlks, u=u, v=v)


class BFV:

    @staticmethod
    def keygen(
            conf: BfvConfig
    ) -> tuple[BfvSecretKey, BfvPublicKey, list[BfvRlk]]:
        s = PolynomialMatrix.random_polynomial_matrix(
            conf.poly_len,
            conf.modulus,
            (conf.mat_size, 1),
            min_val=0,
            max_val=1,
        )
        e = PolynomialMatrix.random_polynomial_matrix(conf.poly_len,
                                                      conf.modulus,
                                                      (conf.mat_size, 1),
                                                      min_val=0,
                                                      max_val=1)
        A = PolynomialMatrix.random_polynomial_matrix(
            conf.poly_len,
            conf.modulus, (conf.mat_size, conf.mat_size),
            min_val=0,
            max_val=conf.modulus)

        # s = PolynomialTensor([[1, 0, 0, 1]], conf.modulus)
        # e = PolynomialTensor([[0, 1, 1, 1]], conf.modulus)
        # A = PolynomialTensor([[723, 177, 914, 847]], conf.modulus)

        b = (-1 * (A @ s + e)) % conf.modulus
        big_mod = conf.p * conf.modulus
        rs = PolynomialMatrix(s.poly_mat, big_mod)

        def _calculate_relin(second_secret: PolynomialMatrix) -> BfvRlk:
            re = PolynomialMatrix.random_polynomial_matrix(
                conf.poly_len, big_mod, (conf.mat_size, 1), 0, 1)
            ra = PolynomialMatrix.random_polynomial_matrix(
                conf.poly_len, big_mod, (conf.mat_size, conf.mat_size))

            rb = (-1 * (ra.T @ rs + re) + conf.p * second_secret) % (big_mod)

            return BfvRlk(ra=ra, rb=rb)

        relins = [
            _calculate_relin(rs * rs),
            _calculate_relin(rs * inverse_poly(rs))
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
                                                       conf.modulus, (1, 1), 0,
                                                       1)
        e2 = PolynomialMatrix.random_polynomial_matrix(conf.poly_len,
                                                       conf.modulus,
                                                       (conf.mat_size, 1), 0,
                                                       1)
        r = PolynomialMatrix.random_polynomial_matrix(conf.poly_len,
                                                      conf.modulus,
                                                      (conf.mat_size, 1), 0, 2)
        # e1 = PolynomialTensor([[0, 1, 0, 0]], conf.modulus)
        # e2 = PolynomialTensor([[1, 1, 0, 1]], conf.modulus)
        # r = PolynomialTensor([[0, 2, 0, 1]], conf.modulus)

        dm = PolynomialMatrix(message, conf.modulus) * (conf.modulus // 2)
        v = (pk.b.T @ r + e1 + dm) % conf.modulus
        u = (pk.A.T @ r + e2) % conf.modulus

        return BfvEncrypted(conf, rlks, u=u, v=v)

    @staticmethod
    def decrypt(sk: BfvSecretKey, m_enc: BfvEncrypted) -> list:
        return (
            round(
                ((((m_enc.v + m_enc.u.T @ sk.sk) % m_enc.config.modulus) * 2) / m_enc.config.modulus)
            ) % 2
        ).poly_mat[0][0] # yapf: disable


if __name__ == "__main__":
    conf = BfvConfig(2, 4, 2**30, 2**60)
    sk, pk, rlk = BFV.keygen(conf)

    m1 = RingPoly([1, 1, 0, 0])
    m_e1 = BFV.encrypt(conf, pk, rlk, m1.poly)

    # print(f"Message: {m1.poly_mat[0]}")
    print(f"+: {(m1+m1) % 2} == {BFV.decrypt(sk, m_e1+m_e1)}")
    print(f"*: {(m1 * m1) % 2} == {BFV.decrypt(sk, m_e1*m_e1)}")
    assert (m1 + m1) % 2 == BFV.decrypt(sk, m_e1 + m_e1)
    assert (m1 * m1) % 2 == BFV.decrypt(sk, m_e1 * m_e1)

    op_count = []
    for j in tqdm(range(500)):

        # Single Test Start
        sk, pk, rlks = BFV.keygen(conf)
        m1 = RingPoly.random_ring_poly(conf.poly_len, 0, 1)
        m_e1 = BFV.encrypt(conf, pk, rlks, m1)
        for i in range(10000):
            m2 = RingPoly.random_ring_poly(conf.poly_len, 0, 1)
            m_e2 = BFV.encrypt(conf, pk, rlks, m2)
            m_e1 = m_e1 * m_e2
            m1 = (m1 * m2) % 2
            # assert BFV.decrypt(sk, m_e1) == m1.poly_mat[0].tolist(), f"{i}: {m1} -- {BFV.decrypt(sk, m_e1)}"
            if BFV.decrypt(sk, m_e1) != m1:
                op_count.append(i)
                break

    print("Average Operations:", sum(op_count) / len(op_count))
