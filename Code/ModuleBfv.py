from dataclasses import dataclass
from functools import reduce
from operator import add

from Polynomial import Polynomial, PolynomialMatrix, RingPoly


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
    ra: PolynomialMatrix
    rb: PolynomialMatrix


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

        def round_with_poly(poly: Polynomial) -> Polynomial:
            return round(poly) % self.config.modulus

        def tp(arr: list) -> PolynomialMatrix:
            return PolynomialMatrix(arr, self.config.modulus)

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
        assert len(d3) == len(self.rlks) - 1

        ip = 1 / self.config.p
        big_mod = self.config.modulus * self.config.p


        v_relin: PolynomialMatrix = reduce(
            add, [
                round_with_poly(ip * (rlk.rb @ d3i.change_modulus(big_mod)))
                for d3i, rlk in zip(d3, self.rlks[1:])
            ],
            round_with_poly(ip *
                            (d2.change_modulus(big_mod).T @ self.rlks[0].rb)))

        u_relin: PolynomialMatrix = reduce(
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

        rs = PolynomialMatrix(s.poly_mat, conf.p * conf.modulus)

        def _calculate_relin(second_secret: PolynomialMatrix) -> BfvRlk:
            re = PolynomialMatrix.random_polynomial_matrix(
                conf.poly_len, conf.p * conf.modulus, (conf.mat_size, 1), 0, 1)
            ra = PolynomialMatrix.random_polynomial_matrix(
                conf.poly_len, conf.p * conf.modulus,
                (conf.mat_size, conf.mat_size))

            # re = Polynomial([[1, 0, 0, 1]], conf.p * conf.modulus)
            # ra = Polynomial([[
            #     837051908812184, 471019529082268, 242724841222094,
            #     606619688219965
            # ]], conf.p * conf.modulus)

            rb = (-1 * (ra.T @ rs + re) + conf.p *
                  (second_secret)) % (conf.p * conf.modulus)
            return BfvRlk(ra=ra, rb=rb)

        relins = [_calculate_relin(rs*rs)] + [
            _calculate_relin(
                PolynomialMatrix(rs.poly_mat[i], rs.modulus)
                @ PolynomialMatrix(rs.poly_mat[j], rs.modulus))
            for i, j in conf.index_distribution
        ]

        return (BfvSecretKey(s), BfvPublicKey(A=A, b=b), relins)

    @staticmethod
    def encrypt(conf: BfvConfig, pk: BfvPublicKey, rlks: list[BfvRlk],
                message: list) -> BfvEncrypted:
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
                ((2 / m_enc.config.modulus) * ((m_enc.v + m_enc.u.T @ sk.sk) % m_enc.config.modulus))
            ) % 2
        ).poly_mat[0][0] # yapf: disable


if __name__ == "__main__":
    conf = BfvConfig(1, 4, 2**30, 2**60)
    sk, pk, rlk = BFV.keygen(conf)

    m1 = RingPoly([1, 1, 0, 0])
    m_e1 = BFV.encrypt(conf, pk, rlk, m1.poly)

    # print(f"Message: {m1.poly_mat[0]}")
    print(f"+: {(m1+m1) % 2} == {BFV.decrypt(sk, m_e1+m_e1)}")
    print(f"*: {(m1 * m1) % 2} == {BFV.decrypt(sk, m_e1*m_e1)}")
    assert (m1 + m1) % 2 == BFV.decrypt(sk, m_e1 + m_e1)
    assert (m1 * m1) % 2 == BFV.decrypt(sk, m_e1 * m_e1)

    conf = BfvConfig(1, 4, 2**30, 2**60)

    op_count = []
    for j in range(200):

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
