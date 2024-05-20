### RLK V1 polynom calc
# https://eprint.iacr.org/2012/144.pdf
from dataclasses import dataclass

# import numpy as np

from Polynomial import Polynomial, PolynomialMatrix, RingPoly


@dataclass
class BfvConfig:
    poly_len: int
    modulus: int
    base: int


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


# def decompose_len(modulus: int, base: int) -> int:
#     return int(np.floor(np.emath.logn(base, modulus))) + 1

# def to_base(poly: Polynomial, base: int) -> np.ndarray:
#     l = decompose_len(poly.modulus, base)

#     def _to_base(arr: np.ndarray, base: int, index=1) -> list:
#         if index >= l:
#             return arr
#         return [arr[-1] % base] + _to_base([arr[-1] // base], base, index + 1)

#     return np.asarray(_to_base(poly.poly_mat, base))

# class BfvEncrypted:

#     def __init__(self, config: BfvConfig, rlk: BfvRlk, u: Polynomial,
#                  v: Polynomial):
#         self.config = config
#         self.u = u
#         self.v = v
#         self.rlk = rlk

#     def __add__(self, other: "BfvEncrypted") -> "BfvEncrypted":
#         assert self.config == other.config
#         return BfvEncrypted(self.config,
#                             self.rlk,
#                             u=(self.u + other.u) % self.config.modulus,
#                             v=(self.v + other.v) % self.config.modulus)

#     def __mul__(self, other: "BfvEncrypted") -> "BfvEncrypted":
#         assert self.config == other.config
#         l = decompose_len(self.config.modulus, self.config.base)

#         def round_with_poly(poly: Polynomial) -> Polynomial:
#             return Polynomial((np.round(poly.poly_mat)) % self.config.modulus,
#                               self.config.modulus)

#         v = round_with_poly((2 / self.config.modulus) * (self.v @ other.v))

#         u = round_with_poly(
#             (2 / self.config.modulus) * (self.v @ other.u + self.u @ other.v))

#         uv = round_with_poly((2 / self.config.modulus) * (self.u @ other.u))
#         uv_base = to_base(uv,
#                           self.config.base).reshape(l, self.config.poly_len, 1)

#         uvu = Polynomial(
#             (
#                 self.rlk.ra.mul_matrix(axis=1).reshape(l, self.config.poly_len, self.config.poly_len)
#                 @ uv_base
#              ).sum(axis=0).T,
#              self.config.modulus
#         ) # yapf: disable
#         uvv = Polynomial(
#             (
#                 self.rlk.rb.mul_matrix(axis=1).reshape(l, self.config.poly_len, self.config.poly_len)
#                 @ uv_base
#             ).sum(axis=0).T,
#             self.config.modulus
#         ) # yapf: disable

#         return BfvEncrypted(self.config,
#                             self.rlk,
#                             u=(u + uvu) % self.config.modulus,
#                             v=(v + uvv) % self.config.modulus)


# class BFV:

#     @staticmethod
#     def keygen(conf: BfvConfig) -> tuple[BfvSecretKey, BfvPublicKey, BfvRlk]:
#         # Key Generation
#         s = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0, 1)
#         e = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0, 1)
#         A = Polynomial.random_polynomial(conf.poly_len, conf.modulus)
#         b = (-1 * (A @ s + e)) % conf.modulus

#         # RLK Generation
#         l = decompose_len(conf.modulus, conf.base)
#         ra = PolynomialTensor.random_polynomial_matrix(conf.poly_len,
#                                                        conf.modulus, (l, ))
#         re = PolynomialTensor.random_polynomial_matrix(conf.poly_len,
#                                                        conf.modulus, (l, ), 0,
#                                                        2)

#         ra_s = PolynomialTensor(
#             (ra.mul_matrix(axis=1) @ s.poly_mat.T).reshape(l, conf.poly_len),
#             conf.modulus)
#         T_is = np.asarray([[conf.base**i for i in range(l)]]).T
#         ti_s2 = PolynomialTensor((s @ s).poly_mat * T_is, conf.modulus)
#         rb = (-1 * (ra_s + re) + ti_s2) % conf.modulus

#         return (BfvSecretKey(s), BfvPublicKey(A, b), BfvRlk(ra, rb))

#     @staticmethod
#     def encrypt(conf: BfvConfig, pk: BfvPublicKey, rlk: BfvRlk,
#                 message: list) -> BfvEncrypted:
#         assert isinstance(message, list)
#         e1 = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0, 1)
#         e2 = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0, 1)
#         r = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0, 2)

#         v = (pk.b @ r + e1 + Polynomial(
#             np.asarray([message]) *
#             (conf.modulus // 2), conf.modulus)) % conf.modulus
#         u = (pk.A @ r + e2) % conf.modulus

#         return BfvEncrypted(conf, rlk, u=u, v=v)

#     @staticmethod
#     def decrypt(sk: BfvSecretKey, m_enc: BfvEncrypted) -> list:
#         return (np.round(
#             ((2 / m_enc.config.modulus) *
#              ((m_enc.v + m_enc.u @ sk.sk) % m_enc.config.modulus)).poly_mat) %
#                 2).tolist()[0]

### RLK V2

@dataclass
class BfvConfig:
    poly_len: int
    modulus: int
    p: int


class BfvEncrypted:

    def __init__(self, config: BfvConfig, rlk: BfvRlk, u: Polynomial,
                 v: Polynomial):
        self.config = config
        self.u = u
        self.v = v
        self.rlk = rlk

    def __add__(self, other: "BfvEncrypted") -> "BfvEncrypted":
        assert self.config == other.config
        return BfvEncrypted(self.config,
                          self.rlk,
                          u=(self.u + other.u) % self.config.modulus,
                          v=(self.v + other.v) % self.config.modulus)

    def __mul__(self, other: "BfvEncrypted") -> "BfvEncrypted":
        assert self.config == other.config

        def round_with_poly(poly: Polynomial) -> Polynomial:
            return round(poly) % self.config.modulus

        v = round_with_poly((2 / self.config.modulus) * (self.v @ other.v))

        u = round_with_poly(
            (2 / self.config.modulus) * (self.v @ other.u + self.u @ other.v))

        uv = round_with_poly(
            (2 / self.config.modulus) * (self.u @ other.u)).change_modulus(
                self.config.p * self.config.modulus)
        uvu = round_with_poly((uv @ self.rlk.ra) * (1 / self.config.p)).change_modulus(self.config.modulus)
        uvv = round_with_poly((uv @ self.rlk.rb) * (1 / self.config.p)).change_modulus(self.config.modulus)

        return BfvEncrypted(self.config,
                          self.rlk,
                          u=(u + uvu) % self.config.modulus,
                          v=(v + uvv) % self.config.modulus)


class BFV:

    @staticmethod
    def keygen(conf: BfvConfig) -> tuple[BfvSecretKey, BfvPublicKey, BfvRlk]:
        s = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0, 1)
        e = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0, 1)
        A = Polynomial.random_polynomial(conf.poly_len, conf.modulus)

        # s = Polynomial([[1, 0, 0, 1]], conf.modulus)
        # e = Polynomial([[0, 1, 1, 1]], conf.modulus)
        # A = Polynomial([[723, 177, 914, 847]], conf.modulus)

        b = (-1 * (A @ s + e)) % conf.modulus

        rs = Polynomial(s.poly_mat, conf.p * conf.modulus)
        re = Polynomial.random_polynomial(conf.poly_len, conf.p * conf.modulus,
                                          0, 1)
        rA = Polynomial.random_polynomial(conf.poly_len, conf.p * conf.modulus)
        # re = Polynomial([[1, 0, 0, 1]], conf.p * conf.modulus)
        # rA = Polynomial([[837051908812184, 471019529082268, 242724841222094, 606619688219965]], conf.p * conf.modulus)

        rb = (-1 * (rA @ rs + re) + conf.p * (rs @ rs)) % (conf.p * conf.modulus)

        return (BfvSecretKey(s), BfvPublicKey(A=A, b=b),
                BfvRlk(rA, rb))


    @staticmethod
    def encrypt(conf: BfvConfig, pk: BfvPublicKey, rlk: BfvRlk,
                message: list) -> BfvEncrypted:
        if isinstance(message, list):
            message = RingPoly(message)
        elif not isinstance(message, RingPoly):
            raise TypeError("Message needs to be list or RingPoly")
        e1 = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0, 1)
        e2 = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0, 1)
        r = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0, 2)
        # e1 = Polynomial([[0, 1, 0, 0]], conf.modulus)
        # e2 = Polynomial([[1, 1, 0, 1]], conf.modulus)
        # r = Polynomial([[0, 2, 0, 1]], conf.modulus)


        v = (pk.b @ r + e1 + Polynomial(message, conf.modulus) * (conf.modulus // 2)) % conf.modulus
        u = (pk.A @ r + e2) % conf.modulus

        return BfvEncrypted(conf, rlk, u=u, v=v)

    @staticmethod
    def decrypt(sk: BfvSecretKey, m_enc: BfvEncrypted) -> list:
        return (
            round(
                ((2 / m_enc.config.modulus) * ((m_enc.v + m_enc.u @ sk.sk) % m_enc.config.modulus))
            ) % 2
        ).poly_mat[0][0] # yapf: disable



if __name__ == "__main__":
    conf = BfvConfig(4, 1000, 1000**4)
    sk, pk, rlk = BFV.keygen(conf)

    # m1 = Polynomial.random_polynomial(conf.poly_len, conf.modulus, 0, 1)
    m1 = RingPoly([1,1,0,0])
    m_e1 = BFV.encrypt(conf, pk, rlk, m1.poly)

    # print(f"Message: {m1.poly_mat[0]}")
    print(f"+: {(m1+m1) % 2} == {BFV.decrypt(sk, m_e1+m_e1)}")
    print(f"*: {(m1 * m1) % 2} == {BFV.decrypt(sk, m_e1*m_e1)}")
    assert (m1 + m1) % 2 == BFV.decrypt(sk, m_e1+m_e1)
    assert (m1 * m1) % 2 == BFV.decrypt(sk, m_e1*m_e1)

    conf = BfvConfig(4, 2**30, 2**64)

    op_count = []
    for j in range(200):

        # Single Test Start
        sk, pk, rlks = BFV.keygen(conf)
        m1 = RingPoly.random_ring_poly(conf.poly_len, 0, 1)
        m_e1 = BFV.encrypt(conf, pk, rlks, m1)
        for i in range(10000):
            m2 = RingPoly.random_ring_poly(conf.poly_len, 0, 1)
            m_e2 = BFV.encrypt(conf, pk, rlks, m2)
            m_e1 = m_e1*m_e2
            m1 = (m1 * m2)%2
            # assert BFV.decrypt(sk, m_e1) == m1.poly_mat[0].tolist(), f"{i}: {m1} -- {BFV.decrypt(sk, m_e1)}"
            if BFV.decrypt(sk, m_e1) != m1:
                op_count.append(i)
                break

    print("Average Operations:", sum(op_count)/len(op_count))
