NEG_INF = float('-inf')
INF = float('inf')
EPS = 1e-9

DEFAULT_TOL = 1e-5
DEFAULT_PSEUDOCOUNT = 1.0

CANONICAL_DNA = frozenset({"A", "C", "G", "T"})
# Расширенный алфавит IUPAC для ДНК:
# R = A/G, Y = C/T, S = G/C, W = A/T, K = G/T, M = A/C,
# B = C/G/T, D = A/G/T, H = A/C/T, V = A/C/G, N = любой,
# дефис '-' часто используется для обозначения гэпов (пропусков выравнивания)
IUPAC_AMBIGUOUS_DNA = CANONICAL_DNA | frozenset({"R", "Y", "S", "W", "K", "M", "B", "D", "H", "V", "N", "-"})