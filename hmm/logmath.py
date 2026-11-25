from hmm.constants import NEG_INF, EPS
from collections.abc import Sequence
from math import log, exp, log1p


def safe_log(p: float) -> float:
    if p < 0.0:
        raise ValueError("Вероятность не может быть отрицательной")
    if p == 0.0:
        return NEG_INF
    if p > 1.0 + EPS:
        raise ValueError("Вероятность не может быть больше 1.0")
    if p > 1.0:
        return 0.0

    return log(p)


def safe_exp(log_p: float) -> float:
    if log_p == NEG_INF or log_p < -745.0:
        return 0.0
    if log_p > EPS:
        raise ValueError("Вероятность не может быть больше 1.0")
    if log_p > 0.0:
        return 1.0

    return exp(log_p)


def logsumexp_pair(log_p1: float, log_p2: float) -> float:
    if log_p1 == NEG_INF and log_p2 == NEG_INF:
        return NEG_INF
    if log_p1 == NEG_INF:
        return log_p2
    if log_p2 == NEG_INF:
        return log_p1

    max_val, min_val = max(log_p1, log_p2), min(log_p1, log_p2)
    diff = max_val - min_val
    if diff > 50:
        return max_val
    term = exp(-diff)

    return max_val + log1p(term)


def logsumexp(logs: Sequence[float]) -> float:
    if not logs:
        return NEG_INF
    valid_logs = [x for x in logs if x != NEG_INF]
    if not valid_logs:
        return NEG_INF

    max_val = max(valid_logs)
    total = 0.0
    for x in valid_logs:
        diff = max_val - x
        if diff <= 50:
            total += exp(-diff)

    return max_val + log(total)