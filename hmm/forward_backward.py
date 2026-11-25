from hmm.constants import NEG_INF
from collections.abc import Sequence
from hmm.core import HMM
from hmm.logmath import logsumexp, safe_exp
from math import fsum


def _encode_sequence(hmm: HMM, sequence: Sequence[str]) -> list[int]:
    """
    Вспомогательная функция перевода последовательности символов в индексы
    """
    symbol_to_idx = hmm.symbol_to_idx
    
    obs_idx = []
    for t, symbol in enumerate(sequence):
        if symbol not in symbol_to_idx:
            raise ValueError(f"Неизвестный символ '{symbol}' на позиции {t}. Ожидался символ из алфавита {hmm.symbols}")
        obs_idx.append(symbol_to_idx[symbol])

    return obs_idx


def forward(hmm: HMM, sequence: Sequence[str]) -> tuple[tuple[tuple[float, ...], ...], float]:
    """
    Алгоритм Forward
    Возвращает вероятность полного правдоподобия и альфа-матрицу (T x N):
        Кортеж (total_log_prob, log_alpha_tuple)
    """
    len_seq = len(sequence)
    if len_seq == 0:
        raise ValueError("Последовательность не может быть пустой")

    # 1. Конвертация символов последовательности в индексы
    obs_idx = _encode_sequence(hmm, sequence)

    log_start = hmm.log_start_prob
    log_trans = hmm.log_trans_mat
    log_emiss = hmm.log_emission_mat

    num_states = len(hmm.states)

    # 2. Создание матрицы альфа (T x N)
    log_alpha = [[NEG_INF] * num_states for _ in range(len_seq)]

    # 3. Инициализация начальных вероятностей на первом шаге (t = 0)
    for i in range(num_states):
        log_alpha[0][i] = log_start[i] + log_emiss[i][obs_idx[0]]

    if all(prob == NEG_INF for prob in log_alpha[0]):
        total_log_prob = NEG_INF
        log_alpha_tuple = tuple(tuple(row) for row in log_alpha)

        return log_alpha_tuple, total_log_prob

    # 4. Прямой шаг и вычисления вероятностей (t = 1 .. T-1)
    for t in range(1, len_seq):
        log_alpha_prev = log_alpha[t - 1]
        log_alpha_curr = log_alpha[t]
        obs_curr = obs_idx[t]

        for i in range(num_states):
            log_alpha_curr[i] = (
                logsumexp([log_alpha_prev[j] + log_trans[j][i] for j in range(num_states)])
                + log_emiss[i][obs_curr]
            )

        # Если все состояния на шаге t невозможны, возвращаем сразу результат
        if all(prob == NEG_INF for prob in log_alpha[t]):
            total_log_prob = NEG_INF
            log_alpha_tuple = tuple(tuple(row) for row in log_alpha)

            return log_alpha_tuple, total_log_prob

    log_alpha_tuple = tuple(tuple(row) for row in log_alpha)

    # 5. Вычисления полного правдоподобия на конце цепи (total_log_prob может иметь NEG_INF)
    total_log_prob = logsumexp(log_alpha_tuple[len_seq - 1])

    return log_alpha_tuple, total_log_prob


def backward(hmm: HMM, sequence: Sequence[str]) -> tuple[tuple[float, ...], ...]:
    """
    Алгоритм Backward
    Возвращает бета-матрицу из кортежей (T x N):
        Матрица (log_beta_tuple)
    """
    len_seq = len(sequence)
    if len_seq == 0:
        raise ValueError("Последовательность не может быть пустой")

    # 1. Конвертация символов последовательности в индексы
    obs_idx = _encode_sequence(hmm, sequence)

    log_trans = hmm.log_trans_mat
    log_emiss = hmm.log_emission_mat

    num_states = len(hmm.states)

    # 2. Создание матрицы бета (T x N)
    log_beta = [[NEG_INF] * num_states for _ in range(len_seq)]

    # 3. Инициализация конечных вероятностей на последнем шаге (t = T-1, P = 1.0)
    for i in range(num_states):
        log_beta[len_seq - 1][i] = 0.0

    # 4. Прямой шаг и вычисления вероятностей (t = T-2 ... 0)
    for t in range(len_seq - 2, -1, -1):
        log_beta_curr = log_beta[t]
        log_beta_next = log_beta[t + 1]
        obs_next = obs_idx[t + 1]

        next_terms = [
            log_emiss[j][obs_next] + log_beta_next[j]
            for j in range(num_states)
        ]

        for i in range(num_states):
            log_beta_curr[i] = (
                logsumexp([log_trans[i][j] + next_terms[j]
                for j in range(num_states)])
            )

        # Если все состояния на шаге t невозможны, выходим заранее из цикла
        if all(prob == NEG_INF for prob in log_beta[t]):
            break

    log_beta_tuple = tuple(tuple(row) for row in log_beta)

    return log_beta_tuple


def posterior_probabilities(log_alpha: Sequence[Sequence[float]], log_beta: Sequence[Sequence[float]],
                            total_log_prob: float) -> tuple[tuple[float, ...], ...]:
    """
    Апостериорные вероятности на оснвое алгоритмов Forward-Backward
    Возвращает гамма-матрицу из кортежей (T x N)
        Матрица (gamma_mat_tuple)
    """
    # 1. Ключевые проверки
    if total_log_prob == NEG_INF:
        raise ValueError("Невозможно вычислить апостериорные вероятности: полное правдоподобие последовательности равно нулю (NEG_INF)")

    if len(log_alpha) == 0 or len(log_alpha[0]) == 0:
        raise ValueError("Матрица альфа не может быть пустой")

    if len(log_beta) == 0 or len(log_beta[0]) == 0:
        raise ValueError("Матрица бета не может быть пустой")

    if len(log_alpha) != len(log_beta) or len(log_alpha[0]) != len(log_beta[0]):
        raise ValueError("Размеры матриц альфа и бета должны совпадать")

    len_seq = len(log_alpha)
    num_states = len(log_alpha[0])

    # 2. Инициализация матрицы гамма (T x N)
    gamma_mat = []

    # 3. Вычисления апостериорных вероятностей
    for t in range(len_seq):
        raw_probs = [safe_exp(log_alpha[t][i] + log_beta[t][i] - total_log_prob) for i in range(num_states)]
        row_sum = fsum(raw_probs)

        if row_sum <= 0.0:
            raise ValueError(f"Критическая ошибка вычислений апостериорных вероятностей на позиции {t}: сумма вероятностей равна {row_sum}")

        normalized_row = tuple(p / row_sum for p in raw_probs)
        gamma_mat.append(normalized_row)

    gamma_mat_tuple = tuple(gamma_mat)

    return gamma_mat_tuple


def posterior_decode(hmm: HMM, gamma_mat: Sequence[Sequence[float]]) -> tuple[str, ...]:
    """
    Апостериорное декодирование, поиск максимума в гамма матрице на каждом шаге и выбор соответствующего состояния
    Возвращает кортеж состояний
    """
    # 1. Ключевые проверки
    if len(gamma_mat) == 0 or len(gamma_mat[0]) == 0:
        raise ValueError("Матрица гамма не может быть пустой")

    if len(gamma_mat[0]) != len(hmm.states):
        raise ValueError(f"Матрица гамма должна иметь такое же количество состояний, что и исходная модель: имеем {len(gamma_mat[0])}, ожидалось {len(hmm.states)}")

    num_states = len(hmm.states)

    # 2. Поиск максимума + запись состояния
    decode_states = tuple(
        hmm.idx_to_state[max(range(num_states), key=lambda i: row[i])]
        for row in gamma_mat
    )

    return decode_states