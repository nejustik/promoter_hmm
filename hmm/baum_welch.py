from hmm.constants import NEG_INF, DEFAULT_PSEUDOCOUNT, DEFAULT_TOL
from collections.abc import Sequence
from hmm.core import HMM
from hmm.logmath import safe_exp
from math import fsum
from hmm.forward_backward import forward, backward, posterior_probabilities


def _normalized_vector(vector: Sequence[float], fallback: Sequence[float]) -> list[float]:
    """
    Вспомогательная фукнция нормализации вектора
    """
    row_sum = fsum(vector)
    if row_sum > 0:
        return [val / row_sum for val in vector]
    return list(fallback)


def _normalized_mat(matrix: Sequence[Sequence[float]], fallback_matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """
    Вспомогательная фукнция нормализации матрицы
    """
    return [_normalized_vector(row, fallback_row) for row, fallback_row in zip(matrix, fallback_matrix)]


def _normalized_laplas_vector(sequence_counts: Sequence[float], pseudocount: float) -> list[float]:
    """
    Вспомогательная фукнция нормализации со сглаживанием Лапласа для вектора
    """
    num_states = len(sequence_counts)
    row_sum_laplas = fsum([*sequence_counts, num_states * pseudocount])

    return [(val + pseudocount) / row_sum_laplas for val in sequence_counts]


def _normalized_laplas_mat(mat_counts: Sequence[Sequence[float]], pseudocount: float) -> list[list[float]]:
    """
    Вспомогательная фукнция нормализации со сглаживанием Лапласа для матрицы
    """
    return [_normalized_laplas_vector(row, pseudocount) for row in mat_counts]


def train_supervised(states: Sequence[str], symbols: Sequence[str], labeled_sequences: Sequence[Sequence[str]], labeled_paths: Sequence[Sequence[str]], *, pseudocount: float = DEFAULT_PSEUDOCOUNT, tol: float = DEFAULT_TOL) -> HMM:
    """
    Обучение модели на основе размеченных последовательностей со сглаживанием Лапласа
    Вход:
        Состояния, алфавит символов, нуклеотидные последовательности, размеченные пути, альфа (псвеоодтсчет)
    Выход:
        Новый объект модели
    """
    # 1. Ключевые проверки
    if pseudocount <= 0:
        raise ValueError(f"Параметр pseudocount должен быть строго положительным, получено {pseudocount}")

    if len(labeled_sequences) != len(labeled_paths):
        raise ValueError(f"Количество нуклеотидных последовательностей ({len(labeled_sequences)}) должно совпадать с количеством размеченных путей ({len(labeled_paths)})")

    if len(labeled_sequences) == 0:
        raise ValueError(f"Обучающая выборка не может быть пустой")

    for i, seq in enumerate(labeled_sequences):
        if len(seq) == 0:
            raise ValueError(f"Последовательность под индексом {i} пуста")

    num_states = len(states)
    num_symbols = len(symbols)

    if num_states == 0:
        raise ValueError("Список состояний не может быть пуст")
    if num_symbols == 0:
        raise ValueError("Алфавит символов не может быть пуст")

    # 2. Инициализация нулевых списков
    start_counts = [0.0] * num_states
    trans_counts = [[0.0] * num_states for _ in range(num_states)]
    emiss_counts = [[0.0] * num_symbols for _ in range(num_states)]

    state_to_idx = {state: i for i, state in enumerate(states)}
    symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}

    # 3. Проход по всем парам последовательность-путь
    for idx, (seq, path) in enumerate(zip(labeled_sequences, labeled_paths)):
        if len(seq) != len(path):
            raise ValueError(f"Длина нуклеотидной последовательности должна совпадать с длиной размеченного пути: ошибка в номере {idx}")

        start_state = path[0]
        start_symbol = seq[0]

        if start_state not in state_to_idx:
            raise ValueError(f"Неизвестное начальное состояние '{start_state}' в пути {idx}. Ожидалось состояния из списка {states}")
        if start_symbol not in symbol_to_idx:
            raise ValueError(f"Неизвестный начальный символ '{start_symbol}' в последовательности {idx}. Ожидался символ из алфавита {symbols}")

        start_state_idx = state_to_idx[start_state]
        start_symbol_idx = symbol_to_idx[start_symbol]

        start_counts[start_state_idx] += 1.0
        emiss_counts[start_state_idx][start_symbol_idx] += 1.0

        state_curr_idx = start_state_idx
        
        for t in range(len(seq) - 1):
            state_next = path[t + 1]
            obs_next = seq[t + 1]

            if state_next not in state_to_idx:
                raise ValueError(f"Неизвестное состояние '{state_next}' на позиции {t + 1} в пути {idx}. Ожидалось состояние из списка {states}")
            if obs_next not in symbol_to_idx:
                raise ValueError(f"Неизвестный символ '{obs_next}' на позиции {t + 1} в последовательности {idx}. Ожидался символ из алфавита {symbols}")

            state_next_idx = state_to_idx[state_next]
            obs_next_idx = symbol_to_idx[obs_next]

            trans_counts[state_curr_idx][state_next_idx] += 1.0
            emiss_counts[state_next_idx][obs_next_idx] += 1.0

            state_curr_idx = state_next_idx

    # 4. Нормализация со сглаживанием Лапласа (count + alpha / total_count + M * alpha)
    start_prob = _normalized_laplas_vector(start_counts, pseudocount)
    trans_prob = _normalized_laplas_mat(trans_counts, pseudocount)
    emiss_prob = _normalized_laplas_mat(emiss_counts, pseudocount)

    # 5. Перевод в словрь
    start_prob_dict = {state: start_prob[i] for i, state in enumerate(states)}
    trans_prob_dict = {
        u: {v: trans_prob[i][j] for j, v in enumerate(states)}
        for i, u in enumerate(states)
    }
    emiss_prob_dict = {
        u: {s: emiss_prob[i][k] for k, s in enumerate(symbols)}
        for i, u in enumerate(states)
    }

    # 6. Возврат обновленной модели
    return HMM(
        states=states,
        symbols=symbols,
        start_probs=start_prob_dict,
        transitions=trans_prob_dict,
        emissions=emiss_prob_dict,
        tol=tol,
    )


def baum_welch(hmm: HMM, sequences: Sequence[Sequence[str]], *, max_iter: int = 100, tol: float = DEFAULT_TOL) -> tuple[HMM, list[float]]:
    """
    Алгоритм Баума-Велша с EM-моделью обучения (Expaction / Maximization)
    Вход:
        Начальная модель, неразмеченные последовательности, макс. кол-во итераций, tol
    Выход:
        Обученная модель, история правдоподобия
    """
    # 1. Ключевые проверки
    if len(sequences) == 0:
        raise ValueError("Список обучающих последовательностей не может быть пустым")

    for idx, seq in enumerate(sequences):
        if len(seq) == 0:
            raise ValueError(f"Последовательность под индексом {idx} пуста")

    if max_iter <= 0:
        raise ValueError(f"Параметр max_iter должен быть строго положительным целым числом, получено {max_iter}")

    if tol < 0.0:
        raise ValueError(f"Параметр tol должен быть неотрицательным, получено {tol}")

    # 2. Инциализируем начальные переменные
    curr_model = hmm
    history_likelihoods = []

    N = len(hmm.states)
    K = len(hmm.symbols)

    symbol_to_idx = hmm.symbol_to_idx

    # 3. Проход по всем итерациям с вычислением EM-шагов
    for iter in range(max_iter):
        valid_seqs_count = 0
        epoch_log_likelihoods = []

        accum_starts = [0.0] * N
        accum_trans = [[0.0] * N for _ in range(N)]
        accum_emiss = [[0.0] * K for _ in range(N)]

        # ШАГ E: Expaction
        log_trans = curr_model.log_trans_mat
        log_emiss = curr_model.log_emission_mat

        for seq in sequences:
            T = len(seq)
            log_alpha, log_likelihood = forward(curr_model, seq)

            if log_likelihood == NEG_INF:
                continue

            log_beta = backward(curr_model, seq)
            gamma = posterior_probabilities(log_alpha, log_beta, log_likelihood)

            valid_seqs_count += 1
            epoch_log_likelihoods.append(log_likelihood)

            # Аккумулирование стартовых вероятностей
            gamma_0 = gamma[0]
            for i in range(N):
                accum_starts[i] += gamma_0[i]

            # Потоковое аккумулирование вероятностей переходов (расчет кси на ходу)
            for t in range(T - 1):
                log_alpha_t = log_alpha[t]
                log_beta_next = log_beta[t + 1]
                obs_next_idx = symbol_to_idx[seq[t + 1]]
                
                for i in range(N):
                    log_alpha_ti = log_alpha_t[i]

                    if log_alpha_ti == NEG_INF:
                        continue

                    trans_row = log_trans[i]
                    accum_row = accum_trans[i]

                    for j in range(N):
                        log_trans_ij = trans_row[j]

                        if log_trans_ij == NEG_INF:
                            continue

                        log_xi = (
                            log_alpha_ti
                            + log_trans_ij
                            + log_emiss[j][obs_next_idx]
                            + log_beta_next[j]
                            - log_likelihood
                        )
                        accum_row[j] += safe_exp(log_xi)

            # аккумулирование вероятнсотей эмиссий
            for t in range(len(seq)):
                obs_idx = symbol_to_idx[seq[t]]
                gamma_t = gamma[t]
                for i in range(N):
                    accum_emiss[i][obs_idx] += gamma_t[i]

        if valid_seqs_count == 0:
            raise ValueError("Обучение невозможно: суммарное правдоподобие последовательностей равно -inf")

        total_epoche_likelihood = fsum(epoch_log_likelihoods)
        history_likelihoods.append(total_epoche_likelihood)

        if iter > 0:
            delta = total_epoche_likelihood - history_likelihoods[-2]
            if 0.0 <= delta <= tol:
                break

        # ШАГ M: Maximization
        old_start = [safe_exp(val) for val in curr_model.log_start_prob]
        old_trans = [[safe_exp(val) for val in row] for row in curr_model.log_trans_mat]
        old_emiss = [[safe_exp(val) for val in row] for row in curr_model.log_emission_mat]

        accum_starts_norm = _normalized_vector(accum_starts, old_start)
        accum_trans_norm = _normalized_mat(accum_trans, old_trans)
        accum_emiss_norm = _normalized_mat(accum_emiss, old_emiss)

        # Перевод в словрь
        accum_starts_norm_dict = {state: accum_starts_norm[i] for i, state in enumerate(curr_model.states)}
        accum_trans_norm_dict = {
            u: {v: accum_trans_norm[i][j] for j, v in enumerate(curr_model.states)}
            for i, u in enumerate(curr_model.states)
        }
        accum_emiss_norm_dict = {
            u: {s: accum_emiss_norm[i][k] for k, s in enumerate(curr_model.symbols)}
            for i, u in enumerate(curr_model.states)
        }

        # Обновляем модель с новыми параметрами
        curr_model = HMM(
            states=hmm.states,
            symbols=hmm.symbols,
            start_probs=accum_starts_norm_dict,
            transitions=accum_trans_norm_dict,
            emissions=accum_emiss_norm_dict,
            tol=hmm.tol,
        )

    return curr_model, history_likelihoods