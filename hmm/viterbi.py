from collections.abc import Sequence
from hmm.core import HMM
from hmm.logmath import NEG_INF


def viterbi(hmm: HMM, sequence: Sequence[str]) -> tuple[float, list[str]]:
    """
    Находит наиболее вероятный путь скрытых состояний для последовательности
    Args:
        hmm: Экземпляр обученной модели HMM с логарифмическими весами
        sequence: Последовательность наблюдаемых символов (нуклеотидов)
    Returns:
        Кортеж (final_best_score_log, best_path):
            final_best_score_log: Максимальный логарифм вероятности найденного пути
            best_path: Список имен скрытых состояний, породивших последовательность
    """
    seq_len = len(sequence)
    if seq_len == 0:
        raise ValueError("Последовательность не может быть пустой")

    # 1. Конвертация символов последовательности в индексы
    symbol_to_idx = hmm.symbol_to_idx
    obs_idx = [-1] * seq_len
    for t, symbol in enumerate(sequence):
        if symbol not in symbol_to_idx:
            raise ValueError(f"Неизвестный символ '{symbol}' на позиции {t}. Ожидались символы из алфавита: {hmm.symbols}")
        obs_idx[t] = symbol_to_idx[symbol]

    log_start = hmm.log_start_prob
    log_trans = hmm.log_trans_mat
    log_emiss= hmm.log_emission_mat

    num_states = len(hmm.states)

    #2. Инициализация базового шага (t = 0)
    first_obs = obs_idx[0]
    v_prev = [NEG_INF] * num_states
    for state_idx in range(num_states):
        v_prev[state_idx] = log_start[state_idx] + log_emiss[state_idx][first_obs]

    if all(start_prob == NEG_INF for start_prob in v_prev):
        raise ValueError(f"Невозможно начать генерацию: символ '{sequence[0]}' на позиции 0 имеет нулевую вероятность появления во всех доступных начальных состояниях")

    # 3. Плоский буфер обратных указателей: backpointers[t * num_states + state_idx]
    backpointers = [-1] * (seq_len * num_states)

    # 4. Прямой проход по всей последоватлеьности (t = 1 .. T-1)
    for t in range(1, seq_len):
        obs = obs_idx[t]
        t_offset = t * num_states
        v_curr = [NEG_INF] * num_states

        for curr_state in range(num_states):
            emiss_score = log_emiss[curr_state][obs]
            if emiss_score == NEG_INF:
                continue

            best_val = NEG_INF
            best_prev_state = -1

            for prev_state in range(num_states):
                prev_val = v_prev[prev_state]
                if prev_val == NEG_INF:
                    continue

                trans_score = log_trans[prev_state][curr_state]
                if trans_score == NEG_INF:
                    continue

                candidate_val = prev_val + trans_score
                if candidate_val > best_val:
                    best_val = candidate_val
                    best_prev_state = prev_state

            if best_val != NEG_INF and best_prev_state != -1:
                v_curr[curr_state] = best_val + emiss_score
                backpointers[t_offset + curr_state] = best_prev_state

        if all(prob == NEG_INF for prob in v_curr):
            raise ValueError(f"Скрытый путь прерван на позиции {t} (символ '{sequence[t]}'). Все возможные переходы в текущие состояния имеют нулевую вероятность")

        v_prev = v_curr

    # 5. Терминация: выбор наилучшего финального состояния
    final_best_score_log = NEG_INF
    final_best_state = -1
    for state in range(num_states):
        if v_prev[state] > final_best_score_log:
            final_best_score_log = v_prev[state]
            final_best_state = state
    if final_best_state == -1:
        raise ValueError("Не удалось найти валидный путь: все финальные состояния недостижимы")

    # 6. Обратный ход (traceback): восстановление последовательности от конца к началу
    path_idx = [-1] * seq_len
    curr_state = final_best_state
    path_idx[seq_len - 1] = curr_state

    for t in range(seq_len - 1, 0, -1):
        curr_state = backpointers[t * num_states + curr_state]
        path_idx[t - 1] = curr_state

    if any(idx == -1 for idx in path_idx):
        raise ValueError("Сбой восстановления пути: обнаружены незаполненные позиции")

    # 7. Конвертация индексов в имена состояний
    idx_to_state = hmm.idx_to_state
    best_path = [idx_to_state[idx] for idx in path_idx]

    return final_best_score_log, best_path