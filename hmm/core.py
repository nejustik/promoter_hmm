from hmm.constants import NEG_INF, DEFAULT_TOL
from collections.abc import Mapping, Sequence
from typing import Any, Self
from math import fsum
from hmm.logmath import safe_log, safe_exp


class HMM:
    def __init__(self, states: Sequence[str], symbols: Sequence[str], start_probs: Mapping[str, float], transitions: Mapping[str, Mapping[str, float]], emissions: Mapping[str, Mapping[str, float]], *, tol: float = DEFAULT_TOL) -> None:
        self.states = tuple(states)
        self.symbols = tuple(symbols)
        self.tol = tol

        self.state_to_idx = {state: i for i, state in enumerate(self.states)}
        self.idx_to_state = {i: state for i, state in enumerate(self.states)}
        self.symbol_to_idx = {sym: i for i, sym in enumerate(self.symbols)}
        self.idx_to_symbol = {i: sym for i, sym in enumerate(self.symbols)}

        self._validate_inputs(start_probs, transitions, emissions, tol)

        self.log_start_prob = self._build_log_start_prob(start_probs)
        self.log_trans_mat = self._build_log_trans_mat(transitions)
        self.log_emission_mat = self._build_log_emission_mat(emissions)

    @staticmethod
    def _normalize_distribution(prob_dict: Mapping[str, float]) -> dict[str, float]:
        """
        Нормализует словарь вероятностей так, чтобы их сумма строго равнялась 1.0
        Защищает от накопления погрешностей округления при переходе log -> exp
        """
        if not prob_dict:
            raise ValueError("Невозможно нормализовать пустое распределение вероятностей")

        for key, val in prob_dict.items():
            if val < 0.0:
                raise ValueError(f"Вероятность при нормализации не может быть отрицательной: ключ '{key}' имеет значение {val}")

        total = fsum(prob_dict.values())
        if total <= 0.0:
            raise ValueError(f"Сумма вероятностей распределения должна быть строго больше 0, получено {total}")
        
        return {key: val / total for key, val in prob_dict.items()}

    def _check_distribution(self, label: str, probs: Mapping[str, float], tol: float) -> None:
        """
        Проверяет диапазон каждой вероятности и сумму блока, равную 1.0
        """
        for key, prob in probs.items():
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f'Вероятность "{key}" в блоке "{label}" вне [0, 1]: {prob}')

        total = fsum(probs.values())
        if abs(1.0 - total) > tol:
            raise ValueError(f'Сумма блока "{label}" должна быть 1.0, получено: {total}')

    def _validate_inputs(self, start_probs: Mapping[str, float], transitions: Mapping[str, Mapping[str, float]], emissions: Mapping[str, Mapping[str, float]], tol: float) -> None:
        """
        Выполняет комплексную проверку входных параметров модели
        """
        # 1. Проверка уникальности состояний и символов
        if not self.states or not self.symbols:
            raise ValueError("Списки states и symbols не могут быть пустыми")
        if len(self.states) != len(set(self.states)):
            raise ValueError("Список states содержит повторяющиеся имена состояний")
        if len(self.symbols) != len(set(self.symbols)):
            raise ValueError("Список symbols содержит повторяющиеся символы")

        valid_states = set(self.states)
        valid_symbols = set(self.symbols)

        # 2. Валидация начальных вероятнсотей (start_prob)
        unknown_start = set(start_probs.keys()) - valid_states
        if unknown_start:
            raise ValueError(f"Неизвестные состояния в start_probs: {sorted(unknown_start)}")

        self._check_distribution("start_prob", start_probs, tol)

        # 3. Валидация матрицы переходов (transitions)
        unknown_trans_from = set(transitions.keys()) - valid_states
        if unknown_trans_from:
            raise ValueError(f"Неизвестные исходные состояния в transitions: {sorted(unknown_trans_from)}")

        for from_state in self.states:
            if from_state not in transitions:
                raise ValueError(f"Для состояния '{from_state}' не заданы исходящие переходы")

            to_dict = transitions[from_state]
            unknown_trans_to = set(to_dict.keys()) - valid_states
            if unknown_trans_to:
                raise ValueError(f"Неизвестные целевые состояния перехода из '{from_state}': {sorted(unknown_trans_to)}")

            self._check_distribution(f"переходы из {from_state}", to_dict, tol)

        # 4. Валидация матрицы эмиссий (emissions)
        unknown_emiss_from = set(emissions.keys()) - valid_states
        if unknown_emiss_from:
            raise ValueError(f"Неизвестные состояния в emissions: {sorted(unknown_emiss_from)}")

        for state in self.states:
            if state not in emissions:
                raise ValueError(f"Для состояния '{state}' не заданы эмиссии")

            to_dict = emissions[state]
            unknown_syms = set(to_dict.keys()) - valid_symbols
            if unknown_syms:
                raise ValueError(f"Неизвестные символы алфавита в состоянии '{state}': {sorted(unknown_syms)}")

            self._check_distribution(f"эмиссии из {state}", to_dict, tol)

    def _build_log_start_prob(self, start_probs: Mapping[str, float]) -> tuple[float, ...]:
        """
        Создает одномерный кортеж логарифмов начальных вероятностей длины N
        Для отсутствующих состояний выставляет NEG_INF
        """
        temp_start = [NEG_INF] * len(self.states)
        
        for state, prob in start_probs.items():
            idx = self.state_to_idx[state]
            temp_start[idx] = safe_log(prob)

        return tuple(temp_start)

    def _build_log_trans_mat(self, transitions: Mapping[str, Mapping[str, float]]) -> tuple[tuple[float, ...], ...]:
        """
        Создает квадратную матрицу N x N переходов в логарифмической шкале
        Для отсутствующих ребер в графе выставляет NEG_INF
        """
        n = len(self.states)
        temp_mat = [[NEG_INF] * n for _ in range(n)]

        for from_state, to_dict in transitions.items():
            i = self.state_to_idx[from_state]
            for to_state, prob in to_dict.items():
                j = self.state_to_idx[to_state]
                temp_mat[i][j] = safe_log(prob)

        return tuple(tuple(row) for row in temp_mat)

    def _build_log_emission_mat(self, emissions: Mapping[str, Mapping[str, float]]) -> tuple[tuple[float, ...], ...]:
        """
        Создает прямоугольную матрицу N x M эмиссий в логарифмической шкале.
        Для отсутствующих эмиссий выставляет NEG_INF
        """
        n = len(self.states)
        m = len(self.symbols)
        temp_mat = [[NEG_INF] * m for _ in range(n)]

        for state, to_dict in emissions.items():
            i = self.state_to_idx[state]
            for sym, prob in to_dict.items():
                k = self.symbol_to_idx[sym]
                temp_mat[i][k] = safe_log(prob)

        return tuple(tuple(row) for row in temp_mat)

    def to_dict(self) -> dict[str, Any]:
        """
        Экспортирует модель в человекочитаемый словарь с линейными вероятностями [0.0, 1.0]
        Значения восстанавливаются из логарифмов через safe_exp
        Нулевые вероятности (с логарифмом -inf) пропускаются для компактности JSON
        """
        # 1. Начальные вероятности (только ненулевые)
        raw_start_probs = {}
        for i, log_p in enumerate(self.log_start_prob):
            if log_p != NEG_INF:
                raw_start_probs[self.idx_to_state[i]] = safe_exp(log_p)
        start_probs = self._normalize_distribution(raw_start_probs)

        # 2. Матрица переходов (только существующие ребра графа)
        transitions = {}
        for i, row in enumerate(self.log_trans_mat):
            from_state = self.idx_to_state[i]
            row_dict = {}
            for j, log_p in enumerate(row):
                if log_p != NEG_INF:
                    row_dict[self.idx_to_state[j]] = safe_exp(log_p)
            transitions[from_state] = self._normalize_distribution(row_dict)

        # 3. Матрица эмиссий (только ненулевые излучения)
        emissions = {}
        for i, row in enumerate(self.log_emission_mat):
            state = self.idx_to_state[i]
            row_dict = {}
            for k, log_p in enumerate(row):
                if log_p != NEG_INF:
                    row_dict[self.idx_to_symbol[k]] = safe_exp(log_p)
            emissions[state] = self._normalize_distribution(row_dict)

        return {
            "states": list(self.states),
            "symbols": list(self.symbols),
            "start_probs": start_probs,
            "transitions": transitions,
            "emissions": emissions,
            "tol": self.tol,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, tol: float = 1e-5) -> Self:
        """
        Воссоздает экземпляр HMM из сериализованного словаря
        """
        model_tol = data.get("tol", tol)
        return cls(
            states=data["states"],
            symbols=data["symbols"],
            start_probs=data["start_probs"],
            transitions=data["transitions"],
            emissions=data["emissions"],
            tol=model_tol,
        )

    def __repr__(self) -> str:
        """
        Строковое представление модели для удобной отладки
        """
        return f"HMM(num_states={len(self.states)}, num_symbols={len(self.symbols)})"