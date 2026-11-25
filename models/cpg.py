from collections.abc import Sequence

from hmm.core import HMM
from hmm.viterbi import viterbi
from bio_io.bed import BedRecord, states_to_bed

DNA_ALPHABET: tuple[str, ...] = ("A", "C", "G", "T")
CPG_PLUS_STATES: tuple[str, ...] = ("A+", "C+", "G+", "T+")
CPG_MINUS_STATES: tuple[str, ...] = ("A-", "C-", "G-", "T-")
CPG_STATES: tuple[str, ...] = CPG_PLUS_STATES + CPG_MINUS_STATES
CPG_PLUS_SET: frozenset[str] = frozenset(CPG_PLUS_STATES)

# Классические динуклеотидные вероятности переходов (Durbin et al.)
PLUS_TRANSITIONS: dict[str, dict[str, float]] = {
    "A": {"A": 0.180, "C": 0.274, "G": 0.426, "T": 0.120},
    "C": {"A": 0.171, "C": 0.368, "G": 0.274, "T": 0.187},
    "G": {"A": 0.161, "C": 0.339, "G": 0.375, "T": 0.125},
    "T": {"A": 0.079, "C": 0.355, "G": 0.384, "T": 0.182},
}

MINUS_TRANSITIONS: dict[str, dict[str, float]] = {
    "A": {"A": 0.300, "C": 0.205, "G": 0.285, "T": 0.210},
    "C": {"A": 0.322, "C": 0.298, "G": 0.078, "T": 0.302},
    "G": {"A": 0.248, "C": 0.246, "G": 0.298, "T": 0.208},
    "T": {"A": 0.177, "C": 0.239, "G": 0.292, "T": 0.292},
}


def build_cpg_model(
    p_enter: float = 0.001,
    p_exit: float = 0.005,
    p_init_plus: float = 0.1,
) -> HMM:
    """
    Создает 8-состоянийную модель HMM для поиска CpG-островов
    """
    if not (0.0 < p_enter < 1.0):
        raise ValueError(f"p_enter должно быть в (0, 1), получено {p_enter}")
    if not (0.0 < p_exit < 1.0):
        raise ValueError(f"p_exit должно быть в (0, 1), получено {p_exit}")
    if not (0.0 < p_init_plus < 1.0):
        raise ValueError(f"p_init_plus должно быть в (0, 1), получено {p_init_plus}")

    start_prob: dict[str, float] = {}
    for base in DNA_ALPHABET:
        start_prob[f"{base}+"] = p_init_plus / 4.0
        start_prob[f"{base}-"] = (1.0 - p_init_plus) / 4.0

    trans_prob: dict[str, dict[str, float]] = {}
    for b_from in DNA_ALPHABET:
        plus_row: dict[str, float] = {}
        minus_row: dict[str, float] = {}
        for b_to in DNA_ALPHABET:
            plus_row[f"{b_to}+"] = (1.0 - p_exit) * PLUS_TRANSITIONS[b_from][b_to]
            plus_row[f"{b_to}-"] = p_exit * MINUS_TRANSITIONS[b_from][b_to]

            minus_row[f"{b_to}-"] = (1.0 - p_enter) * MINUS_TRANSITIONS[b_from][b_to]
            minus_row[f"{b_to}+"] = p_enter * PLUS_TRANSITIONS[b_from][b_to]

        trans_prob[f"{b_from}+"] = plus_row
        trans_prob[f"{b_from}-"] = minus_row

    emission_prob: dict[str, dict[str, float]] = {}
    for state in CPG_STATES:
        emitted_base = state[0]
        emission_prob[state] = {
            base: (1.0 if base == emitted_base else 0.0) for base in DNA_ALPHABET
        }

    return HMM(
        states=CPG_STATES,
        symbols=DNA_ALPHABET,
        start_probs=start_prob,
        transitions=trans_prob,
        emissions=emission_prob,
    )


def sequence_and_mask_to_states(sequence: str, mask: Sequence[int]) -> list[str]:
    """
    Превращает ДНК и бинарную маску из bed_to_mask в разметку состояний для train_supervised
    """
    if len(sequence) != len(mask):
        raise ValueError(
            f"Длина sequence ({len(sequence)}) не совпадает с длиной mask ({len(mask)})"
        )
    states: list[str] = []
    for base, flag in zip(sequence.upper(), mask):
        if base not in DNA_ALPHABET:
            raise ValueError(f"Недопустимый нуклеотид {base!r}; ожидаются A, C, G, T")
        suffix = "+" if flag else "-"
        states.append(f"{base}{suffix}")
    return states


def predict_cpg_islands(
    model: HMM,
    chrom: str,
    sequence: str,
    min_length: int = 1,
) -> tuple[list[BedRecord], float]:
    """
    Запускает Витерби и возвращает найденные CpG-острова в формате BedRecord
    """
    if min_length < 1:
        raise ValueError(f"min_length должно быть >= 1, получено {min_length}")

    seq = sequence.upper()
    log_score, path = viterbi(model, seq)
    records = states_to_bed(
        chrom=chrom,
        states=path,
        target_states=CPG_PLUS_SET,
        name="CpG_island",
        strand="+",
    )
    if min_length > 1:
        records = [rec for rec in records if rec.length >= min_length]
    return records, log_score