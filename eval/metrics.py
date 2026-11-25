from collections.abc import Sequence
from dataclasses import dataclass
import math

from bio_io.bed import BedRecord


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """
    Набор метрик качества на уровне позиций или интервалов
    """

    tp: int
    fp: int
    fn: int
    tn: int | None
    precision: float
    recall: float
    f1: float
    mcc: float | None

    def to_dict(self) -> dict[str, float]:
        data: dict[str, float] = {
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }
        if self.tn is not None:
            data["tn"] = float(self.tn)
        if self.mcc is not None:
            data["mcc"] = self.mcc
        return data


def _build_metrics(
    tp: int,
    fp: int,
    fn: int,
    tn: int | None = None,
) -> ClassificationMetrics:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    mcc: float | None = None
    if tn is not None:
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0.0 else 0.0

    return ClassificationMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=mcc,
    )


def evaluate_positions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> ClassificationMetrics:
    """
    Считает TP, FP, FN, TN, Precision, Recall, F1 и MCC по бинарным маскам
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Длины масок не совпадают: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    tp = fp = fn = tn = 0
    for truth_val, pred_val in zip(y_true, y_pred):
        t_pos = bool(truth_val)
        p_pos = bool(pred_val)
        if t_pos and p_pos:
            tp += 1
        elif not t_pos and p_pos:
            fp += 1
        elif t_pos and not p_pos:
            fn += 1
        else:
            tn += 1

    return _build_metrics(tp=tp, fp=fp, fn=fn, tn=tn)


def interval_overlap(a: BedRecord, b: BedRecord) -> int:
    """
    Возвращает длину пересечения двух BED-интервалов в нуклеотидах
    """
    if a.chrom != b.chrom:
        return 0
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def interval_iou(a: BedRecord, b: BedRecord) -> float:
    """
    Возвращает Intersection over Union (индекс Жаккара) для двух интервалов
    """
    inter = interval_overlap(a, b)
    if inter == 0:
        return 0.0
    union = a.length + b.length - inter
    return inter / union if union > 0 else 0.0


def evaluate_intervals(
    truth: Sequence[BedRecord],
    preds: Sequence[BedRecord],
    min_iou: float = 0.0,
) -> ClassificationMetrics:
    """
    Оценивает совпадение предсказанных интервалов с эталоном (сопоставление 1-к-1).
    При min_iou=0.0 засчитывается любое пересечение длиной >= 1 п.н.
    """
    if not (0.0 <= min_iou <= 1.0):
        raise ValueError(f"min_iou должно быть в [0, 1], получено {min_iou}")

    # Собираем все допустимые пары (score, pred_idx, truth_idx) и жадно матчим лучшие
    candidates: list[tuple[float, int, int]] = []
    for p_idx, pred_rec in enumerate(preds):
        for t_idx, truth_rec in enumerate(truth):
            overlap = interval_overlap(pred_rec, truth_rec)
            if overlap == 0:
                continue
            iou = interval_iou(pred_rec, truth_rec)
            if iou >= min_iou:
                candidates.append((iou, p_idx, t_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_preds: set[int] = set()
    matched_truth: set[int] = set()
    for _, p_idx, t_idx in candidates:
        if p_idx not in matched_preds and t_idx not in matched_truth:
            matched_preds.add(p_idx)
            matched_truth.add(t_idx)

    tp = len(matched_truth)
    fp = len(preds) - tp
    fn = len(truth) - tp
    return _build_metrics(tp=tp, fp=fp, fn=fn, tn=None)