from collections.abc import Set, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

VALID_STRANDS: frozenset[str] = frozenset({"+", "-", "."})


@dataclass(frozen=True, slots=True)
class BedRecord:
    """
    Геномный интервал в формате BED6 (0-based, полуоткрытый [start, end))
    """

    chrom: str
    start: int
    end: int
    name: str = "."
    score: float = 0.0
    strand: str = "."

    def __post_init__(self) -> None:
        if not self.chrom or any(ch.isspace() for ch in self.chrom):
            raise ValueError(f"Некорректное имя хромосомы: {self.chrom!r}")
        if self.start < 0:
            raise ValueError(f"Координата start должна быть >= 0, получено {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"Координата end ({self.end}) должна быть строго больше start ({self.start})"
            )
        if self.strand not in VALID_STRANDS:
            raise ValueError(
                f"Некорректная цепь {self.strand!r}; допустимы: '+', '-', '.'"
            )

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_bed_line(self, bed6: bool = True) -> str:
        if not bed6:
            return f"{self.chrom}\t{self.start}\t{self.end}"
        score_str = f"{self.score:.4f}".rstrip("0").rstrip(".") if self.score != 0 else "0"
        return f"{self.chrom}\t{self.start}\t{self.end}\t{self.name}\t{score_str}\t{self.strand}"


def iter_bed(filepath: str | Path) -> Iterator[BedRecord]:
    """
    Потоково читает записи BED3/BED6, пропуская пустые строки и заголовки
    """
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("track ", "browser ")) or line in ("track", "browser"):
                continue

            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"Строка {line_num}: ожидалось минимум 3 колонки BED, получено {len(parts)}"
                )

            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"Строка {line_num}: координаты start/end должны быть целыми числами"
                ) from exc

            name = parts[3] if len(parts) >= 4 and parts[3] else "."

            score = 0.0
            if len(parts) >= 5 and parts[4] != ".":
                try:
                    score = float(parts[4])
                except ValueError as exc:
                    raise ValueError(
                        f"Строка {line_num}: некорректное значение score={parts[4]!r}"
                    ) from exc

            strand = parts[5] if len(parts) >= 6 else "."

            try:
                yield BedRecord(
                    chrom=chrom,
                    start=start,
                    end=end,
                    name=name,
                    score=score,
                    strand=strand,
                )
            except ValueError as exc:
                raise ValueError(f"Строка {line_num}: {exc}") from exc


def read_bed(filepath: str | Path) -> list[BedRecord]:
    """Читает весь BED-файл в список записей."""
    return list(iter_bed(filepath))


def write_bed(
    records: Iterable[BedRecord],
    filepath: str | Path,
    bed6: bool = True,
    track_name: str | None = None,
) -> None:
    """
    Записывает коллекцию BedRecord в файл
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if track_name is not None:
            handle.write(f'track name="{track_name}"\n')
        for record in records:
            handle.write(record.to_bed_line(bed6=bed6) + "\n")


def states_to_bed(
    chrom: str,
    states: Sequence[str],
    target_states: Set[str],
    name: str = "feature",
    strand: str = ".",
    scores: Sequence[float] | None = None,
    offset: int = 0,
) -> list[BedRecord]:
    """
    Преобразует последовательность скрытых состояний HMM в слитые BED-интервалы
    """
    if offset < 0:
        raise ValueError(f"Смещение offset должно быть >= 0, получено {offset}")
    if scores is not None and len(scores) != len(states):
        raise ValueError(
            f"Длина scores ({len(scores)}) не совпадает с длиной states ({len(states)})"
        )

    records: list[BedRecord] = []
    in_feature = False
    start_idx = 0

    for idx, state in enumerate(states):
        is_target = state in target_states
        if is_target and not in_feature:
            in_feature = True
            start_idx = idx
        elif not is_target and in_feature:
            avg_score = (
                sum(scores[start_idx:idx]) / (idx - start_idx)
                if scores is not None
                else 0.0
            )
            records.append(
                BedRecord(
                    chrom=chrom,
                    start=start_idx + offset,
                    end=idx + offset,
                    name=name,
                    score=avg_score,
                    strand=strand,
                )
            )
            in_feature = False

    if in_feature:
        end_idx = len(states)
        avg_score = (
            sum(scores[start_idx:end_idx]) / (end_idx - start_idx)
            if scores is not None
            else 0.0
        )
        records.append(
            BedRecord(
                chrom=chrom,
                start=start_idx + offset,
                end=end_idx + offset,
                name=name,
                score=avg_score,
                strand=strand,
            )
        )

    return records


def bed_to_mask(records: Iterable[BedRecord], chrom: str, seq_length: int) -> list[int]:
    """
    Превращает BED-интервалы для заданной хромосомы в бинарную маску длины seq_length
    """
    if seq_length < 0:
        raise ValueError(f"Длина последовательности должна быть >= 0, получено {seq_length}")

    mask = [0] * seq_length
    for rec in records:
        if rec.chrom != chrom:
            continue
        clipped_start = max(0, rec.start)
        clipped_end = min(seq_length, rec.end)
        for pos in range(clipped_start, clipped_end):
            mask[pos] = 1
    return mask