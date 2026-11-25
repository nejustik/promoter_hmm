from collections.abc import Iterable, Iterator
from typing import NamedTuple
from hmm.constants import CANONICAL_DNA, IUPAC_AMBIGUOUS_DNA
from pathlib import Path

_COMPLEMENT_TRANS = str.maketrans(
    "ACGTURYMKBDHVNacgturymkbdhvn-",
    "TGCAAYRKMVHDBNtgcaayrkmvhdbn-"
)


class FastaRecord(NamedTuple):
    id: str
    description: str
    seq: str

    def __len__(self) -> int:
        return len(self.seq)


def read_fasta(source_path: str | Path) -> Iterator[FastaRecord]:
    """
    Парсинг FASTA файла с выводом итеративных конструкций FastaRecord, содержащих по одноё последовательности с соответствующим хедером
    """
    with open(source_path, "r", encoding="utf-8") as f:
        curr_id = None
        curr_desc = ""
        line_num = 0
        prev_header_num = 0

        # 1. Первичный цикл поиска первого хедера с его валидностью
        for line_raw in f:
            line_num += 1
            line = line_raw.strip()
            if not line:
                continue

            if not line.startswith('>'):
                    raise ValueError(f"FASTA файл должен начинаться с '>' для хедера, получена последовательность: {line[:20]!r}")

            curr_title = line[1:].strip().split(maxsplit=1)
            if not curr_title:
                raise ValueError(f"Найден пустой хедер для FASTA последовательности (номер строки '{line_num}'), содержащий только '>'")
            curr_id = curr_title[0]
            curr_desc = curr_title[1] if len(curr_title) == 2 else ""
            prev_header_num = line_num

            break

        # 2. Проверка на пустой файл
        if curr_id is None:
            raise ValueError(f"Файл '{source_path}' пуст")

        buffer = []

        # 3. Основной перебор строк
        for i, line_raw in enumerate(f, start=line_num + 1):
            line = line_raw.strip()
            if not line:
                continue

            if line.startswith('>'):
                total_seq = "".join(buffer)
                if not total_seq:
                    raise ValueError(f"Запись '{curr_id}' (номер строки '{prev_header_num}') имеет пустую последовательность")
                yield FastaRecord(curr_id, curr_desc, total_seq)
                
                curr_title = line[1:].strip().split(maxsplit=1)
                if not curr_title:
                    raise ValueError(f"Найден пустой хедер для FASTA последовательности (номер строки '{i}'), содержащий только '>'")
                curr_id = curr_title[0]
                curr_desc = curr_title[1] if len(curr_title) == 2 else ""
                prev_header_num = i

                buffer = []
            else:
                buffer.append("".join(line.split()).upper())

        # 4. Запись последней последовательности в файле 
        total_seq = "".join(buffer)
        if not total_seq:
            raise ValueError(f"Запись '{curr_id}' (номер строки '{prev_header_num}') имеет пустую последовательность")
        yield FastaRecord(curr_id, curr_desc, total_seq)


def is_valid_dna(seq: str, *, allow_ambiguous: bool = False) -> bool:
    """
    Проверять состоит ли последовательность только из разрешенных нуклеотидов
    (априори считаем, что символы в верхнем регистре)
    """
    if not seq:
        return False
    valid_alphabet = IUPAC_AMBIGUOUS_DNA if allow_ambiguous else CANONICAL_DNA

    return set(seq).issubset(valid_alphabet)


def reverse_complement(seq: str) -> str:
    """
    Возвращает обратно-комплементарную последовательность ДНК (5' -> 3')
    """
    if not seq:
        return ""

    return seq.translate(_COMPLEMENT_TRANS)[::-1]


def calculate_gc_content(seq: str, *, count_ambiguous: bool = False) -> float:
    """
    Вычисляет долю GC-нуклеотидов в последовательности
    """
    if not seq:
        return 0.0

    seq_upper = seq.upper()
    g = seq_upper.count("G")
    c = seq_upper.count("C")

    if not count_ambiguous:
        a = seq_upper.count("A")
        t = seq_upper.count("T")

        canonical_len = a + c + g + t
        if canonical_len == 0:
            return 0.0
        
        return (g + c) / canonical_len

    return (g + c) / len(seq_upper)


def write_fasta(records: Iterable[FastaRecord], target_path: str | Path, *, line_width: int = 60) -> None:
    """
    Записывает переданные последовательности через структуру FastaRecord в нужный FASTA файл по пути target_path
    """
    if line_width <= 0:
        raise ValueError(f"Переменная line_width должна иметь положительное значение (>0), получено '{line_width}'")

    if not target_path:
        raise ValueError(f"Передан пустой путь записи FASTA файла (target_path)")

    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            id_clean = record.id.strip()
            if not id_clean:
                raise ValueError(f"Невозможно записать FastaRecord (номер '{i}') с пустым id")

            desc_clean = record.description.strip()
            if desc_clean:
                f.write(f">{id_clean} {desc_clean}\n")
            else:
                f.write(f">{id_clean}\n")

            seq = record.seq
            seq_len = len(seq)

            if not seq_len:
                raise ValueError(f"Невозможно записать FastaRecord (номер '{i}', id '{id_clean}') с пустой последовательностью")
            
            for start in range(0, seq_len, line_width):
                f.write(seq[start:start + line_width])
                f.write("\n")