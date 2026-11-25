from pathlib import Path

from eval.metrics import ClassificationMetrics, evaluate_intervals, evaluate_positions
from hmm.baum_welch import baum_welch, train_supervised
from hmm.core import HMM
from hmm.viterbi import viterbi
from bio_io.bed import BedRecord, bed_to_mask, read_bed, states_to_bed, write_bed
from bio_io.db import Database
from bio_io.fasta import calculate_gc_content, read_fasta
from models.cpg import (
    CPG_PLUS_SET,
    CPG_STATES,
    DNA_ALPHABET,
    build_cpg_model,
    sequence_and_mask_to_states,
)

TRAIN_FASTA = Path("data/chr21_train.fa")
TRAIN_BED = Path("data/chr21_train.bed")
TRAIN_CHROM = "chr21"
TRAIN_OFFSET = 44_000_000

TEST_FASTA = Path("data/chr22_slice.fa")
TEST_BED = Path("data/chr22_cpg.bed")
TEST_CHROM = "chr22"
TEST_OFFSET = 20_000_000

DB_PATH = Path("promoter.db")
MIN_ISLAND_LEN = 200
BW_MAX_BP = 100_000  # Ограничение длины для быстрого прогона EM на чистом Python


def load_clean_dna(fasta_path: Path) -> tuple[str, str, str]:
    record = next(read_fasta(fasta_path))
    clean_seq = "".join(base for base in record.seq if base in {"A", "C", "G", "T"})
    return record.id, record.description, clean_seq


def shift_bed_to_local(records: list[BedRecord], offset: int, seq_len: int) -> list[BedRecord]:
    local: list[BedRecord] = []
    for r in records:
        s = max(0, r.start - offset)
        e = min(seq_len, r.end - offset)
        if e > s:
            local.append(BedRecord(r.chrom, s, e, r.name, r.score, r.strand))
    return local


def run_and_evaluate(
    db: Database,
    seq_id: int,
    model_name: str,
    model: HMM,
    test_seq: str,
    test_truth_local: list[BedRecord],
    test_mask: list[int],
) -> tuple[str, float, ClassificationMetrics, ClassificationMetrics]:
    log_score, path = viterbi(model, test_seq)
    preds_local = states_to_bed(
        chrom=TEST_CHROM,
        states=path,
        target_states=CPG_PLUS_SET,
        name=f"{model_name}_pred",
        strand="+",
        offset=0,
    )
    preds_local = [rec for rec in preds_local if rec.length >= MIN_ISLAND_LEN]

    y_pred = bed_to_mask(preds_local, TEST_CHROM, len(test_seq))
    pos_m = evaluate_positions(test_mask, y_pred)
    int_m = evaluate_intervals(test_truth_local, preds_local, min_iou=0.0)

    preds_global = [
        BedRecord(r.chrom, r.start + TEST_OFFSET, r.end + TEST_OFFSET, r.name, r.score, r.strand)
        for r in preds_local
    ]
    write_bed(preds_global, Path(f"output/{model_name}.bed"), track_name=model_name)

    mod_id = db.save_model(model_name, "cpg", model.to_dict())
    run_id = db.create_run(
        model_id=mod_id,
        sequence_id=seq_id,
        algorithm="viterbi",
        log_likelihood=log_score,
        metrics=pos_m.to_dict(),
    )
    db.save_predictions(run_id, preds_global)
    return model_name, log_score, pos_m, int_m


def print_summary_table(
    results: list[tuple[str, float, ClassificationMetrics, ClassificationMetrics]],
    total_truth: int,
) -> None:
    header = (
        f"{'Модель':<18} | {'Pos P':>6} {'Pos R':>6} {'Pos F1':>7} {'MCC':>7} | "
        f"{'TP/All':>7} {'FP':>4} {'FN':>4} {'Int F1':>7}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for name, _, pos_m, int_m in results:
        mcc_str = f"{pos_m.mcc:.3f}" if pos_m.mcc is not None else "  n/a"
        tp_str = f"{int_m.tp}/{total_truth}"
        print(
            f"{name:<18} | {pos_m.precision:6.3f} {pos_m.recall:6.3f} {pos_m.f1:7.3f} {mcc_str:>7} | "
            f"{tp_str:>7} {int_m.fp:4d} {int_m.fn:4d} {int_m.f1:7.3f}"
        )
    print(sep + "\n")


def main() -> None:
    _, _, train_seq = load_clean_dna(TRAIN_FASTA)
    _, test_desc, test_seq = load_clean_dna(TEST_FASTA)

    train_truth_local = shift_bed_to_local(read_bed(TRAIN_BED), TRAIN_OFFSET, len(train_seq))
    test_truth_local = shift_bed_to_local(read_bed(TEST_BED), TEST_OFFSET, len(test_seq))

    train_mask = bed_to_mask(train_truth_local, TRAIN_CHROM, len(train_seq))
    test_mask = bed_to_mask(test_truth_local, TEST_CHROM, len(test_seq))

    print(
        f"Train ({TRAIN_CHROM}): {len(train_seq)} п.н., "
        f"GC={calculate_gc_content(train_seq):.3f}, островов={len(train_truth_local)}"
    )
    print(
        f"Test  ({TEST_CHROM}): {len(test_seq)} п.н., "
        f"GC={calculate_gc_content(test_seq):.3f}, островов={len(test_truth_local)}"
    )

    # 1. Baseline (константы Дурбина)
    baseline_model = build_cpg_model(p_enter=1e-5, p_exit=0.005)

    # 2. Supervised обучение на chr21
    print("\nОбучение train_supervised на chr21...")
    train_states = sequence_and_mask_to_states(train_seq, train_mask)
    supervised_model = train_supervised(
        states=CPG_STATES,
        symbols=DNA_ALPHABET,
        labeled_sequences=[train_seq],
        labeled_paths=[train_states],
        pseudocount=0.01,
    )

    # 3. Baum-Welch (EM) дообучение
    print("Запуск Baum-Welch (EM)...")
    bw_seq = train_seq[:BW_MAX_BP]
    chunk_size = 10_000
    bw_chunks = [bw_seq[i : i + chunk_size] for i in range(0, len(bw_seq), chunk_size)]
    bw_model, ll_history = baum_welch(supervised_model, bw_chunks, max_iter=8, tol=1e-2)
    print(f"Baum-Welch: {len(ll_history)} итераций, ln P(O|lambda): {ll_history[0]:.1f} -> {ll_history[-1]:.1f}")

    # 4. Оценка на независимом тесте chr22 и запись в SQLite
    with Database(DB_PATH) as db:
        seq_id = db.upsert_sequence(
            seq_id=TEST_CHROM,
            length=len(test_seq),
            gc_content=calculate_gc_content(test_seq),
            description=test_desc,
            source_path=str(TEST_FASTA),
        )
        results = [
            run_and_evaluate(db, seq_id, "cpg_baseline", baseline_model, test_seq, test_truth_local, test_mask),
            run_and_evaluate(db, seq_id, "cpg_supervised", supervised_model, test_seq, test_truth_local, test_mask),
            run_and_evaluate(db, seq_id, "cpg_baum_welch", bw_model, test_seq, test_truth_local, test_mask),
        ]

    print_summary_table(results, total_truth=len(test_truth_local))
    print(f"Все 3 модели и прогоны сохранены в {DB_PATH}, BED-файлы лежат в output/.")


if __name__ == "__main__":
    main()