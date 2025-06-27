from collections import defaultdict
from pyserini.search.lucene import LuceneSearcher
from settings import MAX_SEQUENCE_LENGTH
from utils import get_data, preprocess_case_data, segment_document
import os
import torch
import numpy as np


def evaluate(predictions, golds):
    preds = [set(p["pred"]) for p in predictions]
    labels = [set(gold) for gold in golds]

    if len(preds) == 0 or len(labels) == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = sum(
            [len(p & l) / len(p) for p, l in zip(preds, labels) if len(p) > 0]
        ) / len(preds)
        recall = sum(
            [len(p & l) / len(l) for p, l in zip(preds, labels) if len(l) > 0]
        ) / len(preds)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    return precision, recall, f1


def predict_bm25(searcher, doc, case):
    bm25_score = defaultdict(lambda: 0)
    hits = []
    segments = segment_document(doc, 1, 1)
    for segment in segments:
        _hits = searcher.search(segment[:1024], k=100000)
        hits.extend(_hits)

    for hit in hits:
        if hit.docid.endswith("task2"):
            if hit.docid.split("_candidate")[0] == case:
                hit.docid = hit.docid.split("_task2")[0].split("_candidate")[1]
                bm25_score[hit.docid] = max(hit.score, bm25_score[hit.docid])
    return bm25_score


def predict_all_bm25(
    dataset_path, year, bm25_index_path, eval_segment="test", k1=None, b=None, topk=None
):
    searcher = LuceneSearcher(bm25_index_path)

    if k1 and b:
        print(f"k1: {k1}, b: {b}")
        searcher.set_bm25(k1, b)

    corpus_dir, cases_dir, _ = get_data(dataset_path, year=year, segment=eval_segment)
    bm25_scores = {}
    for case in cases_dir:
        base_case_data = preprocess_case_data(
            corpus_dir / case / "entailed_fragment.txt"
        )
        score = predict_bm25(searcher, base_case_data, case)
        if topk is not None:
            sorted_score = sorted(score.items(), key=lambda x: x[1], reverse=True)[
                :topk
            ]
            score = {x[0]: x[1] for x in sorted_score}
        bm25_scores[case] = score
    return bm25_scores


def predict_all_bert(
    model, tokenizer, dataset_path, year, eval_segment="test", device=None
):
    corpus_dir, cases_dir, _ = get_data(dataset_path, year=year, segment=eval_segment)

    predictions = {}
    for case in cases_dir:
        base_case_data = preprocess_case_data(
            corpus_dir / case / "entailed_fragment.txt"
        )

        candidate_dir = corpus_dir / case / "paragraphs"
        candidate_cases = sorted(os.listdir(candidate_dir))

        predictions[case] = []
        for cand_case in candidate_cases:
            cand_case_data = preprocess_case_data(candidate_dir / cand_case)

            encoded_query = tokenizer(base_case_data, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH).to(device)
            encoded_paragraph = tokenizer(cand_case_data, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH).to(device)

            with torch.no_grad():
                outputs = model(encoded_query, encoded_paragraph)

            predictions[case].append(outputs)

    return predictions


def get_metrics(
    predictions,
    year,
    dataset_path,
    eval_segment="dev",
    topk=1,
):
    print(f"\n[{eval_segment}] k: {topk}")

    corpus_dir, cases_dir, label_data = get_data(
        dataset_path, year=year, segment=eval_segment
    )

    tp, fp, fn = 0, 0, 0
    for case in cases_dir:
        candidate_dir = corpus_dir / case / "paragraphs"
        candidate_cases = sorted(os.listdir(candidate_dir))

        label = [1 if f in label_data[case] else 0 for f in candidate_cases]
        pred = predictions[case]

        tp += np.sum([1 if a == b and a == 1 else 0 for a, b in zip(pred, label)])
        fp += np.sum([1 if a != b and a == 1 else 0 for a, b in zip(pred, label)])
        fn += np.sum([1 if a != b and a == 0 else 0 for a, b in zip(pred, label)])

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * ((p * r) / (p + r))

    print(f"[{eval_segment}] Metrics: {[f1, p, r]} - {[topk]}")
    return [f1, p, r]


def eval_end_model(predictions, year, dataset_path, eval_segment="dev", topk=1):
    if topk is None:
        list_k = [1, 2, 3]

        best_metric = [0, 0, 0]
        best_config = []

        for k in list_k:
            res = get_metrics(
                predictions,
                year,
                dataset_path,
                eval_segment,
                topk,
            )
            if res > best_metric:
                best_metric = res
                best_config = [k]

                with open("./save/best_config.txt", "w") as f:
                    f.write(f"k: {k}")
        print(f"Best metric: {best_metric} with config: {best_config}")
    else:
        res = get_metrics(
            predictions, year, dataset_path, eval_segment, topk
        )
        print(f"Result: {res}")
