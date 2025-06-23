import argparse
from pathlib import Path
import os
import jsonlines
import subprocess
from tqdm import tqdm
import sys
import shutil
from utils import preprocess_case_data, save_json, load_json, get_data
from bert import predict_all_bm25

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--year', type=str, default='2025')
    parser.add_argument('--num_negative', type=int, default=10)
    parser.add_argument('--training_samples_file', type=str, default=None)
    return parser.parse_args()


root = Path(os.path.realpath(__file__)).parents[1]
sys.path.insert(0, str(root))



def create_bm25_indexes(args):
    tmp_dir = root / "data/bm25_indexes/tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    for segment in ["train", "dev", "test"]:
        indexes_dir = root / f"data/bm25_indexes/coliee_task2/{segment}"

        os.makedirs(indexes_dir, exist_ok=True)

        corpus_dir, cases_dir, _ = get_data(dataset_path, year=args.year, segment=segment)

        for case in tqdm(cases_dir):
            candidate_dir = corpus_dir / case / "paragraphs"
            candidate_cases = sorted(os.listdir(candidate_dir))
            for cand_case in candidate_cases:
                cand_case_file = candidate_dir / cand_case
                cand_case_data = preprocess_case_data(cand_case_file)
                cand_num = cand_case.split(".txt")[0]
                dict_ = {
                    "id": f"{case}_candidate{cand_num}.txt_task2",
                    "contents": cand_case_data,
                }

                with jsonlines.open(f"{tmp_dir}/candidate.jsonl", mode="a") as writer:
                    writer.write(dict_)

        subprocess.run(
            [
                "python",
                "-m",
                "pyserini.index.lucene",
                "-collection",
                "JsonCollection",
                "-generator",
                "DefaultLuceneDocumentGenerator",
                "-threads",
                "1",
                "-input",
                f"{tmp_dir}",
                "-index",
                f"{indexes_dir}",
                "-storePositions",
                "-storeDocvectors",
                "-storeRaw",
            ]
        )

def extract_negative_samples(args, segment="train"):
    bm25_index_path = str(root / f"data/bm25_indexes/coliee_task2/{segment}")

    _, cases_dir, label_data = get_data(dataset_path, year=args.year, segment=segment)
    bm25_scores = predict_all_bm25(dataset_path, year=args.year, bm25_index_path=bm25_index_path, eval_segment=segment)

    num_negatives = 10
    sample_dict = {}
    for i, case in tqdm(enumerate(cases_dir)):
        bm25_score = bm25_scores[case]
        top_negatives = sorted(bm25_score.items(), key=lambda x: x[1], reverse=True)[
            :num_negatives
        ]
        negative_ids = [x[0] for x in top_negatives]
        sample_dict[case] = list(set(negative_ids + label_data[case]))

    save_path = root / f"data/task2_{segment}_negatives.json"
    save_json(save_path, sample_dict)
    
    
def build_dataset(dataset_path, year, training_samples_file=None):
    corpus_dir, cases_dir, label_data = get_data(dataset_path, year=year, segment="train")

    training_samples = {}
    if training_samples_file:
        training_samples = load_json(training_samples_file)
    
    dataset = []
    for case in cases_dir:
        base_case_file = corpus_dir / case / "entailed_fragment.txt"
        base_case_data = preprocess_case_data(base_case_file, uncased=False)
        label = label_data[case]

        case_dict = {
            "id": case,
            "text": base_case_data,
            "pos_candidates": [],
            "neg_candidates": [],
        }

        candidate_dir = corpus_dir / case / "paragraphs"
        candidate_cases = sorted(os.listdir(candidate_dir))
        for cand_case in candidate_cases:
            if case in training_samples and cand_case not in training_samples[case]:
                continue
            cand_case_file = candidate_dir / cand_case
            cand_case_data = preprocess_case_data(
                cand_case_file, uncased=False, filter_min_length=10
            )
            if cand_case_data is None:
                continue

            l = "pos_candidates" if cand_case in label else "neg_candidates"
            case_dict[l].append({"id": cand_case, "text": cand_case_data})
        dataset.append(case_dict)
    return dataset

def split_dataset(args):
    label_data = load_json(root / f"data/task2_train_labels_{args.year}.json")
    if args.year == "2024":
        train_labels = {k: v for k, v in label_data.items() if int(k) in range(626)}
        dev_labels = {k: v for k, v in label_data.items() if int(k) in range(626, 726)}
        test_labels = {k: v for k, v in label_data.items() if int(k) in range(726, 826)}
    else:
        train_labels = {k: v for k, v in label_data.items() if int(k) in range(726)}
        dev_labels = {k: v for k, v in label_data.items() if int(k) in range(726, 826)}
        test_labels = {k: v for k, v in label_data.items() if int(k) in range(826, 926)}
        
    save_json(root / f"data/train_labels_{args.year}.json", train_labels)
    save_json(root / f"data/dev_labels_{args.year}.json", dev_labels)
    save_json(root / f"data/test_labels_{args.year}.json", test_labels)
    
if __name__ == '__main__':
    args = parse_args()
    dataset_path = root / f"data/task2_train_files_{args.year}"
    shutil.rmtree(root / "data/bm25_indexes", ignore_errors=True)
    split_dataset(args)
    
    create_bm25_indexes(args)
    extract_negative_samples(args, segment="train")