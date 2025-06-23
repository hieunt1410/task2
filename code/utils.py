import numpy as np
import os
import json
import re
import spacy
from pyserini.search.lucene import LuceneSearcher
from collections import defaultdict
from pathlib import Path
import torch

SPECIAL_CHARACTERS = "/-'#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + \
    '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
PUNCTUATION = ".,!?"

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

def get_data(data_path, year, segment="train"):
    if year == '2024':
        if segment == 'train':
            start_idx, end_idx = 0, 625
        elif segment == 'dev':
            start_idx, end_idx = 625, 725
        elif segment == 'test':
            start_idx, end_idx = 725, 825
    elif year == '2025':
        if segment == 'train':
            start_idx, end_idx = 0, 725
        elif segment == 'dev':
            start_idx, end_idx = 725, 825
        elif segment == 'test':
            start_idx, end_idx = 825, 925
    else:
        raise ValueError(f"Invalid year: {year}")

    corpus_dir = Path(data_path)
    cases_dir = sorted(os.listdir(corpus_dir))

    root_dir = corpus_dir.parent

    if segment == "train":
        label_data = load_json(root_dir / f"train_labels_{year}.json")
        return corpus_dir, cases_dir[start_idx:end_idx], label_data

    elif segment == "dev":
        label_data = load_json(root_dir / f"dev_labels_{year}.json")
        return corpus_dir, cases_dir[start_idx:end_idx], label_data

    else:
        label_data = load_json(root_dir / f"test_labels_{year}.json")
        return corpus_dir, cases_dir[start_idx:end_idx], label_data

def save_txt(file_path, text):
    with open(file_path,"w", encoding="utf-8") as f:
        f.write(text)


def load_txt(file_path, skip=0):
    with open(file_path, encoding="utf-8") as f:
        while skip > 0:
            f.readline()
            skip -= 1
        data = f.read()
    return data


def load_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(file_path, d):
    with open(file_path, "w+", encoding="utf-8") as f:
        json.dump(d, f)


def get_sentences(doc):
    doc = nlp(doc)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


def filter_document(doc, min_sentence_length=None):
    sentences = get_sentences(doc)
    if min_sentence_length:
        sentences = [sent for sent in sentences
                     if len(sent.split()) >= min_sentence_length]
    doc = " ".join(sentences)
    return doc


def handle_base_case(content: str) -> list[str]:
    pattern = r'(?ms)(^\s*\[\d+\]\s*.*?)(?=^\s*\[\d+\]|\Z)'
    
    sections = re.findall(pattern, content)
    # sections = [section.strip().split('\n')[1].strip() for section in sections]
    sections = [section.strip() for section in sections]
    
    return sections

def process_dataset(data_path, labels_path):
    data = []
    labels = {}
    
    with open(labels_path, 'r') as f:
        content = json.load(f)
        
        for case, label in content.items():
            labels[case] = [x.split('.')[0] for x in label]
    
    
    for case in os.listdir(data_path):
        paragraphs = []
        for paragraph in os.listdir(os.path.join(data_path, case, 'paragraphs')):
            with open(os.path.join(data_path, case, 'paragraphs', paragraph), 'r') as f:
                paragraphs.append(f.read())
                
        with open(os.path.join(data_path, case, 'base_case.txt'), 'r') as f:
            base_case = f.read()
            
        with open(os.path.join(data_path, case, 'entailed_fragment.txt'), 'r') as f:
            query = f.read()
            
        data.append({
            'id': case,
            'query': query,
            'base_case': handle_base_case(base_case),
            'paragraphs': paragraphs,
            'labels': labels[case]
        })
            
            
    return data


def segment_document(doc, max_sent_per_segment, stride, max_segment_len=None):
    sentences = get_sentences(doc)
    segments = []
    for i in range(0, len(sentences), stride):
        segment = " ".join(sentences[i:i + max_sent_per_segment])

        if max_segment_len:
            segment = " ".join(segment.split()[:max_segment_len])
        segments.append(segment)
    return segments


def preprocess_case_data(
    file_path,
    max_length=None,
    min_sentence_length=None,
    uncased=False,
    filter_min_length=None,
):
    if not os.path.exists(file_path):
        return None

    text = load_txt(file_path)

    text = (
        text.strip()
        .replace("\n", " ")
        .replace("FRAGMENT_SUPPRESSED", "")
        .replace("FACTUAL", "")
        .replace("BACKGROUND", "")
        .replace("ORDER", "")
    )
    if uncased:
        text = text.lower()
    text = re.sub("\s+", " ", text).strip()
    text = " ".join([w for w in text.split() if w])

    cite_number = re.search("\[[0-9]+\]", text)
    if cite_number:
        text = text[0: cite_number.span()[0]].strip() + ' ' + text[cite_number.span()[1] :].strip()
        
    if filter_min_length:
        words = text.split()
        if len(words) <= filter_min_length:
            return None

    if min_sentence_length:
        text = filter_document(text, min_sentence_length)
    if max_length:
        words = text.split()[:max_length]
        text = " ".join(words)
    if not text.endswith("."):
        text = text + "."
    return text


def format_output(text):
    CLEANR = re.compile("<.*?>")
    cleantext = re.sub(CLEANR, "", text)
    return cleantext.strip().lower()

def format_output_2(text):
    regex = r"Document (\d+)"
    numbers = re.findall(regex, text)

    if numbers:
        return [int(num) - 1 for num in numbers]

    print(f"Parsing error: No valid match found in '{text}'")
    return []


def train_test_split(data, test_size: float, random_state: int) -> tuple:
    np.random.seed(random_state)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    split = int(len(data) * test_size)
    
    return [data[i] for i in indices[split:]], [data[i] for i in indices[:split]]


def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)