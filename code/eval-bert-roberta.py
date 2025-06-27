from model import *
from logger import *
from bert import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'

import sys
from prettytable import PrettyTable
from transformers import AutoTokenizer

eval_mode = 'test'

best_model_path = './save/roberta_best_model.pth'
bert_path = 'FacebookAI/roberta-base'
dataset_path = "./data/task2_train_files_2025"


def show_table(task_names, scores):
    table = PrettyTable()
    table.field_names = task_names
    table.add_row(scores)
    print(table)
    

def main():
    model = Average_BERT(bert_path=bert_path)
    model = torch.nn.DataParallel(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(best_model_path):
        check_point = torch.load(best_model_path, weights_only=False, map_location=device)
        state_dict = check_point['model']
        # if any(key.startswith('module.') for key in state_dict.keys()):
        #     # Remove 'module.' prefix from keys
        #     new_state_dict = {}
        #     for key, value in state_dict.items():
        #         new_key = key.replace('module.', '')
        #         new_state_dict[new_key] = value
        #     state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        logger.info(f'load best model with epoch: {check_point["epoch"]} and dev score: {check_point["score"]}')
    else:
        raise ValueError(f'fail to load {best_model_path}')

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=bert_path)
    model.to(device)
    model.eval()

    predictions = make_predictions(model, tokenizer, dataset_path, year='2025', eval_segment=eval_mode, device=device)

    metrics, k, threshold = eval_end_model(predictions, year='2025', dataset_path=dataset_path, eval_segment=eval_mode)
    logger.info(f"Best metric: {metrics} with k: {k} and threshold: {threshold}")
    
if __name__ == '__main__':
    main()