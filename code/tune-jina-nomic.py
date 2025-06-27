import os
os.environ['NUMEXPR_MAX_THREADS'] = '88'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # A6000 x 2

from model import *
from logger import *
from data_set import *
from loss import TranslatedReLU, SmoothK2Loss

import sys
import random
import numpy as np
from data_set import MyDataSet, BatchCollator

from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

# path
save_path = "./save"
model_path = "jinaai/jina-embeddings-v2-base-en"
dataset_path = "./data/task2_train_files_2025"
bm25_index_path = "./data/bm25_index_2025"
best_model_path = './save/jina_frozen_train.pth'
# 'jinaai/jina-embeddings-v2-base-en' # '../../models/nomic-embed-text-v1' 

# we use DDP to train our model
local_rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)

BERT_LEARNING_RATE = 5e-5
EPOCH, BATCH_SIZE = 10, 1
EVALUATION_PER_STEP = 300
MAX_SEQUENCE_LENGTH = 8192

def set_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_data_set = MyDataSet(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        num_pairs_per_batch=10,
        ns_strategy="hard",
        year='2025',
        training_samples_file="./data/task2_train_negatives.json",
    )

    train_sampler = DistributedSampler(train_data_set)
    train_data_loader = DataLoader(
        dataset=train_data_set,
        batch_size=BATCH_SIZE * dist.get_world_size(),
        sampler=train_sampler,
        collate_fn=BatchCollator(tokenizer, device, MAX_SEQUENCE_LENGTH)
    )
    

    model = Dual_Tower(model_path=model_path)

    # if os.path.exists(best_model_path):
    #     check_point = torch.load(best_model_path, map_location=device)
    #     model.load_state_dict(check_point['model'])  # corresponding to torch.save in train.py
    #     logger.info(f'load best model with epoch: {check_point["epoch"]}')

    #     best_score = check_point['score']
    # else:
        # raise ValueError(f'fail to load {best_model_path}')

    layer_learning_rate = {}
    base_learning_rate = BERT_LEARNING_RATE

    for i in range(11, -1, -1):
        layer_learning_rate[f'.{str(i)}.'] = base_learning_rate
        base_learning_rate *= RATE_DECAY_FACTOR

    optimizer_grouped_parameters = []
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]

    for name, parameter in parameters:
        params = {
            'params': [parameter],
            'lr': BERT_LEARNING_RATE,
            'weight_decay': 0.0 if any(item in name for item in no_decay) else ADAM_WEIGHT_DECAY
        }   

        for layer_name, learning_rate in layer_learning_rate.items():
            if layer_name in name:
                params['lr'] = learning_rate
                break

        optimizer_grouped_parameters.append(params)
    
    optimizer = AdamW(params=optimizer_grouped_parameters,
                      lr=BERT_LEARNING_RATE,
                      betas=(ADAM_BETA_1, ADAM_BETA_2),
                      eps=ADAM_EPSILON,
                      weight_decay=ADAM_WEIGHT_DECAY)
    
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    total_steps = len(train_data_set) * EPOCH // (BATCH_SIZE * dist.get_world_size())
    warm_up_steps = int(total_steps * WARM_UP_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warm_up_steps, num_training_steps=total_steps)

    if local_rank == 0:
        logger.info(f'total steps: {total_steps}, with warm up steps: {warm_up_steps} and decay rate: {RATE_DECAY_FACTOR}')

    steps = 0
    best_metric = [0, 0, 0]
    best_k = 0
    current_steps = 0
    for e_i in range(EPOCH):
        for batch in train_data_loader:
            dist.barrier()
            model.train()

            query_tensors, paragraph_tensors, labels = batch
            prediction = model(query_tensors, paragraph_tensors)

            # special treatment for the left critical point
            mask = (prediction >= 0).type(prediction.dtype)
            prediction = prediction * mask
            
            label = torch.FloatTensor(list(map(int, labels))).cuda()
            label = label.reshape(label.shape[0], 1)
            
            if 'jina' in model_path:
                loss_function = SmoothK2Loss(threshold=0.2, k=3.5)
            elif 'nomic' in model_path:
                loss_function = SmoothK2Loss(threshold=0.2, k=3)
            else:
                raise ValueError('unknown model path')
              
            loss = loss_function(prediction, label)
            loss.backward()

            clip_grad_norm_(parameters=model.parameters(), max_norm=GRADIENT_CLIPPING)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            dist.barrier()
            current_steps += dist.get_world_size()
            steps += dist.get_world_size()

            if steps < EVALUATION_PER_STEP:
                continue

            steps = 0
            model.eval()
            

            if local_rank == 0:
                predictions = make_predictions(model, tokenizer, dataset_path, year='2025', eval_segment="dev", device=device)
                metrics, k = eval_end_model(predictions, year='2025', dataset_path=dataset_path, save_path=save_path, eval_segment="dev")
                
                if metrics[0] > best_metric[0]:
                    best_metric = metrics
                    best_k = k
                    
                    torch.save({'model': model.state_dict(), 'epoch': e_i, 'score': best_metric, 'k': best_k}, best_model_path)


if __name__ == '__main__':
    main()