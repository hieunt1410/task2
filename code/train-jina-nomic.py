import os
os.environ['NUMEXPR_MAX_THREADS'] = '88'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from model import *
from logger import *
from data_set import MyDataSet, BatchCollator
from loss import TranslatedReLU, SmoothK2Loss
from bert import make_predictions, eval_end_model

import random
import numpy as np

from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


# path
save_path = './save'
model_path = 'jinaai/jina-embeddings-v2-base-en'
dataset_path = "./data/task2_train_files_2025"
bm25_index_path = "./data/bm25_index_2025"
best_model_path = './save/jina_frozen_train.pth'

# we use DDP to train our model
local_rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)


def set_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed()
    
    EPOCH = 3
    BATCH_SIZE = 1
    EVALUATION_PER_STEP = 300
    MAX_SEQUENCE_LENGTH = 1792
    
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
    
    for param in model.model.parameters():
        param.requires_grad = False
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=BERT_LEARNING_RATE,
                      betas=(ADAM_BETA_1, ADAM_BETA_2),
                      eps=ADAM_EPSILON,
                      weight_decay=ADAM_WEIGHT_DECAY)
    
    if local_rank == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}")
    
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    total_steps = len(train_data_set) * EPOCH // (BATCH_SIZE * dist.get_world_size())
    warm_up_steps = int(total_steps * WARM_UP_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warm_up_steps, num_training_steps=total_steps)

    if local_rank == 0:
        logger.info(f'total steps: {total_steps}, with warm up steps: {warm_up_steps} and decay rate: {RATE_DECAY_FACTOR}')

    steps = 0
    current_steps = 0
    best_metric = [0, 0, 0]
    best_k = 0
    best_threshold = 0
    
    for e_i in range(EPOCH):
        pbar = tqdm(train_data_loader)
        for batch in pbar:
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
                loss_function = SmoothK2Loss(threshold=0.2, k=2)
            elif 'nomic' in model_path:
                loss_function = SmoothK2Loss(threshold=0.25, k=2)
            else:
                raise ValueError('unknown model path')
                
            loss = loss_function(prediction, label)
            loss.backward()

            clip_grad_norm_(parameters=model.parameters(), max_norm=GRADIENT_CLIPPING)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            dist.barrier()
            pbar.update(1)
            pbar.set_postfix(proportion=current_steps / total_steps, loss=loss.item(), lr=scheduler.get_last_lr()[0])

            current_steps += dist.get_world_size()
            steps += dist.get_world_size()

            if steps < EVALUATION_PER_STEP:
                continue

            steps = 0
            model.eval()
            
 
            
            if local_rank == 0:
                logger.info(f'epoch: {e_i}, steps: {current_steps}, proportion: {current_steps / total_steps}, loss: {loss.item()}')
                
                # if 'jina' in model_path:
                #     torch.save({'model': model.module.state_dict(), 'epoch': e_i, 'score': best_score}, open(os.path.join(save_path, f'jina_frozen_train.pth'), 'wb'))
                # elif 'nomic' in model_path:
                #     torch.save({'model': model.module.state_dict(), 'epoch': e_i, 'score': best_score}, open(os.path.join(save_path, f'nomic_frozen_train.pth'), 'wb'))
                # else:
                #     raise ValueError('unknown model path')
                predictions = make_predictions(model, tokenizer, dataset_path, year='2025', eval_segment="dev", device=device)
                metrics, k, threshold = eval_end_model(predictions, year='2025', dataset_path=dataset_path, eval_segment="dev")
                
                if metrics[0] > best_metric[0]:
                    best_metric = metrics
                    best_k = k
                    best_threshold = threshold
                    
                    torch.save({'model': model.state_dict(), 'epoch': e_i, 'score': best_metric, 'k': best_k, 'threshold': best_threshold}, best_model_path)
                
                logger.info(f"Best metric: {best_metric} with k: {best_k}")

if __name__ == '__main__':
    main()