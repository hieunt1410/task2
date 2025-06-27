import os

os.environ["NUMEXPR_MAX_THREADS"] = "88"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from model import *
from logger import *
from data_set import MyDataSet, BatchCollator
from loss import TranslatedReLU, SmoothK2Loss
from bert import make_predictions, eval_end_model

import random
import numpy as np
from tqdm import tqdm

from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup


# path
save_path = "./save"
model_path = "FacebookAI/roberta-base"
dataset_path = "./data/task2_train_files_2025"
bm25_index_path = "./data/bm25_index_2025"
# '../../models/bert-base-uncased' # '../../models/roberta-base'

# we need this to import senteval
# sys.path.insert(0, "../SentEval")
# import senteval

evaluation_mode = "dev"
# evaluation_tasks = ["STSBenchmark"]
# sent_eval_data_path = "../SentEval/data"

# we use DDP to train our model
os.environ["LOCAL_RANK"] = "0"
local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def set_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed()

    EVALUATION_PER_STEP = 300
    current_steps = 0
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_data_set = MyDataSet(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        num_pairs_per_batch=10,
        ns_strategy="hard",
        year='2025',
        training_samples_file="./data/task2_train_negatives.json",
    )
    # train_data_set = train_data_set[:10]
    
    train_sampler = DistributedSampler(train_data_set)
    train_data_loader = DataLoader(
        dataset=train_data_set,
        batch_size=BATCH_SIZE * dist.get_world_size(),
        sampler=train_sampler,
        collate_fn=BatchCollator(tokenizer, device, MAX_SEQUENCE_LENGTH)
    )

    model = Average_BERT(bert_path=model_path)

    layer_learning_rate = {}
    base_learning_rate = BERT_LEARNING_RATE

    for i in range(11, -1, -1):
        layer_learning_rate[f".{str(i)}."] = base_learning_rate
        base_learning_rate *= RATE_DECAY_FACTOR

    optimizer_grouped_parameters = []
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    for name, parameter in parameters:
        params = {
            "params": [parameter],
            "lr": BERT_LEARNING_RATE,
            "weight_decay": (
                0.0 if any(item in name for item in no_decay) else ADAM_WEIGHT_DECAY
            ),
        }

        for layer_name, learning_rate in layer_learning_rate.items():
            if layer_name in name:
                params["lr"] = learning_rate
                break

        optimizer_grouped_parameters.append(params)

    optimizer = AdamW(
        params=optimizer_grouped_parameters,
        lr=BERT_LEARNING_RATE,
        betas=(ADAM_BETA_1, ADAM_BETA_2),
        eps=ADAM_EPSILON,
        weight_decay=ADAM_WEIGHT_DECAY,
    )

    model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    total_steps = len(train_data_set) * EPOCH // (BATCH_SIZE * dist.get_world_size())
    warm_up_steps = int(total_steps * WARM_UP_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warm_up_steps,
        num_training_steps=total_steps,
    )

    if local_rank == 0:
        logger.info(
            f"total steps: {total_steps}, with warm up steps: {warm_up_steps} and decay rate: {RATE_DECAY_FACTOR}"
        )

    steps = 0
    current_steps = 0
    for e_i in tqdm(range(EPOCH)):
        pbar = tqdm(train_data_loader)
        for batch in pbar:
            dist.barrier()
            model.train()

            query_tensors, paragraph_tensors, labels = batch
            label = torch.FloatTensor(list(map(int, labels))).cuda()
            label = label.reshape(label.shape[0], 1)
            
            prediction = model(query_tensors, paragraph_tensors)

            # special treatment for the left critical point
            mask = (prediction >= 0).type(prediction.dtype)
            prediction = prediction * mask

            loss_function = SmoothK2Loss(threshold=0.25, k=2)
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
                predictions = make_predictions(model, tokenizer, dataset_path, year='2025', eval_segment="dev", device=device)
                # bm25_scores = predict_all_bm25(dataset_path, year='2025', bm25_index_path=bm25_index_path, eval_segment="dev")
                eval_end_model(predictions, year='2025', dataset_path=dataset_path, save_path=save_path, eval_segment="dev") 
        
if __name__ == "__main__":
    main()
