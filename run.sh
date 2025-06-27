source env/bin/activate
torchrun --nproc_per_node=1 code/train-bert-roberta.py
# torchrun --nproc_per_node=1 code/tune-jina-nomic.py