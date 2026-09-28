# MemOPSD

Implementation of MemOPSD and sequential recommendation baselines. The main
experiments use Amazon Reviews 2023 (Industrial_and_Scientific, Video_Games,
Office_Products) and Steam.

## AutoDL setup

Extract the source directory to `/root/autodl-tmp/anonymous-recsys` on an AutoDL
Linux GPU instance. All default resource paths use this directory.

```bash
cd /root/autodl-tmp/anonymous-recsys
conda env create -f environment.yml
conda activate anonymous-recsys
```

The requirements retain the framework versions configured in the code, including
CUDA 12.8 PyTorch wheels. Use a compatible GPU driver. Weights, data, and outputs
are stored separately from source. Place the sentence embedding model under
`/root/autodl-tmp/anonymous-recsys/sentence-t5-base` when constructing semantic
IDs from item metadata.

## Data and semantic IDs

The dataset loaders process public Amazon Reviews 2023 and Steam data.
For an existing experiment, preserve the preprocessing configuration,
interaction sequences, item/user ID mappings, metadata, and semantic IDs.

```text
cache/AmazonReviews2023/<category>/processed/
cache/Steam/processed/
data/memgen_tiger/AmazonReviews2023-<category>/
data/memgen_tiger/Steam/
```

Processed caches include `id_mapping.json`, `all_item_seqs.json`, and
`metadata.sentence.json`. A generative checkpoint must use its corresponding
`.sem_ids` file. LETTER may use different semantic IDs from TIGER, CARE, and
LatentR3. Independently regenerated IDs do not preserve a checkpoint's mapping.

## Train baselines

Use these registered model names: `SASRec`, `BERT4Rec`, `FDSA`, `S3Rec`,
`CLSRec`, `HSTU`, `RPG`, `TIGER`, `LETTER`, `CARE`, `LatentR3`.
`CLSRec` is the CL4SRec implementation.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model=TIGER \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --use_wandb=False
```

For Steam, set `--dataset=Steam` and omit `--category`. Model-specific defaults
are in `genrec/models/<model>/config.yaml`; shared defaults are in
`genrec/default.yaml`. Additional options use `--key=value` syntax or a YAML
file passed with `--config`. Match architecture and preprocessing settings
between training and checkpoint evaluation.

## Build Memory training data

```bash
python scripts/train/build_memgen_tiger_data.py \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --output_dir=data/memgen_tiger/AmazonReviews2023-Industrial_and_Scientific \
  --splits=train,val,test \
  --max_hop=4
```

The script writes Memory subsets under `memory_tiger/` and training labels
under `all_labeled/`. Training labels use leave-one-sequence-out labeling.

## Train a Memory teacher

This example reuses the semantic-ID file produced by baseline training.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_memgen_tiger.py \
  --model=TIGER \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --memgen_data_dir=data/memgen_tiger/AmazonReviews2023-Industrial_and_Scientific \
  --target=memory \
  --splits=train,val,test \
  --sem_ids_path=cache/AmazonReviews2023/Industrial_and_Scientific/processed/sentence-t5-base_256,256,256,256.sem_ids \
  --post_train_stage=sft \
  --skip_sft_if_checkpoint=False \
  --ckpt_dir=ckpt/memory_tiger \
  --use_wandb=False \
  --eval_test
```

The entry point also supports `LETTER`, `CARE`, and `LatentR3`. Set the model,
semantic IDs, configuration, and checkpoint directory consistently.

## Memory-guided post-training

Initialize the student from its baseline checkpoint and select the matching
Memory teacher. Substitute the actual checkpoint filenames for
`ckpt/baseline.pth` and `ckpt/memory_teacher.pth`.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model=TIGER \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --sem_ids_path=cache/AmazonReviews2023/Industrial_and_Scientific/processed/sentence-t5-base_256,256,256,256.sem_ids \
  --post_train_stage=oprd \
  --sft_checkpoint_path=ckpt/baseline.pth \
  --memory_oprd_teacher_path=ckpt/memory_teacher.pth \
  --memory_oprd_labels_path=data/memgen_tiger/AmazonReviews2023-Industrial_and_Scientific/all_labeled/train.jsonl \
  --memory_oprd_non_memory=skip \
  --memory_oprd_logit_weight=0.1 \
  --memory_oprd_temperature=1.0 \
  --oprd_ce_weight=1.0 \
  --oprd_lr=2e-5 \
  --oprd_num_rollouts=1 \
  --oprd_rollout_temperature=0.7 \
  --ckpt_dir=ckpt/post_training \
  --use_wandb=False
```

This is an Industrial TIGER example. Distillation weights and rollout counts
depend on the experiment; use each model/dataset's selected settings rather
than applying this example to every run. Select checkpoints on validation
data (`tune_split=val`).

## Evaluate

Training reports final test metrics after checkpoint selection. To evaluate an
existing checkpoint, substitute its actual filename for `ckpt/final.pth`:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model=TIGER \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --checkpoint_path=ckpt/final.pth \
  --sem_ids_path=cache/AmazonReviews2023/Industrial_and_Scientific/processed/sentence-t5-base_256,256,256,256.sem_ids \
  --eval_only=True \
  --eval_fine_grained=False \
  --test_topk='[5,10,20,50]' \
  --use_wandb=False
```

The main table uses NDCG and Recall at 5, 10, 20, and 50. Preserve the model
configuration, seed, data mapping, semantic IDs, and decoding settings when
comparing results. Aggregate evaluation does not need separate analysis scripts.

## Source layout

- `main.py`: baseline training, post-training, and evaluation.
- `genrec/`: runtime, models, dataset loaders, metrics, and YAML configuration.
- `mem_gen_categorizer.py`: grouping logic imported by the trainer.
- `scripts/train/build_memgen_tiger_data.py`: Memory data and label generation.
- `scripts/train/train_memgen_tiger.py`: Memory teacher training.
- `analysis/semantic_id_coverage.py`: utilities imported by the TIGER tokenizer.

Third-party source attribution in the code is retained.
