# MemOPSD
Generalize or Remember? Rethinking Generative Recommendation from a Data-Centric Perspective

This guide covers the main experiments on **Industrial and Scientific, Video Games, and Office Products** from Amazon Reviews 2023, with **TIGER, LETTER, CARE, and LatentR3** as backbones.

## 1. Environment setup

Use **Linux, Bash, Python 3.11, and an NVIDIA GPU**. Run all commands from the repository root in the same Bash session. Download this repository with GitHub's **Code → Download ZIP**, extract it, and enter the directory containing `main.py`, `genrec/`, `analysis/`, and `scripts/`.

```bash
conda create -y -n memopsd python=3.11
conda activate memopsd

python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  transformers==4.57.0 datasets==4.8.4 accelerate==1.12.0 \
  numpy==1.26.4 pandas==2.3.3 scikit-learn==1.7.2 scipy==1.15.3 \
  sentence-transformers==5.1.2 faiss-cpu==1.12.0 \
  "huggingface-hub>=0.34.0,<1.0" "pyarrow>=21.0.0" \
  sentencepiece protobuf pyyaml requests tqdm wandb tiktoken

python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDA is unavailable"'
```

The PyTorch command uses the official [CUDA 12.8 wheels](https://pytorch.org/get-started/previous-versions/#v2-11-0). The core package versions follow the recorded project requirements; the additional text-encoding and quantization dependencies are included explicitly. Use the installation commands above: the current `requirements.txt` also contains experiment commands and cannot be passed directly to `pip install -r`.

```bash
export ROOT="$PWD"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/cache/huggingface"
export CACHE_ROOT="$ROOT/cache/main_reproduction"
mkdir -p "$CACHE_ROOT" "$ROOT/results"
set -euo pipefail
python -m pip freeze > "$ROOT/results/environment.txt"
```

`wandb` is imported by the implementation, but an account is unnecessary because the commands below set `--use_wandb=False`.

## 2. Download the text encoder and datasets

### 2.1 Download sentence-t5-base

Download [sentence-transformers/sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base) for item text encoding:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path(os.environ["ROOT"])
repo = "sentence-transformers/sentence-t5-base"
manifest = root / "results" / "text_encoder_revision.json"
revision = (json.loads(manifest.read_text())["revision"] if manifest.exists()
            else HfApi().model_info(repo).sha)
snapshot_download(repo_id=repo, revision=revision,
                  local_dir=root / "sentence-t5-base")
manifest.write_text(json.dumps({"repo": repo, "revision": revision}, indent=2))
PY
```

The recommendation models use a lightweight T5 initialized by the code. The pretrained model downloaded here is the **item text encoder**.

### 2.2 Download the three Amazon categories

We use the official [5-core, leave-last-out interaction files](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/benchmark/5core/last_out_w_his) and [item metadata](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/meta_categories). No separately hosted MemOPSD data or checkpoint package is required.

Run this block once. It downloads the three interaction splits for each category, then stores the five metadata fields used by the repository in local Parquet files. The repository subsequently performs its own ID mapping, text cleaning, and sequence construction. Keeping local CSV/Parquet inputs also avoids the legacy remote dataset-script loader.

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

root = Path(os.environ["ROOT"])
cache = Path(os.environ["CACHE_ROOT"])
repo = "McAuley-Lab/Amazon-Reviews-2023"
manifest = root / "results" / "dataset_revision.json"
revision = (json.loads(manifest.read_text())["revision"] if manifest.exists()
            else HfApi().dataset_info(repo).sha)
manifest.write_text(json.dumps({"repo": repo, "revision": revision}, indent=2))
schema = pa.schema([
    ("parent_asin", pa.string()), ("title", pa.string()),
    ("features", pa.list_(pa.string())),
    ("categories", pa.list_(pa.string())),
    ("description", pa.list_(pa.string())),
])

for category in ["Industrial_and_Scientific", "Video_Games", "Office_Products"]:
    raw = cache / "AmazonReviews2023" / category / "raw"
    for split in ["train", "valid", "test"]:
        hf_hub_download(
            repo_id=repo, repo_type="dataset", revision=revision,
            filename=f"benchmark/5core/last_out_w_his/{category}.{split}.csv",
            local_dir=raw,
        )
    target = raw / f"raw_meta_{category}" / "metadata.parquet"
    if target.exists():
        print(f"Already prepared: {category}")
        continue
    source = hf_hub_download(
        repo_id=repo, repo_type="dataset", revision=revision,
        filename=f"raw/meta_categories/meta_{category}.jsonl",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".parquet.part")
    rows = []
    with pq.ParquetWriter(temporary, schema) as writer:
        with open(source, encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                rows.append({key: record[key] for key in schema.names})
                if len(rows) == 10000:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    temporary.replace(target)
    print(f"Prepared: {category}")
PY
```

The downloaded revisions are recorded under `results/` and reused when these blocks run again. The expected input layout is:

```text
cache/main_reproduction/AmazonReviews2023/
├── Industrial_and_Scientific/
│   └── raw/
│       ├── benchmark/5core/last_out_w_his/
│       │   ├── Industrial_and_Scientific.train.csv
│       │   ├── Industrial_and_Scientific.valid.csv
│       │   └── Industrial_and_Scientific.test.csv
│       └── raw_meta_Industrial_and_Scientific/metadata.parquet
├── Video_Games/                 # Same structure
└── Office_Products/             # Same structure
```

## 3. Select one main experiment

Start with TIGER on Industrial and Scientific. To run another experiment, change `MODEL` and `CATEGORY`, then rerun **Sections 3–7**. All twelve combinations are supported.

```bash
MODEL=TIGER
CATEGORY=Industrial_and_Scientific
SEED=2026

export MODEL CATEGORY SEED
RUN_ROOT="$ROOT/ckpt/main_reproduction/$CATEGORY/$MODEL/seed_$SEED"
LOG_ROOT="$ROOT/logs/main_reproduction/$CATEGORY/$MODEL/seed_$SEED"
RESULT_DIR="$ROOT/results/$CATEGORY/$MODEL/seed_$SEED"
MEMORY_DATA="$ROOT/data/memgen_tiger/AmazonReviews2023-$CATEGORY"
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$RESULT_DIR"

case "$CATEGORY/$MODEL" in
  Industrial_and_Scientific/TIGER)    KD_WEIGHT=0.1;  ROLLOUTS=1 ;;
  Industrial_and_Scientific/LETTER)   KD_WEIGHT=0.01; ROLLOUTS=1 ;;
  Industrial_and_Scientific/CARE)     KD_WEIGHT=0.1;  ROLLOUTS=4 ;;
  Industrial_and_Scientific/LatentR3) KD_WEIGHT=0.3;  ROLLOUTS=1 ;;
  Video_Games/TIGER)                 KD_WEIGHT=0.3;  ROLLOUTS=2 ;;
  Video_Games/LETTER)                KD_WEIGHT=0.1;  ROLLOUTS=4 ;;
  Video_Games/CARE)                  KD_WEIGHT=0.3;  ROLLOUTS=4 ;;
  Video_Games/LatentR3)              KD_WEIGHT=0.3;  ROLLOUTS=2 ;;
  Office_Products/TIGER)             KD_WEIGHT=0.2;  ROLLOUTS=2 ;;
  Office_Products/LETTER)            KD_WEIGHT=0.1;  ROLLOUTS=2 ;;
  Office_Products/CARE)              KD_WEIGHT=0.2;  ROLLOUTS=2 ;;
  Office_Products/LatentR3)           KD_WEIGHT=0.2;  ROLLOUTS=2 ;;
  *) printf 'Unsupported experiment: %s/%s\n' "$CATEGORY" "$MODEL"; exit 1 ;;
esac

TEACHER_WARMUP=10000
if [[ "$CATEGORY/$MODEL" == "Industrial_and_Scientific/TIGER" ]]; then
  TEACHER_WARMUP=500
fi

SID_NAME=sentence-t5-base_256,256,256,256.sem_ids
if [[ "$MODEL" == "LETTER" ]]; then
  SID_NAME="letter_$SID_NAME"
fi
SEM_IDS="$CACHE_ROOT/AmazonReviews2023/$CATEGORY/processed/$SID_NAME"

COMMON=(
  --dataset=AmazonReviews2023 --category="$CATEGORY"
  --cache_dir="$CACHE_ROOT" --sent_emb_model="$ROOT/sentence-t5-base"
  --metadata=sentence --split=last_out --kcore=5core
  --rand_seed="$SEED" --reproducibility=True --use_wandb=False
  --eval_fine_grained=False --tune_split=val --load_best_ckpt=True
  --train_batch_size=512 --eval_batch_size=256 --test_batch_size=256
  --lr=0.001 --weight_decay=0.05 --warmup_steps=10000
  --epochs=300 --budget_epochs=None --steps=None
  --eval_interval=4 --patience=20 --max_grad_norm=1.0
  '--topk=[10]' '--test_topk=[5,10,20,50]'
  '--metrics=["ndcg","recall"]' --val_metric=ndcg@10
  --max_item_seq_len=50 --num_beams=50
  --val_num_beams=10 --test_num_beams=50
  --backbone=t5 --num_layers=2 --num_decoder_layers=2
  --d_model=128 --d_ff=1024 --num_heads=6 --d_kv=64
  --dropout_rate=0.1 --n_user_tokens=1
  --sent_emb_dim=768 --sent_emb_pca=32 --sent_emb_batch_size=512
  --rq_faiss=True --rq_n_codebooks=3 --rq_codebook_size=256
  --constrained_generation=False --latentr3_constrained_generation=False
)

# Select the only best-checkpoint file in a stage directory.
# SFT trainers also save a duplicate ending in .sft.pth; exclude it.
pick_checkpoint() {
  python - "$1" <<'PY'
import sys
from pathlib import Path
files = sorted(p for p in Path(sys.argv[1]).glob("*.pth")
               if not p.name.endswith(".sft.pth"))
if len(files) != 1:
    raise SystemExit(f"Expected one checkpoint in {sys.argv[1]}, found {len(files)}. "
                     "Use the exact best checkpoint reported by your training log.")
print(files[0].resolve())
PY
}
```

The explicit layer counts keep the four backbones in the main experiment's two-encoder-layer/two-decoder-layer setting. The separate `main_reproduction` cache prevents accidentally loading semantic IDs generated for another configuration. Within a category, TIGER, CARE, and LatentR3 share one semantic-ID mapping; LETTER uses its own mapping. **The student and its teacher must always use the same semantic-ID file.**

The selected distillation settings are:

| Dataset | Backbone | Distillation weight | Rollouts |
| --- | --- | ---: | ---: |
| Industrial and Scientific | TIGER | 0.1 | 1 |
| Industrial and Scientific | LETTER | 0.01 | 1 |
| Industrial and Scientific | CARE | 0.1 | 4 |
| Industrial and Scientific | LatentR3 | 0.3 | 1 |
| Video Games | TIGER | 0.3 | 2 |
| Video Games | LETTER | 0.1 | 4 |
| Video Games | CARE | 0.3 | 4 |
| Video Games | LatentR3 | 0.3 | 2 |
| Office Products | TIGER | 0.2 | 2 |
| Office Products | LETTER | 0.1 | 2 |
| Office Products | CARE | 0.2 | 2 |
| Office Products | LatentR3 | 0.2 | 2 |

For all combinations, post-training uses learning rate `2e-5`, cross-entropy weight `1.0`, distillation temperature `1.0`, and rollout temperature `0.7`.

## 4. Construct memory training data

```bash
python scripts/train/build_memgen_tiger_data.py \
  --dataset=AmazonReviews2023 --category="$CATEGORY" \
  --cache_dir="$CACHE_ROOT" --metadata=sentence --split=last_out --kcore=5core \
  --output_dir="$MEMORY_DATA" --splits=train,val,test --max_hop=4

test -s "$MEMORY_DATA/all_labeled/train.jsonl"
test -s "$MEMORY_DATA/memory_tiger/train.jsonl"
test -s "$MEMORY_DATA/memory_tiger/val.jsonl"
```

This step runs once per category and can be reused across backbones. The important outputs are:

```text
data/memgen_tiger/AmazonReviews2023-<category>/
├── all_labeled/train.jsonl       # Memory mask for student post-training
├── memory_tiger/train.jsonl      # Memory-only teacher training examples
├── memory_tiger/val.jsonl        # Teacher checkpoint selection
└── memory_tiger/test.jsonl
```

The name `memory_tiger` is shared by all four backbones. Training labels use the script's default leave-one-sequence-out setting and training-only transition support. Keep this default when constructing the teacher training subset.

## 5. Train the backbone on all training examples

```bash
python main.py --model="$MODEL" "${COMMON[@]}" \
  --sem_ids_path="$SEM_IDS" \
  --post_train_stage=sft --skip_sft_if_checkpoint=False \
  --ckpt_dir="$RUN_ROOT/sft" --log_dir="$LOG_ROOT/sft" \
  --run_id="main_${CATEGORY}_${MODEL}_sft_seed${SEED}" \
  2>&1 | tee "$RESULT_DIR/sft.log"

SFT_CKPT="$(pick_checkpoint "$RUN_ROOT/sft")"
test -s "$SEM_IDS"
printf 'Student checkpoint: %s\nSemantic IDs: %s\n' "$SFT_CKPT" "$SEM_IDS"
```

When the semantic-ID file is absent, the tokenizer builds it automatically before training. LETTER also trains its identifier module during this step. The best checkpoint is selected by validation `ndcg@10`; the pipeline then reports test metrics.

For LatentR3, explicitly selecting `--post_train_stage=sft` invokes the shared supervised-training path. This guide uses that path for the pre-distillation student and memory teacher.

## 6. Train the memory teacher

```bash
python scripts/train/train_memgen_tiger.py \
  --model="$MODEL" "${COMMON[@]}" \
  --memgen_data_dir="$MEMORY_DATA" --target=memory --splits=train,val,test \
  --sem_ids_path="$SEM_IDS" \
  --post_train_stage=sft --skip_sft_if_checkpoint=False \
  --warmup_steps="$TEACHER_WARMUP" \
  --ckpt_dir="$RUN_ROOT/teacher" --log_dir="$LOG_ROOT/teacher" \
  --run_id="main_${CATEGORY}_${MODEL}_teacher_seed${SEED}" \
  2>&1 | tee "$RESULT_DIR/teacher.log"

TEACHER_CKPT="$(pick_checkpoint "$RUN_ROOT/teacher")"
printf 'Teacher checkpoint: %s\n' "$TEACHER_CKPT"
```

The teacher is trained from initialization on memory examples and selected on the memory validation subset. Its architecture and semantic IDs match the student. The historical Industrial/TIGER teacher checkpoint is labeled with 500 warm-up steps; the other combinations retain the code's 10,000-step SFT warm-up default.

## 7. Run MemOPSD and evaluate

### 7.1 On-policy self-distillation

```bash
python main.py --model="$MODEL" "${COMMON[@]}" \
  --sem_ids_path="$SEM_IDS" \
  --post_train_stage=oprd --skip_sft_if_checkpoint=True \
  --sft_checkpoint_path="$SFT_CKPT" \
  --memory_oprd_teacher_path="$TEACHER_CKPT" \
  --memory_oprd_labels_path="$MEMORY_DATA/all_labeled/train.jsonl" \
  --memory_oprd_non_memory=skip \
  --memory_oprd_logit_weight="$KD_WEIGHT" --memory_oprd_temperature=1.0 \
  --oprd_ce_weight=1.0 --oprd_lr=2e-5 --oprd_weight_decay=0.0 \
  --oprd_epochs=30 --oprd_patience=3 --oprd_min_delta=0.0 \
  --oprd_warmup_steps=0 --oprd_eval_sft_baseline=False \
  --oprd_num_rollouts="$ROLLOUTS" --oprd_rollout_temperature=0.7 \
  --oprd_train_backbone=True \
  --ckpt_dir="$RUN_ROOT/memopsd" --log_dir="$LOG_ROOT/memopsd" \
  --run_id="main_${CATEGORY}_${MODEL}_memopsd_seed${SEED}" \
  2>&1 | tee "$RESULT_DIR/memopsd.log"

MEMOPSD_CKPT="$(pick_checkpoint "$RUN_ROOT/memopsd")"
printf 'MemOPSD checkpoint: %s\n' "$MEMOPSD_CKPT"
```

`oprd` is the implementation's stage name for MemOPSD. `memory_oprd_non_memory=skip` skips teacher alignment for non-memory examples; their supervised cross-entropy loss remains active. The validation set selects the best post-training checkpoint, and the pipeline evaluates it on the test set automatically.

### 7.2 Evaluate the saved student explicitly

```bash
python main.py --model="$MODEL" "${COMMON[@]}" \
  --sem_ids_path="$SEM_IDS" --post_train_stage=sft \
  --checkpoint_path="$MEMOPSD_CKPT" --eval_only=True \
  --ckpt_dir="$RUN_ROOT/eval" --log_dir="$LOG_ROOT/eval" \
  --run_id="main_${CATEGORY}_${MODEL}_test_seed${SEED}" \
  2>&1 | tee "$RESULT_DIR/test.log"

grep 'Test Results:' "$RESULT_DIR/test.log"
```

Use **`--eval_only=True`**, including `=True`, because additional options in `main.py` require `--key=value`. Evaluation loads only the final student. To reevaluate the backbone, replace `--checkpoint_path="$MEMOPSD_CKPT"` with `--checkpoint_path="$SFT_CKPT"` and use a separate output log.

The main metrics are `recall@5`, `recall@10`, `ndcg@5`, and `ndcg@10`. The commands also report `@20` and `@50`, matching the experiment log. Results are fractions, so `0.0251` corresponds to `2.51%`.

For a fresh rerun with the same seed, use new stage/output directories. If a directory contains several checkpoints, select the exact best-checkpoint path printed by that run instead of choosing the most recent file.

## 8. Reference results

These values are transcribed from the main-experiment worksheet of `GenRec.xlsx` and rounded to four decimal places. They are historical reference scores, not measurements from a fresh execution of this README.

| Dataset | Method | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Industrial and Scientific | TIGER | 0.0265 | 0.0429 | 0.0172 | 0.0224 |
| Industrial and Scientific | TIGER + MemOPSD | 0.0285 | 0.0435 | 0.0183 | 0.0232 |
| Industrial and Scientific | LETTER | 0.0274 | 0.0423 | 0.0177 | 0.0225 |
| Industrial and Scientific | LETTER + MemOPSD | 0.0287 | 0.0441 | 0.0186 | 0.0235 |
| Industrial and Scientific | CARE | 0.0273 | 0.0426 | 0.0179 | 0.0228 |
| Industrial and Scientific | CARE + MemOPSD | 0.0285 | 0.0451 | 0.0187 | 0.0240 |
| Industrial and Scientific | LatentR3 | 0.0274 | 0.0423 | 0.0176 | 0.0224 |
| Industrial and Scientific | LatentR3 + MemOPSD | 0.0285 | 0.0443 | 0.0186 | 0.0237 |
| Video Games | TIGER | 0.0532 | 0.0860 | 0.0346 | 0.0452 |
| Video Games | TIGER + MemOPSD | 0.0565 | 0.0889 | 0.0369 | 0.0473 |
| Video Games | LETTER | 0.0534 | 0.0839 | 0.0350 | 0.0448 |
| Video Games | LETTER + MemOPSD | 0.0574 | 0.0890 | 0.0374 | 0.0476 |
| Video Games | CARE | 0.0572 | 0.0894 | 0.0375 | 0.0478 |
| Video Games | CARE + MemOPSD | 0.0587 | 0.0918 | 0.0386 | 0.0493 |
| Video Games | LatentR3 | 0.0547 | 0.0868 | 0.0353 | 0.0456 |
| Video Games | LatentR3 + MemOPSD | 0.0592 | 0.0915 | 0.0387 | 0.0491 |
| Office Products | TIGER | 0.0288 | 0.0419 | 0.0199 | 0.0241 |
| Office Products | TIGER + MemOPSD | 0.0299 | 0.0432 | 0.0208 | 0.0251 |
| Office Products | LETTER | 0.0292 | 0.0427 | 0.0203 | 0.0246 |
| Office Products | LETTER + MemOPSD | 0.0301 | 0.0433 | 0.0209 | 0.0252 |
| Office Products | CARE | 0.0300 | 0.0433 | 0.0207 | 0.0250 |
| Office Products | CARE + MemOPSD | 0.0306 | 0.0439 | 0.0213 | 0.0255 |
| Office Products | LatentR3 | 0.0299 | 0.0432 | 0.0205 | 0.0248 |
| Office Products | LatentR3 + MemOPSD | 0.0307 | 0.0441 | 0.0213 | 0.0256 |

### Reproducibility status

The commands have been checked against the available source interfaces. Full GPU training and a clean-environment end-to-end run have not been performed for this guide. The recipe fixes the main architecture and PCA dimension to the manuscript setting, uses the worksheet's dataset/backbone parameter summary, and states the inherited warm-up settings explicitly. The original semantic-ID files, checkpoints, complete environment lock, and five-run seed list are not yet publicly available. Recreating those artifacts may change the historical scores; a single run with `SEED=2026` is not a reproduction of a five-run paper average.

## 9. Other comparison baselines

To run the other comparison implementations on the selected category, execute the following after Sections 1–3. These commands use each baseline's current model configuration. They are starting points for rerunning the comparisons; the worksheet does not contain the complete per-run configurations needed to certify the historical baseline scores.

```bash
for BASELINE in SASRec BERT4Rec FDSA S3Rec CLSRec HSTU RPG; do
  BASE_RESULT="$ROOT/results/$CATEGORY/$BASELINE/seed_$SEED"
  mkdir -p "$BASE_RESULT"
  python main.py --model="$BASELINE" \
    --dataset=AmazonReviews2023 --category="$CATEGORY" \
    --cache_dir="$CACHE_ROOT" --sent_emb_model="$ROOT/sentence-t5-base" \
    --rand_seed="$SEED" --reproducibility=True \
    --split=last_out --kcore=5core --tune_split=val \
    --epochs=300 --budget_epochs=None --steps=None \
    --eval_interval=4 --patience=20 --load_best_ckpt=True \
    '--topk=[10]' '--test_topk=[5,10,20,50]' --val_metric=ndcg@10 \
    --eval_fine_grained=False --use_wandb=False \
    --ckpt_dir="$ROOT/ckpt/main_reproduction/$CATEGORY/$BASELINE/seed_$SEED" \
    --log_dir="$ROOT/logs/main_reproduction/$CATEGORY/$BASELINE/seed_$SEED" \
    --run_id="main_${CATEGORY}_${BASELINE}_seed${SEED}" \
    2>&1 | tee "$BASE_RESULT/train.log"
done
```

`CLSRec` is the code identifier for **CL4SRec**, and `BERT4Rec` is case-sensitive. The current S3Rec configuration sets `pretrain_epochs: 0`; enabling its self-supervised pretraining stage requires an explicit nonzero value and should be reported with the comparison setting.

## Acknowledgments

The implementation builds on the MemGen-GR framework, [How Well Does Generative Recommendation Generalize?](https://arxiv.org/abs/2603.19809), and the [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
