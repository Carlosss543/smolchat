# smolchat

smolchat is a small autoregressive transformer pretrained on textual data. Its architecture is largely inspired by GPT-2 but incorporates some modern features such as ReLU squared, RMSNorm and RoPE.

## Some examples of text completion results

To get a feel for smolchat's generations, we prompt it with:

> **Prompt:** `France is`

Here are two sample continuations:

**Sample 1**

> France is a member of the International Labour Organisation (ILO), and the international Union for the International Reduction of Injuries from Wounds and Illnesses.


**Sample 2**

> France is a member of the International Committee of the Red Cross (ICRC). “The World Health Organization is proud to celebrate International Women’s Health

The sentences produced look somewhat plausible and reflect a certain understanding of the world, but the content is often inaccurate or inconsistent.


## Requirements

- Python 3.11 or later
- An NVIDIA GPU compatible with CUDA
- A Weights & Biases (W&B) account for online experiment tracking

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy tiktoken datasets tqdm wandb torch
```

## Prepare the data

The `fineweb.py` script downloads the `sample-10BT` subset of
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
tokenizes documents with GPT-2, and writes tokenized NumPy files. By default,
the files are saved to `~/datasets/edu_fineweb10B/`. Approximately 1% of
documents are reserved for validation.

```bash
python fineweb.py
```

This downloads a large dataset and may require substantial disk space and
memory. The output consists of `train_*.npy` and `val_*.npy` shards, which are
read directly by the data loader. To use a different location, change
`out_dir` in `fineweb.py` and pass the same path to `--data_dir` when training.

## Train

Example command to train on two GPUs:

```bash
torchrun --nproc_per_node=2 train.py \
  --data_dir ~/datasets/edu_fineweb10B/ \
  --output_dir ./pretraining_checkpoints/
```

The script trains on the `train` shards, uses the
`val` shards for validation, generates a sample, and periodically evaluates
accuracy on HellaSwag. HellaSwag is loaded from Hugging Face. Metrics are
logged online to the `smolchat` W&B project; configure your W&B credentials
before starting training.

Main command-line options:

| Option | Default | Description |
| --- | ---: | --- |
| `--data_dir` | required | Directory containing `train_*.npy` and `val_*.npy` shards |
| `--output_dir` | required | Directory where checkpoints are saved |
| `--sequence_length` | `1024` | Sequence length |
| `--global_batch_size` | `524288` | Total number of tokens per optimization step |
| `--micro_batch_size` | `16` | Number of sequences per micro-batch and process |
| `--max_steps` | all available steps | Optional limit on the number of steps |
| `--lr` | `0.001` | Maximum learning rate |
| `--warmup_steps` | `200` | Number of warmup steps |
| `--weight_decay` | `0.1` | Weight decay for applicable parameters |

The global batch size must be divisible by
`micro_batch_size × sequence_length × number_of_processes`. By default,
training runs for at most one epoch. Checkpoints are saved every 500 steps and
at the end of training as `<wandb-run-id>.pt`.

To resume training, provide the W&B run ID associated with the checkpoint:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --data_dir ~/datasets/edu_fineweb10B/ \
  --output_dir ./pretraining_checkpoints/ \
  --resume_from_checkpoint \
  --wandb_run_id YOUR_WANDB_RUN_ID
```

The corresponding checkpoint, `<output_dir>/<wandb-run-id>.pt`, must exist,
and the W&B run must be available to resume.

## Project structure

- `fineweb.py`: FineWeb-Edu dataset download and tokenization
- `data.py`: NumPy shard loading and PyTorch data loader creation
- `train.py`: Training configuration and loop, checkpointing, and tracking
- `engine.py`: Training steps and loss evaluation
- `utils.py`: Distributed initialization, text generation, and HellaSwag evaluation
- `model.py`: Transformer architecture
