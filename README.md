# Meta-MemBank

Meta-MemBank is a research codebase for **memory-augmented multimodal visual question answering (VQA)**. It explores how a structured semantic memory bank can provide visual-language models with reusable information about entities, scenes, and relationships. The repository contains multiple experimental generations of memory banks and model integrations, together with training and benchmark evaluation code.

## Overview

The project combines a vision-language model with an external or model-integrated semantic bank. During training and inference, selected bank entries can provide additional context for answering image-grounded questions. The repository includes implementations based on LLaVA-style models and configurations for Qwen-family language models and different visual encoders.

BankV3 and BankV5 scripts represent different iterations of this approach. Some later configurations expose retrieval controls such as the number of semantic clusters and the top-k values for global, entity, layout, or relation memories. These settings are experiment-specific and should not be assumed to be interchangeable across checkpoints.

## Repository Layout

| Path                      | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `pretrain_*.py`           | Model training implementations for different memory bank and backbone variants |
| `pretrain_*.sh`           | Example DeepSpeed launch configurations                      |
| `custom_llava_*.py`       | Custom multimodal model and processor implementations        |
| `Bank/`                   | Memory bank construction or representation code, when present in the checkout |
| `data.py`, `data_SEED.py` | Dataset loading and preprocessing utilities                  |
| `eval.py`, `eval_*.py`    | General and benchmark-specific evaluation scripts            |
| `GPTscore*.py`            | LLM-based scoring utilities                                  |
| `zero2.json`              | DeepSpeed ZeRO-2 configuration                               |

## Environment

The repository does not currently provide a single pinned `requirements.txt` or `pyproject.toml`. A training environment generally needs Python, PyTorch, Transformers, Datasets, PEFT, DeepSpeed, and the image/evaluation packages required by the selected script. Choose versions that are compatible with the selected checkpoint, CUDA runtime, and GPU drivers.

Training scripts target Linux with CUDA and commonly launch across multiple GPUs. Some scripts refer to author-specific Conda environments, external directories, model caches, and data paths. Update those values before running; the scripts are experiment templates rather than portable one-command installers.

```bash
git clone https://github.com/jylEcho/Meta-MemBank.git
cd Meta-MemBank
```

## Dataset Format

The training loader in `data.py` expects a dataset directory with this structure:

```text
<dataset>/
├── chat.json
└── images/
    ├── image_001.jpg
    └── ...
```

Each record should include an `image` filename and a `conversations` list with a human question and a gpt answer. Image prompts use the `<image>` placeholder. Evaluation scripts may expect different index files or schemas, so check the selected evaluator before preparing data.

## Training

Select a launch script that matches the intended backbone and bank version. For example, a BankV5 experiment can start from:

```bash
bash pretrain_10_reason_bankV5.sh
```

Before launching, inspect and update at least:

- Conda environment and Python packages;
- visible GPU IDs and distributed training settings;
- base model and tokenizer paths;
- processed training data path;
- semantic bank checkpoint path;
- output directory and checkpoint retention settings.

The scripts commonly use DeepSpeed, bf16, gradient checkpointing, and adapter-focused fine-tuning. Exact parameters vary between experiments. Ensure that the selected training implementation and the supplied bank file use compatible formats.

## Evaluation

The repository includes `eval.py` and task-specific evaluators such as `eval_A-OKVQA.py`, `eval_CC12M.py`, `eval_SEED.py`, and `eval_Surg.py`. Evaluation typically requires a trained checkpoint, its matching processor/model implementation, a benchmark index, and the associated image files.

```bash
python eval.py --help
```

Not every script necessarily supports command-line help or the same argument names; inspect its argument parser and defaults before use. `GPTscore*.py` files provide additional model-based scoring workflows where applicable.

## Reproducing Results

Record the exact script, model checkpoint, bank version, dataset preprocessing, software versions, and GPU configuration for each run. Start with a small run to confirm that data loading, model initialization, bank loading, and checkpoint saving work before committing substantial compute. Dataset and base-model licenses remain applicable.
