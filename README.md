# Distilling Answer-Set Programming Rules from LLMs for Neurosymbolic Visual Question Answering

This repository contains code and scripts for the paper **"Distilling Answer-Set Programming Rules from LLMs for Neurosymbolic Visual Question Answering"**. The project focuses on extracting ASP rules from large language models (LLMs) for Visual Question Answering (VQA) with symbolic reasoning.

## Table of Contents

- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Dataset](#dataset)
- [Usage](#usage)
- [Experiments](#experiments)
- [API Keys](#api-keys)
- [Logs](#logs)

## Project Structure

```
.
├── main.py
├── utils.py
├── requirements.txt
├── README.md
├── .env
├── experiments/
├── preprompt/
├── prompt/
├── utils/
└── dataset/         # (to be created by user)
```

- **main.py**: Main entry point for running experiments.
- **utils.py**: Utility functions and API key storage.
- **experiments/**: Bash scripts for running ablation and composite experiments.
- **preprompt/**, **prompt/**: Prompt templates and related files.
- **utils/**: Additional utility modules.
- **dataset/**: Place extracted datasets here (see below).

## Setup Instructions

1. **Create a Conda environment:**
    ```sh
    conda create --name kdasp --file requirements.txt
    conda activate kdasp
    ```

2. **API Keys:**
    - Add your GPT, Mistral, and Huggingface API keys to [`utils.py`](utils.py).

3. **Dataset:**
    - Extract `dataset_AIII.zip` (provided separately) and place the contents in a new folder named `dataset/` at the project root.

4. **Environment Variables (Optional):**
    - You may use a `.env` file for environment-specific settings.

## Dataset

- Download and extract `dataset_kdasp.zip` from [this link](https://drive.google.com/file/d/1KmPhA_c7KV0rWzeOGOTChwKNC_6ifaXd/view?usp=sharing).
- Place the extracted files in a folder named `dataset/` at the root of the repository.


## Usage

- To see the arguments of the main script:
    ```sh
    python main.py --help
    ```

- To run experiments, use the provided bash scripts in the [`experiments/`](experiments/) folder. For example:
    ```sh
    bash experiments/run_experiment_submission.sh
    ```

## Experiments

The [`experiments/`](experiments/) folder contains scripts for the main experiments and various ablation and composite studies. Each script is documented with its purpose in the file header.

## API Keys

API keys for GPT, Mistral, and Huggingface must be added to [`utils.py`](utils.py) as variables. Example:
```python
# utils.py
GPT_API_KEY = "your-gpt-key"
MISTRAL_API_KEY = "your-mistral-key"
HF_API_KEY = "your-huggingface-key"
```


## Logs

All the experiments logs can be found [in this link](https://drive.google.com/file/d/1KmPhA_c7KV0rWzeOGOTChwKNC_6ifaXd/view?usp=sharing).
