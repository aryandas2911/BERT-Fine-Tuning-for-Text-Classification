# BERT Fine-Tuning for Fake News Classification

This repository contains a Jupyter Notebook demonstrating how to fine-tune the pretrained **BERT (`bert-base-uncased`)** model for binary text classification to detect fake news. The model is trained using the Hugging Face `transformers` library, `datasets`, and `evaluate` on a GPU-accelerated environment.

---

## 📌 Project Overview

Fake news detection is a critical application of Natural Language Processing (NLP). This project utilizes **BERT (Bidirectional Encoder Representations from Transformers)** to classify news articles as:
*   **`0` (Reliable / True News)**
*   **`1` (Unreliable / Fake News)**

The model is fine-tuned on the `Reyansh4/Fake-News-Classification` dataset, achieving an validation accuracy of over **99.5%**.

---

## 💻 Hardware & Software Environment

*   **GPU:** NVIDIA Tesla T4 (available on Google Colab or local systems with CUDA support)
*   **PyTorch Version:** `2.11.0+cu128`
*   **Transformers Version:** `5.10.1`

### Libraries Installed
```bash
pip install transformers datasets evaluate scikit-learn pandas numpy torch
```

---

## 📊 Dataset & Preprocessing

The model uses the dataset `Reyansh4/Fake-News-Classification` from Hugging Face Datasets.

### 1. Data Inspection
*   **Total Rows:** 20,800
*   **Columns:** `id`, `title`, `author`, `text`, `label`
*   **Class Balance:** Highly balanced:
    *   **Fake News (`1`):** ~50.06%
    *   **True News (`0`):** ~49.94%

### 2. Cleaning & Preprocessing
*   **Missing Values:** Identified 39 rows with missing values in the `text` column and dropped them (reducing the dataset size to 20,761 rows).
*   **Feature Selection:** Dropped unnecessary metadata columns (`id`, `title`, `author`), leaving only the input `text` and binary `label` columns.
*   **Train-Test Split:** Split the dataset into a **70% training set** (14,532 samples) and a **30% validation set** (6,229 samples), using stratified sampling to maintain class proportions.

### 3. Tokenization & PyTorch Formatting
*   **Tokenizer:** `bert-base-uncased`
*   **Hyperparameters:**
    *   `max_length=256`
    *   `truncation=True`
    *   `padding=True`
*   A custom PyTorch `Dataset` class (`SimpleDataset`) maps tokenizer outputs (`input_ids`, `attention_mask`, `token_type_ids`) alongside target labels.

---

## ⚙️ Model & Training Configuration

*   **Base Model:** `BertForSequenceClassification` from `bert-base-uncased`
*   **Optimizer & Hyperparameters:**
    *   **Learning Rate:** `2e-5`
    *   **Batch Size:** 8 for both training and evaluation
    *   **Epochs:** 3
    *   **Warmup Steps:** 100
    *   **Weight Decay / Evaluation Strategy:** Evaluated and checkpointed at the end of each epoch
    *   **Best Model Metric:** `accuracy` (using `load_best_model_at_end=True`)
*   **Evaluation Metrics:** Accuracy, F1-Score, Precision, and Recall calculated via the Hugging Face `evaluate` library.

---

## 📈 Training Progress and Results

The model was trained for 3 epochs (a total of 5,451 steps) over approximately **43 minutes and 27 seconds** on a Tesla T4 GPU.

### Epoch-by-Epoch Metrics

| Epoch | Training Loss | Validation Loss | Accuracy | F1-Score | Precision | Recall |
| :---: | :-----------: | :-------------: | :------: | :------: | :-------: | :----: |
|   **1**   |   0.035737    |    0.033975     |  99.13%  |  99.14%  |  98.63%   | 99.65% |
|   **2**   |   0.000118    |    0.030727     |  99.45%  |  99.45%  |  99.52%   | 99.39% |
|   **3**   |   0.000062    |    0.028694     |  99.57%  |  99.57%  |  99.52%   | 99.61% |

### Final Evaluation Metrics
```json
{
  "eval_loss": 0.028694303706288338,
  "eval_accuracy": 0.9956654358645047,
  "eval_f1": 0.9956654358645047,
  "eval_precision": 0.995186136071887,
  "eval_recall": 0.9961451975586251
}
```

---

## 💾 Saving Model & Tokenizer

At the end of training, the fine-tuned model and its tokenizer configs are saved directly to the `./results` output directory.

*   **Model Weights & Config:** `trainer.save_model()`
*   **Tokenizer Vocab & Configuration:** `tokenizer.save_pretrained("./results")`

---

## 🚀 How to Run

1.  **Clone this repository** and open [BERT_Fine_Tuning_for_Text_Classification.ipynb](file:///E:/My%20Study%20Resources/Programming%20and%20Hackathons/Projects/AI-ML-DL%20Projects/BERT%20Fine-Tuning%20for%20Text%20Classification/BERT_Fine_Tuning_for_Text_Classification.ipynb) in Jupyter Notebook, JupyterLab, or Google Colab.
2.  Install dependencies:
    ```bash
    pip install transformers datasets evaluate scikit-learn pandas numpy torch
    ```
3.  Execute all cells sequentially.
4.  Once completed, you can load your trained model for inference using:
    ```python
    from transformers import BertForSequenceClassification, BertTokenizer
    
    model = BertForSequenceClassification.from_pretrained("./results")
    tokenizer = BertTokenizer.from_pretrained("./results")
    ```
