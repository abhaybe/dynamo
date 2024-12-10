import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from dataloaders import (
    DomainPreservingCollator,
    DomainWeightedDataset,
    DomainWeightedTrainer,
)
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

random.seed(42)

MODEL_NAME = "unsloth/Llama-3.2-1B"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)
# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

MAX_LENGTH = 512
TOTAL_DATAPOINTS = 20_000
NUM_DOMAINS = 6
NUM_PER_DOMAIN = 20_000  # We'll have 10,000 examples per domain (make sure we can draw entirely from one domain if we want)
LOSS_WINDOW_SIZE = 100
REWEIGHTING_FREQ = 50
ALPHA = 0.05
WEIGHT_LOG_DIR = "./weight_updates.log"


# Update this to test different update policies
def policy(current_weights, loss_windows):
    return current_weights
    return np.array(
        [
            weight - ALPHA * (window[-1] - window[-2])
            for weight, window in zip(current_weights, loss_windows.values())
        ]
    )


data = defaultdict(list)

with open("../data/math_sampled.jsonl", "r") as f:
    lines = f.read().split("\n")
    for line in lines:
        obj = json.loads(line)
        data[obj["domain"]].append(
            {
                "question": obj["question"],
                "answer": obj["answer"],
            }
        )

dataset = load_dataset("json", data_files="../data/math_sampled.jsonl")
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset_train = defaultdict(list)
for obj in tqdm(dataset["train"], desc="Preprocessing train data"):
    dataset_train[obj["domain"]].append(
        {
            "question": obj["question"],
            "answer": obj["answer"],
        }
    )
dataset_train = DomainWeightedDataset(
    max_sequence_length=MAX_LENGTH,
    total_num_datapoints=TOTAL_DATAPOINTS,
    num_domains=6,
    tokenizer=tokenizer,
    loss_window_size=LOSS_WINDOW_SIZE,
    domain_datasets=dataset_train,
    weight_log_dir=WEIGHT_LOG_DIR,
    reweighting_policy=policy,
    reweighting_freq=REWEIGHTING_FREQ,
)


def preprocess(examples):
    input_texts = [f"Question: {q}\nAnswer:" for q in examples["question"]]
    target_texts = examples["answer"]

    inputs = tokenizer(
        input_texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        # return_tensors=None,
    )

    labels = tokenizer(
        target_texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        # return_tensors=None,
    )

    inputs["labels"] = labels["input_ids"]

    return {
        "domain": examples["domain"],
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": inputs["labels"],
    }


dataset_test = dataset["test"].map(
    preprocess,
    batched=True,
    batch_size=8,
    # remove_columns=dataset["test"].column_names  TODO: can we calculate per-domain accuracies afterwards?
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir="./experiment",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    num_train_epochs=5,
    learning_rate=5e-5,
    logging_dir="./logs",
    save_strategy="epoch",
    report_to="tensorboard",
    logging_steps=10,
    evaluation_strategy="epoch",
    gradient_checkpointing=True,
    optim="adamw_torch",
    fp16=True,
)
# data_collator = DomainPreservingCollator(
#     tokenizer=tokenizer, padding="max_length", max_length=MAX_LENGTH
# )
trainer = DomainWeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    # data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./baseline_finetuned_llama3.2")
tokenizer.save_pretrained("./baseline_finetuned_llama3.2")
