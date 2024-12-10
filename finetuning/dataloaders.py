import json
import logging
import math
import random
from collections import defaultdict, deque
from typing import Dict, List

import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# This will sample a datapoint based on what we wanted.
# Will take in sequential indices. Indices are ignored? And then we sample based on our weights
TOTAL_DATAPOINTS = 20_000
NUM_DOMAINS = 6
LOSS_WINDOW_SIZE = 100


class DomainWeightedDataset(Dataset):
    # Pass in the datasets we wanted
    def __init__(
        self,
        max_sequence_length,
        total_num_datapoints,
        num_domains,
        tokenizer,
        loss_window_size,
        domain_datasets: Dict[
            str, List[Dict]
        ],  # String to list of randomly shuffled JSON objects
        weight_log_dir,
        reweighting_policy,  # Function that takes in current weights, loss windows
        reweighting_freq,  # Per how many iterations do we reweight?
        initial_weights=None,
    ):
        """
        Args:
            domain_datasets: Dictionary mapping domain names to their respective datasets
            tokenizer: Tokenizer for processing the text
            initial_weights: Initial sampling weights for each domain (optional)
            samples_per_epoch: Number of samples to draw per epoch (defaults to total dataset size)
        """
        # Read in data
        assert total_num_datapoints <= sum(len(v) for v in domain_datasets.values())
        self.total_num_datapoints = total_num_datapoints
        self.num_domains = num_domains
        assert (
            len(domain_datasets) == num_domains
        ), f"Got {len(domain_datasets)} but expected {num_domains} domains"
        self.domains = list(domain_datasets.keys())
        self.domain_data = domain_datasets
        # Shuffle the domain data as well
        for ls in self.domain_data.values():
            random.shuffle(ls)
        self.domain_positions = {d: 0 for d in self.domains}

        # Initialize weights equally if not provided. Weights are NOT normalized
        self.weights = initial_weights or np.zeros(self.num_domains)

        # Keep track of domain-specific losses
        self.domain_losses = {domain: deque() for domain in self.domains}
        self.loss_window_size = loss_window_size  # Number of recent losses to consider

        # Log all the weights
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=weight_log_dir,
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logger
        self.update_counter = 0
        self.reweighting_policy = reweighting_policy
        self.reweighting_freq = reweighting_freq
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return self.total_num_datapoints

    def __getitem__(self, idx):
        # Sample a domain
        idx = list(WeightedRandomSampler(np.exp(self.weights), 1))[0]
        domain = self.domains[idx]
        # logging.info(f"Sampled {domain}")
        pos = self.domain_positions[domain]
        item = self.domain_data[domain][pos]
        self.domain_positions[domain] += 1  # Advance all the stuff

        input_text = f"Question: {item['question']}\nAnswer:"
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_sequence_length,
            padding="max_length",
            truncation=True,
        )
        target_text = item["answer"]
        targets = self.tokenizer(
            target_text,
            max_length=self.max_sequence_length,
            padding="max_length",
            truncation=True,
        )

        return {
            "domain": domain,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"],
        }

    # Need to update in order to apply backpressure.
    def add_loss(self, domain, loss):
        # Add to sliding window
        for dom in set(domain):
            window = self.domain_losses[dom]
            window.append(loss)
            if len(window) > self.loss_window_size:
                window.popleft()
        self.update_counter += 1  # How many batches have passed
        if self.update_counter % self.reweighting_freq == 0:
            # Update weights
            self.weights = self.reweighting_policy(self.weights, self.domain_losses)
            logging.info(
                f"Iteration {self.update_counter}: weights changed to {np.array_str(self.weights, precision=5)}"
            )


class DomainWeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_steps = defaultdict(int)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to track domain-specific losses"""
        domain = inputs.pop("domain")
        loss = super().training_step(model, inputs, num_items_in_batch)
        # Update weights for each domain in the batch
        if not isinstance(self.train_dataset, DomainWeightedDataset):
            raise Exception("invalid dataset paired with DomainWeightedTrainer")
        self.train_dataset.add_loss(domain, loss.item())
        return loss

    def get_train_dataloader(self):
        """
        Override to preserve custom fields when creating the dataloader
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # Define a custom collate function that preserves all fields
        def custom_collate(features):
            batch = {}
            # First collect all keys we want to preserve
            for k in features[0].keys():
                if k == "domain":
                    batch[k] = [f[k] for f in features]
                else:
                    batch[k] = torch.stack([torch.tensor(f[k]) for f in features])
                # if torch.is_tensor(features[0][k]):
                #     print(k)
                #     batch[k] = torch.stack([f[k] for f in features])
                #     print(type(batch[k]))
                # else:

            return batch

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self._get_train_sampler(),
            collate_fn=custom_collate,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class DomainPreservingCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        # Get the original batched features
        batch = super().__call__(features, return_tensors=return_tensors)

        # Add the domain labels
        if "domain" in features[0]:
            batch["domain"] = torch.tensor([f["domain"] for f in features])

        return batch


def count_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))


def load_domain_data(path):
    """Load training data from domain JSONL files and report token counts."""
    domain_data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            domain = obj["domain"]
            del obj["domain"]
            domain_data.setdefault(domain, []).append(obj)
    return domain_data


def main():
    # Initialize logging
    logging.set_verbosity_info()

    # Model and tokenizer initialization
    model_name = "unsloth/Llama-3.2-1B"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        ),
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare model for training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    # Add LoRA adaptor
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    # Load domain-specific data

    domain_data = load_domain_data("../data/math_sampled.jsonl")

    # Count total tokens
    total_tokens = 0
    domain_tokens = {}
    for domain, data in domain_data.items():
        domain_total = 0
        for item in data:
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            tokens = count_tokens(text, tokenizer)
            domain_total += tokens
        domain_tokens[domain] = domain_total
        total_tokens += domain_total

    print("\nToken counts per domain:")
    for domain, count in domain_tokens.items():
        print(f"{domain}: {count:,} tokens ({len(domain_data[domain]):,} examples)")
    print(f"\nTotal tokens: {total_tokens:,}")

    if total_tokens < 500_000:
        print("\nWarning: Total token count is below recommended minimum (500K)")
    elif total_tokens > 50_000_000:
        print(
            "\nWarning: Total token count is very high. Consider reducing data or increasing learning rate"
        )

    # Create dataset with dynamic weighting
    train_dataset = DomainWeightedDataset(domain_data, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        evaluation_strategy="no",
    )

    # Initialize custom trainer
    trainer = DomainWeightedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the final model and domain weights
    trainer.save_model("./final_lora_model")

    # Save final domain weights
    with open("./final_domain_weights.json", "w") as f:
        json.dump(train_dataset.weights, f, indent=4)

    print("Training completed! Model and domain weights saved.")
