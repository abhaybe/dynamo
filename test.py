from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorForLanguageModeling

def preprocess(examples):
    max_length = 128
    
    input_texts = [f"Question: {q}\nAnswer:" for q in examples['question']]
    target_texts = examples['answer']
    
    inputs = tokenizer(
        input_texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None
    )
    
    labels = tokenizer(
        target_texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None
    )
    
    inputs['labels'] = labels['input_ids']
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': inputs['labels']
    }

if __name__ == "__main__":
    model_name = "unsloth/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Modified model loading to explicitly set device map
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map={'': torch.cuda.current_device()},  # Changed this line
        load_in_8bit=True,
        use_cache=False
    )

    dataset = load_dataset("json", data_files="math_sampled.jsonl")
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        batch_size=8,
        remove_columns=dataset["train"].column_names
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir="./math-qa-llama2",
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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator
    )
    
    trainer.train()

    model.save_pretrained("./fine_tuned_llama2_math_qa")
    tokenizer.save_pretrained("./fine_tuned_llama2_math_qa")