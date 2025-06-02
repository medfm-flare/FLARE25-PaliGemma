"""
PaliGemma2 Fine-tuning Script for Medical Image Analysis

This script fine-tunes a PaliGemma2 model for medical image question answering tasks.
It supports LoRA (Low-Rank Adaptation) for efficient training and 4-bit quantization
to reduce memory usage.
"""

import os
import torch
import argparse
from datasets import load_from_disk
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig
from safetensors.torch import save_file, load_file
from PIL import Image
from torch.utils.data import Dataset


# ================================================================================
# ARGUMENT PARSING
# ================================================================================

def parse_args():
    """Parse command line arguments for fine-tuning configuration."""
    parser = argparse.ArgumentParser(
        description="Fine-tune PaliGemma2 model for medical image analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and Data Configuration
    model_group = parser.add_argument_group('Model and Data Configuration')
    model_group.add_argument(
        "--model_id", 
        type=str, 
        default="google/paligemma2-10b-pt-224", 
        help="Hugging Face model ID for PaliGemma2"
    )
    model_group.add_argument(
        "--train_data_path", 
        type=str,
        default="processed_data_full/train", 
        help="Path to training dataset directory"
    )
    model_group.add_argument(
        "--val_data_path", 
        type=str, 
        default="processed_data_full/validation", 
        help="Path to validation dataset directory"
    )
    model_group.add_argument(
        "--output_dir", 
        type=str, 
        default="fintuned_paligemma2", 
        help="Output directory for saving fine-tuned model and checkpoints"
    )
    model_group.add_argument(
        "--logging_dir", 
        type=str, 
        default="paligemma2_logs", 
        help="Directory for saving training logs"
    )
    
    # Training Hyperparameters
    training_group = parser.add_argument_group('Training Hyperparameters')
    training_group.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=1, 
        help="Total number of training epochs"
    )
    training_group.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=1, 
        help="Training batch size per GPU device"
    )
    training_group.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=32, 
        help="Number of updates steps to accumulate before performing a backward pass"
    )
    training_group.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5, 
        help="Initial learning rate for AdamW optimizer"
    )
    training_group.add_argument(
        "--warmup_ratio", 
        type=float, 
        default=0.03, 
        help="Ratio of total training steps used for linear warmup"
    )
    training_group.add_argument(
        "--max_grad_norm", 
        type=float, 
        default=1.0, 
        help="Maximum gradient norm for gradient clipping"
    )
    
    # Checkpoint and Evaluation Configuration
    checkpoint_group = parser.add_argument_group('Checkpoint and Evaluation')
    checkpoint_group.add_argument(
        "--save_steps", 
        type=int, 
        default=100, 
        help="Number of steps between checkpoint saves"
    )
    checkpoint_group.add_argument(
        "--eval_steps", 
        type=int, 
        default=100, 
        help="Number of steps between evaluations"
    )
    checkpoint_group.add_argument(
        "--logging_steps", 
        type=int, 
        default=10, 
        help="Number of steps between logging training metrics"
    )
    checkpoint_group.add_argument(
        "--save_total_limit", 
        type=int, 
        default=5, 
        help="Maximum number of checkpoints to keep (older ones are deleted)"
    )
    
    # LoRA (Low-Rank Adaptation) Configuration
    lora_group = parser.add_argument_group('LoRA Configuration')
    lora_group.add_argument(
        "--lora_r", 
        type=int, 
        default=16, 
        help="LoRA rank - controls the bottleneck dimension"
    )
    lora_group.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32, 
        help="LoRA alpha - scaling parameter for LoRA weights"
    )
    lora_group.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.05, 
        help="Dropout probability for LoRA layers"
    )
    
    # Checkpoint Resume Configuration
    resume_group = parser.add_argument_group('Checkpoint Resume Configuration')
    resume_group.add_argument(
        "--resume_from_checkpoint", 
        action="store_true", 
        help="Whether to resume training from a checkpoint"
    )
    resume_group.add_argument(
        "--checkpoint_path", 
        type=str, 
        default="fintuned_paligemma2/checkpoint-100", 
        help="Checkpoint path to resume training from"
    )
    
    # Hardware Configuration
    hardware_group = parser.add_argument_group('Hardware Configuration')
    hardware_group.add_argument(
        "--gpu_id", 
        type=str, 
        default="1", 
        help="GPU device ID to use for training"
    )
    
    return parser.parse_args()


# ================================================================================
# DATASET CLASSES
# ================================================================================

class ImageQADataset(Dataset):
    def __init__(self, dataset, processor, image_token="<image>", max_images=5):

        # Filter out samples with too many images to prevent memory issues
        self.dataset = dataset.filter(
            lambda ex: len(ex["image"]) <= max_images if isinstance(ex["image"], list) else True
        )
        self.processor = processor
        self.image_token = image_token
        
        print(f"Dataset loaded with {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]
        
        image_paths = sample["image"] if isinstance(sample["image"], list) else [sample["image"]]        
        images = [Image.open(path).convert("RGB") for path in image_paths]

        num_image_tokens = self.processor.image_seq_length * len(images)
        formatted_question = (
            f"{self.processor.tokenizer.bos_token}"
            f"{self.image_token * num_image_tokens}"
            "Analyze the given medical image and answer the following question:\n"
            f"Question: {sample['question']}\n"
            "Please provide a clear and concise answer."
        )
        
        return {
            "question": formatted_question,
            "answer": sample["answer"],
            "images": images,
        }


# ================================================================================
# DATA COLLATION
# ================================================================================

def create_collate_fn(processor):

    def collate_fn(batch):
        all_images = []
        for sample in batch:
            all_images.extend(sample["images"])
        
        with torch.no_grad():
            pixel_values = processor.image_processor(
                all_images, 
                return_tensors="pt"
            )["pixel_values"].to(torch.bfloat16)
        
        questions = [sample["question"] for sample in batch]
        answers = [sample["answer"] for sample in batch]
        
        tokenized = processor.tokenizer(
            questions,
            text_pair=[answer + processor.tokenizer.eos_token for answer in answers],
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        
        input_ids = tokenized.input_ids.to(torch.long)
        attention_mask = tokenized.attention_mask.to(torch.long)
        token_type_ids = tokenized.token_type_ids.to(torch.long)
        
        labels = input_ids.masked_fill(token_type_ids == 0, -100).to(torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
    
    return collate_fn


# ================================================================================
# CONFIGURATION SETUP
# ================================================================================

def setup_model_configs(args):
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "o_proj", "k_proj", "v_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    return lora_config, quantization_config


def setup_training_args(args):

    return TrainingArguments(
        # Output and Logging
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        
        # Training Schedule
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        
        # Precision and Optimization
        bf16=True,                              
        fp16=False,                             
        optim='paged_adamw_8bit',               
        max_grad_norm=args.max_grad_norm,       
        gradient_checkpointing=True,            
        
        # Checkpointing
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
        
        # Evaluation
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        metric_for_best_model='eval_loss',
        greater_is_better=False,                
        load_best_model_at_end=True,
        
        # Performance Optimization
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        
        # Model-specific Settings
        label_names=["labels"],
    )


# ================================================================================
# MAIN TRAINING FUNCTION
# ================================================================================

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"Using GPU: {args.gpu_id}")
    
    # Initialize processor for tokenization and image processing
    print(f"Loading processor for model: {args.model_id}")
    processor = PaliGemmaProcessor.from_pretrained(args.model_id)
    
    # Set up model configurations
    lora_config, quantization_config = setup_model_configs(args)
    
    # Load datasets
    print("Loading datasets...")
    train_raw_dataset = load_from_disk(args.train_data_path)
    val_raw_dataset = load_from_disk(args.val_data_path)
    
    train_dataset = ImageQADataset(train_raw_dataset, processor)
    val_dataset = ImageQADataset(val_raw_dataset, processor)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Load and configure model
    print("Loading model with 4-bit quantization...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map={"": 0},                     
        low_cpu_mem_usage=True,                 
        attn_implementation="eager",            
        use_safetensors=True,                   
    )
    model.enable_input_require_grads()
    
    # Apply LoRA adapter
    print("Applying LoRA adapter...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Set up training arguments
    training_args = setup_training_args(args)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=create_collate_fn(processor),
        args=training_args,
    )
    
    # Handle checkpoint resuming
    checkpoint_path = None
    if args.resume_from_checkpoint:
        checkpoint_path = args.checkpoint_path
    
    # Start training
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        if args.resume_from_checkpoint:
            print(f"Checkpoint not found at {checkpoint_path}, starting training from scratch")
        else:
            print("Starting training from scratch")
        trainer.train()
    
    print("Training completed successfully!")


# ================================================================================
# SCRIPT ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    main()