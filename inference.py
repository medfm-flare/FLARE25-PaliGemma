import os
import json
import argparse
import glob
import re
from PIL import Image
from tqdm import tqdm
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel


# ================================================================================
# ARGUMENT PARSING
# ================================================================================

def parse_args():
    """Parse command line arguments for prediction script."""
    parser = argparse.ArgumentParser(
        description="PaliGemma2 prediction script for medical image analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model Configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model_id",
        type=str,
        default="google/paligemma2-10b-pt-224",
        help="HuggingFace model identifier for base PaliGemma2 model"
    )
    model_group.add_argument(
        "--checkpoint_path",
        type=str,
        default="./paligemma2_ckpt/checkpoint-100",
        help="Path to LoRA checkpoint for fine-tuned weights"
    )
    
    # Data Configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        "--base_dataset_path", 
        type=str, 
        default="original_dataset",
        help="Base path to dataset directory (e.g., original_dataset)"
    )
    data_group.add_argument(
        "--validation_type",
        type=str,
        choices=["hidden", "public"],
        default="public",
        help="Type of validation dataset to use (hidden or public)"
    )
    data_group.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for prediction results"
    )
    data_group.add_argument(
        "--output_filename",
        type=str,
        default="predictions.json",
        help="Filename for the output predictions file"
    )
    
    # Inference Configuration
    inference_group = parser.add_argument_group('Inference Configuration')
    inference_group.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate"
    )
    inference_group.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on"
    )
    
    return parser.parse_args()


# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def find_json_files(base_path):
    """Recursively find all JSON files in the specified directory."""
    json_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def validate_paths(dataset_path, checkpoint_path):
    """Validate that required paths exist."""
    # Check dataset path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Find JSON files
    json_files = find_json_files(dataset_path)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {dataset_path}")
    
    # Check checkpoint
    checkpoint_exists = os.path.exists(checkpoint_path)
    
    return json_files, checkpoint_exists


# ================================================================================
# ANSWER PARSING FUNCTIONS
# ================================================================================

def parse_answer(output, task_type=None):
    """Parse model output based on task type to extract the final answer."""
    output = output.strip()
    
    # Remove common prefixes
    if "Please provide a clear and concise answer." in output:
        try:
            output = output.split("Please provide a clear and concise answer.")[-1].strip()
        except:
            pass
    
    # Remove leading newlines
    if "\n" in output:
        output = output.split("\n", 1)[-1].strip()
    
    # Task-specific parsing
    task_type = (task_type or "").strip().lower()
    
    if task_type == "classification":
        return _parse_classification(output)
    elif task_type == "multi-label classification":
        return _parse_multi_label_classification(output)
    elif task_type in ["detection", "instance_detection"]:
        return _parse_detection(output)
    elif task_type in ["cell counting", "regression", "counting"]:
        return _parse_numeric(output)
    elif task_type == "report generation":
        return output
    else:
        return output


def _parse_classification(output):
    """Parse classification task output."""
    lines = output.splitlines()
    if len(lines) >= 1:
        last_line = lines[-1].strip()
        return last_line
    return output

def _parse_multi_label_classification(output):
    """Parse multi-label classification task output."""
    lines = output.splitlines()
    labels = []
    for line in lines:
        for part in re.split(r'[;]', line):
            label = part.strip()
            if label:
                labels.append(label)
    return "; ".join(labels)


def _parse_detection(output):
    """Parse detection task output (JSON format expected)."""
    match = re.search(r'\{.*\}|\[.*\]', output, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return json.dumps(parsed)
        except:
            return match.group()
    return output


def _parse_numeric(output):
    """Parse numeric task output (counting, regression)."""
    match = re.search(r'[-+]?[0-9]*\.?[0-9]+', output)
    if match:
        return match.group()
    return "0"


# ================================================================================
# MODEL LOADING FUNCTIONS
# ================================================================================

def load_model_and_processor(model_id, checkpoint_path, device="cuda:0"):
    """Load PaliGemma2 model and processor with optional LoRA weights."""
    print(f"Loading base model: {model_id}")
    
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load base model
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation='eager',
    )
    
    # Load LoRA weights if available
    if os.path.exists(checkpoint_path):
        model = PeftModel.from_pretrained(model, checkpoint_path)
        print(f"Loaded LoRA weights from {checkpoint_path}")
    else:
        print(f"Warning: LoRA checkpoint {checkpoint_path} not found. Using base model only.")
    
    # Load processor
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    
    return model, processor


# ================================================================================
# PREDICTION FUNCTIONS
# ================================================================================

def predict_on_file(input_file, model, processor, max_new_tokens=1024, device="cuda:0"):
    """Perform predictions on a single JSON file containing questions and images."""
    IMAGE_TOKEN = "<image>"
    
    # Load data
    with open(input_file) as f:
        val_data = json.load(f)
    
    print(f"Processing {len(val_data)} samples from {os.path.basename(input_file)}")
    
    # Process each sample
    for sample in tqdm(val_data, desc=f"Predicting {os.path.basename(input_file)}"):
        try:
            # Handle image loading
            img_field = sample["ImageName"]
            if isinstance(img_field, list):
                img_paths = img_field[:5]  # Limit to 5 images max
            else:
                img_paths = [img_field]
            
            # Load and validate images
            imgs = []
            for img_path in img_paths:
                full_path = os.path.join(os.path.dirname(input_file), img_path)
                try:
                    img = Image.open(full_path).convert("RGB")
                    imgs.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")
                    continue
            
            if not imgs:
                print(f"Warning: No valid images for sample, skipping")
                sample["Answer"] = "Error: No valid images"
                continue
            
            # Prepare input
            formatted_question = (
                "Analyze the given medical image and answer the following question:\n"
                f"Question: {sample['Question']}\n"
                "Please provide a clear and concise answer."
            )
            prefix = IMAGE_TOKEN * (processor.image_seq_length * len(imgs))
            input_text = f"{prefix}{processor.tokenizer.bos_token}{formatted_question}\n"
            
            # Process images and text
            pixel_values = processor.image_processor(imgs, return_tensors="pt")["pixel_values"].to(device)
            inputs = processor.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            
            # Generate prediction
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs.input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            
            # Decode output
            output = processor.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Parse answer based on task type
            parsed_answer = parse_answer(output, sample.get("TaskType", ""))
            sample["Answer"] = parsed_answer
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            sample["Answer"] = f"Error: {str(e)}"
    
    return val_data


# ================================================================================
# MAIN PROCESSING FUNCTION
# ================================================================================

def run_predictions(args):
    """
    Main function to run predictions on all JSON files in the dataset directory.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        int: Number of predictions made
    """
    # Construct full dataset path
    dataset_path = os.path.join(args.base_dataset_path, f"validation-{args.validation_type}")
    
    # Validate paths and find JSON files
    print("Validating paths and discovering files...")
    input_files, checkpoint_exists = validate_paths(dataset_path, args.checkpoint_path)
    
    print(f"Found {len(input_files)} JSON files in {dataset_path}:")
    for file in input_files:
        print(f"  - {os.path.relpath(file, dataset_path)}")
    
    # Load model and processor
    print("\nLoading model and processor...")
    model, processor = load_model_and_processor(
        args.model_id, 
        args.checkpoint_path, 
        args.device
    )
    
    # Run predictions on all files
    print(f"\nRunning predictions...")
    all_predictions = []
    
    for input_file in input_files:
        predictions = predict_on_file(
            input_file, 
            model, 
            processor, 
            args.max_new_tokens,
            args.device
        )
        all_predictions.extend(predictions)
    
    # Save results
    print(f"\nSaving results...")
    output_dir = args.output_dir if args.output_dir else "."
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, args.output_filename)
    
    with open(output_file, "w") as f:
        json.dump(all_predictions, f, indent=2)
    
    return len(all_predictions)


# ================================================================================
# SCRIPT ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    args = parse_args()
    
    print("PaliGemma2 Medical Image Prediction")
    print("=" * 50)
    print(f"Base dataset path: {args.base_dataset_path}")
    print(f"Validation type: {args.validation_type}")
    print(f"Full dataset path: {os.path.join(args.base_dataset_path, f'validation-{args.validation_type}')}")
    print(f"Model ID: {args.model_id}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    
    try:
        prediction_count = run_predictions(args)
        print(f"\nSuccessfully completed {prediction_count} predictions")
    except Exception as e:
        print(f"\nError: {e}")
        raise 