import os
import json
import argparse
from PIL import Image
import pandas as pd
import random
from datasets import Dataset
from tqdm import tqdm
import warnings


# ================================================================================
# ARGUMENT PARSING
# ================================================================================

def parse_args():
    """Parse command line arguments for data preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare multimodal medical datasets for PaliGemma2 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data Configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        "--base_dir", 
        type=str, 
        default="original_dataset", 
        help="Base directory containing the organized medical datasets"
    )
    data_group.add_argument(
        "--output_dir", 
        type=str, 
        default="processed_dataset", 
        help="Output directory for processed datasets"
    )
    
    # Sampling Configuration
    sampling_group = parser.add_argument_group('Sampling Configuration')
    sampling_group.add_argument(
        "--max_samples", 
        type=int, 
        default=None, 
        help="Maximum number of samples per dataset (for prototyping, None=use all)"
    )
    sampling_group.add_argument(
        "--val_ratio", 
        type=float, 
        default=0.2, 
        help="Ratio of validation samples relative to max_samples"
    )
    
    # Dataset Selection
    selection_group = parser.add_argument_group('Dataset Selection')
    selection_group.add_argument(
        "--include_datasets", 
        type=str, 
        nargs='+', 
        default=None, 
        help="Specific datasets to include (if None, include all)"
    )
    selection_group.add_argument(
        "--exclude_datasets", 
        type=str, 
        nargs='+', 
        default=None, 
        help="Datasets to exclude from processing"
    )
    
    return parser.parse_args()


# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def validate_image(image_path):
    """Validate that an image can be opened and converted to RGB."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


def get_dataset_configs():
    """Return configuration for all available datasets."""
    return [
        {"name": "bcn20000", "category": "Dermatology", "task_type": "Classification"},
        {"name": "IU_XRay", "category": "Xray", "task_type": "Report_Generation"},
        {"name": "iugc", "category": "Ultrasound", "task_type": "Classification, Detection, Regression"},
        {"name": "chestdr", "category": "Xray", "task_type": "Multi-label Classification"},
        {"name": "endo", "category": "Endoscopy", "task_type": "Classification"},
        {"name": "CMMD", "category": "Mammography", "task_type": "Classification"},
        {"name": "periapical", "category": "Xray", "task_type": "Multi-label Classification"},
        {"name": "neojaundice", "category": "Clinical", "task_type": "Classification"},
        {"name": "chromosome", "category": "Microscopy", "task_type": "instance_detection"},
        {"name": "retino", "category": "Retinography", "task_type": "Classification"},
        {"name": "neurips22cell", "category": "Microscopy", "task_type": "Counting"},
        {"name": "bone_marrow", "category": "Microscopy", "task_type": "classification"},
        {"name": "boneresorption", "category": "Xray", "task_type": "regression"},
        {"name": "dental", "category": "Xray", "task_type": "Classification"},
        {"name": "fundus", "category": "Retinography", "task_type": "Classification"},
        {"name": "BUSI", "category": "Ultrasound", "task_type": "classification"},
        {"name": "BUS-UCLM", "category": "Ultrasound", "task_type": "classification"},
        {"name": "BUSI-det", "category": "Ultrasound", "task_type": "detection"},
        {"name": "BUS-UCLM-det", "category": "Ultrasound", "task_type": "detection"}
    ]


# ================================================================================
# DATA LOADING FUNCTIONS
# ================================================================================

def load_images_and_questions(base_dir, dataset_name, split, task_type, category, max_samples=None):
    """Load images and questions from a specific dataset and split."""
    
    # Determine paths based on split
    if split == "train":
        image_dir = os.path.join(base_dir, "training", category, dataset_name, "imagesTr")
        question_dir = os.path.join(base_dir, "training", category, dataset_name)
    else:  # val
        image_dir = os.path.join(base_dir, "validation-public", category, dataset_name, "imagesVal")
        question_dir = os.path.join(base_dir, "validation-public", category, dataset_name)
    
    # Find JSON file in the directory
    def find_json_file(directory, split_type):
        """Find appropriate JSON file in the directory."""
        if not os.path.exists(directory):
            warnings.warn(f"Directory does not exist: {directory}")
            return None
            
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        
        if not json_files:
            warnings.warn(f"No JSON files found in directory: {directory}")
            return None
        
        return os.path.join(directory, json_files[0])
    
    question_file = find_json_file(question_dir, split)
    
    # Check if files exist
    if not os.path.exists(image_dir) or not question_file or not os.path.exists(question_file):
        warnings.warn(f"Missing required files for {dataset_name} {split}: image_dir={image_dir}, question_file={question_file}")
        return []
    
    print(f"Using question file: {os.path.basename(question_file)}")
    
    # Load questions
    with open(question_file, 'r') as f:
        questions = json.load(f)
    
    # Filter by task type
    if task_type:
        questions = [q for q in questions if q['TaskType'].lower() == task_type.lower()]
    
    # Apply sampling
    if max_samples and len(questions) > max_samples:
        questions = random.sample(questions, max_samples)
    
    # Process questions
    formatted_data = []
    for question in tqdm(questions, desc=f"Processing {dataset_name} {split}"):
        image_name = question["ImageName"]
        
        # Handle multiple images
        if isinstance(image_name, list):
            image_paths = []
            valid = True
            
            for img_name in image_name:
                img_path = os.path.join(image_dir, os.path.basename(img_name))
                if not os.path.exists(img_path):
                    warnings.warn(f"Image file not found: {img_path}")
                    valid = False
                    break
                if not validate_image(img_path):
                    warnings.warn(f"Invalid image file: {img_path}")
                    valid = False
                    break
                image_paths.append(img_path)
            
            if not valid:
                continue
            image_data = image_paths
        else:
            # Handle single image
            image_path = os.path.join(image_dir, os.path.basename(image_name))
            if not os.path.exists(image_path):
                warnings.warn(f"Image file not found: {image_path}")
                continue
            if not validate_image(image_path):
                warnings.warn(f"Invalid image file: {image_path}")
                continue
            image_data = [image_path]
        
        sample = {
            "image": image_data,
            "question": question["Question"],
            "answer": str(question["Answer"]),
            "task_type": question["TaskType"],
            "modality": question["Modality"],
            "dataset": dataset_name
        }
        formatted_data.append(sample)
    
    return formatted_data


# ================================================================================
# MAIN PROCESSING FUNCTION
# ================================================================================

def prepare_dataset(args):
    """Prepare datasets for fine-tuning."""
    
    # Get dataset configurations
    all_configs = get_dataset_configs()
    
    # Filter datasets based on include/exclude lists
    if args.include_datasets:
        all_configs = [c for c in all_configs if c["name"] in args.include_datasets]
    if args.exclude_datasets:
        all_configs = [c for c in all_configs if c["name"] not in args.exclude_datasets]
    
    print(f"Processing {len(all_configs)} datasets")
    
    all_train_data = []
    all_val_data = []
    
    for config in all_configs:
        dataset_name = config["name"]
        category = config["category"]
        task_types = [t.strip() for t in config["task_type"].split(',')]
        
        print(f"\nProcessing dataset: {dataset_name}")
        
        for task_type in task_types:
            # Load training data
            train_data = load_images_and_questions(
                args.base_dir, dataset_name, "train", task_type, category, args.max_samples
            )
            all_train_data.extend(train_data)
            
            # Load validation data
            val_max = int(args.max_samples * args.val_ratio) if args.max_samples else None
            val_data = load_images_and_questions(
                args.base_dir, dataset_name, "val", task_type, category, val_max
            )
            all_val_data.extend(val_data)
    
    # Validate we have data
    if not all_train_data:
        raise ValueError("No training data was loaded")
    
    print(f"\nDataset Summary:")
    print(f"Training samples: {len(all_train_data)}")
    print(f"Validation samples: {len(all_val_data)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert to HuggingFace datasets and save
    train_dataset = Dataset.from_pandas(pd.DataFrame(all_train_data))
    train_dataset.save_to_disk(os.path.join(args.output_dir, "train"))
    print(f"Saved training data to {args.output_dir}/train")
    
    if all_val_data:
        val_dataset = Dataset.from_pandas(pd.DataFrame(all_val_data))
        val_dataset.save_to_disk(os.path.join(args.output_dir, "validation"))
        print(f"Saved validation data to {args.output_dir}/validation")
    
    # Save dataset info
    dataset_info = {
        "train_samples": len(all_train_data),
        "val_samples": len(all_val_data),
        "processed_datasets": [c["name"] for c in all_configs],
        "task_types": list(set(item["task_type"] for item in all_train_data)),
        "modalities": list(set(item["modality"] for item in all_train_data)),
        "config": vars(args)
    }
    
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    return len(all_train_data), len(all_val_data)


# ================================================================================
# SCRIPT ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    args = parse_args()
    
    print("PaliGemma2 Data Preparation")
    print("=" * 50)
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max samples per dataset: {args.max_samples or 'All'}")
    
    try:
        train_count, val_count = prepare_dataset(args)
        print(f"\nSuccessfully prepared {train_count} training and {val_count} validation samples")
    except Exception as e:
        print(f"\nError: {e}")
        raise 