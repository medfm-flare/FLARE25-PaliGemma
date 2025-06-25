import os
import subprocess
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftConfig
from huggingface_hub import login
import torch

def get_folder_size(folder_path):
    """Get folder size in bytes and human readable format"""
    if not os.path.exists(folder_path):
        return 0, "0 B"
    
    try:
        # Use du command for accurate size calculation
        result = subprocess.run(['du', '-sb', folder_path], 
                              capture_output=True, text=True, check=True)
        size_bytes = int(result.stdout.split()[0])
        
        # Convert to human readable format
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                size_human = f"{size_bytes:.1f} {unit}"
                break
            size_bytes /= 1024.0
        
        return int(result.stdout.split()[0]), size_human
    except:
        # Fallback method
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        
        # Convert to human readable
        size_bytes = total_size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                size_human = f"{size_bytes:.1f} {unit}"
                break
            size_bytes /= 1024.0
            
        return total_size, size_human

def download_models():
    """Download base model and FLARE 2025 fine-tuned adapter to local directories"""
    
    print('=' * 50)
    print('Downloading FLARE 2025 fine-tuned model...')
    print('=' * 50)
    
    # Define local paths
    models_dir = '/app/models'
    base_model_path = os.path.join(models_dir, 'paligemma2-10b-pt-224')
    adapter_model_path = os.path.join(models_dir, 'flare25-paligemma2')
    
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    print(f'📁 Models will be saved to: {models_dir}')
    
    # Login to HuggingFace Hub if token is provided
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if hf_token:
        print('Logging in to HuggingFace Hub...')
        login(token=hf_token)
        print('Successfully logged in to HuggingFace Hub')
    else:
        print(' No HuggingFace token provided, attempting without authentication...')
    
    # Download base model
    base_model_id = 'google/paligemma2-10b-pt-224'
    print(f'Downloading base model: {base_model_id}')
    print(f'   Saving to: {base_model_path}')
    
    try:
        # Download processor
        processor = PaliGemmaProcessor.from_pretrained(base_model_id)
        processor.save_pretrained(base_model_path)
        print(f'Processor saved to {base_model_path}')
        
        # Download base model without quantization (quantization will be applied during inference)
        print('Downloading base model (quantization will be applied during inference)...')
        
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map=None  # Don't load to GPU during download
        )
        
        # Save base model
        model.save_pretrained(base_model_path)
        print(f'Base model saved to {base_model_path}')
        
        # Clear memory
        del model
        del processor
        
    except Exception as e:
        print(f'Failed to download base model: {e}')
        raise
    
    # Download FLARE 2025 fine-tuned adapter
    adapter_model_id = 'yws0322/flare25-paligemma2'
    print(f'Downloading FLARE adapter: {adapter_model_id}')
    print(f'   Saving to: {adapter_model_path}')
    
    try:
        # Download adapter config and weights
        config = PeftConfig.from_pretrained(adapter_model_id)
        config.save_pretrained(adapter_model_path)
        
        # Also download adapter model files manually to ensure all files are saved
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=adapter_model_id,
            local_dir=adapter_model_path,
            local_files_only=False
        )
        print(f'FLARE adapter saved to {adapter_model_path}')
        
    except Exception as e:
        print(f'Failed to download FLARE adapter: {e}')
        raise
    
    # Verify downloads
    print('\nVerifying downloads...')
    
    # Check base model files
    base_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    for file in base_files:
        file_path = os.path.join(base_model_path, file)
        if os.path.exists(file_path):
            print(f'  {file}')
        else:
            print(f'  {file} missing')
    
    # Check adapter files
    adapter_files = ['adapter_config.json']
    for file in adapter_files:
        file_path = os.path.join(adapter_model_path, file)
        if os.path.exists(file_path):
            print(f'  {file}')
        else:
            print(f'  {file} missing')
    
    # Calculate and display folder sizes
    print('\nCalculating folder sizes...')
    
    base_size_bytes, base_size_human = get_folder_size(base_model_path)
    adapter_size_bytes, adapter_size_human = get_folder_size(adapter_model_path)
    total_size_bytes = base_size_bytes + adapter_size_bytes
    
    # Convert total size to human readable
    size_bytes = total_size_bytes
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            total_size_human = f"{size_bytes:.1f} {unit}"
            break
        size_bytes /= 1024.0
    
    print(f'Base model size: {base_size_human}')
    print(f'Adapter model size: {adapter_size_human}')
    print(f'Total size: {total_size_human}')
    
    print('=' * 50)
    print('All models downloaded and saved locally!')
    print(f'Base model: {base_model_path} ({base_size_human})')
    print(f'Adapter model: {adapter_model_path} ({adapter_size_human})')
    print(f'Total storage used: {total_size_human}')
    print('=' * 50)

if __name__ == "__main__":
    download_models() 