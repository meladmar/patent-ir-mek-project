# -*- coding: utf-8 -*-
"""splade4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15nrGA2PegbU_6gNE-zM2j7BqSqW5YwIG
"""

from google.colab import drive
import os
import shutil

# Define the mountpoint
mountpoint = '/content/drive'

# Check if the mountpoint is already mounted and unmount if it is
if os.path.exists(mountpoint) and os.path.ismount(mountpoint):
    print(f"Unmounting existing {mountpoint}...")
    try:
        # Attempt to unmount the drive gracefully
        drive.flush_and_unmount()
        # In some cases, force unmount might be needed, but start with graceful
        # !fusermount -uz {mountpoint}
    except Exception as e:
        print(f"Error unmounting: {e}")

# Mount Google Drive
print(f"Mounting Google Drive to {mountpoint}...")
drive.mount(mountpoint, force_remount=True)



# View logs:
!tail -f /content/drive/MyDrive/WPI_60K/splade_indexing.log

# Check validation samples:
!head -n 20 /content/drive/MyDrive/WPI_60K/validation_samples.json

import os
import json
import time
import torch
import logging
from google.colab import drive
from tqdm.notebook import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

# === Configure Logging ===
logging.basicConfig(
    filename='/content/drive-reconnection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Google Drive Connection Manager ===
class DriveManager:
    def __init__(self, mount_point='/content/drive', retries=5):
        self.mount_point = mount_point
        self.retries = retries

    def ensure_mounted(self):
        for attempt in range(self.retries):
            try:
                if not os.path.ismount(self.mount_point):
                    drive.flush_and_unmount()
                    drive.mount(self.mount_point, force_remount=True)
                    logging.info("Google Drive mounted successfully")
                return True
            except Exception as e:
                logging.error(f"Mount attempt {attempt+1} failed: {str(e)}")
                time.sleep(5 * (attempt + 1))
        raise ConnectionError("Failed to mount Google Drive after retries")

# === Path Verification ===
def verify_paths(drive_manager):
    required_paths = [
        "/content/drive/MyDrive/WPI_60K/extracted/extracted",
        "/content/drive/MyDrive/WPI_60K"
    ]
    for path in required_paths:
        if not os.path.exists(path):
            drive_manager.ensure_mounted()
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path not found: {path}")

# === SPLADE Processor with Connection Recovery ===
class SPLADEProcessor:
    def __init__(self):
        self.drive = DriveManager()
        self.setup_paths()
        self.setup_model()

    def setup_paths(self):
        self.drive.ensure_mounted()
        verify_paths(self.drive)

        self.INPUT_DIR = "/content/drive/MyDrive/WPI_60K/extracted/extracted"
        self.OUTPUT_FILE = "/content/drive/MyDrive/WPI_60K/splade_vectors_v3.jsonl"
        self.CHECKPOINT_FILE = "/content/drive/MyDrive/WPI_60K/checkpoint_v3.json"

    def setup_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "naver/splade-cocondenser-ensembledistil"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def process_files(self):
        checkpoint = self.load_checkpoint()
        all_files = self.get_file_list()

        start_idx = self.get_start_index(checkpoint, all_files)
        print(f"🔄 Resuming from file {start_idx} of {len(all_files)}")

        with open(self.OUTPUT_FILE, 'a' if start_idx > 0 else 'w') as out_f:
            for i, file in enumerate(tqdm(all_files[start_idx:], initial=start_idx, total=len(all_files))):
                try:
                    self.drive.ensure_mounted()  # Check connection before each file

                    result = self.process_single_file(file)
                    if result:
                        out_f.write(json.dumps(result) + '\n')
                        out_f.flush()  # Force write

                    checkpoint = self.update_checkpoint(checkpoint, file, i, start_idx)

                except Exception as e:
                    logging.error(f"Failed {file}: {str(e)}")
                    time.sleep(5)  # Cool-down period
                    continue

        print(f"✅ Completed {len(all_files)} files")

    def process_single_file(self, file):
        file_path = os.path.join(self.INPUT_DIR, file)
        with open(file_path) as f:
            data = json.load(f)

        text = f"{data.get('title','')} {data.get('abstract','')} {data.get('claims','')}"
        if not text.strip():
            return None

        sparse_vector = self.splade_encode(text)
        if not sparse_vector:
            return None

        return {
            'id': data.get("docno", file.replace(".json", "")),
            'vector': sparse_vector
        }

    def splade_encode(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                output = self.model(**inputs).logits
            sparse_weights = torch.max(torch.log(1 + torch.relu(output)), dim=1)[0].squeeze()
            nonzero_indices = sparse_weights.nonzero().squeeze().tolist()
            nonzero_indices = [nonzero_indices] if isinstance(nonzero_indices, int) else nonzero_indices

            return {
                self.tokenizer.decode([idx]): sparse_weights[idx].item()
                for idx in nonzero_indices
            }
        except Exception as e:
            logging.error(f"SPLADE encoding failed: {str(e)}")
            return None

    # ... (checkpoint management methods similar to previous version)
    def load_checkpoint(self):
        """Load or initialize checkpoint"""
        if os.path.exists(self.CHECKPOINT_FILE):
            try:
                with open(self.CHECKPOINT_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'processed': [], 'last_file': None}

    def update_checkpoint(self, checkpoint, file, i, start_idx):
        """Update and periodically save checkpoint"""
        checkpoint['processed'].append(file)
        checkpoint['last_file'] = file

        if (i + start_idx) % 100 == 0:
            with open(self.CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint, f)

        return checkpoint

    def get_file_list(self):
        """Get files with retries"""
        for attempt in range(5):
            try:
                return sorted(f for f in os.listdir(self.INPUT_DIR) if f.endswith('.json'))
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
                self.drive.ensure_mounted()

    def get_start_index(self, checkpoint, all_files):
        """Determine resume position"""
        if not checkpoint['last_file']:
            return 0

        try:
            return all_files.index(checkpoint['last_file']) + 1
        except ValueError:
            processed = set(checkpoint['processed'])
            for i, f in enumerate(all_files):
                if f not in processed:
                    return i
            return len(all_files)

# === Execution ===
if __name__ == "__main__":
    processor = SPLADEProcessor()

    try:
        print("🚀 Starting SPLADE processing with connection monitoring")
        processor.process_files()
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        raise

!tail -f "/content/drive/MyDrive/WPI_60K/splade_indexing.log"

processor = SPLADEProcessor()
processor.process_files()

# === Validation Samples ===
def save_validation_sample(doc_id, text, vector):
    sample = {
        'doc_id': doc_id,
        'text_snippet': text[:200] + "...",
        'sample_tokens': dict(list(vector.items())[:5])
    }
    if not os.path.exists(VALIDATION_FILE):
        samples = []
    else:
        try:
            with open(VALIDATION_FILE, 'r') as f:
                samples = json.load(f)
        except:
            samples = []

    samples.append(sample)
    with open(VALIDATION_FILE, 'w') as f:
        json.dump(samples[:100], f)  # Keep last 100 samples