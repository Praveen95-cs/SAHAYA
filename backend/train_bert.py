"""
BERT Fine-tuning for Abuse Classification
Run on Vertex AI for distributed training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from google.cloud import storage
import json

# ===== SYNTHETIC TRAINING DATA =====
# For hackathon demo. Production: use real annotated data

TRAINING_DATA = [
    # Control behaviors
    {"text": "He checks my phone every day", "label": [0.9, 0.1, 0.0, 0.0, 0.0]},
    {"text": "I need to ask permission to see friends", "label": [0.85, 0.15, 0.0, 0.0, 0.0]},
    {"text": "He says I can't work", "label": [0.8, 0.2, 0.0, 0.0, 0.0]},
    
    # Verbal abuse
    {"text": "He calls me stupid all the time", "label": [0.2, 0.8, 0.0, 0.0, 0.0]},
    {"text": "He screams at me constantly", "label": [0.1, 0.85, 0.05, 0.0, 0.0]},
    {"text": "Says I'm worthless", "label": [0.1, 0.9, 0.0, 0.0, 0.0]},
    
    # Threats
    {"text": "He threatened to hurt me", "label": [0.1, 0.2, 0.7, 0.0, 0.0]},
    {"text": "Said I'll regret leaving", "label": [0.0, 0.1, 0.85, 0.05, 0.0]},
    {"text": "Threatened to take the kids", "label": [0.2, 0.1, 0.7, 0.0, 0.0]},
    
    # Physical abuse
    {"text": "He pushed me yesterday", "label": [0.0, 0.1, 0.2, 0.7, 0.0]},
    {"text": "He grabbed my arm hard", "label": [0.0, 0.1, 0.1, 0.8, 0.0]},
    {"text": "He slapped me during argument", "label": [0.0, 0.05, 0.15, 0.8, 0.0]},
    
    # Severe physical
    {"text": "He broke my arm", "label": [0.0, 0.0, 0.1, 0.2, 0.7]},
    {"text": "Had to go to hospital", "label": [0.0, 0.0, 0.05, 0.25, 0.7]},
    {"text": "He choked me until I passed out", "label": [0.0, 0.0, 0.0, 0.1, 0.9]},
]

# Add more synthetic variations
def augment_data(data, num_augmentations=5):
    """Create variations of training data"""
    augmented = []
    synonyms = {
        "checks": ["monitors", "inspects", "watches"],
        "phone": ["cell", "mobile", "device"],
        "angry": ["furious", "mad", "upset"],
        "pushed": ["shoved", "hit", "struck"]
    }
    
    for item in data:
        augmented.append(item)
        # Add slight label noise for robustness
        for _ in range(num_augmentations):
            new_item = item.copy()
            new_item["label"] = [
                min(1.0, max(0.0, l + np.random.normal(0, 0.05)))
                for l in item["label"]
            ]
            # Normalize
            total = sum(new_item["label"])
            new_item["label"] = [l/total for l in new_item["label"]]
            augmented.append(new_item)
    
    return augmented

# ===== DATASET CLASS =====

class AbuseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

# ===== TRAINING FUNCTION =====

def train_model(model, train_loader, val_loader, epochs=3):
    """Train BERT model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.MSELoss()  # Regression for multi-label probabilities
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get logits and apply softmax
            logits = outputs.logits
            predictions = torch.softmax(logits, dim=1)
            
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.softmax(outputs.logits, dim=1)
                val_loss += loss_fn(predictions, labels).item()
        
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
    
    return model

# ===== MAIN TRAINING SCRIPT =====

def main():
    # Prepare data
    augmented_data = augment_data(TRAINING_DATA, num_augmentations=10)
    texts = [item['text'] for item in augmented_data]
    labels = [item['label'] for item in augmented_data]
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=5  # 5 abuse categories
    )
    
    # Create datasets
    train_dataset = AbuseDataset(train_texts, train_labels, tokenizer)
    val_dataset = AbuseDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Train
    print("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, epochs=3)
    
    # Save model
    model_path = "./abuse_classifier_bert"
    trained_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"Model saved to {model_path}")
    
    # Upload to Google Cloud Storage
    bucket_name = "dv-detection-models"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Upload model files
    for file in ["config.json", "pytorch_model.bin", "vocab.txt", "tokenizer_config.json"]:
        blob = bucket.blob(f"bert_classifier/{file}")
        blob.upload_from_filename(f"{model_path}/{file}")
    
    print(f"Model uploaded to gs://{bucket_name}/bert_classifier/")

if __name__ == "__main__":
    main()

# ===== VERTEX AI DEPLOYMENT SCRIPT =====
"""
# deploy_to_vertex.py

from google.cloud import aiplatform

aiplatform.init(project='your-project-id', location='us-central1')

# Upload model
model = aiplatform.Model.upload(
    display_name='abuse-classifier-bert',
    artifact_uri='gs://dv-detection-models/bert_classifier',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-12:latest'
)

# Deploy to endpoint
endpoint = model.deploy(
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1
)

print(f"Model deployed to endpoint: {endpoint.resource_name}")
"""