import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# Load dataset
def load_data():
    # Load training data
    train_df = pd.read_csv('data/training.csv')
    val_df = pd.read_csv('data/validation.csv')
    return train_df, val_df

# Preprocess the data
def preprocess_data(train_df, val_df):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Tokenization
    train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True)
    val_encodings = tokenizer(list(val_df['text']), truncation=True, padding=True)

    return train_encodings, val_encodings, train_df, val_df

# Create dataset class
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Train the model
def train_model(train_encodings, val_encodings, train_labels, val_labels):
    train_dataset = EmotionDataset(train_encodings, train_labels)
    val_dataset = EmotionDataset(val_encodings, val_labels)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)  # Adjust num_labels based on your dataset

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",     # Evaluate at the end of each epoch
        save_strategy="epoch",           # Save the model at each epoch
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train the model
    trainer.train()

    # Save the trained model
    model.save_pretrained('./final_model')

    return model, trainer

# Evaluate the model
def evaluate_model(trainer, val_dataset):
    # Make predictions on the validation set
    predictions, labels, _ = trainer.predict(val_dataset)
    preds = predictions.argmax(-1)
    
    accuracy = accuracy_score(labels, preds)
    print(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # Load and preprocess data
    train_df, val_df = load_data()
    train_encodings, val_encodings, train_df, val_df = preprocess_data(train_df, val_df)

    # Train the model
    model, trainer = train_model(
        train_encodings, 
        val_encodings, 
        train_df['label'].tolist(), 
        val_df['label'].tolist()
    )

    # Evaluate the model
    val_dataset = EmotionDataset(val_encodings, val_df['label'].tolist())
    evaluate_model(trainer, val_dataset)
