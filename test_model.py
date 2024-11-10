import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from sklearn.metrics import accuracy_score

# Load the model
model = DistilBertForSequenceClassification.from_pretrained('C:/Users/Dell/OneDrive/Documents/Documents/Level 6/Final Year Project (FYP)/Reya/POC/final_model')

# Create a tokenizer from the same model type
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  

# Load the test dataset
test_df = pd.read_csv('C:/Users/Dell/OneDrive/Documents/Documents/Level 6/Final Year Project (FYP)/Reya/POC/data/test.csv')
test_texts = test_df['text'].tolist()  
test_labels = test_df['label'].tolist()  

# Tokenize the test data
inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')

# Make predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions.numpy())
print(f"Model Accuracy: {accuracy:.2f}")

# Create a DataFrame for results
results_df = pd.DataFrame({
    'text': test_texts,
    'actual_label': test_labels,
    'predicted_label': predictions.numpy()  # Convert tensor to numpy array
})

# Print the results
print(results_df)

# Save results to a CSV file
results_df.to_csv('test_results.csv', index=False)
print("Test results saved to 'test_results.csv'.")