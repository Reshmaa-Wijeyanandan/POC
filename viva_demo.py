import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the trained model
model = DistilBertForSequenceClassification.from_pretrained('./final_model')
model.eval()

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Map labels with emotions
emotion_map = {
    0: 'Sad', 
    1: 'Joy', 
    2: 'Love', 
    3: 'Anger', 
    4: 'Fear'
}

# Here we create a function to predict emotion from text
def predict_emotion(text):
    # Tokenize the input text
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    
    # Get prediction from the model
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    # Map the emotion label to the prediction
    emotion = emotion_map.get(prediction, "Unknown")
    return emotion

# Interactive loop for live predictions
if __name__ == "__main__":
    print("Emotion Prediction Demo")
    print("Type 'exit' to end the demo.\n")

    while True:
        user_input = input("Please enter a sentence to predict what emotion it carries (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        emotion = predict_emotion(user_input)
        print("Predicted Emotion: " + emotion + "\n")
