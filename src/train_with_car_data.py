import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import bag_of_words, tokenize, stem  # Import from utils.py
from model import NeuralNet

# Load intents and car data JSON
with open("intents.json", "r") as f:
    intents = json.load(f)

with open("cars.json", "r") as f:
    car_data = json.load(f)

all_words = []  # Initialize as an empty list
tags = []  # Initialize as an empty list
xy = []

# Process intents.json data (text-based)
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)  # Add the tag from intents.json
    for pattern in intent["patterns"]:
        w = tokenize(pattern)[0]  # Get tokenized words (first element of the tuple)
        all_words.extend(w)  # Add tokenized words to all_words list
        xy.append((w, tag))

# Process car_data.json data (car-related structured data)
for model in car_data["models"]:
    model_name = model["name"]
    features = model["features"]

    # Create patterns based on car features (for example, questions)
    for feature, value in features.items():
        pattern = f"What is the {feature} of the {model_name}?"
        tag = f"{model_name} - {feature}"  # Create a specific tag for each feature

        # Add the new tag from car_data.json to the tags list
        if tag not in tags:
            tags.append(tag)

        # Tokenize and process the patterns as above
        w = tokenize(pattern)[0]
        all_words.extend(w)  # Add tokenized words to all_words list
        xy.append((w, tag))

# Stem and lower each word, ensuring we only pass strings to the stemmer
ignore_words = ["?", ".", "!"]
all_words = [stem(w) for w in all_words if isinstance(w, str) and w not in ignore_words]

# Remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(tags)  # Ensure tags are sorted for consistency

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Create training data
X_train = []
y_train = []
for pattern_sentence, tag in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)  # This should now work because all tags are included
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")

# Save the trained model data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = "chatdata3.pth"
torch.save(data, FILE)
print(f"Training complete. File saved to {FILE}")
