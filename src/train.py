import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from utils1 import tokenize, bag_of_words, stem  # Import from utils.py


# Load GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


# Tokenized sentence -> word indices
def words_to_indices(words, word_to_idx, max_len):
    # Flatten the list if it's a list of lists
    if isinstance(words[0], list):
        words = [
            item for sublist in words for item in sublist
        ]  # Flatten the list of lists

    # Convert words to indices with padding/truncation
    indices = [word_to_idx.get(word, 0) for word in words]  # 0 for unknown words
    # Pad or truncate to max_len
    if len(indices) < max_len:
        indices.extend(
            [0] * (max_len - len(indices))
        )  # Pad with 0 (unknown word index)
    else:
        indices = indices[:max_len]  # Truncate to max_len
    return indices


# Neural network model class
class NeuralNet(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_classes, embeddings, dropout_rate=0.5
    ):
        super(NeuralNet, self).__init__()

        # Define the embedding layer initialized with GloVe vectors
        self.embeddings = nn.Embedding(
            len(embeddings), 100
        )  # 100 corresponds to the GloVe vector size (e.g., glove.6B.100d.txt)
        self.embeddings.weight.data.copy_(
            torch.from_numpy(np.array(list(embeddings.values())))
        )  # Initialize with GloVe embeddings

        # Define the rest of the model layers
        self.l1 = nn.Linear(
            100, hidden_size
        )  # First layer: input_size (100) from GloVe, hidden_size (e.g., 8)
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Second layer
        self.l3 = nn.Linear(hidden_size, num_classes)  # Output layer

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Get word embeddings using the embedding layer
        x = self.embeddings(x)  # Shape: (batch_size, seq_len, embedding_dim)
        x = torch.mean(
            x, dim=1
        )  # Take the mean of embeddings for sentence-level representation (shape: (batch_size, embedding_dim))

        # Pass through the hidden layers
        out = self.l1(x)
        out = F.relu(out)
        out = self.dropout(out)  # Apply dropout
        out = self.l2(out)
        out = F.relu(out)
        out = self.dropout(out)  # Apply dropout
        out = self.l3(out)  # Output layer

        return out


# Load intents file
with open("intents.json", "r") as f:
    intents = json.load(f)

all_words = []  # Initialize as an empty list
tags = []
xy = []

# Process each sentence in the intents patterns
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        words = tokenize(pattern)  # Tokenize each pattern
        all_words.extend(words)  # Add tokenized words to all_words
        xy.append((words, tag))  # Add pair of (words, tag) to xy

# Stem and lower each word, ensuring we only pass strings to the stemmer
ignore_words = ["?", ".", "!"]
all_words = [stem(w) for w in all_words if isinstance(w, str) and w not in ignore_words]

# Remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Load GloVe embeddings
glove_path = "glove.6B.100d.txt"  # Update this path to your GloVe file
embeddings = load_glove_embeddings(glove_path)

# Create word-to-index mapping
word_to_idx = {word: idx for idx, word in enumerate(embeddings.keys())}

# Set a fixed length for padding/truncation (maximum length of sentences)
MAX_LENGTH = 20  # This value can be adjusted based on your dataset's characteristics

# Create training data
X_train = []
y_train = []
for pattern_sentence, tag in xy:
    # Convert words in the sentence to indices based on the GloVe vocabulary
    bag = words_to_indices(pattern_sentence, word_to_idx, MAX_LENGTH)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels (not one-hot)
    label = tags.index(tag)
    y_train.append(label)

# Convert training data to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
num_epochs = 1000
batch_size = 32
learning_rate = 0.001
input_size = 100  # Use the correct embedding size (100 for glove.6B.100d.txt)
hidden_size = 8
output_size = len(tags)


# Create Dataset and DataLoader
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
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Initialize model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size, embeddings).to(device)

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

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print(f"Training complete. Final loss: {loss.item():.4f}")

# Save model state
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}
FILE = "chatbot_model.pth"
torch.save(data, FILE)
print(f"Model saved to {FILE}")
