import json
import torch
from model import NeuralNet
from utils import tokenize, bag_of_words
from nltk.stem.porter import PorterStemmer
import numpy as np

# Load model and data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
data = torch.load("chatdata1.pth", map_location=device)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(data["model_state"])

# Load car data
try:
    with open("cars.json", "r") as f:
        car_data = json.load(f)
except FileNotFoundError:
    print("Error: cars.json file not found!")
    car_data = {"models": []}

car_names = [
    car["name"] for car in car_data["models"]
]  # Extract car names from the data


# Function to get car features based on the car name
def get_car_features(car_name):
    for car in car_data["models"]:
        print(f"Comparing with: {car['name']}")
        if car_name.lower() == car["name"].lower():
            return car["features"]
    return "Sorry, I couldn't find any information about that car."


# Function to get car price based on the car name
def get_car_price(car_name):
    for car in car_data["models"]:
        if car_name.lower() == car["name"].lower():
            return f"The price of {car_name} is {car['price']}"
    return "Sorry, I couldn't find the price for that car."


def get_car_speed(car_name):
    for car in car_data["models"]:
        if car_name.lower() == car["name"].lower():
            return (
                f"The price of {car_name} is {car['0-60_mph']}"
                and f"The price of {car_name} is {car['top_speed']}"
            )
    return "Sorry, I couldn't find the price for that car."


# Function to predict intent
def predict_intent(sentence):
    tokenized_sentence, recognized_cars = tokenize(sentence)
    print(f"Tokenized sentence: {tokenized_sentence}")
    print(f"Recognized cars: {recognized_cars}")
    bow = bag_of_words(tokenized_sentence, all_words)

    # Ensure the input is treated as a batch of size 1 (e.g., batch_size, input_size)
    bow = torch.from_numpy(bow).to(device).unsqueeze(0)  # Add batch dimension

    output = model(bow)

    # Check the shape of the output
    print(f"Output shape: {output.shape}")

    # Use torch.max() to get the predicted class
    _, predicted = torch.max(output, dim=1)
    print(f"Predicted index: {predicted.item()}")
    print(f"Predicted intent: {tags[predicted.item()]}")

    return tags[predicted.item()], recognized_cars


# Function to answer questions based on the recognized car
def answer_question(sentence):
    intent, recognized_cars = predict_intent(sentence)
    print(f"Predicted intent: {intent}")
    print(f"Recognized cars: {recognized_cars}")

    if intent == "get_car_features" or intent == "get_car_price":
        if recognized_cars:
            car_name = recognized_cars[0]
            print(f"Looking up features/price for: {car_name}")
            if intent == "get_car_features":
                features = get_car_features(car_name)
                print(f"Features found: {features}")
                if features:
                    return format_car_features(car_name, features)
            elif intent == "get_car_price":
                price = get_car_price(car_name)
                return price
            elif intent == "get_car_performance":
                price = get_car_speed(car_name)
                return price

        else:
            return "I couldn't find a car in your question."
    else:
        return "Sorry, I didn't understand the question."


# Function to format the car features into a readable string
def format_car_features(car_name, features):
    features_str = f"The features of {car_name} are:\n"
    for key, value in features.items():
        # Handle long lists
        if isinstance(value, list) and len(value) > 3:
            value = ", ".join(value[:3]) + "..."
        features_str += f"{key.replace('_', ' ').title()}: {value}\n"
    return features_str


# Test prediction and answering
sentence = "Tell me the cost of the 911 Carrera S."
answer = answer_question(sentence)
print(f"Answer: {answer}")
