import json
from utils.nltk_utils import stem
import torch
from src.model import NeuralNet
from utils.utils import bag_of_words, tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import string
import numpy as np
import os


import random
import re

stemmer = PorterStemmer()


import torch
import json


class PorscheBot:
    def __init__(self):
        try:
            # Set up device (GPU if available, else CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

            # Load intents from intents.json file
            with open("data/intents.json", "r") as f:
                self.intents = json.load(f)["intents"]

            # Load car data from cars.json
            with open("data/cars.json", "r") as f:
                self.car_data = json.load(f)

            # Load suggestions from suggestions.json
            with open("data/suggestions.json", "r") as f:
                self.suggestions = json.load(f)

            # Load model state
            data_dir = "weights/chatdata4.pth"
            try:
                # Load the saved model data with proper device mapping
                data = torch.load(data_dir, map_location=self.device)
            except RuntimeError as e:
                print(f"Error loading model: {str(e)}")
                print("Attempting to load on CPU...")
                data = torch.load(data_dir, map_location=torch.device("cpu"))

            # Extract relevant data from the saved model
            self.input_size = data["input_size"]
            self.hidden_size = data["hidden_size"]
            self.output_size = data["output_size"]
            self.all_words = data["all_words"]
            self.tags = data["tags"]
            self.model_state = data["model_state"]

            # Initialize the model with the given size and load state dict
            self.model = NeuralNet(
                self.input_size, self.hidden_size, self.output_size
            ).to(self.device)

            # Try loading model state, handle potential mismatches (e.g., embeddings)
            try:
                self.model.load_state_dict(
                    self.model_state, strict=False
                )  # Use strict=False to handle mismatches
            except Exception as e:
                print(f"Error loading model weights: {str(e)}")

            # Set model to evaluation mode
            self.model.eval()

            # Initialize NLP utilities (if needed)
            # self.nlp = AdvancedNLP()

            # Extract car names for suggestions
            self.car_names = [car["name"] for car in self.car_data["models"]]

        except Exception as e:
            print(f"Error initializing PorscheBot: {str(e)}")
            raise

    # Add any additional methods for processing user input, responding with suggestions, etc.

    def get_car_info(self, car_name):
        """
        Fetch details about the car from the car data (from 'cars.json').
        """
        try:
            # Clean up the input car name
            car_name = car_name.lower().strip()

            for model in self.car_data["models"]:
                if car_name == model["name"].lower():
                    return model

            best_match = None
            best_score = 0

            for model in self.car_data["models"]:
                model_name = model["name"].lower()
                score = 0

                model_words = set(model_name.split())
                input_words = set(car_name.split())

                matching_words = model_words.intersection(input_words)
                if len(matching_words) > 0:
                    score = len(matching_words)

                    if any(word.isdigit() for word in matching_words):
                        score += 2

                    if model_name in car_name:
                        score += 3

                    if "gts" in car_name and "gts" in model_name:
                        score += 2
                        if "carrera" in car_name and "carrera" in model_name:
                            score += 2

                    if "carrera" in car_name and "carrera" in model_name:
                        score += 1
                        if "t" in car_name and "t" in model_name:
                            score += 2

                    if "turbo" in car_name and "turbo" in model_name:
                        score += 2

                    if "gt3" in car_name and "gt3" in model_name:
                        score += 2

                    if score > best_score:
                        best_score = score
                        best_match = model

            if best_score >= 2:
                return best_match

            return None
        except Exception as e:
            print(f"Error getting car info: {str(e)}")
            return None

    # Add this inside your class

    def extract_attribute_question(self, user_message):
        print(f"[DEBUG] User Message: {user_message}")

        # A basic mapping from question to attribute and mode
        attribute_map = {
            "torque": "max_torque",
            "0-60 mph": "0-60 mph",
            "speed": "Top Speed",
            "acceleration": "0-60 mph",
            "price": "Price_Range",
            "power": "power_PS",
            "engine": "max_engine_speed",
            "output": "max_output_per_liter_PS",
            "trunk": "trunk_volume",
            "boot": "trunk_volume",
            "storage": "trunk_volume",
            "wheelbase": "wheelbase",
            "fuel": "fuel_economy",
        }

        # Mode could be "max" or "min"
        mode_map = {
            "highest": "max",
            "lowest": "min",
            "smallest": "min",
            "largest": "max",
            "biggest": "max",
            "most": "max",
            "least": "min",
            "best": "max",
            "worst": "min",
            "top": "max",
            "shortest": "min",
            "expensive": "max",
            "cheapest": "min",
            "fastest": "max",
            "slowest": "min",
            "quickest": "max",
        }

        # Default values if no match found
        attribute = None
        mode = None

        # Try to find the attribute and mode in the message
        for key, value in attribute_map.items():
            if key in user_message.lower():
                attribute = value
                break  # Stop once we find the first matching attribute

        for key, value in mode_map.items():
            if key in user_message.lower():
                mode = value
                break  # Stop once we find the first matching mode

        # Ensure that attribute and mode are found
        if attribute and mode:
            print(
                f"[DEBUG] Matched Keyword: {key} => Attribute: {attribute}, Mode: {mode}"
            )
            return attribute, mode
        else:
            print("[DEBUG] No match found for attribute or mode.")
            return None, None  # Return None for both if no match

    def get_model_by_attribute_extreme(self, attribute, mode="max"):
        models = self.car_data["models"]
        best_model = None
        best_value = None
        print(f"Models with '{attribute}':")

        for model in models:
            value = model["features"].get(attribute)
            if value:
                try:
                    numeric = float(str(value).split()[0])  # Just get the number
                    if (
                        (best_value is None)
                        or (mode == "max" and numeric > best_value)
                        or (mode == "min" and numeric < best_value)
                    ):
                        best_value = numeric
                        best_model = model
                except ValueError:
                    continue

        return best_model

    def find_model_with_attribute(self, attribute, mode="max"):
        print(f"[DEBUG] Searching for model with {mode} '{attribute}'...")
        try:
            best_value = None
            selected_model = None

            for model in self.car_data["models"]:
                value = model["features"].get(attribute)
                if value is None:
                    continue

                # Handle string values with units (e.g. "3.5 seconds", "500 Nm")
                if isinstance(value, str):
                    import re

                    numbers = re.findall(r"\d+(?:\.\d+)?", value)
                    if numbers:
                        value = float(numbers[0])
                    else:
                        continue

                # Choose the best value based on mode
                if (
                    best_value is None
                    or (mode == "max" and value > best_value)
                    or (mode == "min" and value < best_value)
                ):
                    best_value = value
                    selected_model = model

            if selected_model:
                print(
                    f"[DEBUG] Found best model: {selected_model['name']} with value {best_value}"
                )
            else:
                print("[DEBUG] No suitable model found.")

            return selected_model

        except Exception as e:
            print(f"[ERROR] Failed to find model: {e}")
            return None

    def get_dealer_info(self, dealer_name):
        try:
            dealer_name = dealer_name.lower().strip()

            if "dealerships" not in self.car_data:
                raise KeyError("Dealerships data is missing in car_data.")

            # print("Available Dealerships:", self.car_data["dealerships"])

            for dealer in self.car_data["dealerships"]:
                # print(f"Checking dealership: {dealer['name']}")

                if dealer_name == dealer["name"].lower():
                    # print(f"Match found: {dealer}")
                    return dealer

            # print(f"No match found for dealer name: {dealer_name}")

            return None

        except Exception as e:
            # print(f"Error getting dealer info: {str(e)}")
            return None

    def get_response(self, user_message, state=None):
        try:
            response = ""
            suggestions = []

            if state is None:
                state = {
                    "isFirstInteraction": True,
                    "hasProvidedName": False,
                    "hasShownMainMenu": False,
                    "selectedModel": None,
                    "userName": None,
                }

            if not state["hasProvidedName"]:
                state["hasProvidedName"] = True
                state["userName"] = user_message.strip()
                state["hasShownMainMenu"] = True
                response = f"Hi {state['userName']}! How can we help you today?"
                suggestions = [
                    "Explore our models",
                    "Find a dealership",
                    "Schedule a test drive",
                    "Porshe experience",
                    "Learn about customization options",
                    "Build your Porsche",
                ]
                return {
                    "response": response,
                    "suggestions": suggestions,
                    "state": state,
                }

            sentence = tokenize(user_message)  # Tokenize the user message
            print(f"Tokenized sentence: {sentence}")

            if isinstance(sentence, tuple):
                sentence = sentence[0]
            X = bag_of_words(sentence, self.all_words)  # Convert to BoW vector
            print(f"Vocabulary: {self.all_words}")
            print(f"Bag of words: {X}")

            X = X.reshape(1, X.shape[0])  # Reshape to 1xN for model input
            X = torch.from_numpy(X).to(
                self.device
            )  # Convert to tensor and move to the appropriate device

            # Get model output
            output = self.model(X)
            _, predicted = torch.max(output, dim=1)  # Get the predicted intent

            # Extract the predicted tag
            tag = self.tags[predicted.item()]
            print(f"Predicted tag: {tag}")
            # print(f"Predicted output: {output}")
            probs = torch.softmax(output, dim=1)  # Calculate probabilities
            prob = probs[0][
                predicted.item()
            ]  # Get the probability of the predicted intent
            print(f"Predicted probability: {prob.item()}")

            # If confidence is high enough, respond based on the intent
            if prob.item() > 0.9996:
                # Iterate through intents and find the matching one
                for intent in self.intents:
                    print(f"Checking intent: {intent['tag']}")
                    if tag == intent["tag"]:  # Match found
                        response = random.choice(
                            intent["responses"]
                        )  # Choose a random response
                        # If a model is selected, offer related suggestions

                        print(f"User Message: {user_message}")

                        print(f"response: {response}")
                        if state["selectedModel"]:
                            suggestions = [
                                f"What else would you like to know about the {state['selectedModel']}?",
                                "Back to main menu",
                            ]
                        else:
                            suggestions = [
                                "What would you like to explore next?",
                                "Back to main menu",
                            ]
                        return {
                            "response": response,
                            "suggestions": suggestions,
                            "state": state,
                        }

            # Default response if no intent matched or confidence is low
            if prob.item() <= 0.9996:
                attribute, mode = self.extract_attribute_question(user_message)

                if attribute and mode:
                    model = self.find_model_with_attribute(attribute, mode)
                    if model:
                        model_name = model["name"]
                        value = model["features"].get(attribute)
                        response = f"Ah, you’ve got an eye for quality! The {model_name} serves up the {attribute.replace('_', ' ')} with a delightful {value}. Not bad, right?"
                        suggestions = [
                            "Want to play around with more features?",
                            "Back to the main menu",
                        ]
                        return {
                            "response": response,
                            "suggestions": suggestions,
                            "state": state,
                        }

                response = "I do not understand..."
                suggestions = ["Back to main menu"]
                car_info = self.get_car_info(user_message)
                dealer_info = self.get_dealer_info(user_message)

                if car_info:
                    if (
                        "engine" in user_message.lower()
                        or "motor" in user_message.lower()
                        or "cylinder" in user_message.lower()
                        or "displacement" in user_message.lower()
                        or "bore" in user_message.lower()
                    ):
                        response = f"""
                        Meet the {car_info["name"]}'s beastly {car_info["features"]["cylinders"]}-cylinder engine! With a {car_info["features"]["displacement"]}L displacement,  
                        it roars with a {car_info["features"]["bore"]} mm bore and a {car_info["features"]["stroke"]} mm stroke, reaching a wild {car_info["features"]["max_engine_speed"]} rpm.  
                        Buckle up – it's a power trip!
                        """
                    elif (
                        "transmission" in user_message.lower()
                        or "gearbox" in user_message.lower()
                    ):
                        response = f"Shift into high gear with the {car_info['name']}'s {car_info['features']['transmission']} transmission system. Smooth rides await!"
                    elif (
                        "fuel" in user_message.lower()
                        or "economy" in user_message.lower()
                        or "mileage" in user_message.lower()
                    ):
                        response = f"""The {car_info["name"]} is efficient too! With {car_info["features"]["fuel_type"]} fuel and a fuel economy of {car_info["features"]["fuel_economy"]},  
                        you’ll be cruising without constantly visiting the gas station!"""
                    elif (
                        "speed" in user_message.lower()
                        or "accelerate" in user_message.lower()
                        or "0-60" in user_message.lower()
                        or "top" in user_message.lower()
                    ):
                        response = f"""
                        The {car_info["name"]} is all about speed – it hits a top speed of {car_info["features"]["Top Speed"]} and sprints from 0 to 60 mph in just {car_info["features"]["0-60 mph"]} seconds!  
                        Hold on tight, this ride is a blur!
                        """

                    elif "torque" in user_message.lower():
                        response = f"""
                        Ready to feel some serious push? The {car_info["name"]} delivers a thrilling max torque of {car_info["features"]["max_torque"]}.  
                        Don’t worry – your seat will hold you in place!
                        """
                    elif "wheelbase" in user_message.lower():
                        response = f"The {car_info['name']} boasts a wheelbase of {car_info['features']['wheelbase']}, ensuring a smooth and stable ride. Perfect for those winding roads!"
                    elif (
                        "Compare" in user_message.lower()
                        or "compare" in user_message.lower()
                    ):
                        response = """Comparing models is a breeze! Explore these options and more at <a href='https://www.porsche.com/middle-east/compare/?model-range=&price=any' target='_blank'>porsche Model Comparison</a>."""
                    elif (
                        "trunk" in user_message.lower()
                        or "boot" in user_message.lower()
                        or "storage" in user_message.lower()
                    ):
                        response = f"The {car_info['name']} comes with {car_info['features']['trunk_volume']} of trunk space, perfect for all your gear – or just your weekend getaway!"

                    elif "custom" in user_message.lower():
                        response = f"""The {car_info["name"]} offers a variety of customization options! From interior finishes to performance upgrades, you can make it truly yours.  
                        Explore these options and more at <a href='https://www.porsche.com/usa/modelstart/' target='_blank'>porsche configurator</a>."""

                    elif "color" in user_message.lower():
                        best_colors = car_info["features"].get("best_colors", [])
                        available_colors = car_info["features"].get(
                            "available_colors", []
                        )

                        response = f"The {car_info['name']} comes in a range of stunning colors including: <strong>{', '.join(available_colors)}</strong>."

                        if best_colors:
                            response += f"<br><br>🚗 Our top recommended colors for this model are: <strong>{', '.join(best_colors)}</strong>."
                            response += ' <br><br>Explore them all at the <a href="https://configurator.porsche.com/en-WW/model/9921B2/" target="_blank">Porsche Configurator</a>.'

                    elif (
                        "performance" in user_message.lower()
                        or "power" in user_message.lower()
                    ):
                        response = f"""
                        The {car_info["name"]} is a powerhouse! With {car_info["features"]["power_PS"]} PS ({car_info["features"]["power_kW"]} kW) of raw power and a max torque of {car_info["features"]["max_torque"]},  
                        it goes from 0 to 60 mph in just {car_info["features"]["0-60 mph"]} seconds. Oh, and its top speed? {car_info["features"]["Top Speed"]}. Talk about performance!
                        """
                    elif (
                        "price" in user_message.lower()
                        or "cost" in user_message.lower()
                    ):
                        response = f"The {car_info['name']} is priced between {car_info['features']['Price_Range']}. Worth every penny for the ride of your life!"
                    elif (
                        "output" in user_message.lower()
                        or "horsepower" in user_message.lower()
                        or "power" in user_message.lower()
                    ):
                        response = f"The {car_info['name']} cranks out {car_info['features']['power_kW']} kW of pure power. Buckle up — it's not just a ride, it’s an adventure!"

                    elif (
                        "specs" in user_message.lower()
                        or "features" in user_message.lower()
                        or "specification" in user_message.lower()
                    ):
                        response = f"""
                        The {car_info["name"]} is a powerhouse! With {car_info["features"]["power_PS"]} PS ({car_info["features"]["power_kW"]} kW) of raw power and a max torque of {car_info["features"]["max_torque"]},  
                        it goes from 0 to 60 mph in just {car_info["features"]["0-60 mph"]} seconds. Oh, and its top speed? {car_info["features"]["Top Speed"]}. Talk about performance!
                        """
                        response += '<br><br>you can explore more about {car_info["name"]} the features of the at <a href={car_info["features"]["tech_specs"]} target="_blank">Porsche Technical Details</a>'

                    else:
                        # Default comprehensive feature overview
                        response = f"""
                        {car_info["features"]["tagline"]} 
                        Whether you're after speed, power, or just a good-looking machine, it's got it all!  
                        Get more details of {car_info["name"]} at <a href={car_info["features"]["tech_specs"]} target="_blank">Technical Details</a>."""

                    suggestions = [
                        f"What's the top speed of the {car_info['name']}?",
                        f"Tell me about {car_info['name']}'s performance.",
                        f"Can I customize the {car_info['name']}?",
                        "Back to the models list",
                        "Back to the main menu",
                    ]

                    return {
                        "response": response,
                        "suggestions": suggestions,
                        "state": state,
                    }

                if state["hasShownMainMenu"] and not state["selectedModel"]:
                    if "models" in user_message.lower():
                        model_names = [
                            model["name"] for model in self.car_data["models"]
                        ]
                        response = "Here are our available models. Please select the model you are interested in:"
                        suggestions = model_names + ["Back to main menu"]
                        return {
                            "response": response,
                            "suggestions": suggestions,
                            "state": state,
                        }

                    if "dealership" in user_message.lower():
                        dealer_names = [
                            dealer["name"] for dealer in self.car_data["dealerships"]
                        ]
                        response = "Here are our available locations. Please select the model you are interested in:"
                        suggestions = dealer_names + ["Back to main menu"]
                        return {
                            "response": response,
                            "suggestions": suggestions,
                            "state": state,
                        }

                    elif "build" in user_message.lower():
                        response = 'You can Build your a own porsche at  <a href="https://www.porsche.com/middle-east/modelstart/" target="_blank">Build your own porsche</a>'
                        suggestions = ["Back to main menu", "Find a dealership"]
                        return {
                            "response": response,
                            "suggestions": suggestions,
                            "state": state,
                        }

                    elif "porshe experience" in user_message.lower():
                        response = 'You can schedule experience more about porsche at <a href="https://www.porsche.com/middle-east/motorsportandevents/experience/" target="_blank">Porsche Experience</a>'
                        suggestions = ["Back to main menu", "Find a dealership"]
                        return {
                            "response": response,
                            "suggestions": suggestions,
                            "state": state,
                        }

                    elif "back to main menu" in user_message.lower():
                        response = f"Hello, {state['userName']}! How can we help you today, {state['userName']}?"
                        suggestions = [
                            "Explore our models",
                            "Find a dealership",
                            "Schedule a test drive",
                            "Porshe experience",
                            "Learn about customization options",
                            "Build your Porsche",
                        ]
                        return {
                            "response": response,
                            "suggestions": suggestions,
                            "state": state,
                        }

                if dealer_info:
                    print(f"Dealer info: {dealer_info}")
                    print(f"User message: {user_message}")
                    print(f"Dealer name: {dealer_info['name']}")

                    response = response = f"""
                        <strong>{dealer_info["name"]}</strong><br><br>
                        📍 <strong>Location:</strong> {dealer_info["location"]}<br>
                        📞 <strong>Contact:</strong> {dealer_info["contact_number"]}<br>
                        📬 <strong>Address:</strong> {dealer_info["address"]}<br>
                        🕒 <strong>Hours:</strong> {dealer_info["hours"]}<br><br>

                        🚗 <strong>Want to experience it yourself?</strong><br>
                        You can schedule a test drive here: 
                        <a href='https://dealer.porsche.com/in/india/en-GB/Porsche-India-Lead?q=TestDrive&utm_content=Google_Sitelink&utm_source=google&utm_medium=cpc&utm_campaign=AGL-Search-Cayenne-Brand-Mumbai&utm_campaign_id=22359932575&utm_adgroupid=176982114655&utm_keyword=porsche%20price%20in%20india&utm_device=c&utm_placement=&utm_network=g&utm_creativeid=743342308529&utm_matchtype=b&gad_source=1&gclid=Cj0KCQjw2N2_BhCAARIsAK4pEkXqFHqvZBRiopQ_UxkmBaI2ErceF8-E2LGSK2Dj0WgV4FW_3h7OCp4aAhvqEALw_wcB' target='_blank'>Porsche Test Drive</a>
                        """

                    suggestions = [
                        "Back to the models list",
                        "Back to the main menu",
                    ]

                    return {
                        "response": response,
                        "suggestions": suggestions,
                        "state": state,
                    }

                if state["hasShownMainMenu"]:
                    car_info = self.get_car_info(user_message)

                    if car_info:
                        state["selectedModel"] = car_info["name"]
                        response = f"You've selected the {state['selectedModel']}. What would you like to know about it?"
                        suggestions = [
                            f"What's the price of the {state['selectedModel']}?",
                            f"Tell me about the {state['selectedModel']}'s performance",
                            f"What customization options are available for the {state['selectedModel']}?",
                            "Back to models list",
                            "Back to main menu",
                        ]
                        return {
                            "response": response,
                            "suggestions": suggestions,
                            "state": state,
                        }

                    if "back to models list" in user_message.lower():
                        state["selectedModel"] = None
                        response = "Here are our available models. Please select one to learn more:"
                        suggestions = [
                            model["name"] for model in self.car_data["models"]
                        ] + ["Back to main menu"]
                        return {
                            "response": response,
                            "suggestions": suggestions,
                            "state": state,
                        }

                if state["selectedModel"]:
                    car_info = self.get_car_info(state["selectedModel"])
                    if car_info:
                        if "back to main menu" in user_message.lower():
                            state["selectedModel"] = None
                            response = f" hello, {state['userName']}! How can we help you today, {state['userName']}?"
                            suggestions = [
                                "Explore our models",
                                "Find a dealership",
                                "Schedule a test drive",
                                "Porshe experience",
                                "Learn about customization options",
                                "Build your Porsche",
                            ]
                            return {
                                "response": response,
                                "suggestions": suggestions,
                                "state": state,
                            }

                        else:
                            response = f"The {state['selectedModel']} is a high-performance sports car with {car_info['features']['power_PS']} PS of power and a top speed of {car_info['features']['Top Speed']}."

                        return {
                            "response": response,
                            "suggestions": suggestions,
                            "state": state,
                        }

                response = f"Hey {state['userName']}! 🚗💨 Ready to dive into the world of Porsche? I’ve got the lowdown on our models, pricing, and features. Just let me know what sparks your interest! Or, if you're in the mood for more, check us out at <a href='https://www.porsche.com' target='_blank'>porsche.com</a>."

                suggestions = [
                    "Back to main menu",
                ]

                return {
                    "response": response,
                    "suggestions": suggestions,
                    "state": state,
                }

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again or ask a different question.",
                "suggestions": [],
                "state": state,
            }
