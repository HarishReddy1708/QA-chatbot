import json
from nltk_utils import stem
import torch
from model import NeuralNet
from utils import bag_of_words, tokenize
from advanced_nlp import AdvancedNLP
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import string
import numpy as np
import os


import random
import re
import logging

logging.basicConfig(level=logging.DEBUG)

stemmer = PorterStemmer()


class PorscheBot:
    def __init__(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

            # Load intents
            with open("intents.json", "r") as f:
                self.intents = json.load(f)["intents"]

            # Load car data
            with open("cars.json", "r") as f:
                self.car_data = json.load(f)

            with open("suggestions.json", "r") as f:
                self.suggestions = json.load(f)

            # Load pre-trained model data with proper device mapping
            data_dir = "chatdata2.pth"
            try:
                data = torch.load(data_dir, map_location=self.device)
            except RuntimeError as e:
                print(f"Error loading model: {str(e)}")
                print("Attempting to load on CPU...")
                data = torch.load(data_dir, map_location=torch.device("cpu"))

            self.input_size = data["input_size"]
            self.hidden_size = data["hidden_size"]
            self.output_size = data["output_size"]
            self.all_words = data["all_words"]
            self.tags = data["tags"]
            self.model_state = data["model_state"]

            # Initialize the model
            self.model = NeuralNet(
                self.input_size, self.hidden_size, self.output_size
            ).to(self.device)
            self.model.load_state_dict(self.model_state)
            self.model.eval()

            self.nlp = AdvancedNLP()
            self.car_names = [car["name"] for car in self.car_data["models"]]

        except Exception as e:
            print(f"Error initializing PorscheBot: {str(e)}")
            raise

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

    def get_main_menu_suggestions(self):
        return [
            "Explore our models",
            "Find a dealership",
            "Schedule a test drive",
            "Porsche experience",
            "Explore customization options",
            "Build your Porsche",
        ]

    def get_exploration_suggestions(self):
        return random.sample(
            [
                "What would you like to explore next?",
                "Discover the latest models",
                "Back to main menu",
            ],
            len([
                "What would you like to explore next?",
                "Tell me about Porsche performance",
                "Discover the latest models",
                "Back to main menu",
            ]),
        )

    def get_car_related_suggestions(self, car_info):
        return random.sample(
            [
                f"Learn more about the {car_info['name']}'s performance",
                f"What customization options are available for the {car_info['name']}?",
                f"Check out the price range for the {car_info['name']}",
                f"What’s the top speed of the {car_info['name']}?",
                "Back to models list",
                "Back to main menu",
            ],
            len([
                f"Learn more about the {car_info['name']}'s performance",
                f"What customization options are available for the {car_info['name']}?",
                f"Check out the price range for the {car_info['name']}",
                f"What’s the top speed of the {car_info['name']}?",
                "Back to models list",
                "Back to main menu",
            ]),
        )

    def get_build_related_suggestions(self):
        return random.sample(
            [
                "Build your Porsche",
                "Learn more about customization options",
                "Back to main menu",
                "Find a dealership",
            ],
            len([
                "Build your Porsche",
                "Learn more about customization options",
                "Back to main menu",
                "Find a dealership",
            ]),
        )

    def get_experience_related_suggestions(self):
        return random.sample(
            [
                "Schedule a test drive",
                "Learn more about Porsche Experience",
                "Back to main menu",
                "Find a dealership",
            ],
            len([
                "Schedule a test drive",
                "Learn more about Porsche Experience",
                "Back to main menu",
                "Find a dealership",
            ]),
        )

    def get_selected_model_suggestions(self, state):
        return random.sample(
            [
                f"What else would you like to know about the {state['selectedModel']}?",
                f"Ask about {state['selectedModel']}'s performance",
                f"What customization options are available for the {state['selectedModel']}?",
                "Back to models list",
                "Back to main menu",
            ],
            len([
                f"What else would you like to know about the {state['selectedModel']}?",
                f"Ask about {state['selectedModel']}'s performance",
                f"What customization options are available for the {state['selectedModel']}?",
                "Back to models list",
                "Back to main menu",
            ]),
        )

    def get_dealer_info(self, dealer_name):
        try:
            dealer_name = dealer_name.lower().strip()

            if "dealerships" not in self.car_data:
                raise KeyError("Dealerships data is missing in car_data.")

            # print("Available Dealerships:", self.car_data["dealerships"])

            for dealer in self.car_data["dealerships"]:
                # print(f"Checking dealership: {dealer['name']}")

                if dealer_name == dealer["name"].lower():
                    print(f"Match found: {dealer}")
                    return dealer

            # print(f"No match found for dealer name: {dealer_name}")

            return None

        except Exception as e:
            print(f"Error getting dealer info: {str(e)}")
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
                response = f"Hey {state['userName']}! 👋 I'm here to help you with all things Porsche! 😎 What can I assist you with today? 🚗💨"
                suggestions = self.get_main_menu_suggestions()
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
            if prob.item() > 0.9998:
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

            # Handle car model selection
            if state["hasShownMainMenu"] and not state["selectedModel"]:
                logging.debug(
                    "First block executed: Main menu shown, no model selected."
                )
                model_names = [
                    model["name"].lower() for model in self.car_data["models"]
                ]  # Convert model names to lowercase

                # Check if any model name matches the user's input
                for model_name in model_names:
                    if (
                        model_name in user_message.lower()
                    ):  # Match user message with model names
                        # Update the selected model
                        state["selectedModel"] = model_name
                        response = f"You've selected the {model_name}. What would you like to know about it?"
                        suggestions = self.get_selected_model_suggestions(state)
                        logging.debug(f"Model matched: {model_name}")
                        logging.debug(f"Response: {response}")
                        return {
                            "response": response,
                            "suggestions": suggestions,
                            "state": state,
                        }

                # If no model is selected, provide model suggestions
                response = "Exciting times ahead! 🚗💨 Here are our available Porsche models. Take your pick—each one is packed with performance, style, and luxury. 😍 Which one are you most interested in?"
                suggestions = random.sample(model_names, len(model_names)) + [
                    "Back to main menu"
                ]
                return {
                    "response": response,
                    "suggestions": suggestions,
                    "state": state,
                }

            if state["hasShownMainMenu"] and state["selectedModel"]:
                model_name = state["selectedModel"]
                logging.debug(f"Selected model: {model_name}")

                car_info = self.get_car_info(model_name)
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
                        logging.debug(f"Response: {response}")

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
                        response = f"""The {car_info["name"]} comes in {car_info["features"]["available_colors"]} eye-catching colors: {car_info["features"]["available_colors"]}.  
                        Can't decide? You can explore even more color options at the <a href="https://configurator.porsche.com/en-WW/model/9921B2/" target="_blank">Porsche Configurator</a>."""
                    elif "color" in user_message.lower():
                        response = f"""Color your world with the {car_info["name"]} – available in {car_info["features"]["available_colors"]} stunning shades!  
                        Get the perfect match for your vibe at the <a href="https://configurator.porsche.com/en-WW/model/9921B2/" target="_blank">Porsche Configurator</a>."""
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

                    elif "back to models list" in user_message.lower():
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

                    elif "back to main menu" in user_message.lower():
                        state["selectedModel"] = None
                        response = f"How can we help you today, {state['userName']}?"
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
                    # response = f"The {model_name} is a high-performance sports car with {car_info['features']['power_PS']} PS of power and a top speed of {car_info['features']['Top Speed']}."
                    logging.debug(f"Response: {response}")
                    return {
                        "response": response,
                        "suggestions": self.get_selected_model_suggestions(state),
                        "state": state,
                    }

                if "back to models list" in user_message.lower():
                    state["selectedModel"] = None
                    response = "Here are our available models. Please select one to learn more:"
                    suggestions = random.sample(
                        [model["name"] for model in self.car_data["models"]],
                        len(self.car_data["models"]),
                    ) + ["Back to main menu"]
                    return {
                        "response": response,
                        "suggestions": suggestions,
                        "state": state,
                    }

                logging.debug("No model selected, providing default response.")

                response = f"I'm here to help you explore our Porsche lineup. Would you like to know about our models, pricing, or features? Visit <a href='https://www.porsche.com' target='_blank'>porsche.com</a>"
                suggestions = [
                    "Back to main menu",
                ]

                return {
                    "response": response,
                    "suggestions": suggestions,
                    "state": state,
                }

            # More code for further handling...

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {
                "response": "Oops! Something went wrong. 😬 Please try again or ask a different question. 🙏",
                "suggestions": [],
                "state": state,
            }
