from flask import Flask, render_template, request, jsonify, session
import time
from main import PorscheBot
import os
import traceback
import logging

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = "your-secret-key-here"

# Set up logging for better error tracking
logging.basicConfig(level=logging.DEBUG)

# Initialize the bot
try:
    chatbot = PorscheBot()
except Exception as e:
    logging.error(f"Error initializing chatbot: {str(e)}")
    logging.error(traceback.format_exc())
    chatbot = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        if not chatbot:
            return jsonify({
                "response": "I apologize, but the chatbot is currently unavailable. Please try again later.",
                "suggestions": [],
                "state": None,
            })

        # Retrieve user message and state from the request
        user_message = request.json.get("message", "").strip()
        state = request.json.get("state", None)

        # If state is None and user has previously provided a name, populate the state with the session name
        if state is None:
            state = {"userName": session.get("user_name"), "hasProvidedName": False}

        # Print debug logs
        logging.debug(f"Received message: {user_message}")
        logging.debug(f"Received state: {state}")

        if not user_message:
            return jsonify({
                "response": "I didn't receive your message. Please try again.",
                "suggestions": [],
                "state": state,
            })

        # Get response from the chatbot
        result = chatbot.get_response(user_message, state)

        logging.debug(f"Chatbot response state: {result.get('state')}")  # Debug log

        # Ensure the response has the correct structure
        if not isinstance(result, dict):
            result = {"response": str(result), "suggestions": [], "state": state}

        # Ensure 'suggestions' and 'state' exist in the response
        result.setdefault("suggestions", [])
        result.setdefault("state", state)

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            "response": "I apologize, but I encountered an error processing your request. Please try again.",
            "suggestions": [],
            "state": state,
        })


@app.route("/set_name", methods=["POST"])
def set_name():
    name = request.json.get("name", "")
    if name:
        session["user_name"] = name
        return jsonify({"success": True})
    return jsonify({"success": False})


if __name__ == "__main__":
    app.run(debug=True)
