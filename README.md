# Porsche Chatbot

An intelligent chatbot for answering questions about Porsche car models using advanced NLP techniques.

## Features

- Natural language processing for understanding user queries
- Sentiment analysis for more personalized responses
- Question classification for better understanding of user intent
- Automatic suggestion of follow-up questions
- Named Entity Recognition for better car model identification

## Setup

1. Install Python 3.7 or higher
2. Clone this repository
3. Run the setup script to install dependencies and download required NLTK data:

```bash
python setup.py
```

4. Install spaCy's English language model:

```bash
python -m spacy download en_core_web_sm
```

## Running the Application

To start the chatbot, run:

```bash
python app.py
```

## Dependencies

- torch>=1.7.0
- nltk>=3.5
- numpy>=1.19.2
- spacy>=3.0.0
- textblob>=0.15.3
- scikit-learn>=0.24.0

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Verify that NLTK data is downloaded (run `python setup.py` if not)
3. Check that the cars.json file exists in the project directory
4. Ensure you have the correct Python version installed

## Example Questions

- What car models do you offer?
- What's the power of the 911 Carrera?
- How fast is the 911 Carrera S?
- What colors are available for the 911 GT3?
- What is the price range for the 911 Turbo? 