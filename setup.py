import nltk
import subprocess
import sys

def download_nltk_data():
    """Download required NLTK data"""
    required_data = [
        'punkt',  # For tokenization
        'punkt_tab',  # Additional tokenizer data
        'averaged_perceptron_tagger',  # For POS tagging
        'wordnet',  # For lemmatization
        'stopwords',  # For stop words
        'maxent_ne_chunker',  # For named entity recognition
        'words',  # For word corpus
        'treebank'  # For training data
    ]
    
    print("Downloading required NLTK data...")
    for data in required_data:
        try:
            nltk.download(data)
            print(f"Successfully downloaded {data}")
        except Exception as e:
            print(f"Error downloading {data}: {str(e)}")

def install_dependencies():
    """Install required Python packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully installed packages from requirements.txt")
    except subprocess.CalledProcessError as e:
        print(f"Error installing all packages: {str(e)}")

if __name__ == "__main__":
    download_nltk_data()
    install_dependencies()
    print("Setup completed successfully!") 
