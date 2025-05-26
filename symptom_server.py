from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
from pydantic import BaseModel
import uvicorn
import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


class SymptomClassifier:
    def __init__(self, model, vectorizer, label_encoder):
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.stemmer = SnowballStemmer('english')
    
    def preprocess_text(self, text):
        # Remove special chars, keep commas for symptom separation
        text = re.sub(r'[^a-zA-Z,\s]', '', text)
        # Replace commas with spaces (for inputs like "headache,fever")
        text = text.replace(',', ' ')
        # Tokenize and stem
        tokens = word_tokenize(text.lower())
        tokens = [self.stemmer.stem(token) for token in tokens if token.isalpha()]
        return ' '.join(tokens)

# Function to download NLTK resources with error handling
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        print("Attempting to continue with available resources...")

# Download required NLTK data
download_nltk_resources()

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SymptomsRequest(BaseModel):
    symptoms: str

# Load the ensemble model pickle file
try:
    with open('symptom_ensemble.pkl', 'rb') as f:
        classifier = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.post("/predict")
async def predict(request: SymptomsRequest):
    try:
        # Preprocess the input symptoms
        processed_text = classifier.preprocess_text(request.symptoms)
        
        # Vectorize the processed text
        features = classifier.vectorizer.transform([processed_text])
        
        # Get prediction probabilities
        probs = classifier.model.predict_proba(features)[0]
        
        # Get the class names from label encoder
        classes = classifier.label_encoder.classes_
        
        # Get top 3 predictions with confidence scores
        results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "success": True,
            "predictions": [{"disease": d, "confidence": float(c)} for d, c in results]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Ensure NLTK data is downloaded
    download_nltk_resources()
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)