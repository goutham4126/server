from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
from pydantic import BaseModel
import uvicorn
import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
import warnings
# Suppress warnings
warnings.filterwarnings('ignore')

# Must define the same class as used during training
class SymptomClassifier:
    def __init__(self, model, vectorizer, label_encoder):
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.stemmer = SnowballStemmer('english')
    
    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z,\s]', '', text)
        text = text.replace(',', ' ')
        tokens = word_tokenize(text.lower())
        tokens = [self.stemmer.stem(token) for token in tokens if token.isalpha()]
        return ' '.join(tokens)

def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")

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

# Load the model
try:
    with open('symptom_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.post("/predict")
async def predict(request: SymptomsRequest):
    try:
        # Preprocess and predict
        processed = classifier.preprocess_text(request.symptoms)
        vector = classifier.vectorizer.transform([processed])
        probs = classifier.model.predict_proba(vector)[0]
        classes = classifier.label_encoder.classes_
        
        # Format results
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
    # Ensure NLTK data is available
    download_nltk_resources()
    
    # Start server - explicitly bind to port 8000
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False
    )