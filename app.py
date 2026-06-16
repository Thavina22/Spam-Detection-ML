import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Spam Detection API", 
    description="A simple API to classify text messages as Spam or Ham."
)

# 2. Define the input data structure
class Message(BaseModel):
    text: str

# 3. Load your trained model and vectorizer 
# (Make sure to export these as .pkl files from your notebook!)
try:
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    print("Please ensure 'spam_model.pkl' and 'vectorizer.pkl' are in the directory.")

@app.get("/")
def home():
    return {"message": "Spam Detection API is running!"}

# 4. Create the prediction endpoint
@app.post("/predict")
def predict_spam(data: Message):
    # Transform the incoming text using your vectorizer
    transformed_text = vectorizer.transform([data.text])
    
    # Get the prediction (0 for Ham, 1 for Spam)
    prediction = model.predict(transformed_text)[0]
    
    result = "Spam" if prediction == 1 else "Ham"
    
    return {
        "text": data.text,
        "classification": result
    }
