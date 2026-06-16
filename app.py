import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="Spam Detection API")

# Define what the incoming request data should look like
class MessageInput(BaseModel):
    text: str

# Load the saved model and vectorizer we generated in Step 1
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.get("/")
def read_root():
    return {"status": "API is online and ready!"}

@app.post("/predict")
@app.post("/predict")
def predict_spam(input_data: MessageInput):
    # 1. Convert incoming text to features
    transformed_text = vectorizer.transform([input_data.text])
    
    # 2. Get numerical prediction
    prediction = model.predict(transformed_text)[0]
    
    # 3. Map numerical prediction to human-readable text
    # (If your dataset mapped 1 to spam, keep it like this. If it mapped 'spam' directly, check your data format)
    if str(prediction) == "1" or str(prediction).lower() == "spam":
        result = "Spam"
    else:
        result = "Ham"
    
    # 4. Return clean JSON
    return {
        "input_message": input_data.text,
        "prediction": result
    }
