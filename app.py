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
def predict_spam(input_data: MessageInput):
    # 1. Convert the incoming string text into numerical features using the saved vectorizer
    transformed_text = vectorizer.transform([input_data.text])
    
    # 2. Run the transformed features through the saved Naive Bayes model
    prediction = model.predict(transformed_text)[0]
    
    # 3. Return the result as JSON
    return {
        "input_message": input_data.text,
        "prediction": str(prediction)  # Will return your label (e.g., 'spam' or 'ham')
    }
