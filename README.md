# 🛡️ PRODUCTION-GRADE SPAM DETECTION MICROSERVICE

An end-to-end Machine Learning microservice that classifies text payloads into **Spam** or **Ham** (Legitimate). This project showcases the complete productization lifecycle (MLOps), transitioning an experimental pipeline out of a Jupyter Notebook into a live, cloud-hosted API using **FastAPI** and **Render**.

---

### 🚀 LIVE INTERACTIVE DEMO (SWAGGER UI)
Test real-time predictions directly inside your web browser here:  
🔗 **[https://spam-detection-ml.onrender.com/docs](https://spam-detection-ml.onrender.com/docs)**

---

### 🛠️ CORE ENGINEERING SPECIFICATIONS

* **ML Framework:** Python, Scikit-Learn, Pandas, Joblib
* **Algorithmic Model:** CountVectorizer (Feature Extraction) + Multinomial Naive Bayes Classifier
* **API Delivery Layer:** FastAPI, Uvicorn ASGI High-Performance Server
* **Cloud Architecture:** Render Infrastructure with Automated Continuous Deployment via GitHub

---

Navigate to http://127.0.0.1:8000/docs to run the local API deployment.

💡 Engineered to demonstrate modern MLOps architectures.  

### 🔌 RESTful API SCHEMAS

#### `POST /predict`
Consumes an inbound JSON text string and triggers an evaluation using the serialized model weights.

**Example Request Payload (JSON):**
```json
{
  "text": "Congratulations! You've won a free $1000 Walmart gift card. Click here to claim now."
}
