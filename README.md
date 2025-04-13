# cc-fraud-detection

This project uses the **Kaggle credit card transactions dataset** to develop a machine learning model for detecting whether a given transaction is fraudulent (Fraud). The trained model is then deployed via a **real-time** RESTful API using FastAPI.

- **Key Features**  
  - Data preprocessing and exploratory analysis of credit card transactions  
  - Training a **LightGBM** model for fraud detection  
  - FastAPI-based RESTful service that loads the trained model for real-time predictions  

> **Data Source**  
> - [Kaggle - Credit Card Transactions](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions)

---

## How to Use

1. **Download the Dataset**  
   - Download the dataset from Kaggle and place it in the designated `data/` directory (or as specified in your project config).

2. **Data Preprocessing & Feature Engineering (`preprocess.py`)**  
   - Run:
     ```bash
     python preprocess.py
     ```
   - This script will clean and transform the raw dataset, saving the processed data.

3. **Model Training & Prediction (`fds.py`)**  
   - Run:
     ```bash
     python fds.py
     ```
   - This trains the LightGBM model and saves the trained model file to the specified location.

4. **Starting the API Server (`app.py`)**  
   - Launch the FastAPI server to provide real-time predictions:
     ```bash
     python app.py
     ```
   - The server will load the trained model and expose endpoints for fraud detection requests.

5. **API Testing (`test_request.py`)**  
   - Run:
     ```bash
     python test_request.py
     ```
   - This script periodically sends sample requests to the API, simulating real-world usage and verifying the server’s response.
  