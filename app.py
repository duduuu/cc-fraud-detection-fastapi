# app.py
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import uvicorn

from typing import List, Optional, Annotated, Literal
from contextlib import asynccontextmanager

import polars as pl
import joblib
import time
import logging
import warnings

from inference import predict_fraud
from config import CFG

warnings.filterwarnings('ignore', category=UserWarning)                        

# 로깅 및 모니터링
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CFG.fastapi_log_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 입력 데이터 검증
class TransactionData(BaseModel):
    User: Annotated[int, Field(ge=0)]
    Card: Annotated[int, Field(ge=0)]
    Year: Annotated[int, Field(gt=1990, le=2030)]
    Month: Annotated[int, Field(ge=1, le=12)]
    Day: Annotated[int, Field(ge=1, le=31)]
    Time: Annotated[str, Field(
        pattern=r'^\d{2}:\d{2}$', 
        min_length=5, 
        max_length=5
    )]
    Amount: Annotated[str, Field(pattern=r'^\$-?\d+(\.\d+)?$')]
    Use_Chip: Annotated[
        Literal["Swipe Transaction", "Online Transaction", "Chip Transaction"],
        Field(..., alias="Use Chip")
    ]
    Merchant_Name: Annotated[int,Field(..., alias="Merchant Name")]
    Merchant_City: Annotated[str, Field(..., alias="Merchant City")]
    Merchant_State: Annotated[Optional[str], Field(..., alias="Merchant State")]
    MCC: Annotated[int, Field(ge=1000, le=9999)]
    Zip: Annotated[Optional[float], Field(le=99999, multiple_of=1)]
    Errors: Annotated[Optional[str], Field(..., alias="Errors?")]

dframes = {}
ml_models = {}
encoders = {}

# 모델 로딩 및 초기화
@asynccontextmanager
async def lifespan(app: FastAPI):
    dframes["trans"] = pl.read_parquet(CFG.online_trans_df_path)
    dframes["user"] = pl.read_parquet(CFG.online_user_df_path)
    dframes["card"] = pl.read_parquet(CFG.online_card_df_path)
    dframes["merchant"] = pl.read_parquet(CFG.online_merchant_df_path)
    dframes["mcc"] = pl.read_parquet(CFG.online_mcc_df_path)
    
    encoders["ordinal"] = joblib.load(CFG.online_ordinal_encode_path)

    ml_models["lgbm"] = joblib.load(CFG.online_model_path)
    
    yield
    
    dframes.clear()
    encoders.clear()
    ml_models.clear()
    

app = FastAPI(
    title="Credit Card Transaction Fraud Detection Restful API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "ok"}


# /predict 엔드포인트: 예측 요청 처리
@app.post("/predict")
def predict_transaction(data: TransactionData):
    logging.info(f"Request data: {data}")
    
    pred = predict_fraud(data, dframes, ml_models, encoders)
    is_fraud = (pred >= CFG.threshold)
    
    # 로깅 및 모니터링
    logging.info(f"Prediction result: {is_fraud}")
    
    return {"fraud_probability": f"{pred:.6f}", "Is Fraud?": str(is_fraud)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)