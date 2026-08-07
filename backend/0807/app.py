import os
import sys
from train_save import train_and_save_model
from pydantic import BaseModel,Field
from pprint import pprint
import joblib
from fastapi import FastAPI,HTTPException
import uvicorn


current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

model_path = os.path.join(current_dir, "salary_model.joblib")
MODEL_STATE = {}


class TrainConfig(BaseModel):
    alpha: float = Field(1.0, description="正則化強度 alpha (適用於 Lasso 與 Ridge)", ge= 0.001, le=100.0)

class TrainResult(BaseModel):
    message:str = Field(..., description="提示訊息")

def load_model_state():
    )

load_model_state()

app = FastAPI()
@app.post("/train", response_model=TrainResult)
def train_endpoint(config:TrainConfig):
    """
    訓練端點：傳入測試集比例、隨機種子、模型類型與 alpha，線上重新訓練模型，並即時更新服務所使用的模型。
    """
    try:
        # 1. 執行重新訓練並儲存模型
        res = train_and_save_model(
            test_size=config.test_size,
            random_state= config.random_state,
            model_type= config.model_type,
            alpha=config.alpha
        )
         # 2. 線上重新載入最新模型狀態至全域變數
        load_model_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"線上訓練失敗: {str(e)}")

    return res
    
if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)