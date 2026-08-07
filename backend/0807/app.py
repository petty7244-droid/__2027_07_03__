# ------------------------------------
# 載入套件與環境設定
# ------------------------------------
import os
import sys
from train_save import train_and_save_model          # 訓練函式（負責訓練並序列化模型）
from pydantic import BaseModel,Field                 # 用於 API 請求/回應的資料模型與欄位驗證
from pprint import pprint                            # 美化輸出 dict 結構（僅供除錯用）
import joblib                                        # 用於載入/儲存序列化的模型檔案 (.joblib)
from fastapi import FastAPI,HTTPException            # Web API 框架與例外處理
import uvicorn                                       # ASGI 伺服器，用於啟動 FastAPI
import pandas as pd
import numpy as np


# 記錄目前的工作目錄，並把它加入 sys.path，確保可以 import 同資料夾下的 train_save 模組
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 模型檔案的絕對路徑（與工作目錄同層）
model_path = os.path.join(current_dir, "salary_model.joblib")
# 全域變數：存放目前服務使用中的模型與預處理器，方便各端點直接取用
MODEL_STATE = {}


# ------------------------------------
# 定義 Pydantic 資料模型
# ------------------------------------
class TrainConfig(BaseModel):
    """訓練請求的參數模型：由使用者透過 POST /train 傳入。"""
    test_size: float = Field(0.2, description="測試集分割比例", ge=0.1 , le=0.5)          # 測試集佔整體資料比例（限制 0.1~0.5）
    random_state: int = Field(76, description="隨機種子", ge=0)                            # 隨機種子，固定後結果可重現
    model_type: str = Field("LinearRegression", description="模型演算法類型 (LinearRegression, Lasso, Ridge)")  # 選擇演算法
    alpha: float = Field(1.0, description="正則化強度 alpha (適用於 Lasso 與 Ridge)", ge= 0.001, le=100.0)      # 正則化強度

class TrainResult(BaseModel):
    """訓練完成後的回應資料模型：描述訓練結果的各項指標。"""
    status: str = Field(..., description="執行結果狀態")                # 例如 "success"
    r2: float = Field(..., description="測試集 R-squared 決定係數")     # 模型解釋力指標
    coef: list[float] = Field(..., description="特徵權重係數列表")      # 每個特徵的斜率
    intercept: float = Field(..., description="截距")                   # 迴歸線的截距 b
    feature_coefs: dict[str, float] = Field(..., description="特徵及其權重映射")  # 特徵名稱 -> 係數
    model_type: str = Field(..., description="模型演算法類型")          # LinearRegression / Lasso / Ridge
    alpha: float = Field(..., description="正則化強度 alpha")
    train_time: float = Field(..., description="訓練耗時 (秒)")
    message:str = Field(..., description="提示訊息")                    # 對使用者的成功訊息

class SalaryInput(BaseModel):
    """預測請求的資料模型：由使用者透過 POST /predict 傳入。"""
    years_experience: float = Field(..., ge=0.0, le=50.0)   # 工作年資（限制 0~50 年）
    education_level:str                                    # 學歷（高中以下 / 大學 / 碩士以上）
    city: str                                              # 城市（城市A / 城市B / 城市C）

class SalaryOutput(BaseModel):
    """預測結果的回應資料模型。"""
    predicted_salary: float             # 預測月薪
    estimated_annual_salary: float      # 預估年薪（月薪 × 14）
    

def load_model_state():
    """從 joblib 檔案讀取最新模型與預處理器，並同步更新全域變數 MODEL_STATE。"""
    global MODEL_STATE
    # 若模型檔案尚未產生，先自動執行一次訓練產生檔案
    if not os.path.exists(model_path):
        train_and_save_model()

    # 載入 joblib 內封裝的完整字典（含 model、編碼器、標準化器與元數據）
    model_data = joblib.load(model_path)
    # 先清空全域變數，確保不使用舊模型的殘留資料
    MODEL_STATE.clear()
    MODEL_STATE.update(
        {
            "model": model_data["model"],
            "oe": model_data["oe"],
            "ohe": model_data["ohe"],
            "scaler": model_data["scaler"],
            "r2": model_data.get("r2"),
            "feature_names": model_data["feature_names"],
            "feature_coefs": model_data.get("feature_coefs",{}),
            "model_type": model_data.get("model_type"),
            "alpha": model_data.get("alpha")
        }
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
         # 2. 線上重新載入最新模型狀態至全域變數（這樣後續 /predict 就會用到新模型）
        load_model_state()
    except Exception as e:
        # 訓練過程有任何例外，回傳 HTTP 500 給前端
        raise HTTPException(status_code=500, detail=f"線上訓練失敗: {str(e)}")

    return res

@app.post("/predict", response_model=SalaryOutput)
def predict_endpoint(payload:SalaryInput):
    # 從全域狀態取出目前服務使用的預處理器與模型
    oe = MODEL_STATE["oe"]       # OrdinalEncoder：學歷文字 -> 數值
    ohe = MODEL_STATE["ohe"]     # OneHotEncoder：城市文字 -> 0/1 向量
    scaler = MODEL_STATE["scaler"]  # StandardScaler：標準化特徵
    model = MODEL_STATE["model"]    # 已訓練好的迴歸模型

    # 1. 學歷編碼：將「高中以下/大學/碩士以上」轉成 0/1/2
    edu_encoded = int(oe.transform(pd.DataFrame([[payload.education_level]], columns=["EducationLevel"]))[0][0])
    # 2. 城市編碼：將城市轉成 OneHot 向量（城市A -> [1,0,0] 等）
    city_vector = ohe.transform(pd.DataFrame([[payload.city]], columns=["City"]))
    city_cols = ohe.get_feature_names_out(['City'])
    # 3. 組合成與訓練時相同欄位順序的特徵列（工作年資、學歷編碼、城市 OneHot）
    feature_row = [payload.years_experience, edu_encoded] + list(city_vector[0])
    features = pd.DataFrame([feature_row],columns=["YearsExperience", "EducationLevel"] + list(city_cols))
    # 4. 使用訓練時的 scaler 標準化後送入模型預測
    X_scaled = scaler.transform(features)
    predicted_salary = float(model.predict(X_scaled)[0])
    # 5. 回傳月薪與年薪（月薪 × 14 個月）
    return SalaryOutput(
        predicted_salary=predicted_salary,
        estimated_annual_salary= predicted_salary * 14
    )
    
if __name__ == "__main__":
    # 以 reload 模式啟動開發伺服器，修改程式碼後會自動重載
    uvicorn.run("app:app", reload=True)