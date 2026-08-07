# ------------------------------------
# 載入套件與環境設定
# ------------------------------------
import os
import sys
import gradio as gr                             # 圖形化介面（藍色主題）
from train_save import train_and_save_model     # 訓練函式（負責訓練並序列化模型）
from pydantic import BaseModel, Field           # 用於 API 請求/回應的資料模型與欄位驗證
import joblib                                   # 用於載入/儲存序列化的模型檔案 (.joblib)
from fastapi import FastAPI, HTTPException      # Web API 框架與例外處理
import uvicorn                                  # ASGI 伺服器，用於啟動 FastAPI
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
    test_size: float = Field(0.2, description="測試集分割比例", ge=0.1, le=0.5)
    random_state: int = Field(76, description="隨機種子", ge=0)
    model_type: str = Field("LinearRegression", description="模型演算法類型 (LinearRegression, Lasso, Ridge)")
    alpha: float = Field(1.0, description="正則化強度 alpha (適用於 Lasso 與 Ridge)", ge=0.001, le=100.0)


class TrainResult(BaseModel):
    """訓練完成後的回應資料模型：描述訓練結果的各項指標。"""
    status: str = Field(..., description="執行結果狀態")
    r2: float = Field(..., description="測試集 R-squared 決定係數")
    coef: list[float] = Field(..., description="特徵權重係數列表")
    intercept: float = Field(..., description="截距")
    feature_coefs: dict[str, float] = Field(..., description="特徵及其權重映射")
    model_type: str = Field(..., description="模型演算法類型")
    alpha: float = Field(..., description="正則化強度 alpha")
    train_time: float = Field(..., description="訓練耗時 (秒)")
    message: str = Field(..., description="提示訊息")


class SalaryInput(BaseModel):
    """預測請求的資料模型：由使用者透過 POST /predict 傳入。"""
    years_experience: float = Field(..., ge=0.0, le=50.0)   # 工作年資（限制 0~50 年）
    education_level: str                                    # 學歷（高中以下 / 大學 / 碩士以上）
    city: str                                               # 城市（城市A / 城市B / 城市C）


class SalaryOutput(BaseModel):
    """預測結果的回應資料模型。"""
    predicted_salary: float             # 預測月薪
    estimated_annual_salary: float      # 預估年薪（月薪 × 14）


# ------------------------------------
# 模型載入機制
# ------------------------------------
def load_model_state():
    """從 joblib 檔案讀取最新模型與預處理器，並同步更新全域變數 MODEL_STATE。"""
    global MODEL_STATE
    if not os.path.exists(model_path):
        train_and_save_model()
    model_data = joblib.load(model_path)
    MODEL_STATE.clear()
    MODEL_STATE.update(
        {
            "model": model_data["model"],
            "oe": model_data["oe"],
            "ohe": model_data["ohe"],
            "scaler": model_data["scaler"],
            "r2": model_data.get("r2"),
            "feature_names": model_data["feature_names"],
            "feature_coefs": model_data.get("feature_coefs", {}),
            "model_type": model_data.get("model_type"),
            "alpha": model_data.get("alpha"),
        }
    )


load_model_state()


# ------------------------------------
# 建立 FastAPI 後端（提供 /train 與 /predict API）
# ------------------------------------
app = FastAPI(title="薪資預測系統 API", version="1.0.0")


@app.post("/train", response_model=TrainResult)
def train_endpoint(config: TrainConfig):
    """訓練端點：線上重新訓練模型，並即時更新服務所使用的模型。"""
    try:
        res = train_and_save_model(
            test_size=config.test_size,
            random_state=config.random_state,
            model_type=config.model_type,
            alpha=config.alpha,
        )
        load_model_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"線上訓練失敗: {str(e)}")
    return res


@app.post("/predict", response_model=SalaryOutput)
def predict_endpoint(payload: SalaryInput):
    """預測端點：輸入工作年資、學歷與城市，回傳預測月薪與年薪。"""
    oe = MODEL_STATE["oe"]        # OrdinalEncoder：學歷文字 -> 數值
    ohe = MODEL_STATE["ohe"]      # OneHotEncoder：城市文字 -> 0/1 向量
    scaler = MODEL_STATE["scaler"]  # StandardScaler：標準化特徵
    model = MODEL_STATE["model"]    # 已訓練好的迴歸模型

    edu_encoded = int(oe.transform(pd.DataFrame([[payload.education_level]], columns=["EducationLevel"]))[0][0])
    city_vector = ohe.transform(pd.DataFrame([[payload.city]], columns=["City"]))
    city_cols = ohe.get_feature_names_out(["City"])
    feature_row = [payload.years_experience, edu_encoded] + list(city_vector[0])
    features = pd.DataFrame([feature_row], columns=["YearsExperience", "EducationLevel"] + list(city_cols))
    X_scaled = scaler.transform(features)
    predicted_salary = float(model.predict(X_scaled)[0])
    return SalaryOutput(
        predicted_salary=predicted_salary,
        estimated_annual_salary=predicted_salary * 14,
    )


# ------------------------------------
# Gradio 介面（藍色系、卡片式美觀設計）
# ------------------------------------
# 藍色系主題：主色藍 / 次色靛藍 / 中性色石板灰
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Noto Sans TC"), gr.themes.GoogleFont("Segoe UI"), "sans-serif"],
)

# 自訂樣式：藍色漸層背景、圓角卡片、藍色按鈕與聚焦效果
css = """
.gradio-container {
    background: linear-gradient(160deg, #eaf3ff 0%, #d9ecff 40%, #b9d9ff 100%) !important;
    max-width: 100% !important;
}
.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3 {
    color: #1e3a8a;
}
#header-banner {
    background: linear-gradient(90deg, #1d4ed8, #3b82f6, #60a5fa);
    border-radius: 18px;
    padding: 18px 24px;
    color: #ffffff;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
    text-align: center;
}
#header-banner h1 {
    color: #ffffff;
    margin: 0;
    font-size: 1.7rem;
    letter-spacing: 1px;
}
#header-banner p {
    color: #dbeafe;
    margin: 4px 0 0 0;
}
.gradio-container .panel, .gradio-container .block, .gradio-container .gr-block {
    border-radius: 14px !important;
}
.gradio-container button.primary {
    background: linear-gradient(90deg, #1d4ed8, #2563eb) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.gradio-container button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45) !important;
}
.gradio-container input, .gradio-container textarea, .gradio-container select {
    border-radius: 10px !important;
    border: 1px solid #bfdbfe !important;
}
.gradio-container input:focus, .gradio-container textarea:focus, .gradio-container select:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
}
#result-card {
    background: linear-gradient(135deg, #eff6ff, #dbeafe) !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 14px !important;
    padding: 8px;
}
"""


def format_model_info():
    """回傳目前服務使用的模型資訊字串。"""
    return (
        f"**目前模型：** `{MODEL_STATE.get('model_type', '-')}`  "
        f"**R² 分數：** `{MODEL_STATE.get('r2', '-') if MODEL_STATE.get('r2') is None else round(MODEL_STATE.get('r2'), 4)}`  "
        f"**alpha：** `{MODEL_STATE.get('alpha', '-')}`"
    )


def gradio_predict(years_experience: float, education_level: str, city: str):
    """Gradio 預測處理：呼叫 FastAPI 預測端點並格式化結果。"""
    result = predict_endpoint(
        SalaryInput(
            years_experience=years_experience,
            education_level=education_level,
            city=city,
        )
    )
    summary = (
        f"### 預測結果\n"
        f"- **月薪：** {result.predicted_salary:,.2f} 元\n"
        f"- **年薪（×14 個月）：** {result.estimated_annual_salary:,.2f} 元\n\n"
        f"{format_model_info()}"
    )
    return summary


def gradio_train(test_size: float, random_state: int, model_type: str, alpha: float):
    """Gradio 訓練處理：呼叫 FastAPI 訓練端點並格式化結果。"""
    try:
        res = train_endpoint(
            TrainConfig(
                test_size=test_size,
                random_state=random_state,
                model_type=model_type,
                alpha=alpha,
            )
        )
        # train_endpoint 直接呼叫時回傳 dict，透過 TrainResult 驗證並轉為模型物件
        res = TrainResult(**res) if isinstance(res, dict) else res
    except HTTPException as e:
        return f"### ❌ 訓練失敗\n{e.detail}"

    coef_lines = "\n".join(
        f"- **{name}**：{coef:.4f}" for name, coef in res.feature_coefs.items()
    )
    summary = (
        f"### ✅ {res.message}\n"
        f"- **演算法：** `{res.model_type}`（alpha={res.alpha}）\n"
        f"- **R² 分數：** {res.r2:.4f}\n"
        f"- **截距：** {res.intercept:.4f}\n"
        f"- **訓練耗時：** {res.train_time:.4f} 秒\n\n"
        f"#### 特徵權重係數\n{coef_lines}"
    )
    return summary


def build_gradio_demo():
    """建立藍色系美觀的 Gradio 介面（包含預測與訓練兩個頁籤）。"""
    with gr.Blocks(title="薪資預測系統") as demo:
        gr.HTML(
            """
            <div id="header-banner">
                <h1>💼 薪資預測系統</h1>
                <p>基於工作年資、學歷與城市，使用機器學習迴歸模型預測薪資</p>
            </div>
            """
        )

        with gr.Tabs():
            # ---------- 預測頁籤 ----------
            with gr.Tab("📈 薪資預測"):
                with gr.Row():
                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("### 輸入員工資料")
                        years_experience = gr.Slider(
                            minimum=0.0,
                            maximum=50.0,
                            value=5.0,
                            step=0.5,
                            label="工作年資（年）",
                            info="輸入員工累積的工作年資",
                        )
                        education_level = gr.Dropdown(
                            choices=["高中以下", "大學", "碩士以上"],
                            value="大學",
                            label="學歷",
                            info="選擇最高學歷",
                        )
                        city = gr.Dropdown(
                            choices=["城市A", "城市B", "城市C"],
                            value="城市A",
                            label="工作城市",
                            info="選擇任職所在城市",
                        )
                        predict_btn = gr.Button("🚀 開始預測", variant="primary", size="lg")

                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("### 預測結果")
                        predict_output = gr.Markdown("請輸入資料並點擊「開始預測」")
                        gr.HTML(
                            """
                            <div id="result-card">
                                <p style="color:#1e3a8a;margin:0;">
                                    <strong>提示：</strong>月薪為迴歸模型之預測值，
                                    年薪則以 14 個月計算。
                                </p>
                            </div>
                            """
                        )

                predict_btn.click(
                    fn=gradio_predict,
                    inputs=[years_experience, education_level, city],
                    outputs=predict_output,
                )

            # ---------- 訓練頁籤 ----------
            with gr.Tab("🧠 模型訓練"):
                with gr.Row():
                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("### 訓練參數設定")
                        test_size = gr.Slider(
                            minimum=0.1,
                            maximum=0.5,
                            value=0.2,
                            step=0.05,
                            label="測試集比例",
                            info="測試集佔整體資料的比例（0.1 ~ 0.5）",
                        )
                        random_state = gr.Number(
                            value=76,
                            minimum=0,
                            precision=0,
                            label="隨機種子",
                            info="固定隨機種子以確保結果可重現",
                        )
                        model_type = gr.Radio(
                            choices=["LinearRegression", "Lasso", "Ridge"],
                            value="LinearRegression",
                            label="模型演算法",
                            info="選擇迴歸演算法",
                        )
                        alpha = gr.Slider(
                            minimum=0.001,
                            maximum=100.0,
                            value=1.0,
                            step=0.001,
                            label="正則化強度 alpha",
                            info="僅適用於 Lasso 與 Ridge",
                        )
                        train_btn = gr.Button("⚙️ 重新訓練模型", variant="primary", size="lg")

                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("### 訓練結果")
                        train_output = gr.Markdown("請設定參數並點擊「重新訓練模型」")

                train_btn.click(
                    fn=gradio_train,
                    inputs=[test_size, random_state, model_type, alpha],
                    outputs=train_output,
                )

        gr.Markdown(
            "---\n"
            "#### 🔌 API 端點\n"
            "- `POST /train` — 訓練或重新訓練模型（JSON 參數）\n"
            "- `POST /predict` — 預測薪資（JSON 參數）\n"
            "- `GET /docs` — FastAPI 自動產生的 API 文件"
        )

    return demo


# 建立 Gradio 介面並掛載到 FastAPI 應用上：
# 瀏覽器開啟 http://localhost:8000/ 即可使用 Gradio 介面
# （Gradio 6 的 theme 與 css 需於 mount/launch 時傳入）
demo = build_gradio_demo()
app = gr.mount_gradio_app(app, demo, path="/", theme=theme, css=css)


if __name__ == "__main__":
    uvicorn.run("app_gradio:app", host="0.0.0.0", port=8000, reload=True)
