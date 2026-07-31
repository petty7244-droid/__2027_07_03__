import gradio as gr
from fastapi import FastAPI
import uvicorn

#1. 初始化FastAPI應用程式
app = FastAPI(
    title="FastAPI + Gradio 整合範例" ,
    description="利用 FastAPI 做為後端 API 並掛載 Gradio UI" ,
    version= "1.0"
)


# ------------------------
# FastAPI 原生路由 (API端點)
# ------------------------

@app.get("/root")
def read_root():
    return {"message":"歡迎來到 FastAPI 主頁! "}

@app.get("/api/greet")
def api_greet(name:str="可輸入您的姓名"):
    """一個簡單的FastAPI端點"""
    return {"status":"success",
            "result" : f'Hello, {name} from FastAPI'
            }

# ------------------------
# 建立 Gradio 介面
# ------------------------
def predict(name:str, intensity:int):
    """Gradio使用處理函式"""
    greeting = f"Hello~ {name} " * intensity
    return greeting

demo = gr.Interface (
    fn = predict,
    inputs= [
        gr.Textbox(lines=2 , placeholder="請輸入您的姓名...", label="姓名"),
        gr.Slider(1,10, value=3 ,step=1 , label="重複的次數")
    ],
    outputs= gr.Textbox(label="輸出結果"),
    title="Gradio 互動介面",
    description="這是崁入在FastAPI裡面的Gradio"
)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__" :
    uvicorn.run(app, host="0.0.0.0" ,port=8000)
