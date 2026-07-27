"""台灣股票查詢介面（Gradio + yfinance）。

此程式直接向 Yahoo Finance 查詢資料，不需要啟動 FastAPI。
執行方式：
    uv run backend/0721/pratice4_gradio.py
"""

from __future__ import annotations

import re

import gradio as gr
import pandas as pd
import yfinance as yf


PERIODS = {
    "1 天": "1d",
    "1 星期": "5d",
    "1 個月": "1mo",
    "1 年": "1y",
}


def get_stock_data(stock_code: str, period_label: str) -> tuple[str, pd.DataFrame]:
    """直接使用 yfinance 取得台灣股票歷史資料。"""
    code = stock_code.strip()
    if not re.fullmatch(r"\d{4,6}", code):
        raise gr.Error("請輸入 4～6 位數字的台灣股票代碼，例如：2330、0050。")

    symbol = f"{code}.TW"
    try:
        history = yf.Ticker(symbol).history(period=PERIODS[period_label])
    except Exception as exc:
        raise gr.Error("目前無法連線至 Yahoo Finance，請稍後再試。") from exc

    if history.empty:
        raise gr.Error("查無股票資料，請確認股票代碼是否正確。")

    data = history.reset_index().rename(
        columns={
            "Date": "日期",
            "Open": "開盤",
            "High": "最高",
            "Low": "最低",
            "Close": "收盤",
            "Volume": "成交量",
        }
    )
    data["日期"] = pd.to_datetime(data["日期"]).dt.strftime("%Y-%m-%d")
    return symbol, data[["日期", "開盤", "最高", "最低", "收盤", "成交量"]]


def query_stock(stock_code: str, period_label: str):
    """產生股票摘要、走勢圖資料與歷史明細。"""
    symbol, data = get_stock_data(stock_code, period_label)
    latest = data.iloc[-1]
    previous_close = float(data.iloc[-2]["收盤"]) if len(data) > 1 else float(latest["收盤"])
    close = float(latest["收盤"])
    change = close - previous_close
    change_percent = (change / previous_close * 100) if previous_close else 0
    change_class = "up" if change >= 0 else "down"

    summary = f"""
    <div class="result-header">
      <div><span class="symbol">{symbol}</span><span class="date">最新交易日 {latest['日期']}</span></div>
      <div class="price">{close:,.2f} <span class="{change_class}">{change:+.2f} ({change_percent:+.2f}%)</span></div>
    </div>
    <div class="metrics">
      <div><span>開盤</span><strong>{float(latest['開盤']):,.2f}</strong></div>
      <div><span>最高</span><strong>{float(latest['最高']):,.2f}</strong></div>
      <div><span>最低</span><strong>{float(latest['最低']):,.2f}</strong></div>
      <div><span>成交量</span><strong>{int(latest['成交量']):,}</strong></div>
    </div>
    """

    display_data = data.copy()
    for column in ["開盤", "最高", "最低", "收盤"]:
        display_data[column] = pd.to_numeric(display_data[column], errors="coerce").round(2)
    display_data["成交量"] = pd.to_numeric(display_data["成交量"], errors="coerce").fillna(0).astype(int)

    return summary, display_data, display_data


CSS = """
body, .gradio-container {
  background: linear-gradient(135deg, #f5f8ff 0%, #eef5f2 52%, #fff8ee 100%) !important;
  font-family: -apple-system, BlinkMacSystemFont, "Noto Sans TC", sans-serif !important;
}
.app-card { max-width: 1060px; margin: 26px auto; padding: 28px; background: rgba(255,255,255,.88); border: 1px solid #e5ebf2; border-radius: 24px; box-shadow: 0 20px 55px rgba(35,65,105,.12); }
.hero h1 { margin: 0; color: #12263f; font-size: 2rem; }
.hero p { margin: 8px 0 22px; color: #617083; }
.search-button { background: linear-gradient(100deg, #176b87, #27a28c) !important; border: 0 !important; font-weight: 700 !important; }
.result-header { display: flex; justify-content: space-between; gap: 16px; align-items: center; padding: 12px 0 18px; }
.symbol { display: block; color: #12263f; font-size: 1.25rem; font-weight: 800; }.date { color: #758398; font-size: .9rem; }
.price { color: #12263f; font-size: 1.6rem; font-weight: 800; }.price span { display: block; font-size: .88rem; }.up { color: #d94b4b; }.down { color: #198a68; }
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }.metrics div { padding: 12px; border-radius: 12px; background: #f4f8fb; }.metrics span { display: block; color: #718095; font-size: .8rem; }.metrics strong { color: #24364b; font-size: 1.05rem; }
@media (max-width: 640px) { .app-card { padding: 17px; margin: 8px; }.result-header { display: block; }.metrics { grid-template-columns: repeat(2, 1fr); } }
"""


with gr.Blocks(title="台灣股票查詢") as demo:
    with gr.Column(elem_classes="app-card"):
        gr.HTML("""<div class="hero"><h1>台灣股票查詢</h1><p>即時連線 Yahoo Finance，快速瀏覽台股歷史價格與成交量。</p></div>""")
        with gr.Row():
            stock_code = gr.Textbox(label="股票代碼", value="2330", placeholder="例如 2330、0050、2317")
            period = gr.Dropdown(label="查詢期間", choices=list(PERIODS), value="1 個月")
        search = gr.Button("查詢股價", variant="primary", elem_classes="search-button")
        summary = gr.HTML()
        chart = gr.LinePlot(x="日期", y="收盤", title="收盤價走勢", height=360)
        table = gr.Dataframe(label="歷史股價明細", interactive=False, wrap=True)
        gr.Examples(examples=[["2330", "1 個月"], ["0050", "1 年"], ["2317", "1 星期"]], inputs=[stock_code, period])

    search.click(query_stock, inputs=[stock_code, period], outputs=[summary, chart, table])
    stock_code.submit(query_stock, inputs=[stock_code, period], outputs=[summary, chart, table])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=CSS)
