"""台灣股票查詢介面（Gradio + yfinance）。

此程式為獨立網頁介面，不透過 FastAPI，直接查詢 Yahoo Finance。
啟動方式：
    uv run backend/07_21/gradio_stock_app.py
"""

from __future__ import annotations

import os
import re
import socket
from enum import Enum

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


class StockPeriod(str, Enum):
    """yfinance 支援的查詢期間。"""

    one_day = "1d"
    one_week = "5d"
    one_month = "1mo"
    one_year = "1y"


PERIOD_LABELS = {
    StockPeriod.one_day.value: "1 天",
    StockPeriod.one_week.value: "1 星期",
    StockPeriod.one_month.value: "1 個月",
    StockPeriod.one_year.value: "1 年",
}


def _fetch_stock_df(stock_code: str, period: str) -> tuple[str, pd.DataFrame]:
    """直接從 Yahoo Finance 取得台股歷史資料。"""
    code = stock_code.strip()
    if not re.fullmatch(r"[0-9]{4,6}", code):
        raise gr.Error("股票代碼格式錯誤，請輸入 4 到 6 位數字，例如 2330、0050。")

    symbol = f"{code}.TW"
    history = yf.Ticker(symbol).history(period=period)
    if history.empty:
        raise gr.Error("查無資料，請確認股票代碼是否正確，或改用其他查詢期間。")

    df = history.reset_index()
    df.rename(
        columns={
            "Date": "日期",
            "Open": "開盤",
            "High": "最高",
            "Low": "最低",
            "Close": "收盤",
            "Volume": "成交量",
        },
        inplace=True,
    )
    df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
    return symbol, df[["日期", "開盤", "最高", "最低", "收盤", "成交量"]]


def _build_chart(df: pd.DataFrame, symbol: str, period_label: str):
    """建立收盤價與成交量雙軸圖。"""
    fig, ax_price = plt.subplots(figsize=(10, 4.8), dpi=130)
    ax_volume = ax_price.twinx()

    x = pd.to_datetime(df["日期"])
    close = pd.to_numeric(df["收盤"], errors="coerce")
    volume = pd.to_numeric(df["成交量"], errors="coerce")

    bars = ax_volume.bar(x, volume, width=0.8, alpha=0.22, color="#2F5D50")
    line = ax_price.plot(x, close, color="#C84B31", linewidth=2.2, marker="o", markersize=3.8)

    ax_price.set_title(f"{symbol} 期間：{period_label}", fontsize=14, pad=10)
    ax_price.set_ylabel("收盤價", color="#C84B31")
    ax_volume.set_ylabel("成交量", color="#2F5D50")
    ax_price.grid(axis="y", linestyle="--", alpha=0.25)

    for label in ax_price.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    ax_price.legend(line + [bars], ["收盤價", "成交量"], loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def query_stock(stock_code: str, period: str):
    """查詢股票並回傳摘要、圖表與明細資料表。"""
    symbol, df = _fetch_stock_df(stock_code, period)
    period_label = PERIOD_LABELS.get(period, period)

    latest = df.iloc[-1]
    summary = (
        f"### {symbol} 查詢結果\n"
        f"- 期間：{period_label}\n"
        f"- 資料筆數：{len(df)}\n"
        f"- 最新交易日：{latest['日期']}\n"
        f"- 最新收盤價：{float(latest['收盤']):.2f}\n"
        f"- 最新成交量：{int(latest['成交量']):,}"
    )

    chart = _build_chart(df, symbol, period_label)

    output_df = df.copy()
    numeric_cols = ["開盤", "最高", "最低", "收盤"]
    for col in numeric_cols:
        output_df[col] = pd.to_numeric(output_df[col], errors="coerce").round(2)
    output_df["成交量"] = pd.to_numeric(output_df["成交量"], errors="coerce").fillna(0).astype(int)

    return summary, chart, output_df


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700;900&display=swap');

:root {
  --bg-a: #f9f4ec;
  --bg-b: #d7e8df;
  --bg-c: #f2d3b3;
  --card: rgba(255, 255, 255, 0.78);
  --ink: #2f2a25;
  --accent: #c84b31;
  --accent-2: #2f5d50;
}

html, body, .gradio-container {
  font-family: 'Noto Sans TC', sans-serif !important;
  color: var(--ink);
  background:
    radial-gradient(1200px 480px at -5% -10%, var(--bg-c) 0%, transparent 60%),
    radial-gradient(900px 420px at 105% 10%, var(--bg-b) 0%, transparent 58%),
    linear-gradient(160deg, var(--bg-a) 0%, #f6f7ef 100%);
}

.app-shell {
  max-width: 1020px;
  margin: 16px auto;
  padding: 18px;
  border-radius: 18px;
  background: var(--card);
  box-shadow: 0 20px 45px rgba(58, 41, 28, 0.16);
  backdrop-filter: blur(4px);
}

.hero {
  margin-bottom: 8px;
  padding: 12px 8px;
}

.hero h1 {
  margin: 0;
  font-size: 2rem;
  font-weight: 900;
  letter-spacing: 0.04em;
}

.hero p {
  margin: 8px 0 0 0;
  color: #57493d;
}

.stock-btn {
  background: linear-gradient(120deg, var(--accent), #df7f2e) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 700 !important;
  box-shadow: 0 10px 24px rgba(200, 75, 49, 0.32);
}

.stock-btn:hover {
  filter: brightness(1.04);
  transform: translateY(-1px);
}
"""


def _find_available_port(start: int = 7860, end: int = 7890) -> int:
    """從指定範圍內找可用連接埠。"""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"找不到可用連接埠，範圍：{start}-{end}")


with gr.Blocks(title="台灣股票視覺查詢") as demo:
    with gr.Column(elem_classes=["app-shell"]):
        gr.HTML(
            """
            <section class="hero">
              <h1>台灣股票視覺查詢</h1>
              <p>直接連線 Yahoo Finance，不經過 FastAPI，輸入代碼就能看走勢與明細。</p>
            </section>
            """
        )

        with gr.Row():
            stock_code_input = gr.Textbox(
                label="股票代碼（4~6 位數字）",
                value="2330",
                max_lines=1,
                placeholder="例如 2330、0050、2317",
            )
            period_input = gr.Dropdown(
                label="查詢期間",
                choices=[
                    ("1 天", StockPeriod.one_day.value),
                    ("1 星期", StockPeriod.one_week.value),
                    ("1 個月", StockPeriod.one_month.value),
                    ("1 年", StockPeriod.one_year.value),
                ],
                value=StockPeriod.one_month.value,
            )

        search_btn = gr.Button("立即查詢", elem_classes=["stock-btn"])

        summary_output = gr.Markdown(label="摘要")
        chart_output = gr.Plot(label="股價走勢")
        table_output = gr.Dataframe(
            label="歷史明細",
            interactive=False,
            wrap=True,
        )

        gr.Examples(
            examples=[
                ["2330", "1mo"],
                ["0050", "1y"],
                ["2317", "5d"],
            ],
            inputs=[stock_code_input, period_input],
        )

    search_btn.click(
        fn=query_stock,
        inputs=[stock_code_input, period_input],
        outputs=[summary_output, chart_output, table_output],
    )


if __name__ == "__main__":
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    server_port = _find_available_port(start=preferred_port, end=preferred_port + 30)
    if server_port != preferred_port:
        print(f"連接埠 {preferred_port} 已被占用，改用 {server_port} 啟動。")

    demo.launch(server_name="127.0.0.1", server_port=server_port, css=CUSTOM_CSS)