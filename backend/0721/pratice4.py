# 匯入列舉類別（用於定義固定選項，例如查詢期間）
from enum import Enum

# 匯入 yfinance，用於下載 Yahoo Finance 的股票歷史資料
import yfinance as yf
# 匯入 FastAPI 相關元件
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse 


# 建立 FastAPI 應用程式實例，設定標題、說明與版本
app = FastAPI(
    title="台灣股票資料 API",
    description="依股票代碼查詢最近 1 天、1 星期、1 個月或 1 年的股價。",
    version="1.0.0",
)


# 定義查詢期間的列舉，繼承 str 以便 API 文件自動顯示字串值
class StockPeriod(str, Enum):
    """yfinance 支援的查詢期間。"""

    one_day = "1d"       # 1 天
    one_week = "5d"      # 1 星期（5 個交易日）
    one_month = "1mo"    # 1 個月
    one_year = "1y"      # 1 年


# 將列舉值對應到中文標籤，方便回傳時使用
PERIOD_LABELS = {
    StockPeriod.one_day: "1 天",
    StockPeriod.one_week: "1 星期",
    StockPeriod.one_month: "1 個月",
    StockPeriod.one_year: "1 年",
}


def get_stock_history(
    stock_code: str, period: StockPeriod
) -> list[dict[str, object]]:
    """取得台灣股票歷史股價，並轉換成可由 FastAPI 回傳的格式。"""
    # 組合 Yahoo Finance 的股票代號（台灣股票需加上 .TW）
    symbol = f"{stock_code}.TW"
    # 透過 yfinance 下載歷史股價
    history = yf.Ticker(symbol).history(period=period.value)

    # 將 DataFrame 逐筆轉換為字典列表
    records: list[dict[str, object]] = []
    for date, row in history.iterrows():
        records.append(
            {
                "date": date.isoformat(),   # 日期，轉為 ISO 格式字串
                "open": float(row["Open"]),   # 開盤價
                "high": float(row["High"]),   # 最高價
                "low": float(row["Low"]),     # 最低價
                "close": float(row["Close"]), # 收盤價
                "volume": int(row["Volume"]), # 成交量
            }
        )
    return records


# 根路徑：回傳 HTML 表單頁面（不顯示在 /docs 中）
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    """顯示簡單的股票期間查詢頁面。"""
    return """
    <!doctype html>
    <html lang="zh-Hant">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>台灣股票資料</title>
      </head>
      <body>
        <h1>台灣股票資料</h1>
        <form action="/stock" method="get">
          <label for="stock_code">股票代碼：</label>
          <!-- 股票代碼輸入框：預設 2330，限制 4~6 位數字 -->
          <input id="stock_code" name="stock_code" value="2330"
                 pattern="[0-9]{4,6}" maxlength="6" required>
          <br><br>
          <label for="period">查詢期間：</label>
          <select id="period" name="period">
            <option value="1d">1 天</option>
            <option value="5d">1 星期</option>
            <option value="1mo">1 個月</option>
            <option value="1y">1 年</option>
          </select>
          <button type="submit">查詢</button>
        </form>
        <p>API 文件：<a href="/docs">/docs</a></p>
      </body>
    </html>
    """


# /stock 路徑：查詢股票歷史股價，回傳 JSON 格式
@app.get("/stock", summary="查詢台灣股票歷史股價")
def read_stock(
    stock_code: str = Query(
        default="2330",               # 預設股票代碼為台積電
        pattern=r"^[0-9]{4,6}$",      # 限制 4~6 位數字
        description="台灣股票代碼，例如：2330、2317、0050",
    ),
    period: StockPeriod = Query(
        default=StockPeriod.one_month,  # 預設查詢 1 個月
        description="查詢期間：1d、5d、1mo 或 1y",
    ),
) -> dict[str, object]:
    """依股票代碼及指定期間回傳開、高、低、收與成交量。"""
    symbol = f"{stock_code}.TW"
    try:
        # 取得歷史股價資料
        data = get_stock_history(stock_code, period)
    except Exception as exc:
        # 若無法連線至 Yahoo Finance，回傳 502
        raise HTTPException(status_code=502, detail="目前無法取得股票資料") from exc

    # 若查無資料（例如股票代碼錯誤），回傳 404
    if not data:
        raise HTTPException(status_code=404, detail="查無股票資料")

    # 回傳包含股票資訊與歷史資料的 JSON
    return {
        "stock_code": stock_code,
        "symbol": symbol,
        "period": period.value,
        "period_label": PERIOD_LABELS[period],
        "count": len(data),       # 資料筆數
        "data": data,             # 歷史股價列表
    }


# 直接執行此檔案時，啟動 uvicorn 開發伺服器
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)