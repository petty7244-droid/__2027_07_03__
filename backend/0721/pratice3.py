from enum import Enum

import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse


app = FastAPI(
    title="台灣股票資料 API",
    description="依股票代碼查詢最近 1 天、1 星期、1 個月或 1 年的股價。",
    version="1.0.0",
)


class StockPeriod(str, Enum):
    """yfinance 支援的查詢期間。"""

    one_day = "1d"
    one_week = "5d"
    one_month = "1mo"
    one_year = "1y"


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
    symbol = f"{stock_code}.TW"
    history = yf.Ticker(symbol).history(period=period.value)

    records: list[dict[str, object]] = []
    for date, row in history.iterrows():
        records.append(
            {
                "date": date.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            }
        )
    return records


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


@app.get("/stock", summary="查詢台灣股票歷史股價")
def read_stock(
    stock_code: str = Query(
        default="2330",
        pattern=r"^[0-9]{4,6}$",
        description="台灣股票代碼，例如：2330、2317、0050",
    ),
    period: StockPeriod = Query(
        default=StockPeriod.one_month,
        description="查詢期間：1d、5d、1mo 或 1y",
    ),
) -> dict[str, object]:
    """依股票代碼及指定期間回傳開、高、低、收與成交量。"""
    symbol = f"{stock_code}.TW"
    try:
        data = get_stock_history(stock_code, period)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="目前無法取得股票資料") from exc

    if not data:
        raise HTTPException(status_code=404, detail="查無股票資料")

    return {
        "stock_code": stock_code,
        "symbol": symbol,
        "period": period.value,
        "period_label": PERIOD_LABELS[period],
        "count": len(data),
        "data": data,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)