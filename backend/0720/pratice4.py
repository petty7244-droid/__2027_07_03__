from fastapi import FastAPI, HTTPException
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import uvicorn

app = FastAPI()

def get_stock_history(symbol: str = "2330.TW", period: str = "1y") -> pd.DataFrame:
    period_map = {
        "1d": "5d",
        "1w": "5d",
        "1mo": "1mo",
        "1y": "1y"
    }
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period_map.get(period, "1y"))
    return df

@app.get("/stock/{symbol}")
def get_stock_by_symbol(symbol: str, period: str = "1y"):
    try:
        df = get_stock_history(symbol, period)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"找不到股票 {symbol} 的資料")
        
        return {
            "symbol": symbol,
            "period": period,
            "date_range": f"{df.index[0].date()} ~ {df.index[-1].date()}",
            "data_count": len(df),
            "data": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tsmc")
def get_tsmc(period: str = "1y"):
    try:
        df = get_stock_history("2330.TW", period)
        if df.empty:
            raise HTTPException(status_code=404, detail="找不到台積電資料")
        
        return {
            "symbol": "2330.TW",
            "period": period,
            "date_range": f"{df.index[0].date()} ~ {df.index[-1].date()}",
            "data_count": len(df),
            "data": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("pratice4:app", reload=True)
