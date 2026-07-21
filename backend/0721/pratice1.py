from fastapi import FastAPI
import uvicorn
import json
from pydantic import BaseModel,Field,field_validator
from datetime import datetime

app = FastAPI()
class AirSite(BaseModel):
    aqi:int                       # 空氣品質指標 AQI
    county:str                    # 縣市名稱
    date:datetime = Field(alias="datacreationdate")  # 資料建立時間，JSON 欄位名為 datacreationdate
    lat:float = Field(alias = "latitude")            # 緯度，JSON 欄位名為 latitude
    lon:float = Field(alias="longitude")             # 經度，JSON 欄位名為 longitude
    pm25:float = Field(alias="pm2.5")                 # PM2.5 濃度，JSON 欄位名為 pm.5
    pollutant:str                # 主要污染物名稱
    site_name:str = Field(alias="sitename")          # 站點名稱，JSON 欄位名為 sitename
    status:str                   # 空品狀態（如：良好、普通等）

    #自定義驗證器：將空字串轉為 0，避免型別轉換錯誤
    @field_validator('aqi','lat','lon','pm25', mode='before')
    @classmethod
    def empty_to_zero(cls, v):
        return 0 if v == '' else v #簡寫法運算式 if v == "" 則return 0  else v

class Root(BaseModel):
    status:bool = True
    sites:list[AirSite]            # 多個 AirSite 站點組成的列表

# 讀取 JSON 檔案並轉為 Python 字典
with open("空氣品質aqi.json",encoding="utf-8") as file:
    data:dict = json.load(file)

# 從字典中取出 'records' 欄位，為一筆筆空氣品質站點資料的列表
contents:list[dict]= data['records']

# 將每筆 dict 資料透過 **item 解包後建立 AirSite 實例，再裝入 Root 模型
root = Root(sites=[AirSite(**item) for item in contents])


@app.get("/")
def read_root():
    return root


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: str | None = None):
#     return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    uvicorn.run("pratice1:app", port=5000, reload=True)