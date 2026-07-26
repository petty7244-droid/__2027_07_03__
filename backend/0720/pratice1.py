from source import get_camera_position, CameraPosition
from pprint import pprint
import requests
from requests import Response,JSONDecodeError,HTTPError
from pydantic import BaseModel,TypeAdapter

#物件結構化 不要的欄位就不要定義 
class CameraPosition(BaseModel):
    year:int
    bureau:str
    unit:str
    location:str

url = 'https://data.ntpc.gov.tw/api/datasets/1b72abe8-8862-4130-aeb8-178c1240e6c4/json?page=0&size=10'


def main():
    try:
        r:Response  = requests.request(method="GET",url = url)
        data:list[dict]= r.json()
        adapter = TypeAdapter(list[CameraPosition]) #--用TypeAdapter轉為dict
        list_position:list[CameraPosition] = adapter.validate_python(data) #--validate驗證資料
        pprint(list_position)
        
    except JSONDecodeError as e:
      print (f'DecodeError:{e}')
    except HTTPError as e:
      print (f'HTTPError:{e}')
    except Exception as e :
      print (f'Exception:{e}')

if __name__ == "__main__":
    main()