import os
import pandas as pd
import time
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Lasso,LinearRegression
import joblib

def train_and_save_model(
    test_size:float = 0.2,
    random_state:int = 76,
    model_type:str = "LinearRegression",
    alpha:float = 1.0
) -> dict:
    """
    訓練線性迴歸模型 (支援多元線性迴歸、Lasso 迴歸與 Ridge 嶺迴歸) 以預測薪資，
    並將模型與預處理器序列化儲存。
    
    參數:
        test_size: 測試集比例 (0.1 ~ 0.5)
        random_state: 隨機種子 (預設 76，與教學 Notebook 一致)
        model_type: 模型類型 ("LinearRegression", "Lasso", "Ridge")
        alpha: 正則化強度 (適用於 Lasso 與 Ridge)
        
    回傳:
        包含訓練指標、權重與花費時間的字典。
    """
    # ------------------------------------
    # 讀檔:取得csv的絕對路徑
    # ------------------------------------
    
    import os
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir:str = os.getcwd()
    csv_path:str = os.path.join(current_dir, "Salary_Data2.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError("找不到資料檔案")

    data:DataFrame = pd.read_csv(csv_path)
    #display(data)
    
    #開始時間
    star_time:float = time.time()
    
    oe = OrdinalEncoder(categories=[['高中以下','大學','碩士以上']])
    data['EducationLevel'] = oe.fit_transform(data[['EducationLevel']]) #把轉換的資料取代原本的學歷

    # -----------------------------------------------------
    # 3. 建立並擬合 OneHotEncoder (城市：城市A, 城市B, 城市C)
    # -----------------------------------------------------
    from sklearn.preprocessing import OneHotEncoder
    #display(data['City'].unique()) #檢查是否有空值
    
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') #轉換器 轉為陣列
    ohe.fit(pd.DataFrame([["城市A"], ["城市B"], ["城市C"]], columns=["City"])) #訓練:定義有哪些欄位值
    city_encoded = ohe.transform(data[['City']]) #轉換成多欄數值的 Ndarry
    city_cols = ohe.get_feature_names_out(['City'])  #建立欄位名稱
    city_df = pd.DataFrame(city_encoded,columns=city_cols) #建立DataFrame
    data = pd.concat([data,city_df],axis=1).drop('City',axis=1) #把資料串接上來 drop掉原本的
    #display(data)

    # -----------------------------------------------------
    # 4. 開始訓練
    # -----------------------------------------------------
    # 定義特徵欄位與目標變數
    feature_names = ['YearsExperience',	'EducationLevel','City_城市A','City_城市B','City_城市C']
    X = data[feature_names]
    y = data['Salary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = random_state
    )
    
    
    # YearsExperience數字最大 影響權重特別大==>要做標準化
    # train_要一邊訓練一邊記憶 用fit_transform
    # test 只要轉換 transform
    scaler = StandardScaler() #給他一個短名稱
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #設定模型
    model_type = "LinearRegression "
    alpha = 1.0
    model_type_clean = model_type.strip()
    if model_type_clean.lower() == "lasso":
        model = Lasso(alpha=alpha, random_state=random_state)
        actual_model_name = f"Lasso 迴歸(α={alpha})"
        model_type_clean="Lasso"
    elif model_type_clean.lower() == "ridge":
        model = Ridge(alpha=alpha, random_state=random_state)
        actual_model_name = f"Ridge 嶺迴歸(α={alpha})"
        model_type_clean="Ridge"
    else:
        model = LinearRegression()
        actual_model_name = "多元線性迴歸 (OLS)"
        model_type_clean="LinearRegression"
    
    
    print(f'開始訓練{actual_model_name} 測試集比例:{test_size},隨機總子:{random_state})...')
    model.fit(X_train_scaled,y_train) 
    train_time = time.time() - star_time

    # -----------------------------------------------------
    # 5. 取得模型的權重,偏移值,評估值R2
    # -----------------------------------------------------
    
    r2 = model.score(X_test_scaled, y_test)
    coefs = model.coef_ #權重
    intercept = model.intercept_
    feature_coefs = {
        name:float (coef) for name, coef in zip(feature_names,coefs)
    }
    # ----------------------------------------------------------
    # 存模型以及所有相關預處理器與元數據 (Metadata)
    # ----------------------------------------------------------
    import joblib
    
    model_data = {
        "model": model,
        "oe": oe,
        "ohe": ohe,
        "scaler": scaler,
        "r2": float(r2),
        "coef": [float(c) for c in coefs],
        "intercept": float(intercept),
        "feature_names": feature_names,
        "feature_coefs": feature_coefs,
        "model_type": model_type_clean,
        "alpha": float(alpha),
        "train_time": float(train_time),
        "test_size": test_size,
        "random_state": random_state
    }
    
    model_filename = os.path.join(current_dir, "salary_model.joblib")
    print(f"正在將模型、預處理器與元數據序列化並儲存至 {model_filename}...")
    joblib.dump(model_data,model_filename)
    print("模型儲存成功！")

if __name__ == "__main__":
    train_and_save_model()