---
title: "캐글 Home Credit Default Risk 분석"
categories:
  - study
output: 
  html_document:
    keep_md: true
---

출처: Will Koehrsen, Introduction to Manual Feature Engineering, 캐글, 2018.08.01

<details markdown="1">
<summary>접기/펼치기</summary>

<!--summary 아래 빈칸 공백 두고 내용을 적는공간-->

# 사전 준비

## Kaggle 데이터 불러오기

### Kaggle API 설치


```python
!pip install kaggle
```

    Requirement already satisfied: kaggle in /usr/local/lib/python3.6/dist-packages (1.5.9)
    Requirement already satisfied: certifi in /usr/local/lib/python3.6/dist-packages (from kaggle) (2020.6.20)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.24.3)
    Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.15.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.41.1)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.8.1)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.23.0)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.0.1)
    Requirement already satisfied: slugify in /usr/local/lib/python3.6/dist-packages (from kaggle) (0.0.1)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (2.10)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.6/dist-packages (from python-slugify->kaggle) (1.3)
    

### Kaggle Token 다운로드


```python
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
  print('uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  
# kaggle.json을 아래 폴더로 옮긴 뒤, file을 사용할 수 있도록 권한을 부여한다. 
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```



<input type="file" id="files-34d8464b-1b48-4807-ae60-e27fa8b1336a" name="files[]" multiple disabled
   style="border:none" />
<output id="result-34d8464b-1b48-4807-ae60-e27fa8b1336a">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving kaggle.json to kaggle.json
    uploaded file "kaggle.json" with length 63 bytes
    


```python
ls -1ha ~/.kaggle/kaggle.json
```

    /root/.kaggle/kaggle.json
    

### 구글 드라이브 연동


```python
from google.colab import drive # 패키지 불러오기 
from os.path import join  

# 구글 드라이브 마운트
ROOT = "/content/drive"     # 드라이브 기본 경로
print(ROOT)                 # print content of ROOT (Optional)
drive.mount(ROOT)           # 드라이브 기본 경로 

# 프로젝트 파일 생성 및 다운받을 경로 이동
MY_GOOGLE_DRIVE_PATH = 'My Drive/Colab Notebooks/python_basic/kaggle_home-credit-default-risk/data'
PROJECT_PATH = join(ROOT, MY_GOOGLE_DRIVE_PATH)
print(PROJECT_PATH)
```

    /content/drive
    Mounted at /content/drive
    /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_home-credit-default-risk/data
    


```python
%cd "{PROJECT_PATH}"
```

    /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_home-credit-default-risk/data
    

### Kaggle Competitions list 불러오기


```python
!kaggle competitions list
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.9 / client 1.5.4)
    ref                                            deadline             category            reward  teamCount  userHasEntered  
    ---------------------------------------------  -------------------  ---------------  ---------  ---------  --------------  
    contradictory-my-dear-watson                   2030-07-01 23:59:00  Getting Started     Prizes        134           False  
    gan-getting-started                            2030-07-01 23:59:00  Getting Started     Prizes        161           False  
    tpu-getting-started                            2030-06-03 23:59:00  Getting Started  Knowledge        292           False  
    digit-recognizer                               2030-01-01 00:00:00  Getting Started  Knowledge       2248           False  
    titanic                                        2030-01-01 00:00:00  Getting Started  Knowledge      17266            True  
    house-prices-advanced-regression-techniques    2030-01-01 00:00:00  Getting Started  Knowledge       4327            True  
    connectx                                       2030-01-01 00:00:00  Getting Started  Knowledge        366           False  
    nlp-getting-started                            2030-01-01 00:00:00  Getting Started  Knowledge       1130           False  
    rock-paper-scissors                            2021-02-01 23:59:00  Playground          Prizes        232           False  
    riiid-test-answer-prediction                   2021-01-07 23:59:00  Featured          $100,000       1491           False  
    nfl-big-data-bowl-2021                         2021-01-05 23:59:00  Analytics         $100,000          0           False  
    competitive-data-science-predict-future-sales  2020-12-31 23:59:00  Playground           Kudos       9393           False  
    halite-iv-playground-edition                   2020-12-31 23:59:00  Playground       Knowledge         44           False  
    predict-volcanic-eruptions-ingv-oe             2020-12-28 23:59:00  Playground            Swag        198           False  
    hashcode-drone-delivery                        2020-12-14 23:59:00  Playground       Knowledge         80           False  
    cdp-unlocking-climate-solutions                2020-12-02 23:59:00  Analytics          $91,000          0           False  
    lish-moa                                       2020-11-30 23:59:00  Research           $30,000       3454           False  
    google-football                                2020-11-30 23:59:00  Featured            $6,000        925           False  
    conways-reverse-game-of-life-2020              2020-11-30 23:59:00  Playground            Swag        133           False  
    lyft-motion-prediction-autonomous-vehicles     2020-11-25 23:59:00  Featured           $30,000        788           False  
    

### Home Credit Default Risk 데이터셋 불러오기


```python
!kaggle competitions download -c home-credit-default-risk
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.9 / client 1.5.4)
    Downloading installments_payments.csv.zip to /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_home-credit-default-risk/data
     98% 266M/271M [00:02<00:00, 123MB/s]
    100% 271M/271M [00:02<00:00, 104MB/s]
    Downloading previous_application.csv.zip to /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_home-credit-default-risk/data
     98% 75.0M/76.3M [00:00<00:00, 81.3MB/s]
    100% 76.3M/76.3M [00:00<00:00, 89.9MB/s]
    Downloading application_test.csv.zip to /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_home-credit-default-risk/data
    100% 5.81M/5.81M [00:00<00:00, 60.2MB/s]
    
    Downloading bureau.csv.zip to /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_home-credit-default-risk/data
     71% 26.0M/36.8M [00:00<00:00, 49.6MB/s]
    100% 36.8M/36.8M [00:00<00:00, 74.7MB/s]
    Downloading sample_submission.csv to /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_home-credit-default-risk/data
      0% 0.00/524k [00:00<?, ?B/s]
    100% 524k/524k [00:00<00:00, 34.4MB/s]
    Downloading POS_CASH_balance.csv.zip to /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_home-credit-default-risk/data
     99% 107M/109M [00:01<00:00, 95.7MB/s] 
    100% 109M/109M [00:01<00:00, 90.6MB/s]
    Downloading credit_card_balance.csv.zip to /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_home-credit-default-risk/data
     89% 86.0M/96.7M [00:01<00:00, 71.8MB/s]
    100% 96.7M/96.7M [00:01<00:00, 76.9MB/s]
    Downloading HomeCredit_columns_description.csv to /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_home-credit-default-risk/data
      0% 0.00/36.5k [00:00<?, ?B/s]
    100% 36.5k/36.5k [00:00<00:00, 4.81MB/s]
    Downloading application_train.csv.zip to /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_home-credit-default-risk/data
     83% 30.0M/36.1M [00:00<00:00, 37.6MB/s]
    100% 36.1M/36.1M [00:00<00:00, 52.1MB/s]
    Downloading bureau_balance.csv.zip to /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_home-credit-default-risk/data
     99% 56.0M/56.8M [00:01<00:00, 27.5MB/s]
    100% 56.8M/56.8M [00:01<00:00, 47.0MB/s]
    


```python
!ls
```

    application_test.csv.zip     HomeCredit_columns_description.csv
    application_train.csv.zip    installments_payments.csv.zip
    bureau_balance.csv.zip	     POS_CASH_balance.csv.zip
    bureau.csv.zip		     previous_application.csv.zip
    credit_card_balance.csv.zip  sample_submission.csv
    

- zip 파일 압축 풀기 [참고](https://medium.com/hyunjulie/%EC%BA%90%EA%B8%80%EA%B3%BC-%EA%B5%AC%EA%B8%80-colab-%EC%97%B0%EA%B2%B0%ED%95%B4%EC%A3%BC%EA%B8%B0-6a274f6de81d)


```python
!unzip application_test.csv.zip
!unzip application_train.csv.zip
!unzip bureau_balance.csv.zip
!unzip bureau.csv.zip
!unzip credit_card_balance.csv.zip
!unzip installments_payments.csv.zip
!unzip POS_CASH_balance.csv.zip
!unzip previous_application.csv.zip
```

    Archive:  application_test.csv.zip
      inflating: application_test.csv    
    Archive:  application_train.csv.zip
      inflating: application_train.csv   
    Archive:  bureau_balance.csv.zip
      inflating: bureau_balance.csv      
    Archive:  bureau.csv.zip
      inflating: bureau.csv              
    Archive:  credit_card_balance.csv.zip
      inflating: credit_card_balance.csv  
    Archive:  installments_payments.csv.zip
      inflating: installments_payments.csv  
    Archive:  POS_CASH_balance.csv.zip
      inflating: POS_CASH_balance.csv    
    Archive:  previous_application.csv.zip
      inflating: previous_application.csv  
    


```python
!ls
```

    application_test.csv	   credit_card_balance.csv.zip
    application_test.csv.zip   HomeCredit_columns_description.csv
    application_train.csv	   installments_payments.csv
    application_train.csv.zip  installments_payments.csv.zip
    bureau_balance.csv	   POS_CASH_balance.csv
    bureau_balance.csv.zip	   POS_CASH_balance.csv.zip
    bureau.csv		   previous_application.csv
    bureau.csv.zip		   previous_application.csv.zip
    credit_card_balance.csv    sample_submission.csv
    

- 압축파일 삭제하기 [참고](https://shiritori.tistory.com/m/11)


```python
!rm application_test.csv.zip
!rm application_train.csv.zip
!rm bureau_balance.csv.zip
!rm bureau.csv.zip
!rm credit_card_balance.csv.zip
!rm installments_payments.csv.zip
!rm POS_CASH_balance.csv.zip
!rm previous_application.csv.zip
```


```python
!ls
```

    application_test.csv	 HomeCredit_columns_description.csv
    application_train.csv	 installments_payments.csv
    bureau_balance.csv	 POS_CASH_balance.csv
    bureau.csv		 previous_application.csv
    credit_card_balance.csv  sample_submission.csv
    

# Introduction: Manual Feature Engineering

- **bureau.csv**: 'Home Credit'에 제출된 고객(Client)의 다른 금융기관에서의 과거의 대출 기록. (각각의 대출 기록은 각각의 열로 정리되어 있다.)
- **bureau_balance.csv**: 과거 대출들의 월별 데이터. (각 월별 데이터는 각각의 열로 정리되어 있다.)

- Manual(수동화된) Feature Engineering은 지루한 과정일 수 있다. 이것은 많은 사람들이 자동화된 Feature Engineering 기능을 활용하는 주된 이유이다.
- 대출 및 채무 불이행의 주된 원인에 대한 지식을 갖추는데는 한계가 있기 때문에, 최종 학습용 데이터프레임에서 가능한 많은 정보들을 얻는 데 주안점을 두었다.
- 이 커널은 어떤 Feature가 중요한 지를 결정하는 것에 있어서, 사람보다 모델이 고르도록 하는 접근방식을 취한다. 기본적으로 이러한 접근방식에서는 최대한 많은 Feature를 만들고, 모델은 이러한 Feature를 전부 활용한다.
- 수작업(Manual) Feature Engineering의 각 과정은 많은 양의 Pandas 코드와 약간의 인내심, 특히 데이터 처리에 있어서 많은 인내심을 필요로 한다. Feature Engineering은 여전히 전처리 작업을 필요로 한다.


```python
# 데이터 처리
import pandas as pd
import numpy as np

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas에서 나오는 경고문 무시
import warnings
warnings.filterwarnings('ignore')

# 원본
# plt.style.use('fivethirtyeight')

# matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고,
# 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편하다.
plt.style.use('seaborn')
sns.set(font_scale = 2.1)
```

## 예시: 고객의 이전 대출 수량 파악

**Counts of a client's previous loans**  
먼저 고객의 과거 타 금융기관에서의 대출 수량을 간단히 파악하고자 한다. 이 과정은 이 커널에서 반복적으로 사용되는 아래의 pandas 명령어를 포함한다.
- [groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html): Column값에 따라 데이터프레임을 그룹화. 이 과정에서는 `SK_ID_CURR` Column의 값에 따라 고객별로 데이터프레임을 그룹화
- [agg](images/kaggle_home_credit_default_risk/https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html): 그룹화된 데이터의 평균 등을 계산. 'grouped_df.mean()'을 통해 직접 평균을 계산하거나, agg 명령어와 리스트를 활용하여 평균, 최대값, 최소값, 합계 등을 계산 (grouped_df.agg([mean, max, min, sum])).
- [merge](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html): 집계된(aggregated) 값을 해당 고객과 매칭. SK_ID_CURR Column을 활용하여 집계된 값을 원본 트레이닝 데이터로 병합하고, 해당값이 없을 경우에는 NaN값을 입력.

또한 rename 명령어를 통해 Column을 딕셔너리(dict)를 활용하여 변경한다. 이러한 방식은 생성된 변수를 계속해서 추적하는 데 유용하다.


```python
# bureau 파일 읽기
bureau = pd.read_csv('bureau.csv')
bureau.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>SK_ID_BUREAU</th>
      <th>CREDIT_ACTIVE</th>
      <th>CREDIT_CURRENCY</th>
      <th>DAYS_CREDIT</th>
      <th>CREDIT_DAY_OVERDUE</th>
      <th>DAYS_CREDIT_ENDDATE</th>
      <th>DAYS_ENDDATE_FACT</th>
      <th>AMT_CREDIT_MAX_OVERDUE</th>
      <th>CNT_CREDIT_PROLONG</th>
      <th>AMT_CREDIT_SUM</th>
      <th>AMT_CREDIT_SUM_DEBT</th>
      <th>AMT_CREDIT_SUM_LIMIT</th>
      <th>AMT_CREDIT_SUM_OVERDUE</th>
      <th>CREDIT_TYPE</th>
      <th>DAYS_CREDIT_UPDATE</th>
      <th>AMT_ANNUITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>215354</td>
      <td>5714462</td>
      <td>Closed</td>
      <td>currency 1</td>
      <td>-497</td>
      <td>0</td>
      <td>-153.0</td>
      <td>-153.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>91323.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-131</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>215354</td>
      <td>5714463</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-208</td>
      <td>0</td>
      <td>1075.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>225000.0</td>
      <td>171342.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Credit card</td>
      <td>-20</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>215354</td>
      <td>5714464</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-203</td>
      <td>0</td>
      <td>528.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>464323.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-16</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>215354</td>
      <td>5714465</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-203</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>90000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Credit card</td>
      <td>-16</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>215354</td>
      <td>5714466</td>
      <td>Active</td>
      <td>currency 1</td>
      <td>-629</td>
      <td>0</td>
      <td>1197.0</td>
      <td>NaN</td>
      <td>77674.5</td>
      <td>0</td>
      <td>2700000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Consumer credit</td>
      <td>-21</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



- 고객 아이디(SK_ID_CURR)를 기준으로 groupby 실행한다.
- 이전 대출 횟수를 파악하고, Column 이름을 변경한다.



```python
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index = False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>previous_loan_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100002</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100003</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100004</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100005</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



- 훈련용 데이터프레임과 병합(Join)한다.


```python
train = pd.read_csv('application_train.csv')
train = train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')
```

- NaN 값에 0 을 대입한다.


```python
train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>NAME_TYPE_SUITE</th>
      <th>NAME_INCOME_TYPE</th>
      <th>NAME_EDUCATION_TYPE</th>
      <th>NAME_FAMILY_STATUS</th>
      <th>NAME_HOUSING_TYPE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>DAYS_REGISTRATION</th>
      <th>DAYS_ID_PUBLISH</th>
      <th>OWN_CAR_AGE</th>
      <th>FLAG_MOBIL</th>
      <th>FLAG_EMP_PHONE</th>
      <th>FLAG_WORK_PHONE</th>
      <th>FLAG_CONT_MOBILE</th>
      <th>FLAG_PHONE</th>
      <th>FLAG_EMAIL</th>
      <th>OCCUPATION_TYPE</th>
      <th>CNT_FAM_MEMBERS</th>
      <th>REGION_RATING_CLIENT</th>
      <th>REGION_RATING_CLIENT_W_CITY</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>REG_REGION_NOT_LIVE_REGION</th>
      <th>REG_REGION_NOT_WORK_REGION</th>
      <th>LIVE_REGION_NOT_WORK_REGION</th>
      <th>REG_CITY_NOT_LIVE_CITY</th>
      <th>REG_CITY_NOT_WORK_CITY</th>
      <th>LIVE_CITY_NOT_WORK_CITY</th>
      <th>...</th>
      <th>LIVINGAREA_MEDI</th>
      <th>NONLIVINGAPARTMENTS_MEDI</th>
      <th>NONLIVINGAREA_MEDI</th>
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>TOTALAREA_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
      <th>EMERGENCYSTATE_MODE</th>
      <th>OBS_30_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_30_CNT_SOCIAL_CIRCLE</th>
      <th>OBS_60_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_60_CNT_SOCIAL_CIRCLE</th>
      <th>DAYS_LAST_PHONE_CHANGE</th>
      <th>FLAG_DOCUMENT_2</th>
      <th>FLAG_DOCUMENT_3</th>
      <th>FLAG_DOCUMENT_4</th>
      <th>FLAG_DOCUMENT_5</th>
      <th>FLAG_DOCUMENT_6</th>
      <th>FLAG_DOCUMENT_7</th>
      <th>FLAG_DOCUMENT_8</th>
      <th>FLAG_DOCUMENT_9</th>
      <th>FLAG_DOCUMENT_10</th>
      <th>FLAG_DOCUMENT_11</th>
      <th>FLAG_DOCUMENT_12</th>
      <th>FLAG_DOCUMENT_13</th>
      <th>FLAG_DOCUMENT_14</th>
      <th>FLAG_DOCUMENT_15</th>
      <th>FLAG_DOCUMENT_16</th>
      <th>FLAG_DOCUMENT_17</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
      <th>previous_loan_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>351000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.018801</td>
      <td>-9461</td>
      <td>-637</td>
      <td>-3648.0</td>
      <td>-2120</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0193</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0149</td>
      <td>Stone, brick</td>
      <td>No</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>-1134.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>1129500.0</td>
      <td>Family</td>
      <td>State servant</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.003541</td>
      <td>-16765</td>
      <td>-1188</td>
      <td>-1186.0</td>
      <td>-291</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Core staff</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>MONDAY</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0558</td>
      <td>0.0039</td>
      <td>0.01</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0714</td>
      <td>Block</td>
      <td>No</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-828.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>135000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.010032</td>
      <td>-19046</td>
      <td>-225</td>
      <td>-4260.0</td>
      <td>-2531</td>
      <td>26.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-815.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>297000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>0.008019</td>
      <td>-19005</td>
      <td>-3039</td>
      <td>-9833.0</td>
      <td>-2437</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-617.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>513000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.028663</td>
      <td>-19932</td>
      <td>-3038</td>
      <td>-4311.0</td>
      <td>-3458</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Core staff</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1106.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 123 columns</p>
</div>



- 맨 오른쪽 컬럼에 새롭게 만들어진 previous_loan_counts 컬럼을 확일할 수 있다.

## R Value 를 활용한 변수 유용성 평가

**Assessing Usefulness of New Variable with r value**  
- 새롭게 생성된 Column의 변수가 유용한 지 판단하기 위해서, 우선 목표값(target)과 해당 변수간의 [피어슨 상관계수](https://ko.wikipedia.org/wiki/%ED%94%BC%EC%96%B4%EC%8A%A8_%EC%83%81%EA%B4%80_%EA%B3%84%EC%88%98)를 계산하고자 한다.
- 두 변수 사이의 선형관계(linear relationship)는 -1(완벽하게 음의 선형관계)에서부터 +1(완벽히 양의 선형관계) 사이의 값으로 표현된다.
- R Value가 변수의 유용성을 평가하기 위한 최선을 방식은 아니지만, 머신러닝 모델을 발전시키는 데 효과가 있을 지에 대한 대략적인 정보를 줄 수는 있다.
- 목표값에 대한 r-value가 커질수록, 해당 변수가 목표값에 영향을 끼칠 가능성이 높아진다. 그러므로 목표값에 대해 가장 큰 r-value의 절대값을 가지는 변수를 찾고자 한다.
- 또한 커널밀도추정그래프를 활용하여 목표값과의 상관관계를 시각적으로 살펴볼 것이다.

### 커널밀도추정그래프

**Kernal Density Estimate Plots**  
- 커널밀도추정그래프는 단일 변수의 분포를 보여준다. 히스토그램을 부드럽게 한 것으로 생각해보면 될 것이다.
- 범주형 변수의 값 차이에 따른 분포의 차이를 보기 위해, 카테고리에 따라 색을 다르게 칠하도록 하겠다. 예를 들어, target 값이 0인지 1인지에 따라 색을 다르게 칠한 previous_loan_count의 커널밀도추정그래프를 그릴 수 있다.
- 이러한 그래프는 대출을 상환한 그룹(target == 0)과 그렇지 못한 그룹(target == 1)의 분포에 있어 차이점을 보여줄 것이다.
- 이는 변수들이 머신러닝 모델과 관련성을 가지는 지를 보여줄 수 있는 지표로 활용될 수 있다.
- 원본 소스코드에 있던 df.ix 는 더 이상 지원하지 않아서 대신 df.loc를 사용한다.


```python
# 변수의 분포에 대한 그래프 target값에 따라 색을 달리하여 작성한다.
def kde_target(var_name, df):
    '''
    Args

    input:

    var_name = str, 변수가 되는 Column
    df: DataFrame, 대상 데이터 프레임

    return: None
    '''

    # 새롭게 생성된 변수와 target간의 상관계수를 계산한다.
    corr = df['TARGET'].corr(df[var_name])

    # 대출을 상환한 그룹(0)과 그렇지 않은 그룹(1)의 중간값(media)을 계산한다.
    avg_repaid = df.loc[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.loc[df['TARGET'] == 1, var_name].median()

    plt.figure(figsize = (12, 6))

    # target값에 따라 색을 달리하여 그래프 작성
    sns.kdeplot(df.loc[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.loc[df['TARGET'] == 1, var_name], label = 'TARGET == 1')

    # 그래프 라벨링
    plt.xlabel(var_name);
    plt.ylabel('Density');
    plt.title('%s Distribution' % var_name)
    plt.legend();

    # 상관계수 출력
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    
    # 중간값 출력
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)
```

- Random Forest 및 Gradient Boosting Machine에 의해 가장 중요한 변수로 판명된 EXT_SOURCE_3를 활용하여 테스트하도록 하겠다.


```python
kde_target('EXT_SOURCE_3', train)
```

    The correlation between EXT_SOURCE_3 and the TARGET is -0.1789
    Median value for loan that was not repaid = 0.3791
    Median value for loan that was repaid =     0.5460
    


![png](/images/kaggle_home_credit_default_risk/output_42_1.png)


- 새로운 변수 previous_loan_counts를 살펴보겠다.


```python
kde_target('previous_loan_counts', train)
```

    The correlation between previous_loan_counts and the TARGET is -0.0100
    Median value for loan that was not repaid = 3.0000
    Median value for loan that was repaid =     4.0000
    


![png](/images/kaggle_home_credit_default_risk/output_44_1.png)


- 이 그래프를 보면 상관계수가 너무 작고, target값에 따른 분포의 차이도 거의 없는 걸 확인할 수 있다. 이를 통해, 새롭게 생성된 변수(previous_loan_counts Distribution)가 중요하지 않음을 알 수 있다.
- 이제 bureau 데이터프레임으로부터 몇 개의 변수를 새롭게 생성해보도록 하겠다. bureau 데이터프레임의 모든 수치형 변수로부터 평균, 최소, 최대값을 가져올 예정이다.

## 수치 데이터의 대표값을 계산

**Aggregating Numeric Columns**  
- 여기서 '대표값을 계산한다'는 것은 agg를 활용하여 데이터프레임의 평균, 최대값, 최소값, 합계 등을 구하는 것으로 정의한다.
- bureau 데이터 프레임 안의 수치형 변수를 활용하기 위해, 모든 수치 데이터 Column의 대표값을 계산할 것이다.
- 이를 위해 고객 ID별로 그룹화를 수행하고, 그룹화된 데이터프레임의 대표값들을 agg를 

# Putting the Functions Together

# Feature Engineering Outcomes

# Modeling

# Results

</details>