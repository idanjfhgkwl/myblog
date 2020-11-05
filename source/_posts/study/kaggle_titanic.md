---
title: "캐글 타이타닉 분석"
#author: "JustY"
#date: '2020 11 02'
categories:
  - study
output: 
  html_document:
    keep_md: true
marp: false
---

# 사전 준비

## Kaggle 데이터 불러오기

### Kaggle API 설치


```python
!pip install kaggle
```

    Requirement already satisfied: kaggle in /usr/local/lib/python3.6/dist-packages (1.5.9)
    Requirement already satisfied: slugify in /usr/local/lib/python3.6/dist-packages (from kaggle) (0.0.1)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.24.3)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.0.1)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.8.1)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.41.1)
    Requirement already satisfied: certifi in /usr/local/lib/python3.6/dist-packages (from kaggle) (2020.6.20)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.23.0)
    Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.15.0)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.6/dist-packages (from python-slugify->kaggle) (1.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (2.10)
    

### Kaggle Token 다운로드

- 아래 코드는 Kaggle API 토큰을 업로드 하는 코드이다.


```python
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
  print('uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  
# kaggle.json을 아래 폴더로 옮긴 뒤, file을 사용할 수 있도록 권한을 부여한다. 
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```



<input type="file" id="files-dc9ba129-e04f-4e54-91b7-356b0932add7" name="files[]" multiple disabled
   style="border:none" />
<output id="result-dc9ba129-e04f-4e54-91b7-356b0932add7">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving kaggle.json to kaggle.json
    uploaded file "kaggle.json" with length 63 bytes
    

- 실제로 kaggle.json 파일이 업로드 되었는지 확인


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
MY_GOOGLE_DRIVE_PATH = 'My Drive/Colab Notebooks/python_basic/kaggle_titanic/data'
PROJECT_PATH = join(ROOT, MY_GOOGLE_DRIVE_PATH)
print(PROJECT_PATH)
```

    /content/drive
    Mounted at /content/drive
    /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_titanic/data
    


```python
%cd "{PROJECT_PATH}"
```

    /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_titanic/data
    

### kaggle competition list 불러오기

- 캐글 대회 목록 불러오기


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
    titanic                                        2030-01-01 00:00:00  Getting Started  Knowledge      17260            True  
    house-prices-advanced-regression-techniques    2030-01-01 00:00:00  Getting Started  Knowledge       4325            True  
    connectx                                       2030-01-01 00:00:00  Getting Started  Knowledge        366           False  
    nlp-getting-started                            2030-01-01 00:00:00  Getting Started  Knowledge       1130           False  
    rock-paper-scissors                            2021-02-01 23:59:00  Playground          Prizes        226           False  
    riiid-test-answer-prediction                   2021-01-07 23:59:00  Featured          $100,000       1491           False  
    nfl-big-data-bowl-2021                         2021-01-05 23:59:00  Analytics         $100,000          0           False  
    competitive-data-science-predict-future-sales  2020-12-31 23:59:00  Playground           Kudos       9392           False  
    halite-iv-playground-edition                   2020-12-31 23:59:00  Playground       Knowledge         44           False  
    predict-volcanic-eruptions-ingv-oe             2020-12-28 23:59:00  Playground            Swag        198           False  
    hashcode-drone-delivery                        2020-12-14 23:59:00  Playground       Knowledge         80           False  
    cdp-unlocking-climate-solutions                2020-12-02 23:59:00  Analytics          $91,000          0           False  
    lish-moa                                       2020-11-30 23:59:00  Research           $30,000       3454           False  
    google-football                                2020-11-30 23:59:00  Featured            $6,000        925           False  
    conways-reverse-game-of-life-2020              2020-11-30 23:59:00  Playground            Swag        132           False  
    lyft-motion-prediction-autonomous-vehicles     2020-11-25 23:59:00  Featured           $30,000        788           False  
    

### Titanic: Machine Learning from Disaster 데이터셋 불러오기

- 타이타닉 대회 데이터를 가져오는 코드이다.


```python
!kaggle competitions download -c titanic
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.9 / client 1.5.4)
    Downloading train.csv to /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_titanic/data
      0% 0.00/59.8k [00:00<?, ?B/s]
    100% 59.8k/59.8k [00:00<00:00, 7.40MB/s]
    Downloading gender_submission.csv to /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_titanic/data
      0% 0.00/3.18k [00:00<?, ?B/s]
    100% 3.18k/3.18k [00:00<00:00, 439kB/s]
    Downloading test.csv to /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_titanic/data
      0% 0.00/28.0k [00:00<?, ?B/s]
    100% 28.0k/28.0k [00:00<00:00, 3.95MB/s]
    

- 리눅스 명령어 ls는 경로(폴더) 내 모든 데이터 파일을 보여준다.


```python
!ls
```

    gender_submission.csv  test.csv  train.csv
    

## 캐글 데이터 수집 및 EDA

우선 데이터를 수집하기에 앞서서 EDA에 관한 필수 패키지를 설치하자.


```python
import pandas as pd # 데이터 가공, 변환(dplyr)
import pandas_profiling # 보고서 기능 # 아나콘다 할 때... 실습
import numpy as np # 수치 연산 & 배열, 행렬
import matplotlib as mpl # 시각화
import matplotlib.pyplot as plt # 시각화
import seaborn as sns # 시각화

from IPython.core.display import display, HTML
```

### 데이터 수집

여기에서는 우선 `test.csv` & `train.csv` 파일을 받도록 한다. 


```python
# 경로 변경 (프로젝트 파일 생성 및 다운받을 경로 이동)
MY_GOOGLE_DRIVE_PATH = 'My Drive/Colab Notebooks/python_basic/kaggle_titanic'
PROJECT_PATH = join(ROOT, MY_GOOGLE_DRIVE_PATH)
print(PROJECT_PATH)
```

    /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_titanic
    


```python
%cd "{PROJECT_PATH}"
```

    /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_titanic
    


```python
!ls
```

    data  source
    


```python
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
print("data import is done")
```

    data import is done
    

### 데이터 확인

- `Kaggle` 데이터를 불러오면 우선 확인해야 하는 것은 데이터셋의 크기다. 
  + 변수의 갯수
  + 수치형 변수 & 범주형 변수의 개수 등을 파악해야 한다.
- Point 1 - `train`데이터에서 굳이 훈련데이터와 테스트 데이터를 구분할 필요는 없다. 
  + 보통 `Kaggle`에서는 테스트 데이터를 주기적으로 업데이트 해준다.
- Point 2 - 보통 `test` 데이터의 변수의 개수가 하나 더 작다. 



```python
df_train.shape, df_test.shape
```




    ((891, 12), (418, 11))



- 그 후 `train`데이터의 `상위 5개`의 데이터만 확인한다. 


```python
display(df_train.head())
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>


- 그 다음 확인해야 하는 것은 `수치형` 변수와 `범주형` 변수를 구분한다. 
  + 먼저 `numerical_features`를 구분하자.
- 데이터의 변수가 많아서, 일단 숫자형과 문자형으로 분리한 후, EDA를 하려고 한다.
- 아래 코드는 `train`데이터에서 숫자형 변수만 추출하는 코드이다.


```python
numeric_features = df_train.select_dtypes(include = [np.number]) # 수치형 데이터
print(numeric_features.columns)
print("The total number of include numeric features are: ", len(numeric_features.columns))
```

    Index(['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'], dtype='object')
    The total number of include numeric features are:  7
    

- `numeric_features`을 제외한 나머지 변수를 추출하자. (Categorical 등)


```python
categorical_features = df_train.select_dtypes(exclude = [np.number]) # 수치형이 아닌 데이터
print(categorical_features.columns)
print("The total number of exclude numeric features are: ", len(categorical_features.columns))
```

    Index(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], dtype='object')
    The total number of exclude numeric features are:  5
    

- 우선 전체 데이터는 `891`개 변수는 `12`개로 확인했다. 
  + 그 중 수치형 변수는 7개, 문자형 변수는 5개인 것으로 확인된다. 

# 타이타닉 분석 따라하기

- [colab, 전태균님의 타이타닉 분석](https://colab.research.google.com/drive/1cqv5yD9uLHHrVFL-TGM9NPSD1ZyF4AC1)  
- 타이타닉에 탑승한 사람들의 신상정보를 활용하여, 승선한 사람들의 생존여부를 예측하는 모델을 생성할 것이다.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고,
# 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편하다.
plt.style.use('seaborn')
sns.set(font_scale = 2.1)
import missingno as msno

# warnings 무시
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
```

## 데이터셋 확인

- 대부분의 캐글 데이터들은 장 정제되어 있다. 하지만 가끔 null data가 존재한다. 이를 확인하고, 향후 수정한다.
- pandas는 파이썬에서 테이블화 된 데이터를 다루는 데 가장 최적화되어 있으며, 많이 쓰이는 라이브러리이다.
- pandas를 사용하여 Dataset의 간단한 통계적 분석부터, 복잡한 처리들을 간단한 메소드를 사용하여 해낼 수 있다.
- pandas는 파이썬으로 데이터 분석을 한다고 하면 능숙해져야 할 라이브러리이다.


```python
df_train.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



- 이 타이타닉에서 Feature는 Pclass(티켓의 클래스), Age(성별), SibSp(함께 탑승한 형제와 배우자의 수), Parch(함께 탑승한 부모, 아이의 수), Fare(탑승료) 이며, 예측하려는 target label은 Survived(생존여부) 이다.

![](data:image/png;base64,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)

- Feature Engineering은 머신러닝 알고리즘을 작동하기 위해 데이터에 대한 도메인 지식을 활용하여 특징(Feature)을 만들어내는 과정이다. [출처](https://en.wikipedia.org/wiki/Feature_engineering)
- 또는 머신러닝 모델을 위한 데이터 테이블의 컬럼(특징)을 생성하거나 선택하는 작업을 의미한다.
- 간단히 정리하면, 모델의 성능을 높이기 위해, 모델에 입력할 데이터를 만들기 위해 주어진 초기 데이터로부터 특징을 가공하고 생성하는 전체 과정을 의미한다.

### Null data check

describe() 메소드를 쓰면 각 feature가 가진 통계치들을 반환해준다.


```python
df_train.describe()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test.describe()
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>332.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>417.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1100.500000</td>
      <td>2.265550</td>
      <td>30.272590</td>
      <td>0.447368</td>
      <td>0.392344</td>
      <td>35.627188</td>
    </tr>
    <tr>
      <th>std</th>
      <td>120.810458</td>
      <td>0.841838</td>
      <td>14.181209</td>
      <td>0.896760</td>
      <td>0.981429</td>
      <td>55.907576</td>
    </tr>
    <tr>
      <th>min</th>
      <td>892.000000</td>
      <td>1.000000</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>996.250000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.895800</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1100.500000</td>
      <td>3.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1204.750000</td>
      <td>3.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1309.000000</td>
      <td>3.000000</td>
      <td>76.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



- 이 테이블을 보면 PassengerId 숫자와 다른, null data가 존재하는 열(feature)이 있는 것 같다고 하는데 공부가 필요하다.
- 이를 좀 더 보기 편하도록 그래프로 시각화해서 살펴본다.


```python
for col in df_train.columns:
  msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
  print(msg)
```

    column: PassengerId	 Percent of NaN value: 0.00%
    column:   Survived	 Percent of NaN value: 0.00%
    column:     Pclass	 Percent of NaN value: 0.00%
    column:       Name	 Percent of NaN value: 0.00%
    column:        Sex	 Percent of NaN value: 0.00%
    column:        Age	 Percent of NaN value: 19.87%
    column:      SibSp	 Percent of NaN value: 0.00%
    column:      Parch	 Percent of NaN value: 0.00%
    column:     Ticket	 Percent of NaN value: 0.00%
    column:       Fare	 Percent of NaN value: 0.00%
    column:      Cabin	 Percent of NaN value: 77.10%
    column:   Embarked	 Percent of NaN value: 0.22%
    


```python
for col in df_test.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))
    print(msg)
```

    column: PassengerId	 Percent of NaN value: 0.00%
    column:     Pclass	 Percent of NaN value: 0.00%
    column:       Name	 Percent of NaN value: 0.00%
    column:        Sex	 Percent of NaN value: 0.00%
    column:        Age	 Percent of NaN value: 20.57%
    column:      SibSp	 Percent of NaN value: 0.00%
    column:      Parch	 Percent of NaN value: 0.00%
    column:     Ticket	 Percent of NaN value: 0.00%
    column:       Fare	 Percent of NaN value: 0.24%
    column:      Cabin	 Percent of NaN value: 78.23%
    column:   Embarked	 Percent of NaN value: 0.00%
    

- Train, Test 데이터셋에서 Age(둘 다 약 20%), Cabin(둘 다 약 80%), Embarked(Train만 0.22%) null data가 존재하는 것을 볼 수 있다.
- missingno(MSNO)라는 라이브러리를 사용하면 null data의 존재를 더 쉽게 볼 수 있다.


```python
msno.matrix(df = df_train.iloc[:, :], figsize = (8, 8), color = (0.8, 0.5, 0.2))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0543ec17b8>




![png](/images/kaggle_titanic/output_56_1.png)



```python
msno.bar(df = df_train.iloc[:, :], figsize = (8, 8), color = (0.8, 0.5, 0.2))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0541623eb8>




![png](/images/kaggle_titanic/output_57_1.png)



```python
msno.bar(df = df_test.iloc[:, :], figsize = (8, 8), color = (0.8, 0.5, 0.2))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f053fdb0c50>




![png](/images/kaggle_titanic/output_58_1.png)


### Target label 확인

- target label이 어떤 분포(distribution)를 가지고 있는 지 확인해봐야 한다.
- binary classification 문제의 경우에서, 1과 0의 분포가 어떠냐에 따라 모델의 평가 방법이 달라질 수 있다.


```python
f, ax = plt.subplots(1, 2, figsize = (18, 8))

df_train['Survived'].value_counts().plot.pie(explode = [0, 0.1], autopct = '%1.1f%%', ax = ax[0], shadow = True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data = df_train, ax = ax[1])
ax[1].set_title('Count plot - Survived')

plt.show()
```


![png](/images/kaggle_titanic/output_61_0.png)


- 38.4%가 살아남았다는 걸 알 수 있다.
- target label의 분포가 제법 균일(balanced)하다.
- 불균일한 경우, 예를 들어 100 중 1이 99, 0이 1개인 경우에는 만약 모델이 모든 것을 1이라 해도 정확도가 99%가 나오게 된다. 0을 찾는 문제라면 이 모델은 원하는 결과를 줄 수 없게 된다.

## EDA 탐색적 데이터 분석

- Exploratory Data Analysis. 많은 데이터 안의 숨겨진 사실을 찾기 위해서는 적절한 시각화가 필요하다.
- 시각화 라이브러리는 matplotlib, seaborn, plotly 등이 있다. 특정 목적에 맞는 소스코드를 정리해서 필요할 때마다 참고하면 편하다.

### Pclass

- Pclass는 서수형 데이터(ordinal)이다. 카테고리이면서, 순서가 있는 데이터 타입이다.
- Pclass에 따른 생존률의 차이를 살펴보겠다. 엑셀의 피벗 차트와 유사한 작업을 하게 되는데, pandas dataframe엣는 groupby를 사용하면 쉽게 할 수 있다. 또한 pivot이라는 메소드도 있다.
- 'Pclass', "Survived'를 가져온 후, Pclass로 묶는다. 그러고 나면 각 Pclass마다 0과 1이 count가 되는데, 이를 평균 내면 각 Pclass별 생존률이 나온다.

- 아래와 같이 count()를 하면, 각 Pclass에 몇 명이 있는 지 확인할 수 있다.


```python
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = True).count()
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
      <th>Survived</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>216</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
    </tr>
    <tr>
      <th>3</th>
      <td>491</td>
    </tr>
  </tbody>
</table>
</div>



아래와 같이 sum()을 하면, 216명 중 생존한(Survived = 1) 사람의 총합을 주게 된다.


```python
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = True).sum()
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
      <th>Survived</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>136</td>
    </tr>
    <tr>
      <th>2</th>
      <td>87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>119</td>
    </tr>
  </tbody>
</table>
</div>



- pandas의 [crosstab](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.crosstab.html)을 사용하면 위 과정을 좀 더 수월하게 볼 수 있다.


```python
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins = True).style.background_gradient(cmap = 'summer_r')
```

![](data:image/png;base64,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)

- 그룹화된 객체에 mean()을 하게 되면, 각 클래스별 생존률을 얻을 수 있다.


```python
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = True).mean().sort_values(by = 'Survived', ascending = False).plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f053fbfbe80>




![png](/images/kaggle_titanic/output_75_1.png)


- Pclass가 좋을 수록(1st) 생존률이 높은 것을 확인할 수 있다.
- seaborn의 countplot을 이용하면 특정 label에 따른 개수를 확인해볼 수 있다.


```python
y_position = 1.02
f, ax = plt.subplots(1, 2, figsize = (18, 8))
df_train['Pclass'].value_counts().plot.bar(color = ['#CD7F32','#FFDF00','#D3D3D3'], ax = ax[0])
ax[0].set_title('Number of Passengers By Pclass', y = y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue = 'Survived', data = df_train, ax = ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y = y_position)
plt.show()
```


![png](/images/kaggle_titanic/output_77_0.png)


- Pclass가 높을 수록, 생존 확률이 높은 걸 확인할 수 있다? Pclass 1, 2, 3 순서대로 63%, 48%, 25% 이다.
- 생존에 Pclass가 큰 영향을 미친다고 생각해볼 수 있으며, 나중에 모델을 세울 때 이 feature를 사용하는 것이 좋을 것이라 판단할 수 있다.

### Sex

- 성별로 생존률이 어떻게 달라지는 지 확인해보겠다.
- 마찬가지로 pandas groupby와 seaborn countplot을 사용해서 시각화해본다.


```python
f, ax = plt.subplots(1, 2, figsize = (18, 8))
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index = True).mean().plot.bar(ax = ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue = 'Survived', data = df_train, ax = ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()
```


![png](/images/kaggle_titanic/output_81_0.png)


- 여자가 생존할 확률이 높은 걸 확인할 수 있다.


```python
df_train[['Sex', "Survived"]].groupby(['Sex'], as_index = False).mean().sort_values(by = "Survived", ascending = False)
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
      <th>Sex</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>0.742038</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>0.188908</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.crosstab(df_train['Sex'], df_train['Survived'], margins = True).style.background_gradient(cmap = 'summer_r')
```

![](data:image/png;base64,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)

- Pclass와 마찬가지로, Sex도 예측 모델에 쓰일 중요한 feature 임을 알 수 있다.

### Both Sex and Pclass

- Sex, Pclass 2가지에 관하여 생존이 어떻게 달라지는 지 확인해 본다.
- seaborn의 factorplot을 이용하면, 손쉽게 3개의 차원으로 이루어진 그래프를 그릴 수 있다.


```python
sns.factorplot('Pclass', 'Survived', hue = 'Sex', data = df_train,
               size = 6, aspect = 1.5)
```




    <seaborn.axisgrid.FacetGrid at 0x7f053fd72208>




![png](/images/kaggle_titanic/output_89_1.png)


- 모든 클래스에서 여자가 살 확률이 남자보다 높은 걸 알 수 있다.
- 또한 남자, 여자 상관없이 클래스가 높을 수록 살 확률이 높다.
- hue = 'Sex' 대신 col = 'Pclass'로 하면 아래와 같아진다. (column)


```python
sns.factorplot(x = 'Sex', y = "Survived", col = 'Pclass',
               data = df_train, satureation = 5,
               size = 9, aspect = 1)
```




    <seaborn.axisgrid.FacetGrid at 0x7f053fda8da0>




![png](/images/kaggle_titanic/output_91_1.png)


### Age

- Age Feature를 살펴보자.


```python
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))
print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))
print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))
```

    제일 나이 많은 탑승객 : 80.0 Years
    제일 어린 탑승객 : 0.4 Years
    탑승객 평균 나이 : 29.7 Years
    

- 생존에 따른 Age의 히스토그램을 그려보겠다.


```python
fig, ax = plt.subplots(1, 1, figsize = (9, 5))
sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax = ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax = ax)
plt.legend(['Survived == 1', 'Survived == 0'])
plt.show()
```


![png](/images/kaggle_titanic/output_96_0.png)


- 생존자 중 나이가 어린 경우가 많음을 볼 수 있다.


```python
# Age distribution withing classes
plt.figure(figsize = (8, 6))
df_train['Age'][df_train['Pclass'] == 1].plot(kind = 'kde')
df_train['Age'][df_train['Pclass'] == 2].plot(kind = 'kde')
df_train['Age'][df_train['Pclass'] == 3].plot(kind = 'kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])
```




    <matplotlib.legend.Legend at 0x7f053f7dd6a0>




![png](/images/kaggle_titanic/output_98_1.png)


- Pclass가 높을 수록 나이 많은 사람의 비중이 커진다.

- 나이대가 변하면서 생존률이 어떻게 되는 지 보려고 한다.
- 나이 범위를 점점 넓혀가며, 생존률이 어떻게 되는 지 한 번 보자.


```python
cummulate_survival_ratio = []
for i in range(1, 80):
  cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))

plt.figure(figsize = (7, 7))
plt.plot(cummulate_survival_ratio)
plt.title('Survival rate change depending on range of Age', y = 1.02)
plt.ylabel('Survival rate')
plt.xlabel('Range of Age(0~x)')
plt.show()
```


![png](/images/kaggle_titanic/output_101_0.png)


- 나이가 어릴 수록 생존률이 확실히 높은 것을 확인할 수 있다.
- Age는 중요한 Feature로 쓰일 수 있음을 확인했다.

### Pclass, Sex, Age

- Sex, Pclass, Age, Sruvived 모두에 대해서 보고 싶다면, 이를 쉽게 그려주는 seaborn의 biolinplot을 사용한다.
- x축은 우리가 나눠서 보고 싶어하는 case(Pclass, Sex)를 나타내고, y축은 보고 싶어하는 distribution(Age)이다.


```python
f, ax = plt.subplots(1, 2, figsize = (18,8))
sns.violinplot("Pclass", "Age", hue = "Survived", data = df_train, scale = 'count', split = True, ax = ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))
sns.violinplot("Sex", "Age", hue = "Survived", data = df_train, scale = 'count', split = True, ax = ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))
plt.show()
```


![png](/images/kaggle_titanic/output_105_0.png)


- 왼쪽 그림은 Pclass별로 Age의 분포가 어떻게 다른 지, 거기에 생존여부에 따라 구분한 그래프이다.
- 오른쪽 그림은 Sex, Survived에 따른 분포가 어떻게 다른 지 보여주는 그래프이다.
- Survived만 봤을 때, 모든 클래스에서 나이가 어릴 수록 생존을 많이 한 것을 볼 수 있다.
- 오른쪽 그림에서, 명확히 여자가 생존을 많이 한 것을 볼 수 있다.
- 여성과 아이를 먼저 챙겼다고 분석해볼 수 있다.

### Embarked

### Family - SibSp(형제, 자매) + Parch(부모, 자녀)

### Fare

### Cabin

### Ticket
