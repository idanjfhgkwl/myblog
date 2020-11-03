---
title: "판다스 10분 완성"
#author: "JustY"
#date: '2020 11 02'
categories:
  - pandas_10minutes
output: 
  html_document:
    keep_md: true
marp: false
---

이 소개서는 주로 신규 사용자를 대상으로 한 판다스에 대한 간략한 소개로, 더 자세한 방법은 [Cookbook](https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html)에서 볼 수 있습니다.

일반적으로 각 패캐지는 pd, np, plt라는 이름으로 불러옵니다.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## 1. 객체 생성 (Object Creation)

[데이터 구조 소개 섹션](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html)을 참조하세요.

Pandas는 값을 가지고 있는 리스트를 통해 [pandas.Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html)를 만들고, 정수로 만들어진 인덱스를 기본값으로 불러올 것입니다.

```python
s = pd.Series([1,3,5,np.nan,6,8])
```

![](https://user-images.githubusercontent.com/72365720/97833066-c3b7fb80-1d17-11eb-9236-97d18d5280da.png)

datetime 인덱스와 레이블이 있는 열을 가지고 있는 numpy 배열을 전달하여 데이터프레임을 만듭니다.

```python
dates = pd.date_range('20130101', periods=6)
```

![](https://user-images.githubusercontent.com/72365720/97833217-3fb24380-1d18-11eb-968c-16b1bf3af74d.png)

```python
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
```

![](https://user-images.githubusercontent.com/72365720/97833393-96b81880-1d18-11eb-87d7-d4456792609b.png)

Series와 같은 것으로 변환될 수 있는 객체들의 dict로 구성된 데이터프레임을 만듭니다.

```python
df2 = pd.DataFrame({'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo' })
```

![](https://user-images.githubusercontent.com/72365720/97833565-00382700-1d19-11eb-9ba0-dddb0ffea3fc.png)

데이터프레임 결과물의 열은 다양한 데이터 타입 (dtypes)으로 구성됩니다.

```python
df2.dtypes
```

![](https://user-images.githubusercontent.com/72365720/97833651-31b0f280-1d19-11eb-8f60-cdc3e6a2ce43.png)



## 2. 데이터 확인하기 (Viewing Data)

[Basic Section](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html)을 참조하세요.

데이터프레임의 가장 윗 줄과 마지막 줄을 확인하고 싶을 때에 사용하는 방법은 다음과 같습니다.

역자 주: 괄호() 안에는 숫자가 들어갈 수도 있고 안 들어갈 수도 있습니다. 숫자가 들어간다면, 윗 / 마지막 줄의 특정 줄을 불러올 수 있습니다. 숫자가 들어가지 않다면, 기본값인 5로 처리됩니다.

```python
df.head()  # 시작에서 처음 5줄 불러온다.
```

![](https://user-images.githubusercontent.com/72365720/97833976-05e23c80-1d1a-11eb-829e-d799ee5838f7.png)

```python
df.tail()  # 끝에서 마지막 5줄 불러온다.
```

![](https://user-images.githubusercontent.com/72365720/97833981-08449680-1d1a-11eb-9cf9-d60cd4fe6d1b.png)






## 출처

[데잇걸즈2](https://dataitgirls2.github.io/10minutes2pandas/)  
[pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)



