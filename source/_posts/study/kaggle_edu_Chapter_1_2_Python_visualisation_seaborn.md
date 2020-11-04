---
title: "Chapter_1_2_Python_visualisation_seaborn"
#author: "JustY"
#date: '2020 10 30 '
categories:
  - study
output: 
  html_document:
    keep_md: true
marp: false
---



## 1. Matplotlib & Seaborn

### 1-1. 기본 개요
`Matplotlib`: 파이썬 표준 시각화 도구. 파이썬 그래프의 기본 토대. 객체지향 프로그래밍을 지원하므로 세세하게 꾸밀 수 있다. 

`Seaborn`: 파이썬 시각화 도구의 고급 버전. `Matplotlib`에 비해 비교적 단순한 인터페이스를 제공하기 때문에 초보자도 어렵지 않게 배울 수 있다. 

### 1-2. matplotlib & Seabon 설치

설치방법은 윈도우 명령 프롬프트, MacOS, Linux 터미널에서 `pip install matplotlib`입력하면 되지만, 간혹 여러 환경에 따라 달라질 수 있으니 관련 싸이트에서 확인하기를 바란다. 
- matplotlib 설치 방법: https://matplotlib.org/users/installing.html
- seaborn 설치 방법: https://seaborn.pydata.org/installing.html



## 2. 기본적인 시각화 문법
- 시각화 문법은 아래와 같다. 

```python
import seaborn as sns
sns.name_of_graph(x, y, dataset, options)
```

우선 Sample 데이터를 불러와서 데이터를 확인해보자.

```python
import seaborn as sns
from tabulate import tabulate

sns.set()
tips = sns.load_dataset("tips")
print(tabulate(tips.head(), tablefmt="pipe", headers="keys")) # Hugo 블로그 전용
```

![](https://user-images.githubusercontent.com/72365720/97825407-e6d7b080-1d01-11eb-9a03-eec23ec6f457.png)

위 데이터는 매우 간단한 테이블일 수 있지만, 다변량의 그래프를 하나의 이미지 안에서 어떤 형태로 그래프를 작성할 것인지 선택하는 것은 쉽지 않다.

```python
sns.relplot(x="total_bill", y="tip", col="time",
            hue="sex", style="smoker", size="size",
            data=tips);
```

![](https://user-images.githubusercontent.com/72365720/97827219-f9a0b400-1d06-11eb-9188-7702f30cd9df.png)

- 소스코드에 대한 설명을 간단히 하면 아래와 같다. 
  + `relplot`은 다변량의 그래프를 작성할 때 사용한다. 
  + `col` 대신 `row`를 사용해도 된다. 여기에는 `categorical(=범주형)` 자료가 온다. (만약 값이 많으면..?)
  + `hue`는 그래프에 표현되는 색상을 의미한다.  
  + `style`은 범주형 자료를 다르게 표현할 때 사용한다. (예: 동그라미, 별표 등) 대개 범주형 데이터를 지정한다. 
  + `size` 자료의 크기를 의미한다.



## 3. Grouped barplots

```python
import seaborn as sns
sns.set_theme(style="whitegrid") # 축의 색상

# 온라인 저장소에서 예제 데이터 세트를 로드한다.
penguins = sns.load_dataset("penguins")

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=penguins, kind="bar",
    x="species", y="body_mass_g", hue="sex",
    ci="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Body mass (g)")
g.legend.set_title("")
```

![](https://user-images.githubusercontent.com/72365720/97826790-e6d9af80-1d05-11eb-9681-40ccb05055aa.png)

- 소스코드에 대한 설명을 간단히 하면 아래와 같다. 
  + `seaborn.catplot`: 하나의 수치형 변수와 하나 이상의 범주형 변수 간의 관계를 보여주는 그래프
  + `kind`: 그릴 플롯의 종류는 범주 형 좌표축 수준 플로팅 함수의 이름에 해당합니다.
  + `x, y, hue`: 긴 형식의 데이터를 그리기위한 입력입니다.
  + `ci`: 추정값 주위를 그리는 신뢰 구간의 크기입니다. "sd"인 경우 부트 스트랩을 건너 뛰고 관측 값의 표준 편차를 그립니다.
  + `palette`: 다양한 수준의 hue변수 에 사용할 색상 입니다.
  + `alpha`: ?? (그래프 투명도? 숫자가 작으면 연해진다.)
  + `height`: 전체 플롯 세로 높이 (가로도 자동으로 조정된다?)











