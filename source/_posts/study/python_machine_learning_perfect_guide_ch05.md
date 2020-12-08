---
title: "파이썬 머신러닝 완벽가이드 5장"
categories:
  - study
output: 
  html_document:
    keep_md: true
---

출처: 권철민, 『파이썬 머신러닝 완벽 가이드 (개정판)』, 위키북스, 2020.02, 290-376쪽

![](/images/book/K372637591_f.jpg)
[![](/images/bookstore/yes24.png)](http://www.yes24.com/Product/Goods/87044746) [![](/images/bookstore/kyobo.png)](http://www.kyobobook.co.kr/product/detailViewKor.laf?barcode=9791158391928) [![](/images/bookstore/interpark.png)](http://book.interpark.com/product/BookDisplay.do?_method=detail&sc.prdNo=328045193) [![](/images/bookstore/aladin.png)](http://www.aladin.co.kr/shop/wproduct.aspx?ItemId=229787634)

<details markdown="1">
<summary>접기/펼치기</summary>

<!--summary 아래 빈칸 공백 두고 내용을 적는공간-->

# **5장 회귀**

## **01. 회귀 소개**
+ 회귀는 현대 통계학을 이루는 큰 축
+ 회귀 분석은 유전적 특성을 연구하던 영국의 통계학자 갈톤이 수행한 연궁서 유래했다는 것이 일반론



< 회귀에 대한 예시>

> "부모의 키가 크더라도 자식의 키가 대를 이어 무한정 커지지 않으며, 부모의 키가 작더라도 대를 이어 자식의 키가 무한정 작아지지 않는다"

**즉, 회귀 분석은 이처럼 데이터 값이 평균과 같은 일정한 값으로 돌아가려는 경향을 이용한 통계학 기법이다.** 


![캡처](https://user-images.githubusercontent.com/72365693/100684179-2a831000-33bd-11eb-8955-12676993f76b.JPG)
[출처 : 인프런]

ㅡ> X는 피처값(속성)

ㅡ> Y는 결정값

회귀는 회귀 계수의 선형/비선형 여부,
독립변수의 개수, 종속변수의 개수에 따라 여러 가지 유형으로 나눌 수 있다. **회귀에서 가장 중요한 것은 바로 회귀 계수이다** 이 회귀 계수가 선형이냐 아니냐에 따라 선형 회귀와 비선형 회귀로 나눌 수 있다. 그리고 독립변수의 개수가 한 개인지 여러 개인지에 따라 단일 회귀, 다중 회귀로 나뉘게 된다.

|독립변수 개수|회귀계수의 결합|
|-|-|
|1개:단일 회귀|선형:선형 회귀|
|여러 개:다중 회귀|비선형:비선형 회귀|
    <회귀 유형 구분>


**지도학습**은 두 가지 유형으로 나뉘는데, 바로 **분류**와 **회귀**이다. 이 두 가지 기법의 가장 큰 차이는 분류는 예측값이 카테고리와 같은 **이산형 클래스** 값이고 회귀는 **연속형 숫자** 값이라는 것입니다.

![kiki](https://user-images.githubusercontent.com/72365693/100416826-c44b7400-30c2-11eb-918b-a8f812ae44ff.jpg)

여러 가지 회귀 중에서 선형 회귀가 가장 많이 사용된다. 선형 회귀는 실제값과 예측값의 차이(오류의 제곱 값)를 최소화하는 직선형 회귀선을 최적화하는 방식이다. 선형 회귀 모델은 규제 방법에 따라 다시 별도의유형으로 나뉠 수 있다. 규제는 일반적인 선형 회귀의 과적합 문제를 해결하기 위해서 회귀 계수에 패널티값을 적용하는 것을 말한다.

**- 대표적인 선형 회귀모델**

+ **일반 선형 회귀**:예측값과 실제값의 RSS(Residual Sum of Squares)를 최소화할 수 있도록 회귀 계수를 최적화하며 규제(Regularization)  


+ **릿지(Ridge)**: 릿지 회귀는 선형 회귀에 L2 규제를 추가한 회귀 모델, 릿지 회귀는 L2 규제를 적용하는데 L2 규제는 상대적으로 큰 회귀 계수 값의 예측 영향도를 감소시키기 위해서 회귀 계수값을 더 작게 만드는 규제 모델


+ **라쏘(Lasso)**:라쏘 회귀는 선형 회귀에 L1 규제를 적용한 방식, L2 규제가 회귀 계수 값의 크기를 줄이는데 반해 L1규제는 예측 영향력이 작은 피처의 회귀 계수를 0으로 만들어 회귀 예측시 피처가 선택되지 않게 함 → 이러한 특성 때문에 L1 규제는 피처 선택 가능으로도 불림


+ **엘라스틱넷(ElasticNet)**:L2,L1 규제를 함께 결합한 모델, 주로 피처가 많은 데이터 세트에서 적용되며 L1규제로 피처의 개수를 줄임과 동시에 L2 규제로 계수 값의 크기를 조정


+ **로지스틱 회귀(Logistic Regression)**:로지스틱 회귀는 회귀라는 이름이 붙어 있지만 사실은 분류에 사용되는 선형 모델, 로지스틱 회귀는 매우 강력한 분류 알고리즘이며 일반적으로 이진 분류뿐만 아니라 희소 영역의 분류, 예를 들어 텍스트 분류와 같은 영역에서 뛰어난 예측성능을 보임

## **02. 단순선형 회귀를 통한 회귀 이해**

단순선형회귀는 독립변수도 하나 종속변수도 하나인 선형 회귀이다.

**- 예시**


> 주택가격이 주택의 크기로만 결정된다고 할때 일반적으로 주택의 크기가 크면 가격이 높아지는 경향이 있기 때문에 주택가격은 크기에 대해 선형(직선형태)의 관계로 표현할 수 있다.

![1](https://user-images.githubusercontent.com/72365693/100685144-29eb7900-33bf-11eb-8d2d-3a9e7a9f7b26.JPG)
[출처: 인프라]  

<br>

**- 오류합 계산 방법**
1. 절대값을 취하여 더하는 방식
2. 오류값의 제곱을 구해서 더하는 방식(RSS)

일반적으로 미분 등의 계산을 편리하게 하기 위해서 RSS방식으로 오류합을 구한다
즉, Error^2 = RSS
+ RSS는 이제 변수가 w0,w1인 식으로 표현 할 수 있으며 RSS를 최소로 하는 w0,w1 즉 회귀 계수를 학습을 통해서 찾는 것이 머신러닝 기반 회귀의 핵심 사항이다.
+ RSS는 회귀식의 독립변수 X, 종속변수 Y가 중심 변수가 아니라 w 변수(회귀계수)가 중심 변수임을 인지하는 것이 매우 중요(학습 데이터로 입력되는 독립변수와 종속변수는 RSS에서 모두 상수로 간주한다.)
![2](https://user-images.githubusercontent.com/72365693/100685628-14c31a00-33c0-11eb-8a6a-6d93ddfbc99e.JPG)

[출처 : 인프라]

## **03. 비용 최소화하기 - 경사 하강법(Gradient Descent)**

W 파라미터의 개수가 적다면 고차원 방정식으로 비용 함수가 최소가 되는 W 변숫값을 도출할 수 있겠지만 W 파라미터가 많으면 고차원 방정식을 동원하더라도 해결하기가 어렵다. 경사 하강법은 이러한 고차원 방정식에 대한 문제를 해결해 주며서 비용 함수 RSS를 최소화하는 방법을 직관적으로 제공하는 뛰어난 방식이다.

![캡처](https://user-images.githubusercontent.com/72365693/100686489-c7e04300-33c1-11eb-905b-bea79e742997.JPG)

[출처 : 인프라]  


+ 경사 하강법은 반복적으로 비용 함수의 반환 값, 즉 예측값과 실제값의 차이가 작아지는 **방향성**을 가지고 W 파라미터를 지속해서 보정해 나간다.
+ 최초 오류 값이 100이었다면 두 번째 오류 값은 100보다 작은 90, 세 번째는 80과 같은 방식으로 지속해서 오류를 감소시키는 방향으로 W 값을 계속 업데이트해 나간다.
+ 그리고 오류 값이 더 이상 작아지지 않으면 그 오류 값을 최소 비용으로 판단하고 그때의 W 값을 **최적 파라미터로 반환**한다.

> 경사 하강법의 핵심: "어떻게 하면 오류가 작아지는 방향으로 W 값을 보정할 수 있을까?"  

![캡처](https://user-images.githubusercontent.com/72365693/100687679-0f67ce80-33c4-11eb-96c4-fbbeae25b93f.JPG)
[출처 : 인프런]
![캡처](https://user-images.githubusercontent.com/72365693/100690620-42ad5c00-33ca-11eb-9752-5ab4db677acb.JPG)

[출처 : 인프런]  



**- 경사 하강법 수행 프로세스**
![캡처](https://user-images.githubusercontent.com/72365693/100690834-b8b1c300-33ca-11eb-9523-45b4d9f438e7.JPG)

**- 실제값을 Y=4X+6 시뮬레이션하는 데이터 값 생성**



```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(0)
# y = 4X + 6 식을 근사(w1=4, w0=6). random 값은 Noise를 위해 만듬
X = 2 * np.random.rand(100,1) # X는 100개의 랜덤값을 만든다.
# np.random.randn는 노이즈값 이걸 사용하지 않으면 계속 1차 함수로 만들어냄 결국 퍼져보이기 위하여 사용
y = 6 +4 * X+ np.random.randn(100,1) 

# X, y 데이터 셋 산점도로 시각화
plt.scatter(X, y)
```




    <matplotlib.collections.PathCollection at 0x2370c9b30d0>




    
![png](/images/python_machine_learning_perfect_guide_ch05/output_4_1.png)
    



```python
X.shape, y.shape # 100개의 데이터를 다 가지고 있다는 것을 알 수 있다.
```




    ((100, 1), (100, 1))



**- w0과 w1의 값을 최소화 할 수 있도록 업데이트 수행하는 함수 생성.**

+ 예측 배열 y_pred는 np.dot(X, w1.T) + w0 임 100개의 데이터 X(1,2,...,100)이 있다면 예측값은 w0 + X(1)w1 + X(2)w1 +..+ X(100)*w1이며, 이는 입력 배열 X와 w1 배열의 내적임.

+ 새로운 w1과 w0를 update함
![캡처](https://user-images.githubusercontent.com/72365693/100700036-d3dafd80-33df-11eb-8883-864619664f52.JPG)


```python
# w1 과 w0 를 업데이트 할 w1_update, w0_update를 반환. 
def get_weight_updates(w1, w0, X, y, learning_rate=0.01):
    N = len(y)
    # 먼저 w1_update, w0_update를 각각 w1, w0의 shape와 동일한 크기를 가진 0 값으로 초기화
    w1_update = np.zeros_like(w1)
    w0_update = np.zeros_like(w0)
    # 예측 배열 계산하고 예측과 실제 값의 차이 계산
    y_pred = np.dot(X, w1.T) + w0
    diff = y-y_pred
         
    # w0_update를 dot 행렬 연산으로 구하기 위해 모두 1값을 가진 행렬 생성 
    w0_factors = np.ones((N,1))

    # w1과 w0을 업데이트할 w1_update와 w0_update 계산
    w1_update = -(2/N)*learning_rate*(np.dot(X.T, diff))
    w0_update = -(2/N)*learning_rate*(np.dot(w0_factors.T, diff))    
    
    return w1_update, w0_update
```


```python
w0 = np.zeros((1,1))
w1 = np.zeros((1,1))
y_pred = np.dot(X, w1.T) + w0
diff = y-y_pred
print(diff.shape)
w0_factors = np.ones((100,1))
w1_update = -(2/100)*0.01*(np.dot(X.T, diff))
w0_update = -(2/100)*0.01*(np.dot(w0_factors.T, diff))   
print(w1_update.shape, w0_update.shape)
w1, w0
```

    (100, 1)
    (1, 1) (1, 1)
    




    (array([[0.]]), array([[0.]]))



반복적으로 경사 하강법을 이용하여 get_weigth_updates()를 호출하여 w1과 w0를 업데이트 하는 함수 생성


```python
# 입력 인자 iters로 주어진 횟수만큼 반복적으로 w1과 w0를 업데이트 적용함. 
def gradient_descent_steps(X, y, iters=10000):
    # w0와 w1을 모두 0으로 초기화. 
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    
    # 인자로 주어진 iters 만큼 반복적으로 get_weight_updates() 호출하여 w1, w0 업데이트 수행. 
    for ind in range(iters):
        w1_update, w0_update = get_weight_updates(w1, w0, X, y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
              
    return w1, w0
```

**- 예측 오차 비용을 계산을 수행하는 함수 생성 및 경사하강법 수행**


```python
def get_cost(y, y_pred):
    N = len(y) 
    cost = np.sum(np.square(y - y_pred))/N
    return cost

w1, w0 = gradient_descent_steps(X, y, iters=1000)
print("w1:{0:.3f} w0:{1:.3f}".format(w1[0,0], w0[0,0]))
y_pred = w1[0,0] * X + w0
print('Gradient Descent Total Cost:{0:.4f}'.format(get_cost(y, y_pred)))
```

    w1:4.022 w0:6.162
    Gradient Descent Total Cost:0.9935
    


```python
plt.scatter(X, y)
plt.plot(X,y_pred)
```




    [<matplotlib.lines.Line2D at 0x2370ca26040>]




    
![png](/images/python_machine_learning_perfect_guide_ch05/output_13_1.png)
    


**- 미니 배치 확률적 경사 하강법을 이용한 최적 비용함수 도출**


```python
def stochastic_gradient_descent_steps(X, y, batch_size=10, iters=1000):
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    prev_cost = 100000
    iter_index =0
    
    for ind in range(iters):
        np.random.seed(ind)
        # 전체 X, y 데이터에서 랜덤하게 batch_size만큼 데이터 추출하여 sample_X, sample_y로 저장
        stochastic_random_index = np.random.permutation(X.shape[0]) # 임의로 추출
        sample_X = X[stochastic_random_index[0:batch_size]]
        sample_y = y[stochastic_random_index[0:batch_size]]
        # 랜덤하게 batch_size만큼 추출된 데이터 기반으로 w1_update, w0_update 계산 후 업데이트
        w1_update, w0_update = get_weight_updates(w1, w0, sample_X, sample_y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
    
    return w1, w0
```


```python
np.random.permutation(X.shape[0]) # 랜덤 샘플링을 임의로 가져옴
```




    array([66, 71, 54, 88, 82, 12, 36, 46, 14, 67, 10,  3, 62, 29, 97, 69, 70,
           93, 31, 73, 60, 96, 28, 27, 21, 19, 33, 78, 32, 94,  1, 41, 40, 76,
           37, 87, 24, 23, 50,  2, 47, 20, 77, 17, 56, 64, 68, 25, 15, 22, 16,
           98, 63, 92, 86, 38,  6, 57, 95, 44,  9, 42, 81, 99, 35, 84, 59, 48,
           75, 65, 85, 90, 55, 43, 58, 89, 30, 80, 34, 18, 51, 49, 52, 74, 26,
           45, 39,  4, 11, 53, 91, 79,  8,  0,  5, 13, 61, 72,  7, 83])




```python
w1, w0 = stochastic_gradient_descent_steps(X, y, iters=1000)
print("w1:",round(w1[0,0],3),"w0:",round(w0[0,0],3))
y_pred = w1[0,0] * X + w0
print('Stochastic Gradient Descent Total Cost:{0:.4f}'.format(get_cost(y, y_pred)))
```

    w1: 4.028 w0: 6.156
    Stochastic Gradient Descent Total Cost:0.9937
    

## **04. 사이킷런 LinearRegression 클래스**



```
class sklearn.linear_model.LinearRegression(fit_intercept=True,normalize=False, copy_X=True, n_jobs=1)
```

LnearRegression 클래스는 예측값과 실제 값의 RSS를 최소화해 OLS 추정 방식으로 구현한 클래스이다.

LinearRegression 클래스는 fit()메서드로 X,Y 배열을 입력 받으면 회귀 계수인 W를 coef_ 속성에 저장한다.

![캡처](https://user-images.githubusercontent.com/72365693/100702704-85c8f880-33e5-11eb-8141-aa89c3bb3348.JPG)
[출처 :인프런]

### **(1) 회귀평가 지표**
![캡처](https://user-images.githubusercontent.com/72365693/100703039-1dc6e200-33e6-11eb-8202-a1e23139f09c.JPG)



**- 사이킷런 회귀 평가 API**
+ 사이킷런은 아쉽게도 RMSE를 제공하지 않는다. RMSE를 구하기 위해서는 MSE에 제곱근을 씌워서 계산하는 함수를 직접 만들어야 한다.
+ 다음은 각 평가 방법에 대한 사이킷런의 API 및 cross_val_score나 GridSearchCV에서 평가 시 사용되는 scoriong 파라미터의 적용 값이다
![캡처](https://user-images.githubusercontent.com/72365693/100703523-0b00dd00-33e7-11eb-8180-2ba8b60aaa21.JPG)

### **(2) LinearRegression을 이용한 보스턴 주택 가격 예측**



```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_boston
%matplotlib inline

# boston 데이타셋 로드
boston = load_boston()

# boston 데이타셋 DataFrame 변환 
bostonDF = pd.DataFrame(boston.data , columns = boston.feature_names)

# boston dataset의 target array는 주택 가격임. 이를 PRICE 컬럼으로 DataFrame에 추가함. 
bostonDF['PRICE'] = boston.target
print('Boston 데이타셋 크기 :',bostonDF.shape)
bostonDF.head()
```

    Boston 데이타셋 크기 : (506, 14)
    




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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



+ CRIM: 지역별 범죄 발생률
+ ZN: 25,000평방피트를 초과하는 거주 지역의 비율
+ NDUS: 비상업 지역 넓이 비율
+ CHAS: 찰스강에 대한 더미 변수(강의 경계에 위치한 경우는 1, 아니면 0)
+ NOX: 일산화질소 농도
+ RM: 거주할 수 있는 방 개수
+ AGE: 1940년 이전에 건축된 소유 주택의 비율
+ DIS: 5개 주요 고용센터까지의 가중 거리
+ RAD: 고속도로 접근 용이도
+ TAX: 10,000달러당 재산세율
+ PTRATIO: 지역의 교사와 학생 수 비율
+ B: 지역의 흑인 거주 비율
+ LSTAT: 하위 계층의 비율
+ MEDV: 본인 소유의 주택 가격(중앙값)

**- 각 컬럼별로 주택가격에 미치는 영향도 조사**


```python
# 2개의 행과 4개의 열을 가진 subplots를 이용. axs는 4x2개의 ax를 가짐.
fig, axs = plt.subplots(figsize=(16,8) , ncols=4 , nrows=2)
lm_features = ['RM','ZN','INDUS','NOX','AGE','PTRATIO','LSTAT','RAD']
for i , feature in enumerate(lm_features):
    row = int(i/4)
    col = i%4
    # 시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
    sns.regplot(x=feature , y='PRICE',data=bostonDF , ax=axs[row][col])
```


    
![png](/images/python_machine_learning_perfect_guide_ch05/output_24_0.png)
    


- 결과 해석
  - 다른 칼럼보다 RM과 LSTAT의 PRICE 영향도가 가장 두드러지게 나타남
  - RM(방 개수)은 양방향의 선형성이 가장 크다  
  → 방의 크기가 클수록 가격이 증가함  

  - LSTAT는 음방향의 선형성이 가장 큼  
  → LSTAT이 적을수록 PRICE가 증가함
  
  
**- LinearRegression 클래스로 보스턴 주택 가격의 회귀 모델 만들기**


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'],axis=1,inplace=False)

X_train , X_test , y_train , y_test = train_test_split(X_data , y_target ,test_size=0.3, random_state=156)

# 선형회귀 OLS로 학습/예측/평가 수행. 
lr = LinearRegression()
lr.fit(X_train ,y_train )
y_preds = lr.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)

print('MSE : {0:.3f} , RMSE : {1:.3F}'.format(mse , rmse))
print('Variance score : {0:.3f}'.format(r2_score(y_test, y_preds)))
```

    MSE : 17.297 , RMSE : 4.159
    Variance score : 0.757
    


```python
# 절편과 회귀 계수 값 보기
print('절편 값:',lr.intercept_)
print('회귀 계수값:', np.round(lr.coef_, 1))
```

    절편 값: 40.995595172164336
    회귀 계수값: [ -0.1   0.1   0.    3.  -19.8   3.4   0.   -1.7   0.4  -0.   -0.9   0.
      -0.6]
    


```python
# coef_ 속성은 회귀 계수 값만 가지고 있어, 이를 피처별 회귀 계수 값으로 재 매핑
# 회귀 계수를 큰 값 순으로 정렬하기 위해 Series로 생성. index가 컬럼명에 유의
coeff = pd.Series(data=np.round(lr.coef_, 1), index=X_data.columns )
coeff.sort_values(ascending=False)
```




    RM          3.4
    CHAS        3.0
    RAD         0.4
    ZN          0.1
    B           0.0
    TAX        -0.0
    AGE         0.0
    INDUS       0.0
    CRIM       -0.1
    LSTAT      -0.6
    PTRATIO    -0.9
    DIS        -1.7
    NOX       -19.8
    dtype: float64



- 결과 해석
  - RM이 양의 값으로 회귀 계수가 가장 큼
  - NOX 피처의 회귀 계수 - 값이 너무 커보임
  → 최적화를 수행하면서 피처 codfficients의 변화 살필 예정  


**- 5개의 폴드 세트에서 cross_val_score()로 교차 검증하기: MSE, RMSE 측정**  
사이킷런은 cross_val_score()를 이용하는데, RMSE를 제공하지 않으므로 MSE 수치 결과를 RMSE로 변환해야 한다. cross_val_score()의 인자로 scoring='neg_mean_squared_error'를 지칭하면 반환되는 수치 값은 음수이다. 사이킷런은 높은 지표 값일수록 좋은 모델로 평가히는데 반해, **회귀는 MSE 값이 낮을수록 좋은 회귀 모델**로 평가한다.


```python
from sklearn.model_selection import cross_val_score

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'],axis=1,inplace=False)
lr = LinearRegression()

# cross_val_score( )로 5 Fold 셋으로 MSE 를 구한 뒤 이를 기반으로 다시  RMSE 구함. 
neg_mse_scores = cross_val_score(lr, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores  = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

# cross_val_score(scoring="neg_mean_squared_error")로 반환된 값은 모두 음수 
print(' 5 folds 의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 2))
print(' 5 folds 의 개별 RMSE scores : ', np.round(rmse_scores, 2))
print(' 5 folds 의 평균 RMSE : {0:.3f} '.format(avg_rmse))

```

     5 folds 의 개별 Negative MSE scores:  [-12.46 -26.05 -33.07 -80.76 -33.31]
     5 folds 의 개별 RMSE scores :  [3.53 5.1  5.75 8.99 5.77]
     5 folds 의 평균 RMSE : 5.829 
    

- 결과 해석
  - 5개 폴드 세트 교차 검증 수행 결과, 평균 RMSE는 약 5.836이 나옴
  - corss_val_score(scoring="neg_mean_squared_error")로 반환된 값은 모두 음수로 확인됨

## 05. 다항 회귀, 과(대)적합/과소적합
### (1) 다항 회귀 이해
- 현재까지 설명한 회귀는 y = $w_{0} + w_{1}*x_{1} + w_{2}*x_{2} + , ... , + w_{n}*x_{n}$과 같이 독립변수(feature)와 종속변수(target) 관계가 일차 방정식 형태로 표현된 회귀


- 세상의 모든 관계를 직선으로만 표현할 수 없기 때문에 다항 회귀 개념이 필요


- **다항(Polynomial) 회귀**: 회귀가 독립변수의 단항식이 아닌 2차, 3차 방정식과 같은 다항식으로 표현되는 것
  - y = $w_{0} + w_{1}*x_{1} + w_{2}*x_{2} + w_{3}*x_{1}*x_{2} + w_{4}*x_{1}^{2} + w_{5}*x_{2}^{2}$
  
  
- 다항 회귀는 선형 회귀(비선형 회귀가 아님)
  - cf) 회귀에서 선형/비선형을 나누는 기준은 회귀 계수가 선형/비선형인지에 따른 것으로 독립변수의 선형/비선형 여부와는 무관함
  

- 사이킷런은 다항 회귀를 위한 클래스를 명시적으로 제공하지 않아, 비선형 함수를 선형 모델에 적용시키는 방법으로 구현
  - PolynomialFeatures 클래스로 피처를 다항식 피처로 변환함


```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# 다항식으로 변환한 단항식 생성, [[0,1],[2,3]]의 2X2 행렬 생성
X = np.arange(4).reshape(2,2)
print('일차 단항식 계수 feature:\n',X )

# degree = 2 인 2차 다항식으로 변환하기 위해 PolynomialFeatures를 이용하여 변환
poly = PolynomialFeatures(degree=2)
poly.fit(X)
poly_ftr = poly.transform(X)
print('변환된 2차 다항식 계수 feature:\n', poly_ftr)
```

    일차 단항식 계수 feature:
     [[0 1]
     [2 3]]
    변환된 2차 다항식 계수 feature:
     [[1. 0. 1. 0. 0. 1.]
     [1. 2. 3. 4. 6. 9.]]
    

**- 3차 다항 회귀 함수를 임의로 설정하고 회귀 계수 예측하기**


```python
def polynomial_func(X):
    y = 1 + 2*X[:,0] + 3*X[:,0]**2 + 4*X[:,1]**3
    print(X[:, 0])
    print(X[:, 1])
    return y

X = np.arange(0,4).reshape(2,2)

print('일차 단항식 계수 feature: \n' ,X)
y = polynomial_func(X)
print('삼차 다항식 결정값: \n', y)
```

    일차 단항식 계수 feature: 
     [[0 1]
     [2 3]]
    [0 2]
    [1 3]
    삼차 다항식 결정값: 
     [  5 125]
    

**- 일차 단항식 계수를 삼차 다항식 계수로 변환하고, 선형 회귀에 적용**


```python
# 3 차 다항식 변환 
poly_ftr = PolynomialFeatures(degree=3).fit_transform(X)
print('3차 다항식 계수 feature: \n',poly_ftr)

# Linear Regression에 3차 다항식 계수 feature와 3차 다항식 결정값으로 학습 후 회귀 계수 확인
model = LinearRegression()
model.fit(poly_ftr,y)
print('Polynomial 회귀 계수\n' , np.round(model.coef_, 2))
print('Polynomial 회귀 Shape :', model.coef_.shape)
```

    3차 다항식 계수 feature: 
     [[ 1.  0.  1.  0.  0.  1.  0.  0.  0.  1.]
     [ 1.  2.  3.  4.  6.  9.  8. 12. 18. 27.]]
    Polynomial 회귀 계수
     [0.   0.18 0.18 0.36 0.54 0.72 0.72 1.08 1.62 2.34]
    Polynomial 회귀 Shape : (10,)
    

- 결과 해석
  - 일차 단항식 계수 피처는 2개였지만, 3차 다항식 Polynomial 변환 이후에는 다항식 계수 피처가 10개로 늘어남
  - 늘어난 피처 데이터 세트에 LinearRegression을 통해 3차 다항 회귀 형태의 다항 회귀를 적용하면 회귀 계수가 10개로 늘어남
  - 10개의 회귀 계수가 도출됐으며 원래 다항식 계수 값과는 차이가 있지만, 다항 회귀로 근사함을 알 수 있음
  

**- 사이킷런의 Pipeline 객체를 이용해 한 번에 다항 회귀 구현하기**


```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

def polynomial_func(X):
    y = 1 + 2*X[:,0] + 3*X[:,0]**2 + 4*X[:,1]**3 
    return y

# Pipeline 객체로 Streamline 하게 Polynomial Feature변환과 Linear Regression을 연결
model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression())])
X = np.arange(4).reshape(2,2)
y = polynomial_func(X)

model = model.fit(X, y)
print('Polynomial 회귀 계수\n', np.round(model.named_steps['linear'].coef_, 2))
```

    Polynomial 회귀 계수
     [0.   0.18 0.18 0.36 0.54 0.72 0.72 1.08 1.62 2.34]
    

### **(2) 다항 회귀를 이용한 과소적합 및 과적합 이해**
- 다항 회귀는 피처의 직선적 관계가 아닌, 복잡한 다항 관계를 모델링할 수 있음
- 다항식 차수가 높아질수록 매우 복잡한 피처 관계까지 모델링이 가능함
- 단, 다항 회귀의 차수(degree)를 높일수록 학습 데이터에 너무 맞춘 학습이 이루어져서 테스트 데이터 환경에서 예측 정확도가 떨어짐  
→ 차수가 높아질수록 과적합 문제가 크게 발생

**- 다항 회귀의 과소적합과 과적합 문제를 잘 보여주는 예시**  
[- 원본](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py)

- 소스코드 설명
  - 피처 X와 target y가 잡음(Noise)이 포함된 다항식의 코사인 그래프 관계를 가지게 만듦
  - 이에 기반해 다항 회귀의 차수를 변화시키며 그에 따른 회귀 예측 곡선과 예측 정확도를 비교하는 예제
  - 학습 데이터: 30개의 임의 데이터 X, X의 코사인 값에서 약간의 잡음 변동 값을 더한 target 


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
%matplotlib inline

# random 값으로 구성된 X값에 대해 Cosine 변환값을 반환. 
def true_fun(X):
    return np.cos(1.5 * np.pi * X)

# X는 0 부터 1까지 30개의 random 값을 순서대로 sampling 한 데이타 입니다.  
np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))

# y 값은 cosine 기반의 true_fun() 에서 약간의 Noise 변동값을 더한 값입니다. 
y = true_fun(X) + np.random.randn(n_samples) * 0.1
```

**- 예측 결과를 비교할 다항식 차수를 각각 1, 4, 15로 변경하며 예측 결과 비교하기**
  1. 다항식 차수별로 학습 수행 후, cross_val_score()로 MSE 값을 구해 차수별 예측 성능 평가
  2. 0부터 1까지 균일하게 구성된 100개의 테스트용 데이터 세트로 차수별 회귀 예측 곡선 그리기


```python
plt.figure(figsize=(14, 5))
degrees = [1, 4, 15]

# 다항 회귀의 차수(degree)를 1, 4, 15로 각각 변화시키면서 비교합니다. 
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())
    
    # 개별 degree별로 Polynomial 변환합니다. 
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X.reshape(-1, 1), y)
    
    # 교차 검증으로 다항 회귀를 평가합니다. 
    scores = cross_val_score(pipeline, X.reshape(-1,1), y,scoring="neg_mean_squared_error", cv=10)
    coefficients = pipeline.named_steps['linear_regression'].coef_
    print('\nDegree {0} 회귀 계수는 {1} 입니다.'.format(degrees[i], np.round(coefficients),2))
    print('Degree {0} MSE 는 {1:.2f} 입니다.'.format(degrees[i] , -1*np.mean(scores)))
    
    # 0 부터 1까지 테스트 데이터 세트를 100개로 나눠 예측을 수행합니다. 
    # 테스트 데이터 세트에 회귀 예측을 수행하고 예측 곡선과 실제 곡선을 그려서 비교합니다.  
    X_test = np.linspace(0, 1, 100)
    # 예측값 곡선
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model") 
    # 실제 값 곡선
    plt.plot(X_test, true_fun(X_test), '--', label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    
    plt.xlabel("x"); plt.ylabel("y"); plt.xlim((0, 1)); plt.ylim((-2, 2)); plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(degrees[i], -scores.mean(), scores.std()))

plt.show()
```

    
    Degree 1 회귀 계수는 [-2.] 입니다.
    Degree 1 MSE 는 0.41 입니다.
    
    Degree 4 회귀 계수는 [  0. -18.  24.  -7.] 입니다.
    Degree 4 MSE 는 0.04 입니다.
    
    Degree 15 회귀 계수는 [-2.98300000e+03  1.03900000e+05 -1.87417100e+06  2.03717220e+07
     -1.44873987e+08  7.09318780e+08 -2.47066977e+09  6.24564048e+09
     -1.15677067e+10  1.56895696e+10 -1.54006776e+10  1.06457788e+10
     -4.91379977e+09  1.35920330e+09 -1.70381654e+08] 입니다.
    Degree 15 MSE 는 182815433.48 입니다.
    


    
![png](/images/python_machine_learning_perfect_guide_ch05/output_43_1.png)
    


- 실선: 다항 회귀 예측 곡선
- 점선: 실제 데이터 세트 X, Y의 코사인 곡선
- 학습 데이터: 0부터 1까지 30개의 임의 X값과 그에 따른 코사인 Y값에 잡음을 변동 값으로 추가해 구성
- MSE(Mean Squared Error) 평가: 학습 데이터를 10개의 교차 검증 세트로 나누어 측정한 후 평균한 것


- 결과 해석 
  - Degree 1 예측 곡선
    - 단순한 직선으로 단순 선형 회귀와 동일
    - 실제 데이터 세트인 코사인 데이터 세트를 직선으로 예측하기에는 너무 단순함
    - 예측 곡선이 학습 데이터 패턴을 반영하지 못하는 **과소적합 모델**
    - MSE값: 약 0.407
      
  - Degree 4 예측 곡선
    - 실제 데이터 세트와 유사한 모습
    - 변동하는 잡음까지는 예측하지 못했지만, 학습 데이터 세트를 비교적 잘 반영해 코사인 곡선 기반으로 테스트 데이터를 잘 예측한 곡선을 가진 모델
    - MSE값: 0.043 (가장 뛰어난 예측 성능)
   
  - Degree 15 예측 곡선
    - MSE값이 182815432이 될 정도로 이상한 오류 값 발생(과적합 강조를 위해 만든 예측 곡선)
    - 데이터 세트의 변동 잡음까지 지나치게 반영한 결과, 예측 곡선이 학습 데이터 세트만 정확히 예측하고 테스트 값의 실제 곡선과는 완전히 다른 형태의 예측 곡선이 만들어짐
    - 학습 데이터에 너무 충실하게 맞춘 심한 **과적합 모델**
  

- 결론
  - 좋은 예측 모델은 학습 데이터 패턴을 잘 반영하면서도 복잡하지 않은, **균형 잡힌(Balanced) 모델**을 의미
  

### **(3) 편향-분산 트레이드 오프**
- 머신러닝이 극복해야 할 이슈
 
- **고편향(High Bias)성**
  - Degree 1 모델처럼 매우 단순화된 모델로서 지나치게 한 방향성으로 치우친 경향을 보임

- **고분산(High Variance)성**
  - Degree 15 모델처럼 학습 데이터 하나하나 특성을 반영하여 매우 복잡하고 지나치게 높은 변동성을 가짐

![image.png](attachment:image.png)
[- 이미지 출처](http://scott.fortmann-roe.com/docs/BiasVariance.html)

- 일반적으로 편향과 분산은 한 쪽이 높으면, 한 쪽이 낮아지는 경향이 있음
→ 편향이 높으면 분산이 낮아지고(과소적합), 분산이 높으면 편향이 낮아짐(과적합)


- **편향과 분산 관계에 따른 전체 오류 값(Total Error) 변화**
![image.png](attachment:image.png)
[- 이미지 출처](http://scott.fortmann-roe.com/docs/BiasVariance.html)  


<br>

- 편향이 너무 높으면 전체 오류가 높음
- 편향을 낮출수록 분산이 높아지고 전체 오류도 낮아짐
- **골디락스**: 편향을 낮추고 분산을 높이며, 전체 오류가 가장 낮아지는 지점
- 골디락스 지점을 통과하며 분산을 지속적으로 높이면 전체 오류 값이 오히려 증가하며 예측 성능이 다시 저하됨


- 정리
  - 과소적합: 높은 편향/낮은 분산에서 일어나기 쉬움
  - 과적합: 낮은 편향/높은 분산에서 일어나기 쉬움
  → 편향과 분산이 트레이드 오프를 이루며 오류 cost 값이 최대로 낮아지는 모델을 구축하는 것이 중요

## **06. 규제 선형 모델: 릿지, 라쏘, 엘라스틱넷**
### **(1) 규제 선형 모델의 개요**

- 이전까지의 선형 모델 비용 함수는 RSS를 최소화하는(실제값과 예측값의 차이를 최소화하는) 것만 고려
  - 학습 데이터에 지나치게 맞추게 되고, 회귀 계수가 쉽게 커지는 문제가 발생
  - 변동성이 오히려 심해져 테스트 데이터 세트에서 예측 성능이 저하되게 쉬움
  → 비용 함수는 학습 데이터의 잔차 오류값을 최소로 하는 RSS 최소화 방법과 과적합을 방지하기 위해 회귀 계수 값이 커지지 않도록 하는 방법이 균형을 이루어야 함


- 회귀 계수의 크기를 제어해 과적합을 개선하려면, 비용(Cost) 함수의 목표가 아래와 같이 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>M</mi>
  <mi>i</mi>
  <mi>n</mi>
  <mo stretchy="false">(</mo>
  <mi>R</mi>
  <mi>S</mi>
  <mi>S</mi>
  <mo stretchy="false">(</mo>
  <mi>W</mi>
  <mo stretchy="false">)</mo>
  <mo>+</mo>
  <mi>a</mi>
  <mi>l</mi>
  <mi>p</mi>
  <mi>h</mi>
  <mi>a</mi>
  <mo>&#x2217;<!-- ∗ --></mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mi>W</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <msubsup>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <mn>2</mn>
    <mn>2</mn>
  </msubsup>
  <mo stretchy="false">)</mo>
</math>를 최소화하는 것으로 변경될 수 있음

**비용 함수 목표 = <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>M</mi>
  <mi>i</mi>
  <mi>n</mi>
  <mo stretchy="false">(</mo>
  <mi>R</mi>
  <mi>S</mi>
  <mi>S</mi>
  <mo stretchy="false">(</mo>
  <mi>W</mi>
  <mo stretchy="false">)</mo>
  <mo>+</mo>
  <mi>a</mi>
  <mi>l</mi>
  <mi>p</mi>
  <mi>h</mi>
  <mi>a</mi>
  <mo>&#x2217;<!-- ∗ --></mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mi>W</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <msubsup>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <mn>2</mn>
    <mn>2</mn>
  </msubsup>
  <mo stretchy="false">)</mo>
</math>**

- alpha: 학습 데이터 적합 정도와 회귀 계수 값의 크기 제어를 수행하는 튜닝 파라미터
  - alpha가 0(또는 매우 작은 값)이라면 비용 함수 식은 기존과 동일한 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>M</mi>
  <mi>i</mi>
  <mi>n</mi>
  <mo stretchy="false">(</mo>
  <mi>R</mi>
  <mi>S</mi>
  <mi>S</mi>
  <mo stretchy="false">(</mo>
  <mi>W</mi>
  <mo stretchy="false">)</mo>
  <mo>+</mo>
  <mn>0</mn>
  <mo stretchy="false">)</mo>
</math>가 됨
  - alpha가 무한대(또는 매우 큰 값)이라면 비용 함수 식은 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>R</mi>
  <mi>S</mi>
  <mi>S</mi>
  <mo stretchy="false">(</mo>
  <mi>W</mi>
  <mo stretchy="false">)</mo>
</math>에 비해 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>a</mi>
  <mi>l</mi>
  <mi>p</mi>
  <mi>h</mi>
  <mi>a</mi>
  <mo>&#x2217;<!-- ∗ --></mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mi>W</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <msubsup>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <mn>2</mn>
    <mn>2</mn>
  </msubsup>
</math> 값이 너무 커지므로 W 값을 0 또는 매우 작게 만들어야 Cost가 최소화 되는 비용 함수 목표를 달성할 수 있음  
→ alpha 값을 크게 하면 비용 함수는 회귀 계수 W의 값을 작게 해 과적합을 개선할 수 있으며 alpha 값을 작게 하면 회귀 계수 W의 값이 커져도 어느 정도 상쇄가 가능하므로 학습 데이터 적합을 더 개선할 수 있음


- alpha를 0에서부터 지속적으로 값을 증가시키면 회귀 계수 값의 크기를 감소시킬 수 있음
  - **규제(Regularization)**: 비용 함수에 alpha 값으로 패널티를 부여해 회귀 계수 값의 크기를 감소해 과적합을 개선하는 방식
  - 규제 방식
    - L2 규제: <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>a</mi>
  <mi>l</mi>
  <mi>p</mi>
  <mi>h</mi>
  <mi>a</mi>
  <mo>&#x2217;<!-- ∗ --></mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mi>W</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <msubsup>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <mn>2</mn>
    <mn>2</mn>
  </msubsup>
</math>와 같이 W 제곱에 패널티를 부여하는 방식 ← 릿지(Ridge) 회귀  

    - L1 규제: <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>a</mi>
  <mi>l</mi>
  <mi>p</mi>
  <mi>h</mi>
  <mi>a</mi>
  <mo>&#x2217;<!-- ∗ --></mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mi>W</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <msub>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <mn>1</mn>
  </msub>
</math>와 같이 W의 절대값에 패널티를 부여, 영향력이 크지 않은 회귀 계수 값을 0으로 변환 ← 라쏘(Lasso) 회귀

### **(2) 릿지 회귀**
- 사이킷런은 Ridge 클래스로 릿지 회귀를 구현
  - 주요 생성 파라미터: alpha, 릿지 회귀의 alpha L2 규제 계수에 해당
  

**- 보스턴 주택 가격을 Ridge 클래스로 예측하고, cross_val_score()로 평가하기**


```python
# 앞의 LinearRegression예제에서 분할한 feature 데이터 셋인 X_data과 Target 데이터 셋인 Y_target 데이터셋을 그대로 이용 
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# boston 데이타셋 로드
boston = load_boston()

# boston 데이타셋 DataFrame 변환 
bostonDF = pd.DataFrame(boston.data , columns = boston.feature_names)

# boston dataset의 target array는 주택 가격임. 이를 PRICE 컬럼으로 DataFrame에 추가함. 
bostonDF['PRICE'] = boston.target
print('Boston 데이타셋 크기 :',bostonDF.shape)

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'],axis=1,inplace=False)


ridge = Ridge(alpha = 10)
neg_mse_scores = cross_val_score(ridge, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores  = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)
print(' 5 folds 의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 3))
print(' 5 folds 의 개별 RMSE scores : ', np.round(rmse_scores,3))
print(' 5 folds 의 평균 RMSE : {0:.3f} '.format(avg_rmse))
```

    Boston 데이타셋 크기 : (506, 14)
     5 folds 의 개별 Negative MSE scores:  [-11.422 -24.294 -28.144 -74.599 -28.517]
     5 folds 의 개별 RMSE scores :  [3.38  4.929 5.305 8.637 5.34 ]
     5 folds 의 평균 RMSE : 5.518 
    

- 결과 해석
  - 릿지의 5개 폴드 세트 평균 RMSE: 5.524
  - 앞 예제(규제 없는 LinearRegression) 평균인 5.836보다 뛰어난 예측 성능을 보임
  

**- 릿지의 alpha 값을 0, 0.1, 1, 10, 100으로 변화시키며 RMSE와 회귀 계수 값 변화 살펴보기**


```python
# Ridge에 사용될 alpha 파라미터의 값들을 정의
alphas = [0 , 0.1 , 1 , 10 , 100]

# alphas list 값을 iteration하면서 alpha에 따른 평균 rmse 구함.
for alpha in alphas :
    ridge = Ridge(alpha = alpha)
    
    #cross_val_score를 이용하여 5 fold의 평균 RMSE 계산
    neg_mse_scores = cross_val_score(ridge, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
    avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
    print('alpha {0} 일 때 5 folds 의 평균 RMSE : {1:.3f} '.format(alpha,avg_rmse))
```

    alpha 0 일 때 5 folds 의 평균 RMSE : 5.829 
    alpha 0.1 일 때 5 folds 의 평균 RMSE : 5.788 
    alpha 1 일 때 5 folds 의 평균 RMSE : 5.653 
    alpha 10 일 때 5 folds 의 평균 RMSE : 5.518 
    alpha 100 일 때 5 folds 의 평균 RMSE : 5.330 
    

- 결과 해석
  - alpha가 100일 때, 평균 RMSE가 5.332로 가장 좋음
  

**- alpha 값 변화에 따른 피처의 회귀 계수 값을 가로 막대 그래프로 시각화**


```python
# 각 alpha에 따른 회귀 계수 값을 시각화하기 위해 5개의 열로 된 맷플롯립 축 생성  
fig , axs = plt.subplots(figsize=(18,6) , nrows=1 , ncols=5)
# 각 alpha에 따른 회귀 계수 값을 데이터로 저장하기 위한 DataFrame 생성  
coeff_df = pd.DataFrame()

# alphas 리스트 값을 차례로 입력해 회귀 계수 값 시각화 및 데이터 저장. pos는 axis의 위치 지정
for pos , alpha in enumerate(alphas) :
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_data , y_target)
    # alpha에 따른 피처별 회귀 계수를 Series로 변환하고 이를 DataFrame의 컬럼으로 추가.  
    coeff = pd.Series(data=ridge.coef_ , index=X_data.columns )
    colname='alpha:'+str(alpha)
    coeff_df[colname] = coeff
    # 막대 그래프로 각 alpha 값에서의 회귀 계수를 시각화. 회귀 계수값이 높은 순으로 표현
    coeff = coeff.sort_values(ascending=False)
    axs[pos].set_title(colname)
    axs[pos].set_xlim(-3,6)
    sns.barplot(x=coeff.values , y=coeff.index, ax=axs[pos])

# for 문 바깥에서 맷플롯립의 show 호출 및 alpha에 따른 피처별 회귀 계수를 DataFrame으로 표시
plt.show()
```


    
![png](/images/python_machine_learning_perfect_guide_ch05/output_53_0.png)
    


- 결과 해석
  - alpha 값을 계속 증가시킬수록 회귀 계수 값은 지속적으로 작아짐
  - 특히, NOX 피처의 경우 alpha 값을 계속 증가시킴에 따라 회귀 계수가 크게 작아지고 있음
  

**- DataFrame에 저장된 alpha 값 변화에 따른 릿지 회귀 계수 값 구하기**


```python
ridge_alphas = [0 , 0.1 , 1 , 10 , 100]
sort_column = 'alpha:'+str(ridge_alphas[0])
coeff_df.sort_values(by=sort_column, ascending=False)
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
      <th>alpha:0</th>
      <th>alpha:0.1</th>
      <th>alpha:1</th>
      <th>alpha:10</th>
      <th>alpha:100</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RM</th>
      <td>3.809865</td>
      <td>3.818233</td>
      <td>3.854000</td>
      <td>3.702272</td>
      <td>2.334536</td>
    </tr>
    <tr>
      <th>CHAS</th>
      <td>2.686734</td>
      <td>2.670019</td>
      <td>2.552393</td>
      <td>1.952021</td>
      <td>0.638335</td>
    </tr>
    <tr>
      <th>RAD</th>
      <td>0.306049</td>
      <td>0.303515</td>
      <td>0.290142</td>
      <td>0.279596</td>
      <td>0.315358</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>0.046420</td>
      <td>0.046572</td>
      <td>0.047443</td>
      <td>0.049579</td>
      <td>0.054496</td>
    </tr>
    <tr>
      <th>INDUS</th>
      <td>0.020559</td>
      <td>0.015999</td>
      <td>-0.008805</td>
      <td>-0.042962</td>
      <td>-0.052826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.009312</td>
      <td>0.009368</td>
      <td>0.009673</td>
      <td>0.010037</td>
      <td>0.009393</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>0.000692</td>
      <td>-0.000269</td>
      <td>-0.005415</td>
      <td>-0.010707</td>
      <td>0.001212</td>
    </tr>
    <tr>
      <th>TAX</th>
      <td>-0.012335</td>
      <td>-0.012421</td>
      <td>-0.012912</td>
      <td>-0.013993</td>
      <td>-0.015856</td>
    </tr>
    <tr>
      <th>CRIM</th>
      <td>-0.108011</td>
      <td>-0.107474</td>
      <td>-0.104595</td>
      <td>-0.101435</td>
      <td>-0.102202</td>
    </tr>
    <tr>
      <th>LSTAT</th>
      <td>-0.524758</td>
      <td>-0.525966</td>
      <td>-0.533343</td>
      <td>-0.559366</td>
      <td>-0.660764</td>
    </tr>
    <tr>
      <th>PTRATIO</th>
      <td>-0.952747</td>
      <td>-0.940759</td>
      <td>-0.876074</td>
      <td>-0.797945</td>
      <td>-0.829218</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>-1.475567</td>
      <td>-1.459626</td>
      <td>-1.372654</td>
      <td>-1.248808</td>
      <td>-1.153390</td>
    </tr>
    <tr>
      <th>NOX</th>
      <td>-17.766611</td>
      <td>-16.684645</td>
      <td>-10.777015</td>
      <td>-2.371619</td>
      <td>-0.262847</td>
    </tr>
  </tbody>
</table>
</div>



- 결과 해석
  - alpha 값이 증가하며 회귀 계소가 지속적으로 작아짐
  - 단, **릿지 회귀는 회귀 계수를 0으로 만들지 않음**

### **(3) 라쏘 회귀**
- 라쏘 회귀: W의 절댓값에 패널티를 부여하는 L1 규제를 선형 회귀에 적용한 것
  - L1 규제는 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>a</mi>
  <mi>l</mi>
  <mi>p</mi>
  <mi>h</mi>
  <mi>a</mi>
  <mo>&#x2217;<!-- ∗ --></mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mi>W</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <msub>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <mn>1</mn>
  </msub>
</math>를 의미하며, 라쏘 회귀 비용함수 목표는 RSS(W) + <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>a</mi>
  <mi>l</mi>
  <mi>p</mi>
  <mi>h</mi>
  <mi>a</mi>
  <mo>&#x2217;<!-- ∗ --></mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mi>W</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <msub>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <mn>1</mn>
  </msub>
</math> 식을 최소화하는 W를 찾는 것
  - L2 규제가 회귀 계쑤 크기를 감소시키는 데 반해, L1 규제는 불필요한 회귀 계수를 급격히 감소시켜 0으로 만들고 제거함
  - L1 규제는 적절한 피처만 회귀에 포함시키는 피처 선택의 득성을 가짐
  

- 사이킷런은 Lasso 클래스로 라쏘 회귀를 구현
  - 주요 파라미터: alpha, 라쏘 회귀의 alpha L1 규제 계수에 해당


**- Lasso 클래스로 라쏘의 alpha 값을 변화시키며 RMSE와 각 피처의 회귀 계수 출력하기**


```python
from sklearn.linear_model import Lasso, ElasticNet

# alpha값에 따른 회귀 모델의 폴드 평균 RMSE를 출력하고 회귀 계수값들을 DataFrame으로 반환 
def get_linear_reg_eval(model_name, params=None, X_data_n=None, y_target_n=None, 
                        verbose=True, return_coeff=True):
    coeff_df = pd.DataFrame()
    if verbose : print('####### ', model_name , '#######')
    for param in params:
        if model_name =='Ridge': model = Ridge(alpha=param)
        elif model_name =='Lasso': model = Lasso(alpha=param)
        elif model_name =='ElasticNet': model = ElasticNet(alpha=param, l1_ratio=0.7)
        neg_mse_scores = cross_val_score(model, X_data_n, 
                                             y_target_n, scoring="neg_mean_squared_error", cv = 5)
        avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
        print('alpha {0}일 때 5 폴드 세트의 평균 RMSE: {1:.3f} '.format(param, avg_rmse))
        # cross_val_score는 evaluation metric만 반환하므로 모델을 다시 학습하여 회귀 계수 추출
        
        model.fit(X_data_n , y_target_n)
        if return_coeff:
            # alpha에 따른 피처별 회귀 계수를 Series로 변환하고 이를 DataFrame의 컬럼으로 추가. 
            coeff = pd.Series(data=model.coef_ , index=X_data_n.columns )
            colname='alpha:'+str(param)
            coeff_df[colname] = coeff
    
    return coeff_df
# end of get_linear_regre_eval
```


```python
# 라쏘에 사용될 alpha 파라미터의 값들을 정의하고 get_linear_reg_eval() 함수 호출
lasso_alphas = [ 0.07, 0.1, 0.5, 1, 3]
coeff_lasso_df =get_linear_reg_eval('Lasso', params=lasso_alphas, X_data_n=X_data, y_target_n=y_target)
```

    #######  Lasso #######
    alpha 0.07일 때 5 폴드 세트의 평균 RMSE: 5.612 
    alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 5.615 
    alpha 0.5일 때 5 폴드 세트의 평균 RMSE: 5.669 
    alpha 1일 때 5 폴드 세트의 평균 RMSE: 5.776 
    alpha 3일 때 5 폴드 세트의 평균 RMSE: 6.189 
    

- 결과 해석
  - alpha가 0.07일 때, 가장 좋은 평균 RMSE를 보여줌
  

**- alpha 값에 따른 피처별 회귀 계수**


```python
# 반환된 coeff_lasso_df를 첫번째 컬럼순으로 내림차순 정렬하여 회귀계수 DataFrame출력
sort_column = 'alpha:'+str(lasso_alphas[0])
coeff_lasso_df.sort_values(by=sort_column, ascending=False)
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
      <th>alpha:0.07</th>
      <th>alpha:0.1</th>
      <th>alpha:0.5</th>
      <th>alpha:1</th>
      <th>alpha:3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RM</th>
      <td>3.789725</td>
      <td>3.703202</td>
      <td>2.498212</td>
      <td>0.949811</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>CHAS</th>
      <td>1.434343</td>
      <td>0.955190</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>RAD</th>
      <td>0.270936</td>
      <td>0.274707</td>
      <td>0.277451</td>
      <td>0.264206</td>
      <td>0.061864</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>0.049059</td>
      <td>0.049211</td>
      <td>0.049544</td>
      <td>0.049165</td>
      <td>0.037231</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.010248</td>
      <td>0.010249</td>
      <td>0.009469</td>
      <td>0.008247</td>
      <td>0.006510</td>
    </tr>
    <tr>
      <th>NOX</th>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>-0.011706</td>
      <td>-0.010037</td>
      <td>0.003604</td>
      <td>0.020910</td>
      <td>0.042495</td>
    </tr>
    <tr>
      <th>TAX</th>
      <td>-0.014290</td>
      <td>-0.014570</td>
      <td>-0.015442</td>
      <td>-0.015212</td>
      <td>-0.008602</td>
    </tr>
    <tr>
      <th>INDUS</th>
      <td>-0.042120</td>
      <td>-0.036619</td>
      <td>-0.005253</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>CRIM</th>
      <td>-0.098193</td>
      <td>-0.097894</td>
      <td>-0.083289</td>
      <td>-0.063437</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>LSTAT</th>
      <td>-0.560431</td>
      <td>-0.568769</td>
      <td>-0.656290</td>
      <td>-0.761115</td>
      <td>-0.807679</td>
    </tr>
    <tr>
      <th>PTRATIO</th>
      <td>-0.765107</td>
      <td>-0.770654</td>
      <td>-0.758752</td>
      <td>-0.722966</td>
      <td>-0.265072</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>-1.176583</td>
      <td>-1.160538</td>
      <td>-0.936605</td>
      <td>-0.668790</td>
      <td>-0.000000</td>
    </tr>
  </tbody>
</table>
</div>



- 결과 해석
  - alpha의 크기가 증가함에 따라 일부 피처 회귀 계수는 아예 0으로 바뀜
  - NOX 속성은 alpha가 0.07일 때부터 회귀 계수가 0이며, alpha를 증가시키며 INDUS, CHAS와 같은 속성 회귀 계수가 0으로 바뀜
  - 회귀 계수가 0인 피처는 회귀 식에서 제외되며 피처 선택의 효과를 얻을 수 있음

### **(4) 엘라스틱넷 회귀**
- 엘라스틱넷(Elastic Net) 회귀: L2 규제와 L1 규제를 결합한 회귀
- 엘라스틱넷 회귀 비용함수 목표: <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>R</mi>
  <mi>S</mi>
  <mi>S</mi>
  <mo stretchy="false">(</mo>
  <mi>W</mi>
  <mo stretchy="false">)</mo>
  <mo>+</mo>
  <mi>a</mi>
  <mi>l</mi>
  <mi>p</mi>
  <mi>h</mi>
  <mi>a</mi>
  <mn>2</mn>
  <mo>&#x2217;<!-- ∗ --></mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mi>W</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <msubsup>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <mn>2</mn>
    <mn>2</mn>
  </msubsup>
  <mo>+</mo>
  <mi>a</mi>
  <mi>l</mi>
  <mi>p</mi>
  <mi>h</mi>
  <mi>a</mi>
  <mn>1</mn>
  <mo>&#x2217;<!-- ∗ --></mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mi>W</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <msub>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <mn>1</mn>
  </msub>
</math> 식을 최소화하는 W를 찾는 것


- 엘라스틱넷은 라쏘 회귀가 상관관계가 높은 피처들의 경우에, 중요 피처만을 선택하고 다른 피처 회귀 계수는 0으로 만드는 성향이 강함
  - alpha 값에 따라 회귀 계쑤 값이 급격히 변동할 수 있는데, 엘라스틱넷 회귀는 이를 완화하기 위해 L2 규제를 라쏘 회귀에 추가한 것
  - 엘라스틱넷 회귀의 단점은 L1과 L2 규제가 결합된 규제로 인해 수행 시간이 상대적으로 오래 걸림


- 사이킷런은 Elastic Net 클래스로 엘라스틱넷 회귀를 구현
  - 주요 파라미터: aplha, l1_ration
  - Elastic Net 클래스의 aplha는 Ridge와 Lasso 클래스의 alpha 값과는 다름
  - 엘라스틱넷 규제는 a * L1 + b * L2로 정의될 수 있으며, 이 때 a는 L1 규제의 alpha값, b는 L2 규제의 alpha 값
  - 따라서 ElasticNet 클래스의 alpha 파라미터 값은 a + b 값
  - ElasticNet 클래스의 l1_ratio 파라미터 값은 a / (a + b)
  - l1_ratio가 0이면 a가 0이므로 L2 규제와 동일하고, l1_ratio가 1이면 b가 0이므로 L1 규제와 동일
  
  
**- Elastic Net 클래스로 엘라스틱넷 alpha 값을 변화시키며 RMSE와 각 피처의 회귀 계수 출력하기**
- l1_ratio를 0.7로 고정한 이유: 단순히 alpha 값의 변화만 살피기 위해


```python
# 엘라스틱넷에 사용될 alpha 파라미터의 값들을 정의하고 get_linear_reg_eval() 함수 호출
# l1_ratio는 0.7로 고정
elastic_alphas = [ 0.07, 0.1, 0.5, 1, 3]
coeff_elastic_df =get_linear_reg_eval('ElasticNet', params=elastic_alphas,
                                      X_data_n=X_data, y_target_n=y_target)
```

    #######  ElasticNet #######
    alpha 0.07일 때 5 폴드 세트의 평균 RMSE: 5.542 
    alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 5.526 
    alpha 0.5일 때 5 폴드 세트의 평균 RMSE: 5.467 
    alpha 1일 때 5 폴드 세트의 평균 RMSE: 5.597 
    alpha 3일 때 5 폴드 세트의 평균 RMSE: 6.068 
    


```python
# 반환된 coeff_elastic_df를 첫번째 컬럼순으로 내림차순 정렬하여 회귀계수 DataFrame출력
sort_column = 'alpha:'+str(elastic_alphas[0])
coeff_elastic_df.sort_values(by=sort_column, ascending=False)
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
      <th>alpha:0.07</th>
      <th>alpha:0.1</th>
      <th>alpha:0.5</th>
      <th>alpha:1</th>
      <th>alpha:3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RM</th>
      <td>3.574162</td>
      <td>3.414154</td>
      <td>1.918419</td>
      <td>0.938789</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>CHAS</th>
      <td>1.330724</td>
      <td>0.979706</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>RAD</th>
      <td>0.278880</td>
      <td>0.283443</td>
      <td>0.300761</td>
      <td>0.289299</td>
      <td>0.146846</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>0.050107</td>
      <td>0.050617</td>
      <td>0.052878</td>
      <td>0.052136</td>
      <td>0.038268</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.010122</td>
      <td>0.010067</td>
      <td>0.009114</td>
      <td>0.008320</td>
      <td>0.007020</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>-0.010116</td>
      <td>-0.008276</td>
      <td>0.007760</td>
      <td>0.020348</td>
      <td>0.043446</td>
    </tr>
    <tr>
      <th>TAX</th>
      <td>-0.014522</td>
      <td>-0.014814</td>
      <td>-0.016046</td>
      <td>-0.016218</td>
      <td>-0.011417</td>
    </tr>
    <tr>
      <th>INDUS</th>
      <td>-0.044855</td>
      <td>-0.042719</td>
      <td>-0.023252</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>CRIM</th>
      <td>-0.099468</td>
      <td>-0.099213</td>
      <td>-0.089070</td>
      <td>-0.073577</td>
      <td>-0.019058</td>
    </tr>
    <tr>
      <th>NOX</th>
      <td>-0.175072</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>LSTAT</th>
      <td>-0.574822</td>
      <td>-0.587702</td>
      <td>-0.693861</td>
      <td>-0.760457</td>
      <td>-0.800368</td>
    </tr>
    <tr>
      <th>PTRATIO</th>
      <td>-0.779498</td>
      <td>-0.784725</td>
      <td>-0.790969</td>
      <td>-0.738672</td>
      <td>-0.423065</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>-1.189438</td>
      <td>-1.173647</td>
      <td>-0.975902</td>
      <td>-0.725174</td>
      <td>-0.031208</td>
    </tr>
  </tbody>
</table>
</div>



- 결과 해석
  - alpha 0.5일 때, RMSE가 5.468로 가장 좋은 예측 성능을 보임
  - alpha 값에 따른 피처들의 회귀 계수 값이 라쏘보다는 0 되는 값이 적음

### **(5) 선형 회귀 모델을 위한 데이터 변환**
- 선형 회귀 모델과 같은 선형 모델은 일반적으로 피처와 타겟 간에 선형의 관계가 있다 가정하고, 이러한 최적의 선형함수를 찾아내 결과를 예측
- 선형 회귀 모델은 피처값과 타겟값의 분포가 정규 분포(즉 평균을 중심으로 종 모양으로 데이터 값이 분포된 형태) 형태를 매우 선호함
  - 타겟값의 경우 정규 분포 형태가 아니라 특정값의 분포가 치우친 왜곡된 형태의 분포도일 경우 예측 성능에 부정적인 영향을 미칠 가능성이 높음
  - 피처값 역시 왜곡된 분포도로 인해 예측 성능에 부정적인 영향을 미칠 수 있음
  
  
- 일반적으로 선형 회귀 모델을 적용하기전에 데이터에 대한 스케일링/정규화 작업을 수행함
  - 단, 스케일링/정규화 작업을 선행한다고 해서 무조건 예측 성능이 향상되는 것은 아니며 중요한 피처들이나 타겟값의 분포도가 심하게 왜곡됐을 경우에 이러한 변환 작업을 수행함
  - 피처 데이터 셋과 타겟 데이터 셋에 이러한 스케일링/정규화 작업을 수행하는 방법이 다름  


**- 사이킷런을 이용해 피처 데이터 세트에 적용하는 방법 세 가지**
1. StandardScaler 클래스를 이용해 평균이 0, 분산이 1인 표준 정규 분포를 가진 데이터 셋으로 변환하거나 MinMaxScaler 클래스를 이용해 최소값이 0이고 최대값이 1인 값으로 정규화를 수행

2. 스케일링/정규화를 수행한 데이터 셋에 다시 다항 특성을 적용하여 변환하는 방법이다. 보통 1번 방법을 통해 예측 성능에 향상이 없을 경우 이와 같은 방법을 적용

3. 원래 값에 log 함수를 적용하면 보다 정규 분포에 가까운 형태로 값이 분포(= 로그 변환)된다. 실제로 선형 회귀에서는 앞서 소개한 1,2번 방법보다 로그 변환이 훨씬 많이 사용되는 변환 방법(1번 방법: 예측 성능 향상을 크게 기대하기 어려운 경우가 많음, 2번 방법: 피처 개수가 매우 많을 경우에는 다항 변환으로 생성되는 피처의 개수가 기하급수로 늘어나서 과적합의 이슈가 발생할 수 있음)


- 타겟값의 경우 일반적으로 로그 변환을 적용
  - 결정값을 정규 분포나 다른 정규값으로 변환하면 변환된 값을 다시 원본 타겟값으로 원복하기 어려울 수 있음
  - 왜곡된 분포도 형태의 타겟값을 로그 변환하여 예측 성능 향상이 된 경우가 많은 사례에서 검증되었기 때문에 타겟값의 경우는 로그 변환을 적용
  

**- 보스턴 주택가격 피처 데이터 세트에 표준 정규 분포 변환, 최댓값/최솟값 정규화, 로그 변환을 적용한 후 RMSE로 각 경우별 예측 성능 측정하기**
- 사용 함수: get_scaled_data()
  - method 인자로 변환 방법을 결정하며, 표준 정규 분포 변환(Standard), 최댓값/최솟값 정규와(MinMax), 로그 변환(Log) 중에 하나를 선택
  - p_degree: 다항식 특성을 추가할 때, 다항식 차수가 입력됨 (2를 넘기지 않음)
  - np.log1p(): log() 함수만 적용하면 언더 플로우가 발생하기 쉬워 1 + log() 함수를 적용


```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

# method는 표준 정규 분포 변환(Standard), 최대값/최소값 정규화(MinMax), 로그변환(Log) 결정
# p_degree는 다향식 특성을 추가할 때 적용. p_degree는 2이상 부여하지 않음. 
def get_scaled_data(method='None', p_degree=None, input_data=None):
    if method == 'Standard':
        scaled_data = StandardScaler().fit_transform(input_data)
    elif method == 'MinMax':
        scaled_data = MinMaxScaler().fit_transform(input_data)
    elif method == 'Log':
        scaled_data = np.log1p(input_data)
    else:
        scaled_data = input_data

    if p_degree != None:
        scaled_data = PolynomialFeatures(degree=p_degree, 
                                         include_bias=False).fit_transform(scaled_data)
    
    return scaled_data
```


```python
# Ridge의 alpha값을 다르게 적용하고 다양한 데이터 변환방법에 따른 RMSE 추출. 
alphas = [0.1, 1, 10, 100]
#변환 방법은 모두 6개, 원본 그대로, 표준정규분포, 표준정규분포+다항식 특성
# 최대/최소 정규화, 최대/최소 정규화+다항식 특성, 로그변환 
scale_methods=[(None, None), ('Standard', None), ('Standard', 2), 
               ('MinMax', None), ('MinMax', 2), ('Log', None)]
for scale_method in scale_methods:
    X_data_scaled = get_scaled_data(method=scale_method[0], p_degree=scale_method[1], 
                                    input_data=X_data)
    print(X_data_scaled.shape, X_data.shape)
    print('\n## 변환 유형:{0}, Polynomial Degree:{1}'.format(scale_method[0], scale_method[1]))
    get_linear_reg_eval('Ridge', params=alphas, X_data_n=X_data_scaled, 
                        y_target_n=y_target, verbose=False, return_coeff=False)
```

    (506, 13) (506, 13)
    
    ## 변환 유형:None, Polynomial Degree:None
    alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 5.788 
    alpha 1일 때 5 폴드 세트의 평균 RMSE: 5.653 
    alpha 10일 때 5 폴드 세트의 평균 RMSE: 5.518 
    alpha 100일 때 5 폴드 세트의 평균 RMSE: 5.330 
    (506, 13) (506, 13)
    
    ## 변환 유형:Standard, Polynomial Degree:None
    alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 5.826 
    alpha 1일 때 5 폴드 세트의 평균 RMSE: 5.803 
    alpha 10일 때 5 폴드 세트의 평균 RMSE: 5.637 
    alpha 100일 때 5 폴드 세트의 평균 RMSE: 5.421 
    (506, 104) (506, 13)
    
    ## 변환 유형:Standard, Polynomial Degree:2
    alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 8.827 
    alpha 1일 때 5 폴드 세트의 평균 RMSE: 6.871 
    alpha 10일 때 5 폴드 세트의 평균 RMSE: 5.485 
    alpha 100일 때 5 폴드 세트의 평균 RMSE: 4.634 
    (506, 13) (506, 13)
    
    ## 변환 유형:MinMax, Polynomial Degree:None
    alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 5.764 
    alpha 1일 때 5 폴드 세트의 평균 RMSE: 5.465 
    alpha 10일 때 5 폴드 세트의 평균 RMSE: 5.754 
    alpha 100일 때 5 폴드 세트의 평균 RMSE: 7.635 
    (506, 104) (506, 13)
    
    ## 변환 유형:MinMax, Polynomial Degree:2
    alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 5.298 
    alpha 1일 때 5 폴드 세트의 평균 RMSE: 4.323 
    alpha 10일 때 5 폴드 세트의 평균 RMSE: 5.185 
    alpha 100일 때 5 폴드 세트의 평균 RMSE: 6.538 
    (506, 13) (506, 13)
    
    ## 변환 유형:Log, Polynomial Degree:None
    alpha 0.1일 때 5 폴드 세트의 평균 RMSE: 4.770 
    alpha 1일 때 5 폴드 세트의 평균 RMSE: 4.676 
    alpha 10일 때 5 폴드 세트의 평균 RMSE: 4.836 
    alpha 100일 때 5 폴드 세트의 평균 RMSE: 6.241 
    

- 결과 해석
  - 표준 정규 분포와 최솟값/최댓값 정규화로 피처 데이터 세트를 변경해도 성능상의 개선은 없음
  - 표준 정규 분포로 일차 변환 후 2차 다항식 변환 시, alpha = 100에서 4.631로 성능 개선
  - 최솟값/최댓값 정규화로 일차 변환 후 2차 다항식 변환 시, aplha = 1에서 4.320으로 성능 개선
  - 단, 다항식 변환은 피처 개수가 많을 경우 적용하기 힘들며, 데이터 건수가 많아지면 시간이 많이 소모되어 적용하기에 한계가 있음
  - 반면, 로그 변환은 alpha가 0.1, 1, 10인 경우 모두 성능이 좋게 향상됨


- 일반적으로 선형 회귀를 적용하려는 데이터 세트에, 데이터 값 분포가 심하게 왜곡되어 있을 경우에, 로그 변환을 적용하는 편이 더 좋은 결과를 기대할 수 있음

## **07. 로지스틱 회귀**
- 로지스틱 회귀: 선형 회귀 방식을 분류에 적용한 알고리즘 → '분류'에 사용
  - 선형 회귀 계열이나, 선형 회귀와 다른 점은 학습을 통해 선형 함수의 회귀 최적선을 찾지 않고 시그모이드(Sigmoid) 함수 최적선을 찾고 시그모이드 함수 반환 값을 확률로 간주하여 확률에 따라 분류를 결정하는 것
  
![image.png](attachment:image.png)

- 시그모이드 함수
  - y = $\frac{1}{1+e-x}$ (-x는 제곱)
  - 시그모이드 함수는 x 값이 +, -로 아무리 커지거나 작아져도 y 값은 0과 1 사이 값만 반환
  - x 값이 커지면 1에 근사하며 x 값이 작아지면 0에 근사
  - x가 0일 때는 0.5

- 회귀 분제를 분류 문제에 적용하기
  - 종양의 크기에 따라 악성 종양인지(Yes = 1), 아닌지(No = 0)를 회귀를 이용하여 1과 0 값으로 예측하는 것
  - 종양 크기에 따라 악성될 확률이 높다고 하면 아래 왼쪽 그림과 같이 분포하며 선형 회귀 선을 그릴 수 있으나, 해당 회귀 라인은 0과 1을 제대로 분류하지 못함
  - 오른쪽 그림처럼 시그모이드 함수를 이용하면 조금 더 정확하게 0과 1을 분류할 수 있음

![image.png](attachment:image.png)


**- 로지스틱 회귀로 암 여부 판단하기: 위스콘신 유방암 데이터 세트 이용**


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()
```


```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# StandardScaler( )로 평균이 0, 분산 1로 데이터 분포도 변환
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cancer.data)

X_train , X_test, y_train , y_test = train_test_split(data_scaled, cancer.target, test_size=0.3, random_state=0)
```

**- 로지스틱 회귀로 학습 및 예측하고, 정확도와 ROC-AUC 값 구하기**


```python
from sklearn.metrics import accuracy_score, roc_auc_score

# 로지스틱 회귀를 이용하여 학습 및 예측 수행. 
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_preds = lr_clf.predict(X_test)

# accuracy와 roc_auc 측정
print('accuracy: {:0.3f}'.format(accuracy_score(y_test, lr_preds)))
print('roc_auc: {:0.3f}'.format(roc_auc_score(y_test , lr_preds)))
```

    accuracy: 0.977
    roc_auc: 0.972
    

- 사이킷런 LogisticRegression 클래스의 주요 하이퍼 파라미터로 penalty와 C가 존재
- penalty는 규제의 유형을 설정하며 'l2'로 설정 시 L2 규제를, 'l1'으로 설정 시 L1 규제를 뜻함
- C는 규제 강도를 조절하는 alpha 값의 역수로 C = $\frac{1}{alpha}$
  - C 값이 작을수록 규제 강도가 큼을 의미
  

**- 위스콘신 데이터 세트에서 해당 하이퍼 파라미터를 최적화하기**


```python
from sklearn.model_selection import GridSearchCV

params={'penalty':['l2', 'l1'],
        'C':[0.01, 0.1, 1, 1, 5, 10]}

grid_clf = GridSearchCV(lr_clf, param_grid=params, scoring='accuracy', cv=3 )
grid_clf.fit(data_scaled, cancer.target)
print('최적 하이퍼 파라미터:{0}, 최적 평균 정확도:{1:.3f}'.format(grid_clf.best_params_, 
                                                  grid_clf.best_score_))
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:548: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 531, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 1304, in fit
        solver = _check_solver(self.solver, self.penalty, self.dual)
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 442, in _check_solver
        raise ValueError("Solver %s supports only 'l2' or 'none' penalties, "
    ValueError: Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
    
      warnings.warn("Estimator fit failed. The score on this train-test"
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:548: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 531, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 1304, in fit
        solver = _check_solver(self.solver, self.penalty, self.dual)
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 442, in _check_solver
        raise ValueError("Solver %s supports only 'l2' or 'none' penalties, "
    ValueError: Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
    
      warnings.warn("Estimator fit failed. The score on this train-test"
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:548: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 531, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 1304, in fit
        solver = _check_solver(self.solver, self.penalty, self.dual)
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 442, in _check_solver
        raise ValueError("Solver %s supports only 'l2' or 'none' penalties, "
    ValueError: Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
    
      warnings.warn("Estimator fit failed. The score on this train-test"
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:548: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 531, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 1304, in fit
        solver = _check_solver(self.solver, self.penalty, self.dual)
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 442, in _check_solver
        raise ValueError("Solver %s supports only 'l2' or 'none' penalties, "
    ValueError: Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
    
      warnings.warn("Estimator fit failed. The score on this train-test"
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:548: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 531, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 1304, in fit
        solver = _check_solver(self.solver, self.penalty, self.dual)
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 442, in _check_solver
        raise ValueError("Solver %s supports only 'l2' or 'none' penalties, "
    ValueError: Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
    
      warnings.warn("Estimator fit failed. The score on this train-test"
    

    최적 하이퍼 파라미터:{'C': 1, 'penalty': 'l2'}, 최적 평균 정확도:0.975
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:548: FitFailedWarning: Estimator fit failed. The score on this train-test partition for these parameters will be set to nan. Details: 
    Traceback (most recent call last):
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 531, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 1304, in fit
        solver = _check_solver(self.solver, self.penalty, self.dual)
      File "C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py", line 442, in _check_solver
        raise ValueError("Solver %s supports only 'l2' or 'none' penalties, "
    ValueError: Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
    
      warnings.warn("Estimator fit failed. The score on this train-test"
    

- 로지스틱 회귀는 가볍고 빠르며, 이진 분류 예측 성능까지 뛰어남
  - 이진 분류의 기본 모델로 사용하는 경우가 많음
- 로지스틱 회귀는 희소한 데이터 세트 분류에서도 뛰어난 성능을 보임
  - 텍스트 분류에서도 자주 사용

## **08. 회귀 트리**
- 회귀 함수를 기반으로 하지 않고, 결정 트리와 같이 트리를 기반으로 하는 회귀 방식 소개

- 트리 기반이 회귀: 회귀 트리를 이용하는 것
  - 회귀를 위한 트리를 생성하고 이를 기반으로 회귀를 예측하는 것
  - 4장 분류에서 언급한 분류 트리와 비슷하나, 리프 노트에서 예측 결정 값을 만드는 과정에 차이가 있음
    - 분류 트리는 특정 클래스 레이블을 결정하나, 회귀 트리는 리프 노드에 속한 데이터 값의 평균값을 구해 회귀 예측값을 계산
    

- 예시(p.335-336)
  - 피처가 단 하나인 X 피처 데이터 세트와 결정값 Y가 2차원 평면에 있다고 가정
  - 데이터 세트의 X 피처를 결정 트리 기반으로 분할하면 X값의 균일도를 반영한 지니 계수에 따라 분할됨
  - 루트 노드를 Split 0 기준으로 분할하고, 분할된 규칙 노드에서 다시 Split 1과 Split 2 규칙 노드로 분할할 수 있음
  - Split 2는 다시 재귀적으로 Split 3 규칙 노드로 트리 규칙으로 변환될 수 있음
  - 리프 노드 생성 기준에 부합하는 트리 분할이 완료됐다면, 리프 노드에 소속된 데이터 값의 평균값을 구해 최종적으로 리프 노드에 결정 값으로 할당함
  

- **사이킷런 트리 기반 회귀와 분류의 Estimator 클래스**

알고리즘|회귀 Estimator 클래스|분류 Estimator 클래스
:--:|:--:|:--:
Decision Tree|DecisionTreeRegressor|DecisionTreeClassifier
Gradient Boosting|GradientBoostingRegressor|GradientBoostingClassifier
XGBoost|XGBRegressor|XGBClassifier
LightGBM|LGBMRegressor|LGBMClassifier


**- 사이킷런 랜덤 포레스트 회귀 트리인 RandomForestRegressor로 보스턴 주택 가격 예측 수행하기**


```python
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# 보스턴 데이터 세트 로드
boston = load_boston()
bostonDF = pd.DataFrame(boston.data, columns = boston.feature_names)

bostonDF['PRICE'] = boston.target
y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'], axis=1,inplace=False)

rf = RandomForestRegressor(random_state=0, n_estimators=1000)
neg_mse_scores = cross_val_score(rf, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores  = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print(' 5 교차 검증의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 2))
print(' 5 교차 검증의 개별 RMSE scores : ', np.round(rmse_scores, 2))
print(' 5 교차 검증의 평균 RMSE : {0:.3f} '.format(avg_rmse))
```

     5 교차 검증의 개별 Negative MSE scores:  [ -7.88 -13.14 -20.57 -46.23 -18.88]
     5 교차 검증의 개별 RMSE scores :  [2.81 3.63 4.54 6.8  4.34]
     5 교차 검증의 평균 RMSE : 4.423 
    

**- 결정 트리, GBM, XGBoost, LightGBM의 Regressor을 모두 이용해 보스턴 주택 가격 예측 수행**
- 사용 함수: get_model_cv_prediction()
  - 입력 모델과 데이터 세트를 입력 받아, 교차 검증으로 평균 RMSE를 계산하는 함수


```python
def get_model_cv_prediction(model, X_data, y_target):
    neg_mse_scores = cross_val_score(model, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
    rmse_scores  = np.sqrt(-1 * neg_mse_scores)
    avg_rmse = np.mean(rmse_scores)
    print('##### ',model.__class__.__name__ , ' #####')
    print(' 5 교차 검증의 평균 RMSE : {0:.3f} '.format(avg_rmse))
```

**- 다양한 유형의 회귀 트리를 생성하고, 보스턴 주택 가격 예측하기**


```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

dt_reg = DecisionTreeRegressor(random_state=0, max_depth=4)
rf_reg = RandomForestRegressor(random_state=0, n_estimators=1000)
gb_reg = GradientBoostingRegressor(random_state=0, n_estimators=1000)
xgb_reg = XGBRegressor(n_estimators=1000)
lgb_reg = LGBMRegressor(n_estimators=1000)

# 트리 기반의 회귀 모델을 반복하면서 평가 수행 
models = [dt_reg, rf_reg, gb_reg, xgb_reg, lgb_reg]
for model in models:  
    get_model_cv_prediction(model, X_data, y_target)
```

    #####  DecisionTreeRegressor  #####
     5 교차 검증의 평균 RMSE : 5.978 
    #####  RandomForestRegressor  #####
     5 교차 검증의 평균 RMSE : 4.423 
    #####  GradientBoostingRegressor  #####
     5 교차 검증의 평균 RMSE : 4.269 
    #####  XGBRegressor  #####
     5 교차 검증의 평균 RMSE : 4.251 
    #####  LGBMRegressor  #####
     5 교차 검증의 평균 RMSE : 4.646 
    

**- feature_importances_를 이용해 보스턴 주택 가격 모델의 피처별 중요도 시각화하기**
- 회귀 트리 Regressor 클래스는 선형 회귀와 다른 처리 방식으로, 회귀 계수를 제공하는 coef_ 속성이 없으나, feature_importances_를 이용해 피처별 중요도를 알 수 있음


```python
import seaborn as sns
%matplotlib inline

rf_reg = RandomForestRegressor(n_estimators=1000)

# 앞 예제에서 만들어진 X_data, y_target 데이터 셋을 적용하여 학습합니다.   
rf_reg.fit(X_data, y_target)

feature_series = pd.Series(data=rf_reg.feature_importances_, index=X_data.columns )
feature_series = feature_series.sort_values(ascending=False)
sns.barplot(x= feature_series, y=feature_series.index)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x167a0bb8940>




    
![png](/images/python_machine_learning_perfect_guide_ch05/output_87_1.png)
    


**- 회귀 트리 Regressor가 예측값을 판단하는 방법을 선형 회귀와 비교하여 시각화하기**
- 보스턴 데이터 세트를 100개만 샘플링하고 RM과 PRICE 칼럼만 추출
  - 2차원 평면상에서 X축에 독립변수인 RM, Y축에 종속변수인 PRICE만 가지고 더 직관적으로 예측값을 시각화하기 위한 것


```python
import matplotlib.pyplot as plt
%matplotlib inline

bostonDF_sample = bostonDF[['RM','PRICE']]
bostonDF_sample = bostonDF_sample.sample(n=100,random_state=0)
print(bostonDF_sample.shape)
plt.figure()
plt.scatter(bostonDF_sample.RM , bostonDF_sample.PRICE,c="darkorange")
```

    (100, 2)
    




    <matplotlib.collections.PathCollection at 0x167a237f970>




    
![png](/images/python_machine_learning_perfect_guide_ch05/output_89_2.png)
    


**- LinearRegression과 DecisionTreeRegressor를 max_depth 2, 7로 학습하기** 


```python
import numpy as np
from sklearn.linear_model import LinearRegression

# 선형 회귀와 결정 트리 기반의 Regressor 생성. DecisionTreeRegressor의 max_depth는 각각 2, 7
lr_reg = LinearRegression()
rf_reg2 = DecisionTreeRegressor(max_depth=2)
rf_reg7 = DecisionTreeRegressor(max_depth=7)

# 실제 예측을 적용할 테스트용 데이터 셋을 4.5 ~ 8.5 까지 100개 데이터 셋 생성. 
X_test = np.arange(4.5, 8.5, 0.04).reshape(-1, 1)

# 보스턴 주택가격 데이터에서 시각화를 위해 피처는 RM만, 그리고 결정 데이터인 PRICE 추출
X_feature = bostonDF_sample['RM'].values.reshape(-1,1)
y_target = bostonDF_sample['PRICE'].values.reshape(-1,1)

# 학습과 예측 수행. 
lr_reg.fit(X_feature, y_target)
rf_reg2.fit(X_feature, y_target)
rf_reg7.fit(X_feature, y_target)

pred_lr = lr_reg.predict(X_test)
pred_rf2 = rf_reg2.predict(X_test)
pred_rf7 = rf_reg7.predict(X_test)
```


```python
fig , (ax1, ax2, ax3) = plt.subplots(figsize=(14,4), ncols=3)

# X축값을 4.5 ~ 8.5로 변환하며 입력했을 때, 선형 회귀와 결정 트리 회귀 예측 선 시각화
# 선형 회귀로 학습된 모델 회귀 예측선 
ax1.set_title('Linear Regression')
ax1.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
ax1.plot(X_test, pred_lr,label="linear", linewidth=2 )

# DecisionTreeRegressor의 max_depth를 2로 했을 때 회귀 예측선 
ax2.set_title('Decision Tree Regression: \n max_depth=2')
ax2.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
ax2.plot(X_test, pred_rf2, label="max_depth:3", linewidth=2 )

# DecisionTreeRegressor의 max_depth를 7로 했을 때 회귀 예측선 
ax3.set_title('Decision Tree Regression: \n max_depth=7')
ax3.scatter(bostonDF_sample.RM, bostonDF_sample.PRICE, c="darkorange")
ax3.plot(X_test, pred_rf7, label="max_depth:7", linewidth=2)
```

- 정리
  - 선형 회귀: 예측 회귀선을 직선으로 표현
  - 회귀 트리: 분할되는 데이터 지점에 따라 브랜치를 만들며 계단 형태로 회귀선을 만듦
  - DecisionTreeRegressor의 max_depth = 7인 경우, 학습 데이터 세트의 이상치(outlier) 데이터도 학습하면서 복잡한 계단 형태의 회귀선을 만들어 과적합 되기 쉬운 모델이 됨

## **09. 회귀 실습- 자전거 대여 수요 예측**
- **데이터 설명**
  - 기간: 2011년 1월 - 2012년 12월
  - 날짜/시간, 기온, 습도, 풍속 등 정보
  - 1시간 간격으로 자전거 대여 횟수 기록


- **데이터의 주요 칼럼 (결정값: count)**
  - datetime: hourly date + timestamp
  - season: 1 = 봄, 2 = 여름, 3 = 가을, 4 = 겨울
  - holiday: 1= 토/일요일의 주말 제외한 국경일 등의 휴일, 0 = 휴일 아닌 날
  - workingday: 1 = 토/일요일의 주말 및 휴일이 아닌 주중, 0 = 주말 및 휴일
  - weather: 1 = 맑음, 약간 구름 낀 흐림, 2 = 안개, 안개 + 흐림, 3 = 가벼운 눈, 가벼운 비 + 천둥, 4 = 심한 눈/비, 천둥/번개
  - temp: 온도(섭씨)
  - atemp: 체감온도(섭씨)
  - humidity: 상대습도
  - windspeed: 풍속
  - casual: 사전 등록되지 않은 사용자 대여 횟수
  - registered: 사전 등록된 사용자 대여 횟수
  - count: 대여 획수

### (1) 데이터 클렌징 및 가공
- bike_train.csv 데이터 세트로 모델을 학습한 후, 대여 횟수(count) 예측


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

bike_df = pd.read_csv('./data/bike_train.csv')
print(bike_df.shape)
bike_df.head(3)
```

    (10886, 12)
    




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
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 데이터 타입 살펴보기
bike_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 12 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   datetime    10886 non-null  object 
     1   season      10886 non-null  int64  
     2   holiday     10886 non-null  int64  
     3   workingday  10886 non-null  int64  
     4   weather     10886 non-null  int64  
     5   temp        10886 non-null  float64
     6   atemp       10886 non-null  float64
     7   humidity    10886 non-null  int64  
     8   windspeed   10886 non-null  float64
     9   casual      10886 non-null  int64  
     10  registered  10886 non-null  int64  
     11  count       10886 non-null  int64  
    dtypes: float64(3), int64(8), object(1)
    memory usage: 1020.7+ KB
    

- 데이터 타입 확인
  - Null 데이터 없음
  - datetime 칼럼만 object형, 년-월-일 시:분:초 형식 가공 필요


```python
# 문자열을 datetime 타입으로 변경. 
bike_df['datetime'] = bike_df.datetime.apply(pd.to_datetime)

# datetime 타입에서 년, 월, 일, 시간 추출
bike_df['year'] = bike_df.datetime.apply(lambda x : x.year)
bike_df['month'] = bike_df.datetime.apply(lambda x : x.month)
bike_df['day'] = bike_df.datetime.apply(lambda x : x.day)
bike_df['hour'] = bike_df.datetime.apply(lambda x: x.hour)
bike_df.head(3)
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
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# datetime 삭제
# casule + registered = count이므로 casule, registered 값도 삭제

drop_columns = ['datetime','casual','registered']
bike_df.drop(drop_columns, axis=1,inplace=True)
```

**- 다양한 회귀 모델을 데이터 세트에 적용해 예측 성능 측정하기**
- 캐글에서 요구한 성능 평가 방법은 RMSLE(Root Mean Square Log Error)로 오류 값 로그에 대한 RMSE
  - 단, 사이킷런은 RMSLE를 제공하지 않아 RMSLE를 수행하는 성능 형가 함수를 만들어야 함


```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# log 값 변환 시 NaN등의 이슈로 log() 가 아닌 log1p() 를 이용하여 RMSLE 계산
def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

# 사이킷런의 mean_square_error() 를 이용하여 RMSE 계산
def rmse(y,pred):
    return np.sqrt(mean_squared_error(y,pred))

# MSE, RMSE, RMSLE 를 모두 계산 
def evaluate_regr(y,pred):
    rmsle_val = rmsle(y,pred)
    rmse_val = rmse(y,pred)
    # MAE 는 scikit learn의 mean_absolute_error() 로 계산
    mae_val = mean_absolute_error(y,pred)
    print('RMSLE: {0:.3f}, RMSE: {1:.3F}, MAE: {2:.3F}'.format(rmsle_val, rmse_val, mae_val))
```

### (2) 로그 변환, 피처 인코딩, 모델 학습/예측/평가
- 회귀 모델을 이용해 자전거 대여 횟수 예측하기
  - 먼저, 결괏값이 정규 분포로 되어 있는지 확인해야 함
  - 카테고리형 회귀 모델은 원-핫 인코딩으로 피처를 인코딩해야 함


**- 사이킷런의 LinearRegression 객체로 회귀 예측하기**


```python
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LinearRegression , Ridge , Lasso

y_target = bike_df['count']
X_features = bike_df.drop(['count'],axis=1,inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=0)

lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

evaluate_regr(y_test ,pred)
```

    RMSLE: 1.165, RMSE: 140.900, MAE: 105.924
    

- 결과 해석
  - 실제 Target 데이터 값인 대여 횟수(Count)를 감안하면 예측 오류로서는 비교적 큰 값


**- 실제값과 예측값이 어느 정도 차이 나는지 DataFrame 칼럼으로 만들어서 오류 값이 가장 큰 순으로 5개만 확인하기**


```python
def get_top_error_data(y_test, pred, n_tops = 5):
    # DataFrame에 컬럼들로 실제 대여횟수(count)와 예측 값을 서로 비교 할 수 있도록 생성. 
    result_df = pd.DataFrame(y_test.values, columns=['real_count'])
    result_df['predicted_count']= np.round(pred)
    result_df['diff'] = np.abs(result_df['real_count'] - result_df['predicted_count'])
    # 예측값과 실제값이 가장 큰 데이터 순으로 출력. 
    print(result_df.sort_values('diff', ascending=False)[:n_tops])
    
get_top_error_data(y_test,pred,n_tops=5)
```

          real_count  predicted_count   diff
    1618         890            322.0  568.0
    3151         798            241.0  557.0
    966          884            327.0  557.0
    412          745            194.0  551.0
    2817         856            310.0  546.0
    

- 결과 해석
  - 가장 큰 상위 5 오류값은 546 - 568로 실제값을 감안하면 오륙 꽤 큼
  - 회귀에서 큰 예측 오류가 발생할 경우, Target 값의 분포가 왜곡된 형태를 이루는지를 확인해야 함
  - Target 값 분포는 정규 분포 형태가 가장 좋으며, 왜곡된 경우에는 회귀 예측 성능이 저하되는 경우가 쉽게 발생함
  

**- 판다스 DataFrame의 hist()를 이용해 자전거 대여 모델의 Target 값인 count 칼럼이 정규 분포를 이루는지 확인하기**


```python
y_target.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x167a2792220>




    
![png](/images/python_machine_learning_perfect_guide_ch05/output_108_1.png)
    


- 결과 해석
  - count 칼럼 값이 정규 분포가 아닌, 0 - 200 사이에 왜곡된 것을 알 수 있음
  - 왜곡된 값을 정규 분포 형태로 바꾸는 방법: 로그를 적용해 변환하는 것
    - Numpy의 log1p()이용
    - 변경된 Target 값을 기반으로 학습하고, 예측한 값은 expm1() 함수를 이용해 원래의 scale 값으로 원상 복구


**- lop1p()를 적용한 'count'값이 분포 확인하기**


```python
y_log_transform = np.log1p(y_target)
y_log_transform.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x167a272a3d0>




    
![png](/images/python_machine_learning_perfect_guide_ch05/output_110_1.png)
    


- 정규 분포 형태는 아니지만, 왜곡 정도가 많이 향상됨

**- 위 데이터로 다시 학습하고 평가하기**


```python
# 타겟 컬럼인 count 값을 log1p 로 Log 변환
y_target_log = np.log1p(y_target)

# 로그 변환된 y_target_log를 반영하여 학습/테스트 데이터 셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target_log, test_size=0.3, random_state=0)
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

# 테스트 데이터 셋의 Target 값은 Log 변환되었으므로 다시 expm1를 이용하여 원래 scale로 변환
y_test_exp = np.expm1(y_test)

# 예측 값 역시 Log 변환된 타겟 기반으로 학습되어 예측되었으므로 다시 exmpl으로 scale변환
pred_exp = np.expm1(pred)

evaluate_regr(y_test_exp ,pred_exp)
```

    RMSLE: 1.017, RMSE: 162.594, MAE: 109.286
    

- RMSLE 오류는 줄어들었으나, RMSE는 오히려 더 늘어남


**- 각 피처의 회귀 계수 값을 시각화해 확인하기**


```python
coef = pd.Series(lr_reg.coef_, index=X_features.columns)
coef_sort = coef.sort_values(ascending=False)
sns.barplot(x=coef_sort.values, y=coef_sort.index)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x167a23ae1c0>




    
![png](/images/python_machine_learning_perfect_guide_ch05/output_114_1.png)
    


- 결과 해석
  - Year 피처 회귀 계수 값이 독보적으로 큼
    - Year는 2011, 2012 두 개의 값으로, year에 따라 자전거 대여 횟수가 크게 영향을 받는다고 할 수 없음
    - Category 피처지만 숫자형 값으로 되어 있고 2011, 2012가 매우 큰 숫자라 영향을 주게 됨
    - 원-핫 인코딩을 적용해 변환하여야 함


**- 여러 칼럼 원-핫 인코딩하고 선형 회귀 모델(LinearRegression, Ridge, Lasso 모두 학습해 예측 성능 확인하기**
- 사용 함수: get_model_predict()
  - 모델과 학습/테스트 데이터 세트를 입력하면 성능 평가 수치를 반환하는 함수


```python
# 'year', month', 'day', hour'등의 피처들을 One Hot Encoding
X_features_ohe = pd.get_dummies(X_features, columns=['year', 'month','day', 'hour', 'holiday',
                                              'workingday','season','weather'])
```


```python
# 원-핫 인코딩이 적용된 feature 데이터 세트 기반으로 학습/예측 데이터 분할. 
X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log,
                                                    test_size=0.3, random_state=0)

# 모델과 학습/테스트 데이터 셋을 입력하면 성능 평가 수치를 반환
def get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=False):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if is_expm1 :
        y_test = np.expm1(y_test)
        pred = np.expm1(pred)
    print('###',model.__class__.__name__,'###')
    evaluate_regr(y_test, pred)
# end of function get_model_predict    

# model 별로 평가 수행
lr_reg = LinearRegression()
ridge_reg = Ridge(alpha=10)
lasso_reg = Lasso(alpha=0.01)

for model in [lr_reg, ridge_reg, lasso_reg]:
    get_model_predict(model,X_train, X_test, y_train, y_test,is_expm1=True)
```

    ### LinearRegression ###
    RMSLE: 0.590, RMSE: 97.688, MAE: 63.382
    ### Ridge ###
    RMSLE: 0.590, RMSE: 98.529, MAE: 63.893
    ### Lasso ###
    RMSLE: 0.635, RMSE: 113.219, MAE: 72.803
    

- 결과 해석
  - 원-핫 인코딩 적용 후, 선형 회귀 예측 성능이 많이 향상됨


**- 원-핫 인코딩으로 피처가 늘어났으므로, 회귀 계수 상위 25개 피처를 추출해 시각화하기**


```python
coef = pd.Series(lr_reg.coef_ , index=X_features_ohe.columns)
coef_sort = coef.sort_values(ascending=False)[:20]
sns.barplot(x=coef_sort.values , y=coef_sort.index)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x167a25a88b0>




    
![png](/images/python_machine_learning_perfect_guide_ch05/output_119_1.png)
    


- 결과 해석
  - 선형 회귀 모델 시 month_9, month_8, month_7 등의 월 관련 피처와 workingday 관련 피처, hour 관련 피처의 회귀 계수가 높은 것을 알 수 있음
    - 월, 주말/주중, 시간대 등 상식선에서 자전거 타는 데 필요한 피처의 회귀 계수가 높아짐
    → 선형 회귀 수행 시에는 피처를 어떻게 인코딩하는가가 성능에 큰 영향을 미칠 수 있음
    
    
**- 회귀 트리로 회귀 예측 수행하기**


```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 랜덤 포레스트, GBM, XGBoost, LightGBM model 별로 평가 수행
rf_reg = RandomForestRegressor(n_estimators=500)
gbm_reg = GradientBoostingRegressor(n_estimators=500)
xgb_reg = XGBRegressor(n_estimators=500)
lgbm_reg = LGBMRegressor(n_estimators=500)

for model in [rf_reg, gbm_reg, xgb_reg, lgbm_reg]:
    # XGBoost의 경우 DataFrame이 입력 될 경우 버전에 따라 오류 발생 가능. ndarray로 변환.
    get_model_predict(model,X_train.values, X_test.values, y_train.values, y_test.values,is_expm1=True)
```

    ### RandomForestRegressor ###
    RMSLE: 0.356, RMSE: 50.371, MAE: 31.261
    ### GradientBoostingRegressor ###
    RMSLE: 0.330, RMSE: 53.324, MAE: 32.736
    ### XGBRegressor ###
    RMSLE: 0.342, RMSE: 51.732, MAE: 31.251
    ### LGBMRegressor ###
    RMSLE: 0.319, RMSE: 47.215, MAE: 29.029
    

- 결과 해석
  - 앞의 선형 회귀 모델보다 회귀 예측 성능이 개선됨
  - 단, 회귀 트리가 선형 트리보다 나은 성능을 가진다는 의미가 아님
    - 데이터 세트 유형에 따라 결과는 얼마든지 달라질 수 있음

## **10. 회귀 실습- 캐글 주택 가격: 고급 회귀 기법**
- 데이터 설명
  - 변수: 79개
  - 미국 아이오와주의 에임스(Ames) 지방 주택 가격 정보
  [- 피처별 설명 확인하기](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=train.csv)
  
- 성능 평가
  - RMSLE(Root Mean Squared Log Error) 기반
  - 가격이 비싼 주택일수록 예측 결과 오류가 전체 오류에 미치는 비중이 높으므로, 이를 상쇄하기 위해 오류 값을 로그 변환한 RMSLE를 이용


### (1) 데이터 사전 처리(Preprocessing)


```python
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

house_df_org = pd.read_csv('./data/house_price.csv')
house_df = house_df_org.copy()
house_df.head(3)
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 81 columns</p>
</div>




```python
# 데이터 세트 전체 크기와 칼럼 타입, Null이 있는 칼럼과 건수를 내림차순으로 출력

print('데이터 세트의 Shape:', house_df.shape)
print('\n전체 feature 들의 type \n',house_df.dtypes.value_counts())
isnull_series = house_df.isnull().sum()
print('\nNull 컬럼과 그 건수:\n ', isnull_series[isnull_series > 0].sort_values(ascending=False))
```

    데이터 세트의 Shape: (1460, 81)
    
    전체 feature 들의 type 
     object     43
    int64      35
    float64     3
    dtype: int64
    
    Null 컬럼과 그 건수:
      PoolQC          1453
    MiscFeature     1406
    Alley           1369
    Fence           1179
    FireplaceQu      690
    LotFrontage      259
    GarageYrBlt       81
    GarageType        81
    GarageFinish      81
    GarageQual        81
    GarageCond        81
    BsmtFinType2      38
    BsmtExposure      38
    BsmtFinType1      37
    BsmtCond          37
    BsmtQual          37
    MasVnrArea         8
    MasVnrType         8
    Electrical         1
    dtype: int64
    

- 데이터 타입 확인
  - 테이터 세트는 1460개의 레코드와 81개의 피처로 구성
  - 피처 타입은 숫자형과 문자형 모두 존재
    - Target을 제외한 80개 피처 중, 43개가 문자형이고 37개가 숫자형
  - 1480개 데이터 중, PoolQC, MiseFeature, Alley, Fence는 1000개가 넘는 Null 값을 가짐
    - Null 값이 너무 많은 피처는 drop


**- 회귀 모델 적용 전, 타깃 값 분포가 정규 분포인지 확인하기**
- 아래 그래프에서 볼 수 있듯, 데이터 값 분포가 왼쪽으로 치우친 형태로 정규 분포에서 벗어나 있음


```python
plt.title('Original Sale Price Histogram')
sns.distplot(house_df['SalePrice'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x167a3cf4340>




    
![png](/images/python_machine_learning_perfect_guide_ch05/output_127_1.png)
    


**- 로그 변환(Log Transformation)을 적용하여, 정규 분포가 아닌 결괏값을 정규 분포 형태로 변환하기**
- Numpy의 log1p()로 로그 변환한 결괏값 기반으로 학습
- 예측 시에는 결괏값을 expm1()로 환원


```python
plt.title('Log Transformed Sale Price Histogram')
log_SalePrice = np.log1p(house_df['SalePrice'])
sns.distplot(log_SalePrice, color = 'g')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x167a3df6160>




    
![png](/images/python_machine_learning_perfect_guide_ch05/output_129_1.png)
    


- SalePrice를 로그 변환해 정규 분포 형태로 결괏값이 분포함을 확인할 수 있음

**- 다음 작업**
1. SalePrice를 로그 변환하고 DataFrame에 반영
2. Null 값이 많은 피처인 PoolQC, MiseFeature, Alley, Fence, FireplaceQu 삭제
3. 단순 식별자인 Id 삭제
4. LotFrontage Null 값은 259개로 비교적 많으나, 평균값으로 대체
5. 나머지 피처 Null 값은 많지 않으므로 숫자형의 경우 평균값으로 대체


```python
# SalePrice 로그 변환
original_SalePrice = house_df['SalePrice']
house_df['SalePrice'] = np.log1p(house_df['SalePrice'])

# Null 이 너무 많은 컬럼들과 불필요한 컬럼 삭제
house_df.drop(['Id','PoolQC' , 'MiscFeature', 'Alley', 'Fence','FireplaceQu'], axis=1 , inplace=True)
# Drop 하지 않는 숫자형 Null컬럼들은 평균값으로 대체
house_df.fillna(house_df.mean(),inplace=True)

# Null 값이 있는 피처명과 타입을 추출
null_column_count = house_df.isnull().sum()[house_df.isnull().sum() > 0]
print('## Null 피처의 Type :\n', house_df.dtypes[null_column_count.index])
```

    ## Null 피처의 Type :
     MasVnrType      object
    BsmtQual        object
    BsmtCond        object
    BsmtExposure    object
    BsmtFinType1    object
    BsmtFinType2    object
    Electrical      object
    GarageType      object
    GarageFinish    object
    GarageQual      object
    GarageCond      object
    dtype: object
    

**- 문자형 피처는 원-핫 인코딩으로 변환하기**
- 사용 함수: get_dummies()
  - 자동으로 문자열 피처를 원-핫 인코딩으로 변환하면서 Null 값을 'None' 칼럼으로 대체해주어 Null 값을 대체하는 별도의 로직이 필요 없음


- 원-핫 인코딩을 적용하면 칼럼이 증가하기 때문에, 변환 후 늘어난 칼럼 값까지 확인하기


```python
print('get_dummies() 수행 전 데이터 Shape:', house_df.shape)
house_df_ohe = pd.get_dummies(house_df)
print('get_dummies() 수행 후 데이터 Shape:', house_df_ohe.shape)

null_column_count = house_df_ohe.isnull().sum()[house_df_ohe.isnull().sum() > 0]
print('## Null 피처의 Type :\n', house_df_ohe.dtypes[null_column_count.index])
```

    get_dummies() 수행 전 데이터 Shape: (1460, 75)
    get_dummies() 수행 후 데이터 Shape: (1460, 271)
    ## Null 피처의 Type :
     Series([], dtype: object)
    

- 결과 해석
  - 원-핫 인코딩 후 피처가 75개에서 272개로 증가
  - Null 값을 가진 피처는 없음

### **(2) 선형 회귀 모델 학습/예측/평가**

**RMSE 평가 함수 생성**



```python
def get_rmse(model):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test , pred)
    rmse = np.sqrt(mse)
    print('{0} 로그 변환된 RMSE: {1}'.format(model.__class__.__name__,np.round(rmse, 3)))
    return rmse

def get_rmses(models):
    rmses = [ ]
    for model in models:
        rmse = get_rmse(model)
        rmses.append(rmse)
    return rmses
```

**LinearRegression, Ridge, Lasso 학습, 예측, 평가**


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop('SalePrice',axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)

# LinearRegression, Ridge, Lasso 학습, 예측, 평가
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)

ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)

models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)
```

    LinearRegression 로그 변환된 RMSE: 0.132
    Ridge 로그 변환된 RMSE: 0.128
    Lasso 로그 변환된 RMSE: 0.176
    




    [0.1318957657915436, 0.12750846334053045, 0.17628250556471395]



**회귀 계수값과 컬럼명 시각화를 위해 상위 10개, 하위 10개(-값으로 가장 큰 10개) 회귀 계수값과 컬럼명을 가지는 Series생성 함수.**


```python
def get_top_bottom_coef(model):
    # coef_ 속성을 기반으로 Series 객체를 생성. index는 컬럼명. 
    coef = pd.Series(model.coef_, index=X_features.columns)
    
    # + 상위 10개 , - 하위 10개 coefficient 추출하여 반환.
    coef_high = coef.sort_values(ascending=False).head(10)
    coef_low = coef.sort_values(ascending=False).tail(10)
    return coef_high, coef_low
```

**인자로 입력되는 여러개의 회귀 모델들에 대한 회귀계수값과 컬럼명 시각화**


```python
def visualize_coefficient(models):
    # 3개 회귀 모델의 시각화를 위해 3개의 컬럼을 가지는 subplot 생성
    fig, axs = plt.subplots(figsize=(24,10),nrows=1, ncols=3)
    fig.tight_layout() 
    # 입력인자로 받은 list객체인 models에서 차례로 model을 추출하여 회귀 계수 시각화. 
    for i_num, model in enumerate(models):
        # 상위 10개, 하위 10개 회귀 계수를 구하고, 이를 판다스 concat으로 결합. 
        coef_high, coef_low = get_top_bottom_coef(model)
        coef_concat = pd.concat( [coef_high , coef_low] )
        
        # 순차적으로 ax subplot에 barchar로 표현. 한 화면에 표현하기 위해 tick label 위치와 font 크기 조정. 
        axs[i_num].set_title(model.__class__.__name__+' Coeffiecents', size=25)
        axs[i_num].tick_params(axis="y",direction="in", pad=-120)
        for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):
            label.set_fontsize(22)
        sns.barplot(x=coef_concat.values, y=coef_concat.index , ax=axs[i_num])

# 앞 예제에서 학습한 lr_reg, ridge_reg, lasso_reg 모델의 회귀 계수 시각화.    
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)
```


    
![png](/images/python_machine_learning_perfect_guide_ch05/output_142_0.png)
    


**5 폴드 교차검증으로 모델별로 RMSE와 평균 RMSE출력**


```python
from sklearn.model_selection import cross_val_score

def get_avg_rmse_cv(models):
    for model in models:
        # 분할하지 않고 전체 데이터로 cross_val_score( ) 수행. 모델별 CV RMSE값과 평균 RMSE 출력
        rmse_list = np.sqrt(-cross_val_score(model, X_features, y_target,
                                             scoring="neg_mean_squared_error", cv = 5))
        rmse_avg = np.mean(rmse_list)
        print('\n{0} CV RMSE 값 리스트: {1}'.format( model.__class__.__name__, np.round(rmse_list, 3)))
        print('{0} CV 평균 RMSE 값: {1}'.format( model.__class__.__name__, np.round(rmse_avg, 3)))

# 앞 예제에서 학습한 lr_reg, ridge_reg, lasso_reg 모델의 CV RMSE값 출력           
models = [lr_reg, ridge_reg, lasso_reg]
get_avg_rmse_cv(models)
```

    
    LinearRegression CV RMSE 값 리스트: [0.135 0.165 0.168 0.111 0.198]
    LinearRegression CV 평균 RMSE 값: 0.155
    
    Ridge CV RMSE 값 리스트: [0.117 0.154 0.142 0.117 0.189]
    Ridge CV 평균 RMSE 값: 0.144
    
    Lasso CV RMSE 값 리스트: [0.161 0.204 0.177 0.181 0.265]
    Lasso CV 평균 RMSE 값: 0.198
    

**각 모델들의 alpha값을 변경하면서 하이퍼 파라미터 튜닝 후 다시 재 학습/예측/평가**


```python
from sklearn.model_selection import GridSearchCV

def get_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params, 
                              scoring='neg_mean_squared_error', cv=5)
    grid_model.fit(X_features, y_target)
    rmse = np.sqrt(-1* grid_model.best_score_)
    print('{0} 5 CV 시 최적 평균 RMSE 값: {1}, 최적 alpha:{2}'.format(model.__class__.__name__,
                                        np.round(rmse, 4), grid_model.best_params_))
    return grid_model.best_estimator_

ridge_params = { 'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20] }
lasso_params = { 'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10] }
best_rige = get_best_params(ridge_reg, ridge_params)
best_lasso = get_best_params(lasso_reg, lasso_params)
```

    Ridge 5 CV 시 최적 평균 RMSE 값: 0.1418, 최적 alpha:{'alpha': 12}
    Lasso 5 CV 시 최적 평균 RMSE 값: 0.142, 최적 alpha:{'alpha': 0.001}
    


```python
# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행. 
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
ridge_reg = Ridge(alpha=12)
ridge_reg.fit(X_train, y_train)
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, y_train)

# 모든 모델의 RMSE 출력
models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

# 모든 모델의 회귀 계수 시각화 
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)
```

    LinearRegression 로그 변환된 RMSE: 0.132
    Ridge 로그 변환된 RMSE: 0.124
    Lasso 로그 변환된 RMSE: 0.12
    


    
![png](/images/python_machine_learning_perfect_guide_ch05/output_147_1.png)
    


**숫자 피처들에 대한 데이터 분포 왜곡도 확인 후 높은 왜곡도를 가지는 피처 추출**


```python
from scipy.stats import skew

# object가 아닌 숫자형 피쳐의 컬럼 index 객체 추출.
features_index = house_df.dtypes[house_df.dtypes != 'object'].index

# house_df에 컬럼 index를 [ ]로 입력하면 해당하는 컬럼 데이터 셋 반환. apply lambda로 skew( )호출 
skew_features = house_df[features_index].apply(lambda x : skew(x))

# skew 정도가 1 이상인 컬럼들만 추출. 
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top.sort_values(ascending=False))
```

    MiscVal          24.451640
    PoolArea         14.813135
    LotArea          12.195142
    3SsnPorch        10.293752
    LowQualFinSF      9.002080
    KitchenAbvGr      4.483784
    BsmtFinSF2        4.250888
    ScreenPorch       4.117977
    BsmtHalfBath      4.099186
    EnclosedPorch     3.086696
    MasVnrArea        2.673661
    LotFrontage       2.382499
    OpenPorchSF       2.361912
    BsmtFinSF1        1.683771
    WoodDeckSF        1.539792
    TotalBsmtSF       1.522688
    MSSubClass        1.406210
    1stFlrSF          1.375342
    GrLivArea         1.365156
    dtype: float64
    

**왜곡도가 1인 피처들은 로그 변환 적용하고 다시 하이퍼 파라미터 튜닝 후 재 학습/예측/평가**


```python
house_df[skew_features_top.index] = np.log1p(house_df[skew_features_top.index])
```


```python
# Skew가 높은 피처들을 로그 변환 했으므로 다시 원-핫 인코딩 적용 및 피처/타겟 데이터 셋 생성,
house_df_ohe = pd.get_dummies(house_df)
y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop('SalePrice',axis=1, inplace=False)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)

# 피처들을 로그 변환 후 다시 최적 하이퍼 파라미터와 RMSE 출력
ridge_params = { 'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20] }
lasso_params = { 'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10] }
best_ridge = get_best_params(ridge_reg, ridge_params)
best_lasso = get_best_params(lasso_reg, lasso_params)
```

    Ridge 5 CV 시 최적 평균 RMSE 값: 0.1275, 최적 alpha:{'alpha': 10}
    Lasso 5 CV 시 최적 평균 RMSE 값: 0.1252, 최적 alpha:{'alpha': 0.001}
    


```python
# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행. 
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
ridge_reg = Ridge(alpha=10)
ridge_reg.fit(X_train, y_train)
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, y_train)

# 모든 모델의 RMSE 출력
models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

# 모든 모델의 회귀 계수 시각화 
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)
```

    LinearRegression 로그 변환된 RMSE: 0.128
    Ridge 로그 변환된 RMSE: 0.122
    Lasso 로그 변환된 RMSE: 0.119
    


    
![png](/images/python_machine_learning_perfect_guide_ch05/output_153_1.png)
    


**이상치 데이터 검출을 위해 주요 피처인 GrLivArea값에 대한 산포도 확인**


```python
plt.scatter(x = house_df_org['GrLivArea'], y = house_df_org['SalePrice'])
plt.ylabel('SalePrice', fontsize=15)
plt.xlabel('GrLivArea', fontsize=15)
plt.show()
```


    
![png](/images/python_machine_learning_perfect_guide_ch05/output_155_0.png)
    


**이상치 데이터 삭제 후 재 학습/예측/평가**


```python
# GrLivArea와 SalePrice 모두 로그 변환되었으므로 이를 반영한 조건 생성. 
cond1 = house_df_ohe['GrLivArea'] > np.log1p(4000)
cond2 = house_df_ohe['SalePrice'] < np.log1p(500000)
outlier_index = house_df_ohe[cond1 & cond2].index

print('아웃라이어 레코드 index :', outlier_index.values)
print('아웃라이어 삭제 전 house_df_ohe shape:', house_df_ohe.shape)
# DataFrame의 index를 이용하여 아웃라이어 레코드 삭제. 
house_df_ohe.drop(outlier_index, axis=0, inplace=True)
print('아웃라이어 삭제 후 house_df_ohe shape:', house_df_ohe.shape)
```

    아웃라이어 레코드 index : [ 523 1298]
    아웃라이어 삭제 전 house_df_ohe shape: (1460, 271)
    아웃라이어 삭제 후 house_df_ohe shape: (1458, 271)
    


```python
y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop('SalePrice',axis=1, inplace=False)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)

ridge_params = { 'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20] }
lasso_params = { 'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10] }
best_ridge = get_best_params(ridge_reg, ridge_params)
best_lasso = get_best_params(lasso_reg, lasso_params)
```

    Ridge 5 CV 시 최적 평균 RMSE 값: 0.1125, 최적 alpha:{'alpha': 8}
    Lasso 5 CV 시 최적 평균 RMSE 값: 0.1122, 최적 alpha:{'alpha': 0.001}
    


```python
# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행. 
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
ridge_reg = Ridge(alpha=8)
ridge_reg.fit(X_train, y_train)
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, y_train)

# 모든 모델의 RMSE 출력
models = [lr_reg, ridge_reg, lasso_reg]
get_rmses(models)

# 모든 모델의 회귀 계수 시각화 
models = [lr_reg, ridge_reg, lasso_reg]
visualize_coefficient(models)
```

    LinearRegression 로그 변환된 RMSE: 0.129
    Ridge 로그 변환된 RMSE: 0.103
    Lasso 로그 변환된 RMSE: 0.1
    


    
![png](/images/python_machine_learning_perfect_guide_ch05/output_159_1.png)
    


### **회귀 트리 학습/예측/평가**

**XGBoost와 LightGBM 학습/예측/평가**


```python
from xgboost import XGBRegressor

xgb_params = {'n_estimators':[1000]}
xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.05, 
                       colsample_bytree=0.5, subsample=0.8)
best_xgb = get_best_params(xgb_reg, xgb_params)
```

    XGBRegressor 5 CV 시 최적 평균 RMSE 값: 0.1178, 최적 alpha:{'n_estimators': 1000}
    


```python
from lightgbm import LGBMRegressor

lgbm_params = {'n_estimators':[1000]}
lgbm_reg = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=4, 
                         subsample=0.6, colsample_bytree=0.4, reg_lambda=10, n_jobs=-1)
best_lgbm = get_best_params(lgbm_reg, lgbm_params)
```

    LGBMRegressor 5 CV 시 최적 평균 RMSE 값: 0.1163, 최적 alpha:{'n_estimators': 1000}
    

**트리 회귀 모델의 피처 중요도 시각화**


```python
# 모델의 중요도 상위 20개의 피처명과 그때의 중요도값을 Series로 반환.
def get_top_features(model):
    ftr_importances_values = model.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values, index=X_features.columns  )
    ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
    return ftr_top20

def visualize_ftr_importances(models):
    # 2개 회귀 모델의 시각화를 위해 2개의 컬럼을 가지는 subplot 생성
    fig, axs = plt.subplots(figsize=(24,10),nrows=1, ncols=2)
    fig.tight_layout() 
    # 입력인자로 받은 list객체인 models에서 차례로 model을 추출하여 피처 중요도 시각화. 
    for i_num, model in enumerate(models):
        # 중요도 상위 20개의 피처명과 그때의 중요도값 추출 
        ftr_top20 = get_top_features(model)
        axs[i_num].set_title(model.__class__.__name__+' Feature Importances', size=25)
        #font 크기 조정.
        for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):
            label.set_fontsize(22)
        sns.barplot(x=ftr_top20.values, y=ftr_top20.index , ax=axs[i_num])

# 앞 예제에서 get_best_params( )가 반환한 GridSearchCV로 최적화된 모델의 피처 중요도 시각화    
models = [best_xgb, best_lgbm]
visualize_ftr_importances(models)
```


    
![png](/images/python_machine_learning_perfect_guide_ch05/output_165_0.png)
    


### **회귀 모델들의 예측 결과 혼합을 통한 최종 예측**


```python
def get_rmse_pred(preds):
    for key in preds.keys():
        pred_value = preds[key]
        mse = mean_squared_error(y_test , pred_value)
        rmse = np.sqrt(mse)
        print('{0} 모델의 RMSE: {1}'.format(key, rmse))

# 개별 모델의 학습
ridge_reg = Ridge(alpha=8)
ridge_reg.fit(X_train, y_train)
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, y_train)
# 개별 모델 예측
ridge_pred = ridge_reg.predict(X_test)
lasso_pred = lasso_reg.predict(X_test)

# 개별 모델 예측값 혼합으로 최종 예측값 도출
pred = 0.4 * ridge_pred + 0.6 * lasso_pred
preds = {'최종 혼합': pred,
         'Ridge': ridge_pred,
         'Lasso': lasso_pred}
#최종 혼합 모델, 개별모델의 RMSE 값 출력
get_rmse_pred(preds)
```

    최종 혼합 모델의 RMSE: 0.10007930884470519
    Ridge 모델의 RMSE: 0.10345177546603272
    Lasso 모델의 RMSE: 0.10024170460890039
    


```python
xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.05, 
                       colsample_bytree=0.5, subsample=0.8)
lgbm_reg = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=4, 
                         subsample=0.6, colsample_bytree=0.4, reg_lambda=10, n_jobs=-1)
xgb_reg.fit(X_train, y_train)
lgbm_reg.fit(X_train, y_train)
xgb_pred = xgb_reg.predict(X_test)
lgbm_pred = lgbm_reg.predict(X_test)

pred = 0.5 * xgb_pred + 0.5 * lgbm_pred
preds = {'최종 혼합': pred,
         'XGBM': xgb_pred,
         'LGBM': lgbm_pred}
        
get_rmse_pred(preds)

```

    최종 혼합 모델의 RMSE: 0.1017007808403327
    XGBM 모델의 RMSE: 0.10738299364833828
    LGBM 모델의 RMSE: 0.10382510019327311
    

### **스태킹 모델을 통한 회귀 예측**


```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# 개별 기반 모델에서 최종 메타 모델이 사용할 학습 및 테스트용 데이터를 생성하기 위한 함수. 
def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds ):
    # 지정된 n_folds값으로 KFold 생성.
    kf = KFold(n_splits=n_folds, shuffle=False, random_state=0)
    #추후에 메타 모델이 사용할 학습 데이터 반환을 위한 넘파이 배열 초기화 
    train_fold_pred = np.zeros((X_train_n.shape[0] ,1 ))
    test_pred = np.zeros((X_test_n.shape[0],n_folds))
    print(model.__class__.__name__ , ' model 시작 ')
    
    for folder_counter , (train_index, valid_index) in enumerate(kf.split(X_train_n)):
        #입력된 학습 데이터에서 기반 모델이 학습/예측할 폴드 데이터 셋 추출 
        print('\t 폴드 세트: ',folder_counter,' 시작 ')
        X_tr = X_train_n[train_index] 
        y_tr = y_train_n[train_index] 
        X_te = X_train_n[valid_index]  
        
        #폴드 세트 내부에서 다시 만들어진 학습 데이터로 기반 모델의 학습 수행.
        model.fit(X_tr , y_tr)       
        #폴드 세트 내부에서 다시 만들어진 검증 데이터로 기반 모델 예측 후 데이터 저장.
        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1,1)
        #입력된 원본 테스트 데이터를 폴드 세트내 학습된 기반 모델에서 예측 후 데이터 저장. 
        test_pred[:, folder_counter] = model.predict(X_test_n)
            
    # 폴드 세트 내에서 원본 테스트 데이터를 예측한 데이터를 평균하여 테스트 데이터로 생성 
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1,1)    
    
    #train_fold_pred는 최종 메타 모델이 사용하는 학습 데이터, test_pred_mean은 테스트 데이터
    return train_fold_pred , test_pred_mean
```

**기반 모델은 리지, 라소, XGBoost, LightGBM 으로 만들고 최종 메타 모델은 라소로 생성하여 학습/예측/평가**


```python
# get_stacking_base_datasets( )은 넘파이 ndarray를 인자로 사용하므로 DataFrame을 넘파이로 변환. 
X_train_n = X_train.values
X_test_n = X_test.values
y_train_n = y_train.values

# 각 개별 기반(Base)모델이 생성한 학습용/테스트용 데이터 반환. 
ridge_train, ridge_test = get_stacking_base_datasets(ridge_reg, X_train_n, y_train_n, X_test_n, 5)
lasso_train, lasso_test = get_stacking_base_datasets(lasso_reg, X_train_n, y_train_n, X_test_n, 5)
xgb_train, xgb_test = get_stacking_base_datasets(xgb_reg, X_train_n, y_train_n, X_test_n, 5)  
lgbm_train, lgbm_test = get_stacking_base_datasets(lgbm_reg, X_train_n, y_train_n, X_test_n, 5)
```

    Ridge  model 시작 
    	 폴드 세트:  0  시작 
    	 폴드 세트:  1  시작 
    	 폴드 세트:  2  시작 
    	 폴드 세트:  3  시작 
    	 폴드 세트:  4  시작 
    Lasso  model 시작 
    	 폴드 세트:  0  시작 
    	 폴드 세트:  1  시작 
    	 폴드 세트:  2  시작 
    	 폴드 세트:  3  시작 
    	 폴드 세트:  4  시작 
    XGBRegressor  model 시작 
    	 폴드 세트:  0  시작 
    	 폴드 세트:  1  시작 
    	 폴드 세트:  2  시작 
    	 폴드 세트:  3  시작 
    	 폴드 세트:  4  시작 
    LGBMRegressor  model 시작 
    	 폴드 세트:  0  시작 
    	 폴드 세트:  1  시작 
    	 폴드 세트:  2  시작 
    	 폴드 세트:  3  시작 
    	 폴드 세트:  4  시작 
    


```python
# 개별 모델이 반환한 학습 및 테스트용 데이터 세트를 Stacking 형태로 결합.  
Stack_final_X_train = np.concatenate((ridge_train, lasso_train, 
                                      xgb_train, lgbm_train), axis=1)
Stack_final_X_test = np.concatenate((ridge_test, lasso_test, 
                                     xgb_test, lgbm_test), axis=1)

# 최종 메타 모델은 라쏘 모델을 적용. 
meta_model_lasso = Lasso(alpha=0.0005)

#기반 모델의 예측값을 기반으로 새롭게 만들어진 학습 및 테스트용 데이터로 예측하고 RMSE 측정.
meta_model_lasso.fit(Stack_final_X_train, y_train)
final = meta_model_lasso.predict(Stack_final_X_test)
mse = mean_squared_error(y_test , final)
rmse = np.sqrt(mse)
print('스태킹 회귀 모델의 최종 RMSE 값은:', rmse)
```

    스태킹 회귀 모델의 최종 RMSE 값은: 0.0979915406689774
    

### **정리**

+ 선형 회귀와 비용 함수 RSS
+ 경사 하강법
+ 다항회귀와 과소적합/과대적합
+ 규제 -L2규제를 적용한 릿지, L1규제를 적용한 라쏘, L1과 L2규제가 결합된 엘라스틱넷 회귀
+ 분류를 위한 로지스틱 회귀
+ CART 기반의 회귀 트리
+ 왜곡도 개선을 위한 데이터 변환과 원-핫 인코딩
+ 실습 예제를 통한 데이터 정제와 변환 그리고 선형회귀/회귀트리/혼합모델/스태킹 모델 학습/예측/평가비교

</details>