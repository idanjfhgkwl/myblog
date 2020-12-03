---
title: "ML 모델 을 평가하기 위한 올바른 측정 항목을 선택하는 방법"
categories:
  - study
output: 
  html_document:
    keep_md: true
---

# ML 모델 을 평가하기 위한 올바른 측정 항목을 선택하는 방법
(How to Choose Right Metric for Evaluating ML Model)

## 도입부

https://www.kaggle.com/vipulgandhi/how-to-choose-right-metric-for-evaluating-ml-model

이 [Scikit-learn 페이지](https://scikit-learn.org/stable/modules/model_evaluation.html)는 훌륭한 참조를 제공합니다.

일반적인 기능 엔지니어링, 선택, 모델 구현을 수행하고 확률 또는 클래스 형태로 출력을 얻은 후 다음 단계는 테스트 데이터 세트를 사용하여 일부 메트릭을 기반으로 모델이 얼마나 효과적인지 확인하는 것입니다. 메트릭은 모델의 성능을 설명합니다.  
<br>
모델은 정확도 _ 점수라는 메트릭을 사용하여 평가할 때 만족스러운 결과를 제공 할 수 있지만 logarithmic_loss와 같은 다른 메트릭 또는 다른 이러한 메트릭에 대해 평가할 때 좋지 않은 결과를 제공 할 수 있습니다. 따라서 기계 학습 모델을 평가하기 위해 올바른 메트릭을 선택하는 것이 매우 중요합니다.  
<br>
측정 항목 선택은 기계 학습 알고리즘의 성능을 측정하고 비교하는 방법에 영향을줍니다. 결과에서 다른 특성의 중요성에 가중치를 부여하는 방법에 영향을줍니다.  
<br>
**분류 메트릭**
- 정확성.
- 대수 손실.
- ROC, AUC.
- 혼란 매트릭스.
- 분류 보고서.

**회귀 지표**
- 평균 절대 오차.
- 평균 제곱 오차.
- 평균 제곱근 오차.
- 루트 평균 제곱 로그 오류.
- R 광장.
- R 제곱을 조정했습니다.

**분류 문제**에서는 , 우리는 (자신이 생성 출력의 종류에 따라) 알고리즘의 두 가지 유형을 사용
- **클래스 출력** : SVM 및 KNN과 같은 알고리즘은 클래스 출력을 생성합니다. 예를 들어, 이진 분류 문제에서 출력은 0 또는 1입니다. SKLearn의 / 기타 알고리즘은 이러한 클래스 출력을 확률로 변환 할 수 있습니다.
- **확률 출력** : 로지스틱 회귀, 랜덤 포레스트, 그라디언트 부스팅, Adaboost 등과 같은 알고리즘은 확률 출력을 제공합니다. 확률 출력은 임계 확률을 생성하여 클래스 출력으로 변환 할 수 있습니다.

회귀 문제에서 출력은 본질적으로 항상 연속적이며 추가 처리가 필요하지 않습니다.

## 분류 메트릭
(Classification Metrices)

- 데이터 세트 : 피마 인디언 당뇨병 예측.  
- 평가 알고리즘 : 로지스틱 회귀, SGDClassifier, RandomForestClassifier.


```python
from google.colab import drive # 패키지 불러오기 
from os.path import join  

# 구글 드라이브 마운트
ROOT = "/content/drive"     # 드라이브 기본 경로
print(ROOT)                 # print content of ROOT (Optional)
drive.mount(ROOT)           # 드라이브 기본 경로 

# 프로젝트 파일 생성 및 다운받을 경로 이동
MY_GOOGLE_DRIVE_PATH = 'My Drive/Colab Notebooks/python_basic/kaggle_how-to-choose-right-metric-for-evaluating-ml-model_vipulgandhi/data'
PROJECT_PATH = join(ROOT, MY_GOOGLE_DRIVE_PATH)
print(PROJECT_PATH)
```

    /content/drive
    Mounted at /content/drive
    /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_how-to-choose-right-metric-for-evaluating-ml-model_vipulgandhi/data
    


```python
%cd "{PROJECT_PATH}"
```

    /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_how-to-choose-right-metric-for-evaluating-ml-model_vipulgandhi/data
    


```python
!ls
```

    diabetes.csv
    


```python
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


diabetes_data = pd.read_csv('diabetes.csv')

X =  diabetes_data.drop(["Outcome"],axis = 1)
y = diabetes_data["Outcome"]

# 훈련 세트를 사용하여 다양한 하이퍼 파라미터로 여러 모델을 훈련하고 검증 세트에서 가장 잘 수행되는 모델과 하이퍼 파라미터를 선택합니다.
# 모델 유형과 하이퍼 파라미터가 선택되면 전체 훈련 세트에서 이러한 하이퍼 파라미터를 사용하여 최종 모델을 훈련시키고 일반화 된 오류는 테스트 세트에서 최종적으로 측정됩니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 56)

# StratifiedKFold 클래스는 계층화 된 샘플링을 수행하여 각 클래스의 대표 비율을 포함하는 폴드를 생성합니다.
cv = StratifiedKFold(n_splits=10, shuffle = False, random_state = 76)

# 로지스틱 회귀
clf_logreg = LogisticRegression()
# 적합 모델
clf_logreg.fit(X_train, y_train)
# 검증 세트에 대한 클래스 예측을합니다.
y_pred_class_logreg = cross_val_predict(clf_logreg, X_train, y_train, cv = cv)
# 클래스 1에 대한 예측 확률, 양성 클래스의 확률
y_pred_prob_logreg = cross_val_predict(clf_logreg, X_train, y_train, cv = cv, method="predict_proba")
y_pred_prob_logreg_class1 = y_pred_prob_logreg[:, 1]

# SGD 분류기
clf_SGD = SGDClassifier()
# 적합 모델
clf_SGD.fit(X_train, y_train)
# 검증 세트에 대한 클래스 예측을합니다.
y_pred_class_SGD = cross_val_predict(clf_SGD, X_train, y_train, cv = cv)
# 클래스 1에 대한 예측 확률
y_pred_prob_SGD = cross_val_predict(clf_SGD, X_train, y_train, cv = cv, method="decision_function")

# 랜덤 포레스트 분류기
clf_rfc = RandomForestClassifier()
# 적합 모델
clf_rfc.fit(X_train, y_train)
# 검증 세트에 대한 클래스 예측을합니다.
y_pred_class_rfc = cross_val_predict(clf_rfc, X_train, y_train, cv = cv)
# 클래스 1에 대한 예측 확률
y_pred_prob_rfc = cross_val_predict(clf_rfc, X_train, y_train, cv = cv, method="predict_proba")
y_pred_prob_rfc_class1 = y_pred_prob_rfc[:, 1]
```

**빠른 참고** : SkLearn의 "predict_log_proba"는 확률의 로그를 제공합니다. 확률이 매우 작아 질 수 있으므로 종종 더 편리합니다.

### Null 정확도
(Null accuracy)

- 항상 가장 빈번한 클래스를 예측하여 얻을 수있는 정확도.
- 이것은 항상 0/1을 예측하는 멍청한 모델이 "null_accuracy"%의 시간에 맞을 것이라는 것을 의미합니다.


```python
from sklearn.base import BaseEstimator
import numpy as np

class BaseClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
    
base_clf = BaseClassifier()
cross_val_score(base_clf, X_train, y_train, cv=10, scoring="accuracy").mean()


# Method 2
# calculate null accuracy (for binary / multi-class classification problems)
# null_accuracy = y_train.value_counts().head(1) / len(y_train)
```




    0.6509981851179674



### 분류 정확도
(Classification Accuracy)

분류 정확도 또는 정확도는 총 입력 샘플 수에 대한 올바른 예측 수의 비율입니다.  

$$Accuracy = \frac{Number\ of\ correct\ predictions}{Total\ number\ of\ predictions\ made} = \frac{TP + TN}{TP + TN + FP + FN}$$  
![다운로드](https://user-images.githubusercontent.com/72365720/100689145-3b388380-33c7-11eb-9f61-aec4666ba452.png)  

정확도 측정 항목을 사용하는 경우: 각 클래스에 속하는 샘플 수가 거의 동일한 경우  
정확도 측정 항목을 사용하지 않는 경우: 하나의 클래스 만 대부분의 샘플을 보유 할 때.  
<br>
**예**:  
훈련 세트에 클래스 A의 샘플이 98 %이고 클래스 B의 샘플이 2 %라고 가정합니다. 그러면 우리 모델은 클래스 A에 속하는 모든 훈련 샘플을 간단히 예측하여 98 %의 훈련 정확도를 쉽게 얻을 수 있습니다.  
동일한 모델이 클래스 A의 60 % 샘플과 클래스 B의 40 % 샘플이있는 테스트 세트에서 테스트되면 테스트 정확도가 60 %로 떨어집니다. 분류 정확도는 우리에게 높은 정확도를 달성한다는 잘못된 감각을 줄 수 있습니다.


```python
# 정확도 계산

acc_logreg = cross_val_score(clf_logreg, X_train, y_train, cv = cv, scoring = 'accuracy').mean()
acc_SGD = cross_val_score(clf_SGD, X_train, y_train, cv = cv, scoring = 'accuracy').mean()
acc_rfc = cross_val_score(clf_rfc, X_train, y_train, cv = cv, scoring = 'accuracy').mean()

acc_logreg, acc_SGD, acc_rfc
```




    (0.7797035692679977, 0.611222020568663, 0.7606473079249849)



### 로그 손실 / 로그 손실 / 로지스틱 손실 / 교차 엔트로피 손실

- 로그 손실로 작업 할 때 분류기는 모든 샘플에 대해 각 클래스에 확률을 할당해야합니다.
- 로그 손실은 실제 레이블과 비교하고 잘못된 분류에 페널티를 적용하여 모델 확률의 불확실성을 측정합니다.
- 로그 손실은 둘 이상의 레이블에 대해서만 정의됩니다.
- 로그 손실은 예측 확률이 향상됨에 따라 점차 감소하므로 로그 손실이 0에 가까울수록 정확도가 높아지고 로그 손실이 0에서 멀어지면 정확도가 낮아집니다.
- 로그 손실은 (0, ∞] 범위에 있습니다.

M 클래스에 속하는 N 개의 샘플이 있다고 가정하면 로그 손실은 다음과 같이 계산됩니다.  
$$ Log\ Loss = \frac{-1}{N} \sum_{i=1}^{N} \sum_{i=1}^{M}  y_{ij} * \log(\hat{y_{ij}})$$   
- $y_{ij}$ ,샘플 i가 클래스 j에 속하는지 여부를 나타냅니다.
- $p_{ij}$ ,샘플 i가 클래스 j에 속하는 확률을 나타냅니다.

음수 부호 부정  $\log(\hat{y_{ij}})$  항상 음수 인 출력.  $\hat{y_{ij}}$  확률 (0-1)을 출력하고,  $\log(x)$  0 <x <1 인 경우 음수입니다.  

**예**:  
학습 레이블은 0과 1이지만 학습 예측은 0.4, 0.6, 0.89 등입니다. 모델의 오류 측정 값을 계산하기 위해 0.5보다 큰 값을 갖는 모든 관측 값을 1로 분류 할 수 있습니다. 우리는 오 분류를 증가시킬 위험이 높습니다. 확률이 0.4, 0.45, 0.49 인 많은 값이 1의 참값을 가질 수 있기 때문입니다.  
이것이 logLoss가 등장하는 곳입니다.  
이제 LogLoss의 공식을 자세히 살펴 보겠습니다. 값에 대한 4 가지 주요 사례가있을 수 있습니다. $y_{ij}$  과  $p_{ij}$ 
- 사례 1 :  $y_{ij}$j =1 ,  $p_{ij}$  = 높음
- 사례 2 :  $y_{ij}$ =1 ,  $p_{ij}$  = 낮음
- 사례 3 :  $y_{ij}$ =0 ,  $p_{ij}$  = 낮음
- 사례 4 :  $y_{ij}$ =0 ,  $p_{ij}$  = 높음

LogLoss는 불확실성을 어떻게 측정합니까?  
케이스 1과 케이스 3이 더 많이있는 경우 로그 로스 공식 내부의 합계 (및 평균)는 케이스 2와 케이스 4가 추가 된 경우에 비해 훨씬 더 커질 것입니다. 이제이 값은 좋은 예측을 나타내는 Case 1 및 Case 3만큼 큽니다. (-1)을 곱하면 값을 가능한 한 작게 만듭니다. 이것은 이제 직관적으로 의미합니다.-값이 작을수록 모델이 더 좋습니다. 즉, 로그 손실이 더 작고, 모델이 더 좋습니다. 즉, 불확실성이 더 작고, 모델이 더 좋습니다.


```python
# logloss 계산

logloss_logreg = cross_val_score(clf_logreg, X_train, y_train, cv = cv, scoring = 'neg_log_loss').mean()
logloss_rfc = cross_val_score(clf_rfc, X_train, y_train, cv = cv, scoring = 'neg_log_loss').mean()

# SGDClassifier의 힌지 손실은 확률 추정을 지원하지 않습니다.
# Scikit-learn의 CalibratedClassifierCV에서 SGDClassifier를 기본 추정기로 설정하여 확률 추정치를 생성 할 수 있습니다.

from sklearn.calibration import CalibratedClassifierCV

new_clf_SGD = CalibratedClassifierCV(clf_SGD)
new_clf_SGD.fit(X_train, y_train)
logloss_SGD = cross_val_score(new_clf_SGD, X_train, y_train, cv = cv, scoring = 'neg_log_loss').mean()

logloss_logreg, logloss_SGD, logloss_rfc
```




    (-0.48368646454082465, -0.6383384003043665, -0.4664817973667718)



### ROC 곡선

### AUC

### 혼동 매트릭스

### 분류 보고서

### 정밀도-재현율 트레이드 오프

### 결론

## 회귀 지표

### 평균 절대 오차

### 평균 제곱 오차

### RMSE

### 평균 제곱근 로그 오차

### R_ 제곱

### 조정 된 R- 제곱

## NLP 메트릭

## 보너스

### 다중 클래스 분류

### 다중 라벨 분류

### 다중 출력 분류
