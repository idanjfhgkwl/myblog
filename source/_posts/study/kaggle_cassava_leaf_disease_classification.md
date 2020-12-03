---
title: "캐글 카사바 잎 질병 분류 파파고 번역"
categories:
  - study
output:
  html_document:
    keep_md: true
---

# 구글 연동 (생략)


```python
!pip install kaggle
```


```python
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
  print('uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  
# kaggle.json을 아래 폴더로 옮긴 뒤, file을 사용할 수 있도록 권한을 부여한다. 
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```


```python
ls -1ha ~/.kaggle/kaggle.json
```


```python
from google.colab import drive # 패키지 불러오기 
from os.path import join  

# 구글 드라이브 마운트
ROOT = "/content/drive"     # 드라이브 기본 경로
print(ROOT)                 # print content of ROOT (Optional)
drive.mount(ROOT)           # 드라이브 기본 경로 

# 프로젝트 파일 생성 및 다운받을 경로 이동
MY_GOOGLE_DRIVE_PATH = 'My Drive/Colab Notebooks/python_basic/kaggle_cassava-leaf-disease-classification/data'
PROJECT_PATH = join(ROOT, MY_GOOGLE_DRIVE_PATH)
print(PROJECT_PATH)
```


```python
%cd "{PROJECT_PATH}"
```

# **카사바 잎 질병 분류**  
Cassava Leaf Disease Classification  
https://www.kaggle.com/c/cassava-leaf-disease-classification  
<br>
이미지에 존재하는 질병 유형 식별  
Identify the type of disease present on a Cassava Leaf image

## **Description**
<br><br>
아프리카에서 두 번째로 많은 탄수화물을 공급하고 있는 카사바는 가혹한 조건을 견뎌낼 수 있기 때문에 소작농들이 재배하는 주요 식량안보 작물이다.  
As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions.    
<br>
아프리카 사하라 사막 이남의 가정 농장의 80% 이상이 이 녹농 뿌리를 기르고 있지만 바이러스성 질병은 수확량이 저조한 주요 원인이다.  
At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields.  
<br>
데이터 과학의 도움으로, 일반적인 질병들이 치료될 수 있도록 식별하는 것이 가능할 수도 있다.  
With the help of data science, it may be possible to identify common diseases so they can be treated.  
<br><br>

기존의 질병감지 방법은 농업인이 정부출연 농업전문가의 도움을 받아 식물을 육안으로 검사하고 진단할 수 있도록 해야 한다.  
Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants.  
<br>
이것은 노동집약적이고, 공급량이 적고, 비용이 많이 드는 것으로 고통받고 있다.  
This suffers from being labor-intensive, low-supply and costly.  
<br>
추가적인 도전으로서, 아프리카 농부들은 낮은 대역폭의 모바일 퀄리티 카메라에만 접근할 수 있기 때문에 농부들을 위한 효과적인 해결책은 상당한 제약 조건 하에서 좋은 성과를 거두어야 한다.  
As an added challenge, effective solutions for farmers must perform well under significant constraints, since African farmers may only have access to mobile-quality cameras with low-bandwidth.  
<br><br>

이번 대회에서는 우간다의 정기 조사 때 수집한 21,367개의 라벨 이미지 데이터 세트를 소개한다.  
In this competition, we introduce a dataset of 21,367 labeled images collected during a regular survey in Uganda.  
<br>
대부분의 이미지는 그들의 정원을 사진 찍는 농부들로부터 크라우드소싱되었고, 캄팔라 소재 마케레대학의 AI 연구소와 협력하여 국립작물자원연구소(NaCRRI)의 전문가들이 주석을 달았다.  
Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala.  
<br>
이것은 농부들이 실생활에서 진단해야 할 것을 가장 현실적으로 나타내는 형식이다.  
This is in a format that most realistically represents what farmers would need to diagnose in real life.  

## **Data Description**  

비교적 저렴한 카메라의 사진을 사용하여 카사바 공장의 문제점을 식별할 수 있는가?  
Can you identify a problem with a cassava plant using a photo from a relatively inexpensive camera?  
<br>
이 대회는 많은 아프리카 국가들의 식량 공급에 물질적인 해를 끼치는 여러 질병들을 구별하는 것에 도전할 것이다.  
This competition will challenge you to distinguish between several diseases that cause material harm to the food supply of many African countries.  
<br>
어떤 경우에는 더 이상의 확산을 막기 위해 감염된 식물을 태우는 것이 주요 치료법인데, 이것은 농부들에게 꽤 유용한 빠른 자동 전환이 될 수 있다.  
In some cases the main remedy is to burn the infected plants to prevent further spread, which can make a rapid automated turnaround quite useful to the farmers.

## **Files**  

**[train/test]_images** the image files.  
<br>
테스트 이미지의 전체 세트는 노트북이 채점을 위해 제출되었을 때만 사용할 수 있다.  
The full set of test images will only be available to your notebook when it is submitted for scoring.  
<br>
테스트 세트에서 약 15,000개의 이미지를 볼 수 있을 것으로 예상한다.  
Expect to see roughly 15,000 images in the test set.  
<br>
**train.csv**
<br>
- `image_id` the image file name.  
- `label` 질병의 ID 코드 (the ID code for the disease)  
<br>  

**sample_submission.csv**  
공개된 테스트 세트 내용을 고려할 때 적절한 형식의 샘플 제출.
<br>
A properly formatted sample submission, given the disclosed test set content.
<br>
- `image_id` the image file name.  
- `label` 질병의 예상 ID 코드 (the predicted ID code for the disease)  
<br>  

**[train/test]_tfrecords**  
tfrecord 형식의 이미지 파일  
the image files in tfrecord format.  
<br> 
**label_num_to_disease_map.json**  
각 질병 코드와 실제 질병 이름 간의 매핑.  
The mapping between each disease code and the real disease name.

# **카사바 잎 질병 - EDA(탐색적 데이터 분석)**
Cassava Leaf Disease - Exploratory Data Analysis   
https://www.kaggle.com/ihelon/cassava-leaf-disease-exploratory-data-analysis

Cassava Leaf 질병 분류 과제를 위한 빠른 탐색 데이터 분석  
Quick Exploratory Data Analysis for Cassava Leaf Disease Classification challenge  
<br>
이 대회는 많은 아프리카 국가들의 식량 공급에 물질적인 해를 끼치는 여러 질병들을 구별하는 것에 도전할 것이다.  
This competition will challenge you to distinguish between several diseases that cause material harm to the food supply of many African countries.  
  
어떤 경우에는 더 이상의 확산을 막기 위해 감염된 식물을 태우는 것이 주요 치료법인데, 이것은 농부들에게 꽤 유용한 빠른 자동 전환이 될 수 있다.  
In some cases the main remedy is to burn the infected plants to prevent further spread, which can make a rapid automated turnaround quite useful to the farmers.

## Overview


```python
import os
import json

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from sklearn import metrics as sk_metrics
```


```python
BASE_DIR = "../input/cassava-leaf-disease-classification/"
```

이번 대회에는 5개의 수업이 있다: **4개의 질병**과 **1개의 건강**  
클래스 번호와 클래스 이름 간의 매핑은 파일 label_num_to_disease_map.json에서 찾을 수 있다.  
  
In this competition we have 5 classes: **4 diseases** and **1 healthy**  
We can find the mapping between the class number and its name in the file label_num_to_disease_map.json


```python
with open(os.path.join(PROJECT_PATH, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    
print(json.dumps(map_classes, indent=4))
```

    {
        "0": "Cassava Bacterial Blight (CBB)",
        "1": "Cassava Brown Streak Disease (CBSD)",
        "2": "Cassava Green Mottle (CGM)",
        "3": "Cassava Mosaic Disease (CMD)",
        "4": "Healthy"
    }
    


```python
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")
```

**이하 용량이 커서 결과 나중에**

![K-20201126-092849](https://user-images.githubusercontent.com/72365720/100294328-ce3d7c00-2fc9-11eb-8b04-5c67610d2bda.png)

처음 300개의 이미지 치수를 살펴봅시다.  
아래에서 볼 수 있듯이 모든 이미지는 크기가 동일하다(600, 800, 3)  
Let's take a look at the dimensions of the first 300 images  
As you can see below, all images are the same size (600, 800, 3)  


```python
img_shapes = {}
for image_name in os.listdir(os.path.join(BASE_DIR, "train_images"))[:300]:
    image = cv2.imread(os.path.join(BASE_DIR, "train_images", image_name))
    img_shapes[image.shape] = img_shapes.get(image.shape, 0) + 1

print(img_shapes)
```

![K-20201126-093038](https://user-images.githubusercontent.com/72365720/100294402-0e9cfa00-2fca-11eb-82fc-af4ebe528d11.png)

교육 데이터 프레임을 로드하고 실제 클래스 이름이 포함된 열을 추가합시다.  
Let's load the training dataframe and add a column with the real class name to it.  



```python
df_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))

df_train["class_name"] = df_train["label"].astype(str).map(map_classes)

df_train
```

![K-20201126-093157](https://user-images.githubusercontent.com/72365720/100294470-3ee49880-2fca-11eb-9aba-2abc9b6e96e6.png)

각 반의 사진 수를 살펴보자.  
Let's look at the number of pictures in each class.


```python
plt.figure(figsize=(8, 4))
sn.countplot(y="class_name", data=df_train);
```

![__results___14_0](https://user-images.githubusercontent.com/72365720/100294542-6c314680-2fca-11eb-969f-5dd5f3e83144.png)

우리가 알 수 있듯이 데이터 집합은 상당히 큰 불균형을 가지고 있다.  
As we can see, the dataset has a fairly large imbalance.


## 일반 시각화 (General Visualization)


```python
def visualize_batch(image_ids, labels):
    plt.figure(figsize=(16, 12))
    
    for ind, (image_id, label) in enumerate(zip(image_ids, labels)):
        plt.subplot(3, 3, ind + 1)
        image = cv2.imread(os.path.join(BASE_DIR, "train_images", image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(f"Class: {label}", fontsize=12)
        plt.axis("off")
    
    plt.show()
```


```python
tmp_df = df_train.sample(9)
image_ids = tmp_df["image_id"].values
labels = tmp_df["class_name"].values

visualize_batch(image_ids, labels)
```

![__results___18_0](https://user-images.githubusercontent.com/72365720/100294635-bf0afe00-2fca-11eb-917d-1783e7009cfa.png)

## Cassava Bacterial Blight (CBB)

![inbox_1865449_be9cdd94efb9b1660066ad10b55c8626_bact_bright](https://user-images.githubusercontent.com/72365720/100295143-39884d80-2fcc-11eb-936f-0059a1b61a6b.jpeg)  
The image from discussion: [Cassava Lead Diseases: Overview](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/198143)


```python
tmp_df = df_train[df_train["label"] == 0]
print(f"Total train images for class 0: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(9)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values

visualize_batch(image_ids, labels)
```

![K-20201126-094743](https://user-images.githubusercontent.com/72365720/100295287-9b48b780-2fcc-11eb-9285-f29207d79ba6.png)

![__results___21_1](https://user-images.githubusercontent.com/72365720/100295234-79e7cb80-2fcc-11eb-86e3-ff0e144ed678.png)

## Cassava Brown Streak Disease (CBSD)

![inbox_1865449_feba3dafc914d04517659650d137b77a_brown_st](https://user-images.githubusercontent.com/72365720/100295367-c7fccf00-2fcc-11eb-9ee6-752ff3e9a02e.jpeg)  
The image from discussion: [Cassava Lead Diseases: Overview](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/198143)


```python
tmp_df = df_train[df_train["label"] == 1]
print(f"Total train images for class 1: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(9)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values

visualize_batch(image_ids, labels)
```

![K-20201126-095102](https://user-images.githubusercontent.com/72365720/100295429-ecf14200-2fcc-11eb-81d4-9db76e8e0637.png)

![__results___24_1](https://user-images.githubusercontent.com/72365720/100295437-f2e72300-2fcc-11eb-8b1a-4cfb178de9ea.png)

## Cassava Green Mottle (CGM)

![inbox_1865449_4f2975866feb2a1d4ef4111c2d57db29_green_mottle](https://user-images.githubusercontent.com/72365720/100295584-4eb1ac00-2fcd-11eb-8bad-a137d5d6d0fb.jpeg)  
The image from discussion: [Cassava Lead Diseases: Overview](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/198143)


```python
tmp_df = df_train[df_train["label"] == 2]
print(f"Total train images for class 2: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(9)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values

visualize_batch(image_ids, labels)
```

![K-20201126-095448](https://user-images.githubusercontent.com/72365720/100295638-71dc5b80-2fcd-11eb-96de-ecbc4b6b3a2f.png)

![__results___27_1](https://user-images.githubusercontent.com/72365720/100295641-7274f200-2fcd-11eb-8768-13096f77e77e.png)


## Cassava Mosaic Disease (CMD)

![inbox_1865449_36990f77ded6667e5c30d19b5405d4d3_mosaic_disease](https://user-images.githubusercontent.com/72365720/100295912-3b531080-2fce-11eb-94a4-6ebb00a0c249.jpeg)  
The image from discussion: [Cassava Lead Diseases: Overview](https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/198143)


```python
tmp_df = df_train[df_train["label"] == 3]
print(f"Total train images for class 3: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(9)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values

visualize_batch(image_ids, labels)
```

![K-20201126-100145](https://user-images.githubusercontent.com/72365720/100295991-6a698200-2fce-11eb-992a-29d8bf0db854.png)

![__results___30_1](https://user-images.githubusercontent.com/72365720/100295994-6b9aaf00-2fce-11eb-8310-2d703ee9233a.png)

## Healthy


```python
tmp_df = df_train[df_train["label"] == 4]
print(f"Total train images for class 4: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(9)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values

visualize_batch(image_ids, labels)
```

![K-20201126-100242](https://user-images.githubusercontent.com/72365720/100296037-8c630480-2fce-11eb-818c-144ea91386cc.png)

![__results___32_1](https://user-images.githubusercontent.com/72365720/100296042-8ec55e80-2fce-11eb-985b-ca44e8c3c63e.png)

## 확대 예제 (Augmentation Examples)

이미지 증가는 기존 교육 사례에서 새로운 교육 사례를 만드는 과정이다.  
Image augmentation is a process of creating new training examples from the existing ones.  

새로운 샘플을 만들려면 원본 이미지를 약간 변경하십시오.  
To make a new sample, you slightly change the original image.  

예를 들어, 새로운 이미지를 조금 더 밝게 만들 수 있다.  
For instance, you could make a new image a little brighter;  

원본 이미지에서 조각을 잘라낼 수 있다.  
you could cut a piece from the original image;  

원래 이미지를 미러링하는 등의 방법으로 새로운 이미지를 만들 수 있다.  
you could make a new image by mirroring the original one, etc. [source](https://albumentations.ai/docs/introduction/image_augmentation/)

![augmentation](https://user-images.githubusercontent.com/72365720/100296265-14e1a500-2fcf-11eb-9423-bdea28961091.jpg)  
The image from the [Albumentations Documentation](https://albumentations.ai/docs/introduction/image_augmentation/)


```python
def plot_augmentation(image_id, transform):
    plt.figure(figsize=(16, 4))
    img = cv2.imread(os.path.join(BASE_DIR, "train_images", image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    x = transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    x = transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    
    plt.show()
```

우리는 몇몇 수업의 수가 상당히 제한되어 있기 때문에, 우리는 증강을 사용할 수 있다.  
Since we have a fairly limited number of some classes, we can use augmentation  

이 절은 연금술 라이브러리를 사용한 증축의 예를 보여준다.  
This section shows examples of augmentation using the albumentations library  
<br>

아래 예제는 지정학적 가장자리 보완과 함께 회전-시프트-척도 확대를 사용한다.  
The example below uses rotate-shift-scale augmentation with specular edge complementation.  

이런 종류의 사진을 보면, 이런 증가는 꽤 자연스러워 보인다.  
For this kind of pictures, this augmentation looks quite natural.


```python
transform_shift_scale_rotate = A.ShiftScaleRotate(
    p=1.0, 
    shift_limit=(-0.3, 0.3), 
    scale_limit=(-0.1, 0.1), 
    rotate_limit=(-180, 180), 
    interpolation=0, 
    border_mode=4, 
)

plot_augmentation("1003442061.jpg", transform_shift_scale_rotate)
```

![__results___39_0](https://user-images.githubusercontent.com/72365720/100296395-6e49d400-2fcf-11eb-95b5-28647fac52d4.png)

또 다른 유용한 증가는 ThoughDropout일 수 있다.  
Another useful augmentation could be CoarseDropout.  

이 확대 덕분에, 당신은 모델의 수명을 복잡하게 만들 수 있어서 그녀가 이미지의 일부 세부사항을 너무 자세히 보지 않도록 할 수 있다.  
Thanks to this augmentation, you can complicate the life of the model so that she does not look too closely at some of the details of the image.  

아래의 예를 보자.  
Let's look at the example below:


```python
transform_coarse_dropout = A.CoarseDropout(
    p=1.0, 
    max_holes=100, 
    max_height=50, 
    max_width=50, 
    min_holes=30, 
    min_height=20, 
    min_width=20,
)

plot_augmentation("1003442061.jpg", transform_coarse_dropout)
```

![__results___41_0](https://user-images.githubusercontent.com/72365720/100296476-aea95200-2fcf-11eb-85ea-e4585ee247f3.png)

우리는 두 개 이상의 증강을 하나의 과정으로 구성할 수 있다.  
We can compose two or more augmentations into one process.  

예를 들어 shift-scale-rotate 및 ThoughDropout을 일관되게 사용합시다.  
For example, let's use shift-scale-rotate and CoarseDropout consistently:


```python
transform = A.Compose(
    transforms=[
        transform_shift_scale_rotate,
        transform_coarse_dropout,
    ],
    p=1.0,
)

plot_augmentation("1003442061.jpg", transform)
```

![__results___43_0](https://user-images.githubusercontent.com/72365720/100296522-d4cef200-2fcf-11eb-9d3c-d47762b11e9c.png)

## Submission Example

제출 템플릿 로드  
Load the submission template


```python
df_sub = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv", index_col=0)
df_sub
```

![K-20201126-101306](https://user-images.githubusercontent.com/72365720/100296604-05169080-2fd0-11eb-9a3c-2a5ded46bda4.png)

제출 파일에서 하나의 파일만 볼 수 있기 때문에  
As we can see only one file in the submission file


```python
os.listdir(os.path.join(BASE_DIR, "test_images"))
```

![K-20201126-101414](https://user-images.githubusercontent.com/72365720/100296650-26777c80-2fd0-11eb-901c-9fba48d1bd80.png)

코드 대회인데다 시험 데이터가 숨겨져 있기 때문이다.  
This is because it is a Code Competition, and the test data is hidden  

노트북에서 볼 수 없는 테스트 데이터 집합으로 작업을 수정해야 함  
Your notebook should correct working with unseen test dataset  

테스트 이미지의 전체 세트는 노트북이 채점을 위해 제출되었을 때만 사용할 수 있다.  
The full set of test images will only be available to your notebook when it is submitted for scoring.  

테스트 세트에서 약 15,000개의 이미지를 볼 수 있을 것으로 예상한다.  
Expect to see roughly 15,000 images in the test set.  
<br>

이 경기의 척도는 **정확성**이다.  
The metric of this competition is **Accuracy**.  

정확도 - 총 표본 수에 대해 올바르게 예측된 표본 수의 비율  
Accuracy - the ratio of the number of samples predicted correctly to the total number of samples

![K-20201126-101734](https://user-images.githubusercontent.com/72365720/100296861-9e45a700-2fd0-11eb-8c80-051105d4dcf5.png)

모든 예에 대해 하나의 클래스만 선택하면 교육 세트의 정확도를 계산해 봅시다.  
Let's calculate the accuracy on a training set if we select only one class for all examples.


```python
for pred_class in range(0, 5):
    y_true = df_train["label"].values
    y_pred = np.full_like(y_true, pred_class)
    print(f"accuracy score (predict {pred_class}): {sk_metrics.accuracy_score(y_true, y_pred):.3f}")
```

![K-20201126-101838](https://user-images.githubusercontent.com/72365720/100296919-c3d2b080-2fd0-11eb-80a4-373fbb48049f.png)

우리는 계층의 불균형이 크기 때문에 가장 빈번한 수업을 예측하면 이 경우 정확도가 더 크다.  
Since we have a large imbalance of classes, if we predict the most frequent class, then our accuracy is greater in this case  

테스트 세트에 있는 모든 이미지의 라벨로 가장 인기 있는 트레이닝 세트의 클래스를 선택하자.  
Let's choose the most popular class of training set as the label for all images in test set


```python
df_sub["label"] = 3
```

그리고 나서 제출 파일에 결과를 쓰세요.  
And then write result to the submission file


```python
df_sub.to_csv("submission.csv")
```

평가를 위해 결과를 제출하면 공용 라이더보드(열차 위는 0.615)에서 0.614의 정확도를 얻는다.  
If you submit the result for evaluation, you will get an accuracy of 0.614 on a public liderboard (on the train it is 0.615).  

이는 공공 시험 분포에 대한 계층의 불균형도 있음을 나타낼 수 있다.  
This may indicate that there is also an imbalance of classes on the public test distribution.

# WORK IN PROGRESS...
