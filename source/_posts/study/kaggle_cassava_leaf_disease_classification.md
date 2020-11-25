---
title: "캐글 카사바 잎 질병 분류 파파고 번역"
categories:
  - study
output: 
  html_document:
    keep_md: true
---

```python
!pip install kaggle
```

    Requirement already satisfied: kaggle in /usr/local/lib/python3.6/dist-packages (1.5.9)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.24.3)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.0.1)
    Requirement already satisfied: slugify in /usr/local/lib/python3.6/dist-packages (from kaggle) (0.0.1)
    Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.15.0)
    Requirement already satisfied: certifi in /usr/local/lib/python3.6/dist-packages (from kaggle) (2020.11.8)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.8.1)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.41.1)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.23.0)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.6/dist-packages (from python-slugify->kaggle) (1.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (2.10)
    


```python
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
  print('uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  
# kaggle.json을 아래 폴더로 옮긴 뒤, file을 사용할 수 있도록 권한을 부여한다. 
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```



<input type="file" id="files-b05e94e7-734f-4e5e-9420-e376c5775847" name="files[]" multiple disabled
   style="border:none" />
<output id="result-b05e94e7-734f-4e5e-9420-e376c5775847">
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

    /content/drive
    Mounted at /content/drive
    /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_cassava-leaf-disease-classification/data
    


```python
%cd "{PROJECT_PATH}"
```

    /content/drive/My Drive/Colab Notebooks/python_basic/kaggle_cassava-leaf-disease-classification/data
    

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

**이하 용량이 커서 결과 생략**
