---
title: "8장 기본 명령어"
#author: "JustY"
#date: '2020 10 30 '
categories:
  - sql_10minutes
output: 
  html_document:
    keep_md: true
---


## 8.1 SELECT

### 테이블 전체를 검색하려면 *

``` sql
SELECT *
FROM   TB_CUSTOMER;
```

### 필드가 많은 테이블에서 필요한 내용만 검색하려면 ,

``` sql
SELECT CUSTOMER_CD,
       CUSTOMER_NM,
       PHONE_NUMBER,
       EMAIL
FROM   TB_CUSTOMER;
```

### 필드 제목을 한글로 바꾸려면 AS (생략가능)




