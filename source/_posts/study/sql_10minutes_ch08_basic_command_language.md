---
title: "8장 기본 명령어"
#author: "JustY"
#date: '2020 10 30 '
categories:
  - study
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

``` sql
SELECT CUSTOMER_CD AS 고객코드,
       CUSTOMER_NM AS 고객명,
       PHONE_NUMBER AS 전화번호,
       EMAIL AS 이메일
FROM   TB_CUSTOMER;
```

수정한 게 안 올라간다




