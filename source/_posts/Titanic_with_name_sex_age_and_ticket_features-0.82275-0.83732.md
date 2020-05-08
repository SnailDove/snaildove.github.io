---
title: kaggle首战Titanic 0.82275-Top3% & 0.83732-Top2%
mathjax: true
mathjax2: true
categories: 中文
tags: [kaggle]
date: 2019-01-06
commets: true
copyright: true
toc: true
top: 12
---

![](http://q9kvrafcq.bkt.clouddn.com/gitpages/kaggle/Titanic/1.png)

![](http://q9kvrafcq.bkt.clouddn.com/gitpages/kaggle/Titanic/2.png)

本文用数据分析探索规律，效果好于一堆的随机森林和xgboost，超过参加这个比赛的很多ensemble模型，至少排在前156/10021（Top 2%），最终只选择 name，sex，age，Ticket 4个特征，构建出新的特征，然后进行规则判断，即多个嵌套的if-else，再一次感受到了特征工程的强大。省了数据缺失弥补，其他繁琐的数据预处理，数据清洗，后续的调参和集成模型。
需要注意的是：需要自己定制交叉验证函数。

具体方案细节，查看我的jupyter notebook： 
1. [Titanic_with_name_sex_age_and_ticket_features-0.82275.ipynb](https://nbviewer.jupyter.org/github/SnailDove/github-blog/blob/master/Titanic_with_name_sex_age_and_ticket_features-0.82275.ipynb)
2. [Titanic_with_name_sex_age_and_ticket_features-0.83732.ipynb](https://nbviewer.jupyter.org/github/SnailDove/github-blog/blob/master/Titanic-0.837.ipynb)
