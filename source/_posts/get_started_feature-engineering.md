---
title: 从特征工程到XGBoost参数调优
mathjax: true
mathjax2: true
categories: 中文
tags: [Machine Learning.feature engineering, XGBoost]
date: 2018-12-18
comments: true
copyright: true
toc: true
top: 11
---

# 前言

本文陈述脉络：理论结合kaggle上一个具体的比赛。

# 正文

## 数据科学的一般流程

![1548322618454](http://q9kvrafcq.bkt.clouddn.com/gitpages/ML/get_started_xgboost/1548322618454.png)

## 指南

- 特征工程
- 评价指标
- XGBoost参数调优
- XGBoost并行处理 

## 特征工程

结合以下案例分析：

[Two Sigma Connect: Rental Listing Inquiries](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries)

任务：根据公寓的listing 内容，预测纽约市某公寓租赁listing的受欢迎程度
标签： interest_level，该listing被咨询的次数

选择这个案例是因为**小而精**，虽然只有14维特征，但是基本上都涉及**各种类型特征**。

1. 有三个取值：: 'high', 'medium', 'low'，是一个多类分类任务
2. Listing内容有：
    - 浴室和卧室的数目bathrooms， bedrooms
    - 地理位置（ longitude 、 latitude ）
    - 地址： display_address、 street_address
    - building_id、 listing_id、 manager_id
    - Created：创建日期
    - Description：更多描述信息
    - features: 公寓的一些特征描述
    - photos: a list of photo links
    - 价格：price 

###  数据分析方法

对数据进行探索性的分析的工具包：pandas、 matplotlib／seaborn

1. 读取训练数据，取少量样本进行观测，并查看数据规模和数据类型

    - 标签、特征意义、特征类型等

2. 分析每列特征的分布

    - 直方图

    - 包括标签列（对分类问题，可看出类别样本是否均衡）

    - 检测奇异点（outliers）

    - 分析每两列特征之间的相关性
        – 特征与特征之间信息是否冗余
        – 特征与标签是否线性相关

### histogram 直方图

1. 直方图：每个取值在数据集中出现的次数，可视为概率函
    数（PDF）的估计（seaborn可视化工具比较简单）

    ```python3
    import seaborn as sns
    %matplotlib inline（ seaborn 是基于matplotlib 的）
    sns.distplot(train.price.values, bins=50, kde=True)
    ```
2. 核密度估计

  - Kernel Density Estimation, KDE
  - 对直方图的加窗平滑 

![1548331998361](http://q9kvrafcq.bkt.clouddn.com/gitpages/ML/get_started_xgboost/1548331998361.png)

**在分类任务中，我们关心不同类别的特征分布**

**violinplot** 提供不同类别条件下特征更多的分部信息 核密度估计（KDE） 三个4分位数（quartile）：1/4, 1/2, 3/4 1.5倍四分数间距（nterquartile range, IQR） IQR ：第三四分位数和第一分位数的区别（即Q1~Q3的差距），表示变量的分散情况，播放差更稳健的统计量

```python
order = ['low', 'medium', 'high']
sns.violinplot(x='interest_level', y='price', data=train, order = order) 
```

![1548331822211](http://q9kvrafcq.bkt.clouddn.com/gitpages/ML/get_started_xgboost/1548331822211.png)

### outliers 奇异点

奇异点：或称离群点，指远离大多数样本的样本点。通常认为这些点是噪声，对模型有坏影响

1. 可以通过直方图或散点图发现奇异点

    - 直方图的尾巴
    - 散点图上孤立的点

    ```python
    plt.figure(figsize=(8,6))
    plt.scatter(range(train_df.shape[0]), train_df.price.values, color = color[6])
    plt.xlabel('the number of train data', fontsize=12)
    plt.ylabel('price', fontsize=12)
    plt.show()
    ```

    ![1548331724663](http://q9kvrafcq.bkt.clouddn.com/gitpages/ML/get_started_xgboost/1548331724663.png)

2. 可以通过只保留某些分位数内的点去掉奇异点

    - 如0.5%-99.5%，或>99%

```python
ulimit = np.percentile(train.price.values, 99)
train['price'].loc[train['price']>ulimit] = ulimit 
```

### correlation 相关性

- 相关性可以通过计算相关系数或打印散点图来发现

    ![1548330721971](http://q9kvrafcq.bkt.clouddn.com/gitpages/ML/get_started_xgboost/1548330721971.png)

- 相关系数：两个随机变量x,y之间的**线性**相关程度，**不线性相关并不代表不相关**，可能高阶相关，如 $y=x^2$

    1. $r = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=0}^{n}}(x_i-\bar{x})^2\sum_{i=0}^{n}(y_i-\bar{y})^2}, -1\le r \le 1$

    2. 通常 $|r| > 0.5$ ，认为两者相关性比较强

    3. $r\cases{=0, &\text{完全线性不相关}\\ >0, &\text{正相关}\\ <0, &\text{负相关}}$

- 相关性只能是数值型特征之间相关性

    - 我们希望特征与标签强相关，分类直方图可以从某种程度上看出特征与标签的相关性：不同类别的直方图差异大

        ```python
        order = ['low', 'medium', 'high']
        sns.stripplot(train_df.interest_level, train_df.price.values, jitter=True, order=order)
        plt.title("Price VS Interest Level")
        plt.show()
        ```

        ![1548331676891](http://q9kvrafcq.bkt.clouddn.com/gitpages/ML/get_started_xgboost/1548331676891.png)

    - 特征与特征之间强相关的话意味着信息冗余
        1. 可以两个特征可以只保留一个特征

        2. 或采用主成分分析（PCA）等降维

            ```python
            contFeaturelist = []
            contFeaturelist.append('bathrooms')
            contFeaturelist.append('bedrooms')
            contFeaturelist.append('price')
            
            correlationMatrix = train_df[contFeaturelist].corr().abs()
            plt.subplots()
            sns.heatmap(correlationMatrix, annot=True)
            
            #Mask unimportant features
            sns.heatmap(correlationMatrix, mask=correlationMatrix < 1, cbar = False)
            plt.show()
            ```

            ![1548332212260](http://q9kvrafcq.bkt.clouddn.com/gitpages/ML/get_started_xgboost/1548332212260.png)

### 数据类型

XGBoost 模型内部将所有的问题都建模成一个回归预测问题，输入特征只能是数值型。如果给定的数据是不同的类型，必须先将数据变成数值型。 

### 类别型特征（ categorical features）

- **LabelEncoder**： 对不连续的数字或者文本进行编号
    1. 可用在对字符串型的标签编码（测试结果需进行反变换）
    2. 编号默认有序数关系
    3. 存储量小

- 如不希望有序数关系： **OneHotEncoder**：将类别型整数输入从 1 维 K 维的稀疏编码（K 种类别）
    1. 对XGBoost，OneHotEncoder不是必须，因为XGBoost对特征进行排序从而进行分裂建树；如果用OneHotEncoder得到稀疏编码，XGBoost建树过程中对稀疏特征处理速度块
    2. 输入必须是数值型数据（对字符串输入，先调用LabelEncoder变成数字，再用OneHotEncoder ）
    3. 存储要求高 
- **低基数（low-cardinality ）**类别型特征： OneHotEncoder 
    1. 1维到K维， K为该特征不同的取值数目 
    2. 通常在K <10的情况下采用 
- **高基数（high-cardinality）**类别型特征：通常有成百上千个不同的取值，可先降维，如：邮政编码、街道名称…
    1. 聚类（Clustering）： 1 维 到 K维，K为聚类的类别数
    2. 主成分分析（principle component analysis, PCA）：但对大矩阵操作费资源
    3. 均值编码：在贝叶斯的架构下，利用标签变量，有监督地确定最适合特定特征的编码方式。均值编码详细参考：
        - [Mean Encoding: A Preprocessing Scheme for High-Cardinality Categorical Features ](http://helios.mm.di.uoa.gr/~rouvas/ssi/sigkdd/sigkdd.vol3.1/barreca.pdf) 
        - [平均数编码：针对高基数定性特征（类别特征）的数据预处理/特征工程](https://zhuanlan.zhihu.com/p/26308272)

### 日期型特征

1. 日期特征：年月日
2. 时间特征：小时分秒
3. 时间段：早中晚
4. 星期，工作日／周末 

```python
train_test['Date'] = pd.to_datetime(train_test['created'])
train_test['Year'] = train_test['Date'].dt.year
train_test['Month'] = train_test['Date'].dt.month
train_test['Day'] = train_test['Date'].dt.day
train_test['Wday'] = train_test['Date'].dt.dayofweek
train_test['Yday'] = train_test['Date'].dt.dayofyear
train_test['hour'] = train_test['Date'].dt.hour
 
train_test = train_test.drop(['Date', 'created'], axis=1)
```

### 文本型特征
- 可用词云（wordcloud）可视化
    - 文本词频统计函数，自动统计词的个数，以字典形式内部存储，在显示的时候词频大的词的字体更大 

```python
# wordcloud for street address
plt.figure()
wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40)
wordcloud.generate(text_street)
wordcloud.recolor(random_state=0)
plt.imshow(wordcloud)
plt.title("Wordcloud for street address", fontsize=30)
plt.axis("off")
plt.show()
```

![1548332336363](http://q9kvrafcq.bkt.clouddn.com/gitpages/ML/get_started_xgboost/1548332336363.png)

- TF-IDF

    1. 通俗易懂原理参考：廖雪峰老师的TF-IDF，概率解释参考：CoolShell 陈皓的 TF-IDF

    2. 实战参考官网和[使用sklearn提取文本的tfidf特征](https://www.jianshu.com/p/c7e2771eccaa)

下面是个例子
```python
from sklearn.feature_extraction.text import TfidfVectorizer

X_train = ['This is the first document.', 'This is the second document.']
X_test = ['This is the third document.']
vectorizer = TfidfVectorizer()
# 用X_train数据来fit
vectorizer.fit(X_train)
# 得到tfidf的矩阵
tfidf_train = vectorizer.transform(X_train)
tfidf_test = vectorizer.transform(X_test)

tfidf_train.toarray()
```
### 数据预处理

`from sklearn.preprocessing import …`

1. 数据标准化
2. 数据归一化
3. 数据二值化
4. 数据缺失

XGBoost对数据预处理要求少，以上操作都不是必须

### 特征工程小结

- 如果知道数据的物理意义（领域专家），可能可以设计更多特征
    - 如Higgs Boson任务中有几维特征是物理学家设计的，还有些有高能物理
        研究经验的竞赛者设计了其他一些特征
    - 如房屋租赁任务中，利用常识可设计出一些特征，例子：租金/卧室数目=单价
- 如果不是领域专家，一些通用的规则：
    - 字符串型特征：Label编码
    - 时间特征：年月日、时间段（早中晚）…
    - 数值型特征：加减乘除，多项式，log, exp
    - 低基数类别特征：one-hot编码
    - 高基数类别特征：先降维，再one-hot编码；均值编码 
- **非结构化特征**
    - 文本
    - 语音
    - 图像／视频
    - fMRI
    - …
- **利用领域知识设计特征**
    - 如曾经流行的图像目标检测特征HOG…
- **利用深度学习从数据中学习特征表示**
    - 采用end-to-end方式一起学习特征和分类／回归／排序
    - 学习好特征可以送入XGBoost学习器 

### 信息泄漏

训练数据特征不应该包含标签的信息
– 如Rent Listing Inquries任务中图片压缩文件里文件夹的创造时间：加入这个特征后，模型普遍能提高0.01的public LB分数

## 特征工程案例实践

这是我的 jupyter notebook: [Rent Listing Inquries](https://nbviewer.jupyter.org/github/SnailDove/github-blog/blob/master/4_FE_RentListingInqueries.ipynb)

## 评价指标

### 回归问题的评价指标

损失函数可以作为评价指标，以下约定俗成： $\hat{y_i}$ 是预测值，$y$ 是标签值

1. L1: mean absolute error (MAE)  

     $MAE = \frac{1}{N}\sum_{i=0}^{N}|\hat{y_i}-y_i|$

2. L2: Root Mean Squared Error(RMSE)

    $RMSE = \sqrt{\frac{1}{N}\sum_{i=0}^{N}|\hat{y_i}-y_i|}$

3. Root Mean Sqared Logarithmic Error (RMSLE)

    - $RMLSE = \sqrt{\frac{1}{N}\sum_{i=0}^{N} \big( \log(\hat{y_i}+1) - \log(y_i+1) \big) ^2}$
    - https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError
    - 当不想在给预测值与真值差距施加很大惩罚时，采用RMSLE


![1548607142536](http://q9kvrafcq.bkt.clouddn.com/gitpages/ML/get_started_xgboost/1548607142536.png)

### 分类任务的评价指标

同样地，损失函数可以作为评价指标

1. logistic/负log似然损失

    $\text{logloss}= -{1\over N}\sum_{i=0}^{N}\sum_{j=0}^{M}y_{ij}\log{P_{ij}}$，M是类别数，$y_{ij}$ 为二值函数，当 i 个样本是第 j 类时 $y_{ij}=1$ ，否则取 $0$ ；$p_{ij}$ 为模型预测的第 i 个样本为第 j 类的概率。当 $M=2$ 时， $\text{logloss} = -{1\over N}\sum_{i=0}^{N}\big(y_i\log p_i + (1-y_i)\log(1-p_i)\big)$ ，$y_i$ 为第 i 个样本类别，$p_i$ 为模型预测的第 i 个样本为第 1 类的概率。

2. 0-1损失对应的Mean Consequential Error (MCME) 

    $\text{MCE}=-\frac{1}{N}\sum\limits_{\hat{y_i}\ne y_i}1$ 

### 两类分类任务中更多评价指标

1. ROC／AUC
2. PR曲线
3. MAP@n 

- 0-1损失：假设两种错误的代价相等
    ​    False Positive （FP） & False Negative（FN）

- 有些任务中可能某一类错误的代价更大

    - 如蘑菇分类中将毒蘑菇误分为可食用代价更大
    - 因此单独列出每种错误的比例：混淆矩阵

- 混淆矩阵（confusion matrix）

    - 真正的正值（true positives）

    - 假的正值（false positives）

    - 真正的负值（true negatives）

    - 假的负值（false negatives ） 

- SciKit-Learn实现了多类分类任务的混淆矩阵

    `sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)`

    1.  y_true： N个样本的标签真值
    2.  y_pred： N个样本的预测标签值
    3.  labels：C个类别在矩阵的索引顺序，缺省为y_true或y_pred类别出现的顺序
    4.  sample_weight： N个样本的权重 

![1548607271453](http://q9kvrafcq.bkt.clouddn.com/gitpages/ML/get_started_xgboost/1548607271453.png)

#### Receiver Operating Characteristic (ROC)



上面我们讨论给定阈值 $τ$ 的TPR和FPR

- 如果不是只考虑一个阈值，而是在一些列阈值上运行检测器，并画出TPR和FPR为阈值 $τ$ 的隐式函数，得到ROC曲线在此处键入公式。 

    ![1548607383824](http://q9kvrafcq.bkt.clouddn.com/gitpages/ML/get_started_xgboost/1548607383824.png)

#### PR曲线

Precision and Recall (PR曲线)：用于**稀有事件检测**，如目标检测、信息检索

- 负样本非常多，因此  FPR = FP /N_ 很小，比较 TPR 和 FPR 不是很有用的信息 （ROC曲线中只有左边很小一部分有意义） $\rightarrow$ 只讨论正的样本

- Precision（精度，查准率，正确 率）：以信息检索为例，对于一个查询，返回了一系列的文档，正确率指的是返回结果中相关文档占的比例 

    - $\text{precision}= TP /\hat{N}_+$ 预测结果真正为正的比例 

- Recall（召回率，查全率）：返回结果中相关文档占所有相关文档的比例

  - $\text{Recall}=TP/N_+$  被正确预测到正样本的比例

- Precision and Recall (PR曲线) 

  - 阈值变化时的P 和R ，只考虑了返回结果中相关文档的个数，没有考虑文档之间的序。

    - 对一个搜索引擎或推荐系统而言，返回的结果必然是有序的，而且越相关的文档排的越靠前越好，于是有了 AP 的概念。
    - AP: Average Precision，对不同召回率点上的正确率进行平均。

    ![1548607476814](http://q9kvrafcq.bkt.clouddn.com/gitpages/ML/get_started_xgboost/1548607476814.png)

#### Average Precision 

Precision只考虑了返回结果中相关文档的个数，没有考虑文档之间的序。

- 对一个搜索引擎或推荐系统而言，返回的结果必然是有序的，而且越相关的文档排的越靠前越好，于是有了AP的概念。
- AP: Average Precision，对不同召回率点上的正确率进行平均
    - $AP = \int_{0}^{1}p(k)dr = \sum_{k=0}^{n}p(k)\Delta r(k)$
    - 即 PR 曲线下的面积（Recall: AUC 为 ROC 下的面积）
    - 其中 k 为返回文档中序位，n 为返回文档的数目，$p(k)$ 为列表中k截至点的precision，$\Delta r(k)$ 表示从 $k-1$ 到 $k$ 召回率的变化
- 上述离散求和表示等价于 $AP=\sum_{k=0}^{n}p(k)rel(k)/\text{相关文档的数目}$ ，其中 $rel(k)$ 为示性函数，即第 $k$ 个位置为相关文档则取1，否则取0。 
- 计算每个位置上的 precision，如果该位置的文档是不相关的则该位置 precision=0，然后对所有的位置的precision 再做 average 。

### MAP: Mean Average Precision 

- $MAP = (\sum_{q=0}^{Q}AP(q)/(Q))$ ，其中 $Q$ 为查询的数目，$n$ 为文档数目。

### MAP@K （MAPK） 

- 在现代web信息检索中，recall其实已经没有意义，因为相关文档有成千上万个，很少有人会关心所有文档

- Precision@K：在第K个位置上的Precision

    对于搜索引擎，考虑到大部分作者只关注前一、两页的结果，所以Precision @10， Precision @20对大规模搜索引擎非常有效 

- MAP@K：多个查询Precision@K的平均 

### F1 分数/调和平均

- 亦被称为F1 score, balanced F-score or F-measure
- Precision 和 Recall 加权平均： $F1=\frac{2*(\text{Precision * Recall)}}{(\text{Precision + Recall)}}$
    - 最好为1，最差为0
    - 多类：每类的F1平均值 

### Scikit-Learn: Scoring

- 用交叉验证（cross_val_score和GridSearchCV）评价模型性能时，用scoring参数定义评价指标。

- 评价指标是越高越好，因此用一些损失函数当评价指标时，需要再加负号，如neg_log_loss，neg_mean_squared_error

- 详见sklearn文档：http://scikit-learn.org/stable/modules/model_evaluation.html

| Scoring| Function | Comment|
| ------------------------------ | ------------------------------------------------------------ | -------------------------------- |
| **Classification**             |                                                              |                                  |
| ‘accuracy’                     | [`metrics.accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score) |                                  |
| ‘balanced_accuracy’            | [`metrics.balanced_accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score) | for binary targets               |
| ‘average_precision’            | [`metrics.average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) |                                  |
| ‘brier_score_loss’             | [`metrics.brier_score_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss) |                                  |
| ‘f1’                           | [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) | for binary targets               |
| ‘f1_micro’                     | [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) | micro-averaged                   |
| ‘f1_macro’                     | [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) | macro-averaged                   |
| ‘f1_weighted’                  | [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) | weighted average                 |
| ‘f1_samples’                   | [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) | by multilabel sample             |
| ‘neg_log_loss’                 | [`metrics.log_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss) | requires `predict_proba` support |
| ‘precision’ etc.               | [`metrics.precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) | suffixes apply as with ‘f1’      |
| ‘recall’ etc.                  | [`metrics.recall_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score) | suffixes apply as with ‘f1’      |
| ‘roc_auc’                      | [`metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score) |                                  |
| **Clustering**                 |                                                              |                                  |
| ‘adjusted_mutual_info_score’   | [`metrics.adjusted_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score) |                                  |
| ‘adjusted_rand_score’          | [`metrics.adjusted_rand_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score) |                                  |
| ‘completeness_score’           | [`metrics.completeness_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score) |                                  |
| ‘fowlkes_mallows_score’        | [`metrics.fowlkes_mallows_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html#sklearn.metrics.fowlkes_mallows_score) |                                  |
| ‘homogeneity_score’            | [`metrics.homogeneity_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score) |                                  |
| ‘mutual_info_score’            | [`metrics.mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html#sklearn.metrics.mutual_info_score) |                                  |
| ‘normalized_mutual_info_score’ | [`metrics.normalized_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score) |                                  |
| ‘v_measure_score’              | [`metrics.v_measure_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score) |                                  |
| **Regression**                 |                                                              |                                  |
| ‘explained_variance’           | [`metrics.explained_variance_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score) |                                  |
| ‘neg_mean_absolute_error’      | [`metrics.mean_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error) |                                  |
| ‘neg_mean_squared_error’       | [`metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error) |                                  |
| ‘neg_mean_squared_log_error’   | [`metrics.mean_squared_log_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error) |                                  |
| ‘neg_median_absolute_error’    | [`metrics.median_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error) |                                  |
| ‘r2’                           | [`metrics.r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score) |                                  |

### SciKit-Learn：sklearn.metrics
- metrics模块还提供为其他目的而实现的预测误差评估函数
    分类任务的评估函数如表所示，其他任务评估函数请见：http://scikitlearn.org/stable/modules/classes.html#module-sklearn.metrics 

#### Classification metrics

See the [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) section of the user guide for further details.

| [`metrics.accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)(y_true, y_pred[, …]) | Accuracy classification score.                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.auc`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc)(x, y[, reorder]) | Compute Area Under the Curve (AUC) using the trapezoidal rule |
| [`metrics.average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)(y_true, y_score) | Compute average precision (AP) from prediction scores        |
| [`metrics.balanced_accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score)(y_true, y_pred) | Compute the balanced accuracy                                |
| [`metrics.brier_score_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss)(y_true, y_prob[, …]) | Compute the Brier score.                                     |
| [`metrics.classification_report`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report)(y_true, y_pred) | Build a text report showing the main classification metrics  |
| [`metrics.cohen_kappa_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score)(y1, y2[, labels, …]) | Cohen’s kappa: a statistic that measures inter-annotator agreement. |
| [`metrics.confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)(y_true, y_pred[, …]) | Compute confusion matrix to evaluate the accuracy of a classification |
| [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)(y_true, y_pred[, labels, …]) | Compute the F1 score, also known as balanced F-score or F-measure |
| [`metrics.fbeta_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#sklearn.metrics.fbeta_score)(y_true, y_pred, beta[, …]) | Compute the F-beta score                                     |
| [`metrics.hamming_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss)(y_true, y_pred[, …]) | Compute the average Hamming loss.                            |
| [`metrics.hinge_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html#sklearn.metrics.hinge_loss)(y_true, pred_decision[, …]) | Average hinge loss (non-regularized)                         |
| [`metrics.jaccard_similarity_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html#sklearn.metrics.jaccard_similarity_score)(y_true, y_pred) | Jaccard similarity coefficient score                         |
| [`metrics.log_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss)(y_true, y_pred[, eps, …]) | Log loss, aka logistic loss or cross-entropy loss.           |
| [`metrics.matthews_corrcoef`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef)(y_true, y_pred[, …]) | Compute the Matthews correlation coefficient (MCC)           |
| [`metrics.precision_recall_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)(y_true, …) | Compute precision-recall pairs for different probability thresholds |
| [`metrics.precision_recall_fscore_support`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support)(…) | Compute precision, recall, F-measure and support for each class |
| [`metrics.precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)(y_true, y_pred[, …]) | Compute the precision                                        |
| [`metrics.recall_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)(y_true, y_pred[, …]) | Compute the recall                                           |
| [`metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)(y_true, y_score[, …]) | Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores. |
| [`metrics.roc_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)(y_true, y_score[, …]) | Compute Receiver operating characteristic (ROC)              |
| [`metrics.zero_one_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.zero_one_loss.html#sklearn.metrics.zero_one_loss)(y_true, y_pred[, …]) | Zero-one classification loss.                                |

#### Regression metrics

See the [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) section of the user guide for further details.

| [`metrics.explained_variance_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score)(y_true, y_pred) | Explained variance regression score function                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.mean_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error)(y_true, y_pred) | Mean absolute error regression loss                          |
| [`metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)(y_true, y_pred[, …]) | Mean squared error regression loss                           |
| [`metrics.mean_squared_log_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error)(y_true, y_pred) | Mean squared logarithmic error regression loss               |
| [`metrics.median_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error)(y_true, y_pred) | Median absolute error regression loss                        |
| [`metrics.r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)(y_true, y_pred[, …]) | R^2 (coefficient of determination) regression score function. |

#### Multilabel ranking metrics

See the [Multilabel ranking metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics) section of the user guide for further details.

| [`metrics.coverage_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.coverage_error.html#sklearn.metrics.coverage_error)(y_true, y_score[, …]) | Coverage error measure                  |
| ------------------------------------------------------------ | --------------------------------------- |
| [`metrics.label_ranking_average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html#sklearn.metrics.label_ranking_average_precision_score)(…) | Compute ranking-based average precision |
| [`metrics.label_ranking_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_loss.html#sklearn.metrics.label_ranking_loss)(y_true, y_score) | Compute Ranking loss measure            |

**<font color="red">XGBoost 原理部分参考：</font>** [XGBoost第一课](/2018/10/02/get-started-XGBoost/)

## XGBoost支持的目标函数


objective参数，这个参数在 XGBoost 里面属于任务参数（Learning Task Parameters）

 [default=reg:linear]

- `reg:linear`: linear regression
- `reg:logistic`: logistic regression
- `binary:logistic`: logistic regression for binary classification, output probability
- `binary:logitraw`: logistic regression for binary classification, output score before logistic transformation
- `binary:hinge`: hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
- `count:poisson` –poisson regression for count data, output mean of poisson distribution 计数问题的poisson回归，输出结果为poisson分布。 
    - `max_delta_step` is set to 0.7 by default in poisson regression (used to safeguard optimization)
- `survival:cox`: Cox regression for right censored survival time data (negative values are considered right censored). Note that predictions are returned on the hazard ratio scale (i.e., as HR = exp(marginal_prediction) in the proportional hazard function `h(t) = h0(t) * HR`).
- `multi:softmax`: set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes) 让XGBoost采用softmax目标函数处理多分类问题 
- `multi:softprob`: same as softmax, but output a vector of `ndata * nclass`, which can be further reshaped to `ndata * nclass` matrix. The result contains predicted probability of each data point belonging to each class. 和softmax一样，但是输出的是ndata * nclass的向量，可以将该向量
    reshape成ndata行nclass列的矩阵。没行数据表示样本所属于每个类别的概率。 
- `rank:pairwise`: Use LambdaMART to perform pairwise ranking where the pairwise loss is minimized
- `rank:ndcg`: Use LambdaMART to perform list-wise ranking where [Normalized Discounted Cumulative Gain (NDCG)](http://en.wikipedia.org/wiki/NDCG) is maximized
- `rank:map`: Use LambdaMART to perform list-wise ranking where [Mean Average Precision (MAP)](http://en.wikipedia.org/wiki/Mean_average_precision#Mean_average_precision) is maximized
- `reg:gamma`: gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be [gamma-distributed](https://en.wikipedia.org/wiki/Gamma_distribution#Applications).
- `reg:tweedie`: Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be [Tweedie-distributed](https://en.wikipedia.org/wiki/Tweedie_distribution#Applications).

## XGBoost自定义目标函数

- 在GBDT训练过程，当每步训练得到一棵树，要调用目标函数得到其梯度作为下一棵树拟合的目标
- XGBoost在调用obj函数时会传入两个参数：preds和dtrain

    - preds为当前模型完成训练时，所有训练数据的预测值

    - dtrain为训练集，可以通过dtrain.get_label()获取训练样本的label

    - 同时XGBoost规定目标函数需返回当前preds基于训练label的一阶和二阶梯度

**例子**

参考官网：https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py

```python
#user define objective function, given prediction, return gradient and second order gradient
#this is log likelihood loss
def logregobj(preds, dtrain): #自定义损失函数
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels #梯度
    hess = preds * (1.0-preds) #2阶导数
    return grad, hess
```
调用的时候：
```python
# training with customized objective, we can also do step by step training
# simply look at xgboost.py's implementation of train
bst = xgb.train(param, dtrain, num_round, watchlist, obj=logregobj, feval=evalerror)
```

## XGBoost支持的评价函数

`eval_metric`参数，这个参数在 XGBoost 里面属于任务参数（Learning Task Parameters）

 [default according to objective]

- Evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and error for classification, mean average precision for ranking)
- User can add multiple evaluation metrics. Python users: remember to pass the metrics in as list of parameters pairs instead of map, so that latter `eval_metric` won’t override previous one
- The choices are listed below:
    - `rmse`: [root mean square error](http://en.wikipedia.org/wiki/Root_mean_square_error)
    - `mae`: [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
    - `logloss`: [negative log-likelihood](http://en.wikipedia.org/wiki/Log-likelihood)
    - `error`: Binary classification error rate. It is calculated as `#(wrong cases)/#(all cases)`. For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
    - `error@t`: a different than 0.5 binary classification threshold value could be specified by providing a numerical value through ‘t’.
    - `merror`: Multiclass classification error rate. It is calculated as `#(wrong cases)/#(all cases)`.
    - `mlogloss`: [Multiclass logloss](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).
    - `auc`: [Area under the curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_curve)
    - `aucpr`: [Area under the PR curve](https://en.wikipedia.org/wiki/Precision_and_recall)
    - `ndcg`: [Normalized Discounted Cumulative Gain](http://en.wikipedia.org/wiki/NDCG)
    - `map`: [Mean Average Precision](http://en.wikipedia.org/wiki/Mean_average_precision#Mean_average_precision)
    - `ndcg@n`, `map@n`: ‘n’ can be assigned as an integer to cut off the top positions in the lists for evaluation.
    - `ndcg-`, `map-`, `ndcg@n-`, `map@n-`: In XGBoost, NDCG and MAP will evaluate the score of a list without any positive samples as 1. By adding “-” in the evaluation metric XGBoost will evaluate these score as 0 to be consistent under some conditions.
    - `poisson-nloglik`: negative log-likelihood for Poisson regression
    - `gamma-nloglik`: negative log-likelihood for gamma regression
    - `cox-nloglik`: negative partial log-likelihood for Cox proportional hazards regression
    - `gamma-deviance`: residual deviance for gamma regression
    - `tweedie-nloglik`: negative log-likelihood for Tweedie regression (at a specified value of the `tweedie_variance_power` parameter)

## XGBoost自定义评价函数

例子参考官网：https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py

```python
# user defined evaluation function, return a pair metric_name, result
# NOTE: when you do customized loss function, the default prediction value is margin
# this may make builtin evaluation metric not function properly
# for example, we are doing logistic loss, the prediction is score before logistic transformation
# the builtin evaluation error assumes input is after logistic transformation
# Take this in mind when you use the customization, and maybe you need write customized evaluation function
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'my-error', float(sum(labels != (preds > 0.0))) / len(labels)

# training with customized objective, we can also do step by step training
# simply look at xgboost.py's implementation of train
bst = xgb.train(param, dtrain, num_round, watchlist, obj=logregobj, feval=evalerror)
```

调用的时候：

```python
# training with customized objective, we can also do step by step training
# simply look at xgboost.py's implementation of train
bst = xgb.train(param, dtrain, num_round, watchlist, obj=logregobj, feval=evalerror)
```

## XGBoost参数调优

### XGBoost参数列表

| 参数             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| **max_depth** | 树的最大深度。树越深通常模型越复杂，更容易过拟合。           |
| **learning_rate** | 学习率或收缩因子。学习率和迭代次数／弱分类器数目n_estimators相关。 缺省：0.1 |
| **n_estimators** | 弱分类器数目. 缺省:100                                       |
| slient           | 参数值为1时，静默模式开启，不输出任何信息                    |
| objective        | 待优化的目标函数，常用值有： binary:logistic 二分类的逻辑回归，返回预测的概率 multi:softmax 使用softmax的多分类器，返回预测的类别(不是概率)。 multi:softprob 和 multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率。支持用户自定义目标函数 |
| nthread          | 用来进行多线程控制。 如果你希望使用CPU全部的核，那就不用缺省值-1，算法会自动检测它。 |
| booster          | 选择每次迭代的模型，有两种选择： gbtree：基于树的模型，为缺省值。 gbliner：线性模型 |
| **gamma**        | 节点分裂所需的最小损失函数下降值                             |
| **min_child_weight** | 叶子结点需要的最小样本权重（hessian）和                      |
| max_delta_step   | 允许的树的最大权重                                           |
| **subsample**     | 构造每棵树的所用样本比例（样本采样比例），同GBM |
| **colsample_bytree** | 构造每棵树的所用特征比例                        |
| **colsample_bylevel** | 树在每层每个分裂的所用特征比例                  |
| **reg_alpha**     | L1/L0正则的惩罚系数                             |
| **reg_lambda**    | L2正则的惩罚系数                                |
| scale_pos_weight  | 正负样本的平衡                                  |
| base_score        | 每个样本的初始估计，全局偏差                    |
| random_state      | 随机种子                                        |
| seed              | 随机种子                                        |
| missing           | 当数据缺失时的填补值。缺省为np.nan              |
| kwargs            | XGBoost Booster的Keyword参数                    |

### 参数类别
1.  通用参数：这部分参数通常我们不需要调整，默认值就好
2. 学习目标参数：与任务有关，定下来后通常也不需要调整
3. booster参数：弱学习器相关参数，需要仔细调整，会影响模型性能 

### 通用参数
- booster：弱学习器类型

    - 可选gbtree（树模型）或gbliner（线性模型） 或 dart （参考我的另一篇博文： [XGBoost第一课](/2018/10/02/get-started-XGBoost/)）
    -  默认为gbtree（树模型为非线性模型，能处理更复杂的任务）

- silent：是否开启静默模式

    - 1：静默模式开启，不输出任何信息
    -  默认值为0：输出一些中间信息，以助于我们了解模型的状态

- nthread：线程数

    - 默认值为-1，表示使用系统所有CPU核 

### 学习目标参数

- objective: 损失函数
    - 	支持分类／回归／排
- eval_metric：评价函数
- seed：随机数的种子
    - 默认为0
    - 设置seed可复现随机数据的结果，也可以用于调整参数 

### booster参数

弱学习器的参数，尽管有两种booster可供选择，这里只介绍gbtree

1. learning_rate : 收缩步长 vs. n_estimators：树的数目

   - 较小的学习率通常意味着更多弱分学习器

   - 通常建议学习率较小（ $\eta < 0.1$ ）弱学习器数目n_estimators大 

        $f_m(x_i)=f_{m-1}(x_i)+\eta\beta_m WeakLearner_m(x_i) $
   - 可以设置较小的学习率，然后用交叉验证确定n_estimators

2. 行（subsample）列（colsample_bytree、colsample_bylevel）下采样比例

    - 默认值均为1，即不进行下采样，使用所有数据
    - 随机下采样通常比用全部数据的确定性过程效果更好，速度更快
    - 建议值：0.3 - 0.8

3. 树的最大深度： max_depth 

    - max_depth越大，模型越复杂，会学到更具体更局部的样本 
    - 需要使用交叉验证进行调优，默认值为6，建议3-10 

4. min_child_weight ：孩子节点中最小的样本权重和 

    - 如果一个叶子节点的样本权重和小于min_child_weight则分裂过程
        结束 

### Kaggle竞赛优胜者的建议

- Tong He（XGBoost R语言版本开发者）： 三个最重要的参数为：树的数目、树的深度和学习率。建议参数调整策略为： 

    1. 采用默认参数配置试试 

    2. 如果系统过拟合了，降低学习率 

    3. 如果系统欠拟合，加大学习率 

        **油管上作者视频**：[Kaggle Winning Solution Xgboost Algorithm - Learn from Its Author, Tong He](https://www.youtube.com/watch?v=ufHo8vbk6g4)

- Owen Zhang （常使用XGBoost）建议： 

    1. n_estimators和learning_rate：固定n_estimators为100（数目不大，因为树的深度较大，每棵树比较复杂），然后调整learning_rate 

    2. 树的深度max_depth：从6开始，然后逐步加大 

    3. $\text{min_child_weight}={1\over\sqrt{\text{rare_events}}}$ ，其中 rare_events 为稀有事件的数目

    4. 列采样  ${\text{colsample_bytree}\over \text{colsample_bylevel}}$ ：在 $[0.3,0.5]$ 之间进行网格搜索

    5. 行采样subsample：固定为1 

    6. gamma: 固定为0.0 

        **油管上大神的视频**：[Learn Kaggle techniques from Kaggle #1, Owen Zhang](https://www.youtube.com/watch?v=LgLcfZjNF44)

### 参数调优的一般方法
1. 选择较高的**学习率(learning rate)**，并选择对应于此学习率的理想的**树数量**
    - 学习率以工具包默认值为0.1。
    - XGBoost直接引用函数“cv”可以在每一次迭代中使用交叉验证，并返回理想的树数量（因为交叉验证很慢，所以可以import两种XGBoost：直接引用xgboost（用“cv”函数调整树的数目）和XGBClassifier —xgboost的sklearn包（用GridSearchCV调整其他参数 ）。
3. 对于给定的学习率和树数量，进行**树参数调优**(max_depth,
    min_child_weight, gamma, subsample, colsample_bytree, colsample_bylevel)
4. xgboost的**正则化参数**(lambda, alpha)的调优
5. 降低学习率，确定理想参数

### XGBoost参数调优案例分析

- 竞赛官网：[Otto Group Product Classification Challenge ](https://www.kaggle.com/c/otto-group-productclassification-challenge ) 是关于电商商品分类的案例，其中

1. Target：共9个商品类别
2. 93个特征：整数型特征 

详细请看我的jupyter notebook: []()

- kaggle Titanic 案例

详细请看我的jupyter notebook: []()

## XGBoost并行处理 

### XGBoost工程实现

1. XGBoost用C++实现，显示地采用OpenMP API做并行处理

    - 建单棵树时并行（Random Forest在建不同树时并行，但Boosting增加树是一个串行操作）

    - XGBoost的scikit-learn接口中的参数 nthread 可指定线程数
        - -1 表示使用系统所有的核资源
        - model = XGBClassifier(nthread=-1) 
2. 在准备建树数据时高效（近似建树、稀疏、 Cache、数据分块）
3. 交叉验证也支持并行（由scikit-learn 提供支持） 

    - scikit-learn 支持的k折交叉验证也支持多线程

        - cross_val_score() 函数中的参数：n_ jobs = -1 表示使用系统所有的CPU核
        - results = cross_val_score(model, X, label_encoded_y, cv=kfold,
            scoring= ’neg_log_loss’ , n_jobs=-1, verbose=1) 
4. 并行处理的三种配置
    - 交叉验证并行，XGBoost建树不并行
    - 交叉验证不并行，XGBoost建树并行
    - 交叉验证并行，XGBoost建树并行
5. Otto数据集上的10折交叉验证实验结果：
    - Single Thread XGBoost, Parallel Thread CV: 359.854589
    - Parallel Thread XGBoost, Single Thread CV: 330.498101
    - Parallel Thread XGBoost and CV: 313.382301，并行 XGBoost 比并行交叉验证好，两者都并行更好 

### 例子

查看

```python
# evaluate the effect of the number of threads
 results = []
 num_threads = [1, 2, 3, 4]
 for n in num_threads:
	start = time.time()
	model = XGBClassifier(nthread=n)
	model.fit(X_train, y_train)
	elapsed = time.time() - start
	print(n, elapsed)
	results.append(elapsed)
```

![1548608348576](http://q9kvrafcq.bkt.clouddn.com/gitpages/ML/get_started_xgboost/1548608348576.png)

## XGBoost总结
- XGBoost是一个用于监督学习的非参数模型
    - 目标函数（损失函数、正则项）
    - 参数（树的每个分支分裂特征及阈值）
    - 优化：梯度下降
    - 参数优化
        - 决定模型复杂度的重要参数：learning_rate, n_estimators,
            max_depth, min_child_weight, gamma, reg_alpha, reg_lamba
        - 随机采样参数也影响模型的推广性： subsample, colsample_bytree, colsample_bylevel

- 其他未涉及的部分
    - 分布式XGBoost
        - AWS
        - YARN Cluster
        - …
    - GPU加速
    - 并行计算与内存优化的细节
        - 主要关注XGBoost的对外接口
    
- XGBoost资源
    - XGBoost官方文档：https://xgboost.readthedocs.io/en/latest/
        - Python API：http://xgboost.readthedocs.io/en/latest/python/python_api.html
    - Github： https://github.com/dmlc/xgboost
        - 很多有用的资源：https://github.com/dmlc/xgboost/blob/master/demo/README.md
        - GPU加速：https://github.com/dmlc/xgboost/blob/master/plugin/updater_gpu/README.md
    - XGBoost原理：XGBoost: A Scalable Tree Boosting System
       - https://arxiv.org/abs/1603.02754
         

### 其他资源

- XGBoost参数调优：
    - https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuningxgboost-with-codes-python/
    - 中文版：http://blog.csdn.net/u010657489/article/details/51952785
- Owen Zhang, Winning Data Science Competitions
    - https://www.slideshare.net/OwenZhang2/tips-for-data-sciencecompetitions?from_action=save
- XGBoost User Group：
    - https://groups.google.com/forum/#!forum/xgboost-user/ 
