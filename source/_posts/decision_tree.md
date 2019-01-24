---
title: 决策树学习
mathjax: true
mathjax2: true
categories: 中文
date: 2015-05-01 20:16:00
tags: [Machine Learning]
commets: true
toc: true
---


## 决策树学习

决策树学习通常包含三个方面：特征选择、决策树生成和决策树剪枝。决策树学习思想主要来源于：Quinlan在1986年提出的ID算法、在1993年提出的C4.5算法和Breiman等人在1984年提出的CART算法。

## 特征选择

为了解释清楚各个数学概念，引入例子

表5.1  贷款申请样本数据表（来自[李航《统计方法》](https://book.douban.com/subject/10590856/)）

![img](http://pltr89sz6.bkt.clouddn.com/gitpage/TongJiXueXiFangfa/decision_tree/1.png)

​       上表有15个样本数据组成的贷款申请训练数据D。数据包括贷款申请人的4个特征：年龄、有工作与否、有房子与否、信贷情况，其中最后一列类别的意思是：是否同意发放贷款，这个就是决策树最后要给出的结论，即目标属性——是否发放贷款，即决策树最末端的叶子节点只分成2类：同意发放贷款与不同意发放贷款。

### 信息熵（entropy）

​    引入概念，对于第一个要用到的概念：信息熵在另外一篇博客——[数据压缩与信息熵](http://2.mybrtzl.sinaapp.com/%e8%bd%ac%e6%95%b0%e6%8d%ae%e5%8e%8b%e7%bc%a9%e4%b8%8e%e4%bf%a1%e6%81%af%e7%86%b5/)中详细解释了信息熵为什么度量的是不确定性，下文也不再赘述，直接引用。

​     设D为按照目标类别（或称目标属性）对训练数据（即样本数据）进行的划分，则D的信息熵（information entropy）表示为：
$$info(D)=-\sum\limits_{i=1}^{m}p_ilog_2(p_i)$$

​     其中pi表示第i个类别在整个训练数据中出现的概率，可以用属于此类别元素的数量除以训练数据（即样本数据）总数量作为估计。

具体问题具体分析

在上表中目标类别：是否发放贷款，将9个发放归为一类，剩余6个不发放归为一类，这样进行分类的信息熵为：

$$H(D)=-\frac{9}{15}log_2\frac{9}{15}-\frac{6}{15}log_2\frac{6}{15}=0.971$$

注：这个根据目标类别分类得出的信息熵，在样本给出的情况下就已经知晓，根据概率统计，也称经验熵。

​      现在我们假设将训练数据D按属性A进行划分，则按A属性进行分裂出的v个子集（即树中的v个分支），这些子集按目标类别（发放与不发放两类）进行分类所对应的熵的期望（即：按属性A划分出不同子集的信息熵的平均值）：

$$info_A(D)=\sum\limits_{j=1}^{v}\frac{|D_j|}{|D|}info(D_j)$$

注：这个实际上是经验条件熵，因为确认是在A属性划分出子集的前提下再按照目标类别分类得出的熵的期望，见下文信息增益计算就可以一目了然。

### 信息增益（information gain）

为上述两者的差值：

$$gain(A)=info(D)-info_A(D)$$

具体问题具体分析

-   按照年龄属性（记为A1）划分：青年（D1表示），中年（D2表示），老年（D3表示）

$$\begin{align}g(D, A_1) &= H(D) - [\frac{5}{15}H(D_1) + \frac{5}{15}H(D_2) + \frac{5}{15}H(D_3)] \\ &= 0.971 - [ \frac{5}{15}(-\frac{2}{5}log_2\frac{2}{5}-\frac{3}{5}log_2\frac{3}{5})+\frac{5}{15}(-\frac{3}{5}log_2\frac{3}{5} - \frac{2}{5}log_2\frac{2}{5}) + \frac{5}{15}(-\frac{4}{5}log_2\frac{4}{5} - \frac{1}{5}log_2\frac{1}{5})] \\ &= 0.971 - 0.888 \\ &= 0.083   \end{align}$$

-   按照是否有工作（记为A2）划分：有工作（D1表示），无工作（D2表示）

$$\begin{align}g(D, A_2) &= H(D) - [\frac{5}{15}H(D_1) + \frac{5}{15}H(D_2) ] \\ &= 0.971 - [ \frac{5}{15}\times 0+\frac{10}{15}(-\frac{4}{10}log_2\frac{4}{10} - \frac{6}{10}log_2\frac{6}{10})] \\ &= 0.324   \end{align}$$

-   按照是否有自己房子（记为A3）划分：有自己房子（D1表示），无自己房子（D2表示）

$$\begin{align}g(D, A_3) &= 0.971 - [ \frac{6}{15}\times 0+\frac{9}{15}(-\frac{3}{9}log_2\frac{3}{9} - \frac{6}{9}log_2\frac{6}{9})] \\ &= 0.971 - 0.551 \\ &= 0.420   \end{align}$$

-   同理，根据最后一个属性：信贷情况算出其信息增益：

$$g(D, A_4) = 0.971 - 0.608 = 0.363$$

所以可以看出信息增益度量的是：信息熵的降低量，这个降低是经过某个属性对原数据进行划分得出的。信息熵的降低，即确定性的提高，进一步讲，就是类别的数量在下降，那么确定为哪一类的可能性就提高，这样就更容易分类了。ID3算法就是基于信息增益来衡量属性（即特征）划分数据的能力，进而为特征（即属性）选择提供原则。

-   ### 增益比率（gain ratio）

信息增益选择方法有一个很大的缺陷，它总是会倾向于选择属性值多的属性，如果我们在上面的数据记录中加一个姓名属性，假设15条记录中的每个人姓名不同，那么信息增益就会选择姓名作为最佳属性，因为按姓名分裂后，每个组只包含一条记录，而每个记录只属于一类（要么发放要么不发放），因此不确定性最低，即纯度最高，（注：为什么最高呢？大家可以根据导数计算一下，最大值的情况，这里不赘述）以姓名作为测试分裂的结点下面有15个分支。但是这样的分类没有意义，它没有任何泛化能力。增益比率对此进行了改进，它引入一个分裂信息：

$$SplitInfo_R(D)=-\sum\limits_{j=1}^{k}\frac{|D_j|}{D}\times log_2(\frac{|D_j|}{D})$$

注：分裂信息即按照某个属性划分的信息熵，而本文前面叙述的熵全部是按照目标属性进行分类的信息熵。

　　增益比率定义为信息增益与分裂信息的比率：

　$$GainRatio(R)=\frac{Gain(R)}{SplitInfo_R(D)}$$

我们找GainRatio最大的属性作为最佳分裂属性。如果一个属性的取值很多，那么SplitInfoR(D)会大，从而使GainRatio(R)变小。不过增益比率也有缺点，SplitInfo(D)可能取0，此时没有计算意义；且当SplitInfo(D)趋向于0时，GainRatio(R)的值变得不可信，改进的措施就是在分母加一个平滑，这里加一个所有分裂信息的平均值：

$$GainRatio(R)=\frac{Gain(R)}{\overline{SplitInfo(D)}+SplitInfo_R(D)}$$



C4.5算法就是按照信息增益比来计算各属性的分类能力，进而为特征（即属性）选择提供原则。

-   ### 基尼指数（Gini coefficient）

定义（基尼指数）：在分类问题中，假设有K个类，样本点属于第K类的概率为p(k)，则概率分布的基尼指数定义为

$$[Gini(p)=\sum\limits_{k=1}^{K}p_k(1-p_k)=1-\sum\limits_{k=1}^{K}p_k^2]$$

对于2分类问题，若样本属于第一类的概率是p，则概率分布的基尼指数为：

$$Gini(p)=2p(1-p)$$

对于给定的样本集合D的基尼指数为：

$$Gini(D)=1-\sum\limits_{k=1}^{K}\left(\frac{|C_k|}{|D|}\right)^2$$

这里，C(k)是D中属于第k类的样本子集，K是类的个数。

如果样本集合D根据特征A是否取某一可能值α被分割成D1和D2两部分，即

$$D_1 = \{(x, y) \in D| A(x)=a\}, D_2=D - D_1$$

则在特征A的条件下，集合D的基尼指数定义为

$$Gini(D, A) = \frac{|D_1|}{|D|}Gini(D_1) + \frac{|D_2|}{D}Gini(D_2)$$

基尼指数Gini(D)表示集合D的不确定性，基尼指数Gini(D, A)表示经A=α分割后集合D的不确定性。基尼指数值越大，样本集合的不确定性也就越大，这一点与熵相似。

## ID3算法

-   ### 信息增益算法

    （来自[李航《统计方法》](https://book.douban.com/subject/10590856/)）

![img](http://pltr89sz6.bkt.clouddn.com/gitpage/TongJiXueXiFangfa/decision_tree/2.png)

-   ### ID3算法

    （来自[李航《统计方法》](https://book.douban.com/subject/10590856/)）

![img](http://pltr89sz6.bkt.clouddn.com/gitpage/TongJiXueXiFangfa/decision_tree/3.png)

-   ### 示例

    （来自[李航《统计方法》](https://book.douban.com/subject/10590856/)）

![img](http://pltr89sz6.bkt.clouddn.com/gitpage/TongJiXueXiFangfa/decision_tree/4.png)

## C4.5生成算法

（来自[李航《统计方法》](https://book.douban.com/subject/10590856/)）

![img](http://pltr89sz6.bkt.clouddn.com/gitpage/TongJiXueXiFangfa/decision_tree/5.png)

##  CART生成算法

（来自[李航《统计方法》](https://book.douban.com/subject/10590856/)）

![img](http://pltr89sz6.bkt.clouddn.com/gitpage/TongJiXueXiFangfa/decision_tree/6.png)

### 示例

### ![img](http://pltr89sz6.bkt.clouddn.com/gitpage/TongJiXueXiFangfa/decision_tree/7.png)
