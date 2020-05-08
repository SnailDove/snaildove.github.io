---
title: Transformer论文简记
mathjax: true
mathjax2: true
categories: 中文
tags: [Machine Learning, NLP]
date: 2019-03-10
comments: true
copyright: true
toc: true
top: 11
---

## 资源

Transformer来自论文: [All Attention Is You Need](https://arxiv.org/abs/1706.03762)

别人的总结资源：

1. 谷歌官方AI博客: [Transformer: A Novel Neural Network Architecture for Language Understanding](http://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
2. [Attention机制详解（二）——Self-Attention与Transformer](https://zhuanlan.zhihu.com/p/47282410)  谷歌软件工程师
3. 一个是Jay Alammar可视化地介绍Transformer的博客文章 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)，非常容易理解整个机制，建议先从这篇看起，[这是中文翻译版本](https://zhuanlan.zhihu.com/p/54356280)；
4. [放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941) 中科院软件所 · 自然语言处理 /搜索 10年工作经验的博士（阿里，微博）；
5. Calvo的博客：[Dissecting BERT Part 1: The Encoder](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3)，尽管说是解析Bert，但是因为Bert的Encoder就是Transformer，所以其实它是在解析Transformer，里面举的例子很好；
6. 再然后可以进阶一下，参考哈佛大学NLP研究组写的“[The Annotated Transformer.](http://nlp.seas.harvard.edu/2018/04/03/attention.html) ”，代码原理双管齐下，讲得也很清楚。
7. [《Attention is All You Need》浅读（简介+代码）](https://kexue.fm/archives/4765) 这个总结的角度也很棒。

## 总结

这里总结的思路：自顶向下方法

### model architecture

一图胜千言，6层编码器和解码器，论文中没有说为什么是6这个特定的数字
![](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)

#### Encoder

![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm.png)

#### Decoder

如果我们想做堆叠了**2**个Encoder和**2**个Decoder的Transformer，那么它可视化就会如下图所示：
![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)

翻译输出的时候，前一个时间步的输出，要作为下一个时间步的解码器端的输入，下图展示第2~6步：![](http://q9kvrafcq.bkt.clouddn.com/gitpages/nlp/transformer/transformer_decoding_2.gif)

下面是一个单层：Nx 表示 N1, ... , N6 层

![](http://q9kvrafcq.bkt.clouddn.com/gitpages/nlp/transformer/All-attetion-is-you-need_%20architecture.jpg)



### parts

#### Multi-head Attention

其实就是多个Self-Attention结构的结合，每个head学习到在不同表示空间中的特征，所谓“多头”（Multi-Head），就是做h次同样的事情（参数不共享），然后把结果拼接。
![](https://kexue.fm/usr/uploads/2018/01/2809060486.png)

#### Self-Attention

实际上是scaled dot-product attention 缩放的点积注意力：

![](http://q9kvrafcq.bkt.clouddn.com/gitpages/nlp/transformer/All-attetion-is-you-need_Scaled-Dot-Product-Attention.jpg)



#### Add

residual connection: skip connection 跳跃了解

#### Norm

layer norm 归一化层

#### Positional encoding

google的这个位置编码很魔幻，是两个周期函数：sine cosine
数学系出生的博主的解释：[《Attention is All You Need》浅读（简介+代码）](https://kexue.fm/archives/4765)，相比之下Bert的位置编码直观的多。
