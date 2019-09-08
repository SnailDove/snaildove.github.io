---
title: Tsinghua linear-algebra-2 3rd-lecture Singular-Values-Decomposition
mathjax: true
mathjax2: true
categories: 中文
date: 2017-08-03 20:16:00
tags: [linear_algebra]
toc: true
top: 7
---

## 前言

笔记源自：清华大学公开课：线性代数2——第三讲：奇异值分解，**本文将每一步推导都详细列出来了**。

**清华大学线性代数2公开课笔记总结**

第1讲：[正定矩阵](https://snaildove.github.io/2017/08/01/positive_definite_matrix/)
第2讲：[相似矩阵]()
第3讲：[奇异值分解](https://snaildove.github.io/2017/08/03/singular_values_decomposition/)
第4讲：[线性变换1](https://snaildove.github.io/2017/08/04/linear_transformation_1st_part/)
第5讲：[线性变换2](https://snaildove.github.io/2017/08/05/linear_transformation_2nd_part/)
第6讲：[伪逆](https://snaildove.github.io/2017/08/06/pseudo_inverse/)
第7讲：[工程中的矩阵](https://snaildove.github.io/2017/08/07/engineering_matrices/)
第8讲：[图与网络](https://snaildove.github.io/2017/08/08/graph_and_network/)
第9讲：[Markov矩阵和正矩阵](https://snaildove.github.io/2017/08/06/Markov_matrix/)
第10讲：[Fourier级数](https://snaildove.github.io/2017/08/02/Fourier_series/)
第11讲：[计算机图像](https://snaildove.github.io/2017/08/11/computer_graphics/)
第12讲：[复数与复矩阵](https://snaildove.github.io/2017/08/12/complex_and_complex_matrix/)

## 正文

对角矩阵是我们最喜欢的一类矩阵，对能够相似于对角阵的矩阵能方便地计算其幂和指数，对不能相似于对角阵的方阵。上节课我们讨论了如何求出其尽可能简单的相似标准形及Jordan标准形以上讨论的都是方阵。那么对m乘n的矩阵我们如何来对它进行对角化呢？

线性代数中最重要的一类矩阵分解即奇异值分解，从而回答以上的问题。**对角矩阵是我们最喜欢的一类矩阵，因为给定一个对角阵立即就可以得到它的特征值，行列式，幂和指数函数等等。对角矩阵的运算跟我们熟悉的数的运算有很多相似之处，而一个n阶的矩阵相似于对角阵当且仅当它存在着n个线性无关的特征向量。** 
![preface_SVD](http://pwmpcnhis.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-3/1.png) 
特别地，实对称矩阵一定会正交相似于对角阵，也就是说给你一个实对称矩阵，一定存在着正交矩阵$Q$把它的列向量记成$v_1$到$v_n$，它能够满足$Q^TAQ$等于$\lambda$，$\lambda$是一个对角阵，它的对角元是$A$的特征值，那么其中$Q$的列向量$v_i$，它是矩阵$A$的属于特征值，$\lambda_i$的特征向量，也就是满足$Av_i$等于$\lambda_iv_i$。我们现在有个问题是说，如果对于$m \times n$的一个矩阵，我们如何来"对角化"它。那么也就是说在什么意义上，我们能够尽可能地。把$m \times n$的一个矩形的阵向对角阵靠拢，今天我们来讨论矩阵的奇异值分解它是线性代数应用中，最重要的一类矩阵分解。

## $AA^T$与$A^TA$的特性

### $AA^T$与$A^TA$的特征值 
![1st_property_of_AAT](http://pwmpcnhis.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-3/2.png) 
### $AA^T$与$A^TA$非0特征值集合 
![2nd_property_of_AAT](http://pwmpcnhis.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-3/3.png) 
### $A^TA$与$AA^T$的特征向量 
![orthonormal_eigenvectors_of_AAT](http://pwmpcnhis.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-3/4.png)

令$u_i:={Av_i \over \sigma_i}\in\,R^m(1 \le i \le r) $，则 $AA^Tu_i=A(A^T\frac{Av_i}{\sigma_i})=A\frac{A^TAv_i}{\sigma_i}=A\frac{\sigma_i^2v_i}{\sigma_i}={\sigma_i}^2{Av_i \over \sigma_i}={\sigma_i}^2u_i$，得出：$AA^Tu_i={\sigma_i}^2u_i$。又因为：${u_i}^T{u_j}=\frac{(Av_i)^T}{\sigma_i}{Av_j \over \sigma_j}={v_i^T(A^TAv_j) \over \sigma_i\sigma_j}=\frac{\sigma_j^2{v_i}^Tv_j}{\sigma_i\sigma_j}={\sigma_j\over \sigma_i}v_i^Tv_j\rightarrow u_i^Tu_j=\begin{cases}0, & i\ne j\\ 1, & i=j\end{cases}$故：$\{u_i|1\le i \le r\}$ 是$AA^T$的单位正交特征向量。

根据假设（$v_1,\,...\,,v_n$是$A^TA$的单位交基，$\sigma_1^2,\,...\,,\sigma_n^2$是$AA^T$的特征值）得：$A^TAv_i=\sigma_i^2v_i(1\le i\le r) \rightarrow v_i^TA^TAv_i=v_i^T\sigma_i^2v_i=\sigma_i^2v_i^Tv_i \rightarrow ||Av_i||^2=\sigma_i^2 \rightarrow|Av_i|=\sigma_i$

### 从$AA^T$得出SVD

$(1)u_i:={Av_i \over \sigma_i}\in\,R^m(1 \le i \le r) \rightarrow Av_i=\sigma_iu_i\\ (2)A^TAv_i={\sigma_i}^2v_i, (i\le i \le r)\rightarrow A^T{Av_i\over \sigma_i}=\sigma_iv_i\rightarrow A^Tu_i=\sigma_iv_i$

由上式子得：$U$是$A$列空间的一组单位正交基，$V$是$A^T$的列空间的一组单位正交基。$\sigma_i$是$Av_i$的长度，计$\begin{pmatrix}\sigma_1&&&&\\&.&&&\\&&.&&\\&&&.&\\&&&&\sigma_r\end{pmatrix}$为$\Sigma$，得：$A_{m\times n}V_{n\times r}=U_{m\times r}\Sigma_{r\times r}\rightarrow A_{m\times n}=U_{m\times r}\Sigma_{r\times r} {V^{-1}}_{r\times n}\\=U_{m\times r}\Sigma_{r\times r} {V^{T}}_{r\times n}$

向量形式：$A=\sum_{i=1}^r \sigma_i u_i{v_i}^T$

![get_svd_from_AAT](http://pwmpcnhis.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-3/5.png)

## SVD形式

![formula_svd](http://pwmpcnhis.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-3/6.png)

## 例题

![example_svd](http://pwmpcnhis.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-3/7.png)

求$u_3$两种方法：

方法1：$AA^Tu_3=\begin{pmatrix}1&0\\0&1\\1&-1\end{pmatrix}\begin{pmatrix}1&0&1\\0&1&-1\end{pmatrix}u_3=\begin{pmatrix}1&0&1\\0&1&-1\\1&-1&2\end{pmatrix}u_3=0u_3\rightarrow u_3={1\over\sqrt{3}}\begin{pmatrix}1\\ -1\\ -1\end{pmatrix}$

方法2：$u_j:=\begin{pmatrix}x\\y\\z\end{pmatrix}, \sum_{i=1}^{r=3}u_iu_j=0 (i\ne j), ||u_j||^2=1\rightarrow u_{j=3}={1\over\sqrt{3}}\begin{pmatrix}1\\ -1\\ -1\end{pmatrix}$

## svd几何意义

![example_geometry_svd](http://pwmpcnhis.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-3/8.png) 
![geometry_svd](http://pwmpcnhis.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-3/9.png)

## svd应用

### svd与矩阵的四个基本子空间

![4_subspaces_svd](http://pwmpcnhis.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-3/10.png)

### svd与图像压缩

![img_compression_by_svd](http://pwmpcnhis.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-3/11.png)

### 奇异值与特征值关系

![singular_values_and_eigenvalues](http://pwmpcnhis.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-3/12.png)

### 奇异值与奇异矩阵

![singular_values](http://pwmpcnhis.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-3/13.png)
