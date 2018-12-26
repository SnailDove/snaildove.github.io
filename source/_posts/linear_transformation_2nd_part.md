---
title: Tsinghua linear-algebra-2 5th-lecture linear-transformation-2nd-part
mathjax: true
mathjax2: true
categories: 中文
date: 2017-08-05 20:16:00
tags: [linear_algebra]
toc: true
---

笔记源自：清华大学公开课：线性代数2——第五讲：线性变换2


## 前言

对于给定的线性变换选取适当的基底使得其矩阵表示尽可能简单，我们引入了线性变换的矩阵表示
对于从$n$维的向量空间$V$到$m$维的向量空间$W$的线性变换$\sigma$，我们取定$V$的一组基$v_1$到$v_n$取定W的一组基$w_1$到$w_m$，那线性变换$σ$作用在$v_1$到$v_n$上可以被$w_1,…,w_m$线性表示，表示的系数我们被一个$m×n$的矩阵$A$去描述，那么这样线性变换$σ$就跟这个$m×n$的矩阵$A$一一对应。线性变换的矩阵表示要依赖于我们基底的选取，一般说来如果基做了改变，同一个线性变换它会有不同的矩阵表示，那我们希望找出线性变换与基底选取无关的性质，这样当我们借助矩阵来研究线性变换的这些性质的时候就可以利用好基底下面尽可能简单的矩阵表示。

## 恒等变换与基变换

恒等变换就是不变，那么不变的线性变换对应单位矩阵。

![identical_linear_transformation](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/1.png) 

>the 9th property of determinant: the determinant of $AB$ is det $A$ times det $B$: $|AB| = |A||B|$ 

因此：由于$(\sigma_1\,....\,\sigma_n)$ 和 $(\beta_1\,...\,\beta_n)$ 都是基向量，因此都是列满秩，又是 $n$ 维，所以可逆，再推出$P$可逆。否则 $|\alpha_1\,...\,\alpha_n|\ne|\beta_1\,...\,\beta_n||P|$
![example_identical_linear_transformation](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/2.png)  

## 基变换的应用

### 一张256x256的灰度图像

![256x256_application_of_change_of_basis](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/3.png) 

**注意：$C^N$是$n$维元素可为复数的基**

### 图像的其中3种基底

![img_basis](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/4.png) 

小波基好求它的逆，傅里叶基也好求它的逆。如果是$4\times4$纯色图像直接用小波基或者傅里叶基的第一个分量$w_1$和$\xi_1$做基底，表示成$c_1w_1=W\begin{pmatrix}c_1\\0\\0\\0\end{pmatrix}$和$c_1\xi_1=\xi\begin{pmatrix}c_1\\0\\0\\0\end{pmatrix}$。而像素之间变换比较剧烈的图像可用小波基中的 $c(w_3+w_4)$ 和傅里叶基中的 $c\xi_3$ 。

### jpeg

![jpeg_process](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/5.png) 

图像本身是用系数矩阵$c$表示，那么所谓的压缩和传输图像也是压缩和传输这个矩阵$c$。压缩做的就是用尽可能少的信息（数据）去代表原有的信息（数据），这个过程会丢失一些不重要的信息（数据），对应到矩阵上就是$c$的非0项元素比较少（这个要求用更少数量的基底向量就能接近描述出原来的矩阵，越少越好）。由于 $c=W^{-1}x$ 因此能不能快速计算基底的逆也很重要，而小波基和傅里叶基正符合此特点。

## 线性变换在不同基下的矩阵

![the_same_linear_transform_of_different_bases](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/6.png)  

**定理：$n$向量空间$V$上的线性变换$\sigma$在$V$的不同基下的矩阵是相似矩阵。**

![the_same_linear_transform_of_different_bases_at_different_aspects](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/7.png) 

由上图可得：
$I_1$ 和 $I_2$ 是恒等变换
$(\beta_1\,...\,\beta_n)=I_1(\beta_1\,...\,\beta_n)=(\alpha_1\,...\,\alpha_n)P$
$(\alpha_1\,...\,\alpha_n)=I_2(\alpha_1\,...\,\alpha_n)=(\beta_1\,...\,\beta_n)P^{-1}$
线性变换复合角度：$\sigma=I_2\,\sigma\,I_1\,\rightarrow\,B=P^{-1}AP$

## 同一个线性变换在不同基下的不变性

当我们借助于矩阵来研究线性变换的时候，我们希望研究线性变换与基底选取无关的性质。由以上的讨论我们知道这个向量空间$V$到自身的线性变换在不同基下的矩阵表示是互为相似矩阵的。因此，所谓与基底选取无关的性质也就是相似变换下不变的性质，那么这样自然地研究相似不变量是线性代数中很重要的内容。我们知道对于一个矩阵而言特征多项式、特征值、迹、行列式、矩阵的秩等等都是矩阵的相似不变量，这样我们就称一个n维向量空间$V$上线性变换在$V$的一组基下的矩阵$A$，把矩阵表示$A$的特征多项式、特征值、迹行列式等等就叫做这个线性变换的特征多项式、特征值 、迹、行列式。
![properties_of_the_same_linear_transformation_of_different_bases](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/8.png)

## 矩阵分解与基变换

给定一个$R^n$到$R^m$的线性变换$σ$，它在$R^n$中的标准基$e_1$到$e_n$和$R^m$的标准基$ẽ_1,…,ẽ_m$下的矩阵是 $A$ ，
$σ$作用在$e_1 … e_n$上面就等于$\tilde{e}_1,…, \tilde{e}_m$去乘以矩阵$A$，也就是说$σ$作用在$e_j$上，就等于$A$的第j列，也就是$A$去乘以$e_j$，因此这个线性变换就可以表示成对任何的$n$维向量$v$，那么$σ$作用在$v$上就是矩阵$A$去乘以$V$ ：
$$\sigma(e_1\,...\,e_n)=(\tilde{e}_1 \,...\,\tilde{e}_m)A\rightarrow\sigma(e_j)=Ae_j$$

![input_space_form_of_linear_transformation](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/9.png)

接下来做基变换，第一个改变输入基，第二个改变输出基，第三个输入输出基都改。

![matrix_decomposition_and_basis_transformation](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/10.png)

### 对角化矩阵视为线性变换
![vector_basis_of_linear_transformation](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/11.png)

由上可得 $\sigma(x_1\,...\,x_n)=(x_1\,...\,x_n)\Lambda=S\Lambda$ ，$x$ 为特征向量基，另外基变换 ${id}_1(S)=S=\{e\}S$
$σ$这个线性变换在A的特征向量作为的新基下面，它的矩阵表示是 $\Lambda$ 这个对角阵。而$σ$从 $R^n$ 到 $R^n$在标准基下的矩阵是$A$，$σ$在特征向量基下的矩阵表示是对角阵 $\Lambda$。那么输入$x$这组基，输出$e$这组基，这个恒同变换，它的矩阵表示是 $S$ 。如果输入$e$这组基 ，输出$x$这组基这个恒同变换，它的矩阵表示是$S^{-1}$。

### 奇异值分解视为线性变换

![SVD_decomposition_as_linear_transformation](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/12.png)

## 线性变换的核与像

### 定义

![definition_kernel_and_image](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/13.png)

### 线性变换的零度与秩

![nullity_and_rank_of_linear_transformation](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/14.png)

#### 线性变换秩的证明

![proof_nullity_and_rank_of_linear_transformation](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/15.png)

注：$L(\sigma(v_1),\,...\,,\sigma(v_n))$ 符号含义：由 $\sigma(v_1),\,...\,,\sigma(v_n)$ 线性张成。

#### 线性变换的维度公式

![dim(kernel)+dim(image)=dimV.png](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/16.png) 
![kernel+image!=V.png](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/17.png) 

#### 单射满射可逆

**中学学过的单射双射满射**

![injective_surjective_bijective](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/18.png)

**线性变换下的单射（injective），满射（surjective）与逆（inverse）**

![injective_surjective_inverse_of_linear_transformation_are_equivalent](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/19.png)

第一个等价符号证明（反证法）：如果单射无法推出核只有$\{0\}$，那么假设$\exists\,\alpha(\ne0)\in{ker\,\sigma}$ 那么$\sigma(\alpha)=0$，又因为$\sigma(0)=0$,  即$\sigma(\alpha\,or\,0)=0$与单射矛盾。反之，如果$\sigma(v_1)=0, \sigma(v_2)=0$，根据线性变换的定义或者性质得：$\sigma(v_1-v_2)=0\rightarrow v_1-v_2\in ker\,\sigma=\{0\}\rightarrow v_1=v_2\rightarrow \sigma$ 是单射。因此：$\sigma$是单射$\Leftarrow\Rightarrow ker\,\sigma=\{0\}$

**例子：**

![example_of_injective_surjective_inverse_of_linear_transformation_are_equivalent](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/20.png)

## 不变子空间

### 定义

![definition_of_invariant_subspace](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/21.png)

### 不变子空间的意义

![candy_of_invariant_subspace](http://pkaunwk1s.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-5/22.png)


那从这里头我们看到，我们希望把大空间分解成不变子空间的直和，从而能够取出合适的基底，从而使得线性变换
在这组基底下的矩阵表示能够成为对角块的形状，那么对于线性变换的研究就转化成它限制在不变子空间上的研究
以此为基础，看一下幂零变换的结构。
