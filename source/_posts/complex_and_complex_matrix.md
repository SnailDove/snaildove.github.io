---
title: Tsinghua linear-algebra-2 12th-lecture complex_and_complex-matrix
mathjax: true
mathjax2: true
categories: 中文
date: 2017-08-12 20:16:00
tags: [linear_algebra]
toc: true
---

笔记源自：清华大学公开课：线性代数2——第12讲：复数与复矩阵

*之前接触的大部分线性代数知识都只考虑实数情形*，但复数情形不可避免会遇到。例如$\begin{pmatrix}cos\theta&-sin\theta\\sin\theta&cos\theta\end{pmatrix}$没有实特征值（除了极特殊情形），目的：比较实数和复数情形的异同，**注意学习复数和实数的区别联系**。 

## 复数复习
1.  -   $i^2=-1$， 一个复数$a+bi=z$，$a$是**实部(real part)**，$b$是**虚部(imaginary part)**，可以把实部$a$看成x轴分量，虚部$b$看成y轴分量。复数的**共轭(complex conjugate)** $z=a+bi\rightarrow \bar{z}=a-bi$，**长度** $|z|=\sqrt{a^2+b^2}=(a-bi)(a+bi)=z\bar{z}$（$z$的长度不能定义为$\sqrt{(a+bi)^2}$，长度必须是正值，如果把复数$z$看成一个2维向量，那么它的长度显然就是定义中给出的）， 矩阵的共轭定义为： $A=(a_{ij})_{n\times n}, a_{ij}\in C \rightarrow \bar{A}=(\overline{a_{ij}})_{n\times n}$，**性质**：$\overline{AB}=\bar{A}\bar{B}\ z\bar{z}=|z|^2$。
    -   {长度为1（单位圆上）的复数}$\rightarrow${二阶旋转矩阵}，且保持乘法。$z=cos\theta+isin\theta\rightarrow A_2=\begin{pmatrix}cos\theta&-sin\theta\\sin\theta&cos\theta\end{pmatrix}$。验证性质：$z_1=e^{i\theta_1},z_2=e^{i\theta_2}\rightarrow A_{z_1}=\begin{pmatrix}cos\theta&-sin\theta\\sin\theta&cos\theta\end{pmatrix}, A_{z_2}=\begin{pmatrix}cos\theta&-sin\theta\\sin\theta&cos\theta\end{pmatrix}\\\rightarrow z_1z_2=e^{i(\theta_1+\theta_2)}=\begin{pmatrix}cos(\theta_1+\theta_2)&-sin(\theta_1+\theta_2)\\sin(\theta_1+\theta_2)&cos(\theta_1+\theta_2)\end{pmatrix}=A_{z_1z_2}$
    -   **欧拉公式(Euler formula)** ：$e^{i\theta}=cos\theta+isin\theta$，**极分解(polar decomposition)**： $z=re^{i\theta}=r(cos\theta+isin\theta)\rightarrow z^n=r^ne^{in\theta}=r^n(cos(n\theta)+isin(n\theta)) $，这里z的公式中三角函数部分长度为1，所以r即z的长度，这样任何一个复数都可以用$re^{i\theta}$表示。
    -   单位根$x^n=1$有n个复根$e^{2k\pi i\over n}, k=0,1,2,\ldots,n-1$，令$\omega=e^{2\pi i\over n}\rightarrow 1+\omega+\omega^2+\cdots+\omega^{n-1}=0$，例如：求$(1+i)^8\leftarrow1+i=\sqrt{2}e^{i{\pi\over 4}}, (1+i)^8={(\sqrt{2})}^8e^{i2\pi}=16$。$\frac{x^{2n+1}-1}{x-1}$。
2.  -   代数基本定理：$a_nx^n+\cdots+a_1x+a_0=0, a_i\in C$有n个复数根(可能重复)，设$a_i\in R, a_nx^n+\cdots+a_1x+a_0=0$ 的非实数的复根也是成对出现，即若$z=a+bi(b\ne0)$是它的根，则$\bar{z}=a-bi$也是它的根，复数根是成对出现的。$\Rightarrow$ 奇次实系数方程总有一个实根。（注：公开课字幕内容如下：因为我们知道复根是成对出现的，所以对一个实系数方程，它的复根实际上是2的倍数，因为它是成对出现的，但是奇数次实系数呢，所以它必然除了复根应该有一个实根，不然的话它只有偶数的根，这样就跟它奇数次矛盾）。
    -   实系数多项式（次数$\ge 1$）的$f(x)$可分解成$f(x)=a(x-\lambda_1)^{n_1}\cdots(x-\lambda_s)^{n_s}(x^2-b_1x+c)^{e_1}\cdots(x^2-b_tx+c)^{e_{t}}$，$\lambda_i$即实数根，后$t$项即复数根给出来的，后面这种形式无法写成实根的一次形式，也就是它的判别式小于0（有复数根），不能写成前$s$项的形式。例如：$x^m-1=\prod\limits_{k=0}^{m-1}(x-\omega_k), \omega_k=e^{i2k\pi\over m}$$\omega_{m-k}=e^{i2(m-k)\pi\over m}=e^{i{2\pi(1-{k\over m})}}=cos(2\pi(1-{k\over m}))+isin(2\pi(1-{k\over m}))=cos({2k\pi\over m})-isin({2k\pi\over m})=\overline{\omega_k}, \\ {k\over m} <1\Rightarrow (x-\omega_k)(x-\omega_{m-k})=x^2-(\omega_k+\omega_{m-k})x+(\omega_k\omega_{m-k})=x^2-2cos({2k\pi\over m}x)+1$， 同理可得：$x^m+1=\prod\limits_{k=0}^{m-1}(x-\xi_k), \xi_k=e^{i(\pi+2k\pi)\over m}$ 。

例题：证明$cos{\pi\over 2n+1}cos{2\pi\over 2n+1}\cdots cos{n\pi\over 2n+1}={1\over 2^n}$
要证明这个需要以下3点：

(1)$-1-e^{i2\theta}=-1-cos2\theta-isin2\theta=-2cos\theta(cos\theta+isin\theta)\Rightarrow |-1-cos2\theta-isin2\theta|=2|cos\theta|$

(2)设$\omega=cos{2\pi\over 2n+1}+isin{2\pi\over 2n+1}=e^{i2{\pi\over 2n+1}}\Rightarrow|-1-\omega|=2|cos({\pi\over {2n+1}})|$，那么$x^{2n}+x^{2n-1}+\cdots+1=(x-\omega)(x-\omega^2)\cdots(x-\omega^{2n})\quad (*)$ 推导如下：${x^{2n+1}-1}=(x-1)(x-\omega)(x-\omega^2)\cdots(x-\omega^{2n})\Rightarrow \frac{x^{2n+1}-1}{x-1}=(x-\omega)(x-\omega^2)\cdots(x-\omega^{2n})\\\Rightarrow {1(1-x^{2n+1})\over {1-x}}=(x-\omega)(x-\omega^2)\cdots(x-\omega^{2n})$

(3)$cos{(2n+1-k)\pi\over 2n+1}=cos{k\pi\over 2n+1}$
令$(*)$等式中$x=-1$，且取两边长度$1=|(-1-\omega)(-1-\omega^2)\cdots(-1-\omega^{2n})$中右边每一项利用(1)式子得到$|-1-\omega|=2|cos{\pi\over {2n+1}}|,\\ |-1-\omega^2|=2|cos{2\pi\over {2n+1}}|,\\\ldots\\|-1-\omega^n|=2|cos{n\pi\over {2n+1}}|$

从n+1项起根据(3)得：

$$|-1-\omega^{n+1}|=2|cos{(n+1)\pi\over {2n+1}}|=2|cos{(2n+1-n)\pi\over {2n+1}}|=2|cos(\pi-{n\pi\over {2n+1}})|=2|cos({n\pi\over {2n+1}})|=|-1-\omega^n|$$

$$|-1-\omega^{n+2}|=2|cos{(n+2)\pi\over {2n+1}}|=2|cos{[(2n+1)-(n-1)]\pi\over {2n+1}}|=2|cos(\pi-{(n-1)\pi\over {2n+1}})|=2|cos{(n-1)\pi\over {2n+1}}|=|-1-\omega^{n-1}|$$
$$\cdots\cdots$$

$|-1-\omega^{2n}|=2|cos{2n\pi\over {2n+1}}|=2|cos{(2n+1-1)\pi\over {2n+1}}|=2|cos(\pi-{\pi\over {2n+1}})|=2|cos({\pi\over {2n+1}})|=|-1-\omega|$

## 复矩阵

### Hermitian矩阵

复数矩阵$A=(a_{ij})_{m\times n},a_{ij}\in C$, 那么称$\overline{A^T}(=\bar{A}^T)$ 为 **Hermitian 矩阵**，记为$A^H$。例如： $Z=\begin{pmatrix}1+i\\i\end{pmatrix}\rightarrow Z^H=\begin{pmatrix}1-i&-i\end{pmatrix}$，而且发现$ZZ^H=||Z||^2$，这个可以类比实数中的$x^Tx=||x||^2$。**性质**：$(A^H)H=A, (AB)^H=B^HA^H$（按照共轭转置即可求得），正如在$R^n$的定义内积，在$C$上也可以定义**内积**：$u,v\in C^n, u^Hv=(\bar{u}_1\cdots\bar{u}_n)\begin{pmatrix}v_1\\\vdots\\v_n\end{pmatrix}=\bar{u}_1v_1+\cdots+\bar{u}_nv_n$，**内积的性质**：$u^Hv=\overline{v^Hu}$。

### 厄米特Hermite矩阵

在实数矩阵中有对称矩阵的概念和作用，复数矩阵有类似的——**厄米特矩阵(Hermite matrix)**，定义为：$A=A^H$，即一个矩阵的共轭转置等于它本身，那么称这种矩阵为Hermite阵。例：$\begin{pmatrix}2&1+i\\1-i&3\end{pmatrix}$。

-   性质1：Hermite阵对角线元素为实数。

-   性质2：$z\in C, A=A^H\Rightarrow z^HAz$ 是一个实数。证明如下：${\overline{z^HAz}}^T=(z^HAz)^H=z^HA^Hz=z^HAz$ 

-   性质3：设$A,B$是Hermite阵，则$A+B$也是，证明：$(A+B)^H=A^H+B^H=A+B$。进一步，若$AB=BA$（即乘法可交换的时候），则$AB$是Hermite阵。$\Rightarrow A^n$是Hermite阵。

-   性质4：设$A$是一个$n$阶复矩阵，$AA^H, A+A^H$是Hermite阵，联系对比实对称矩阵的$AA^T, A^TA, A+A^T$。 

-   性质5：**一个Hermite矩阵A的特征值是实数**。证明：设$Az=\lambda_0z$，则$z^HAz=\lambda_0z^Hz$。$z^HAz$和$z^Hz$均为实数$\Rightarrow \lambda_0 (z_0\ne 0)$是实数。

-   性质6：**一个Hermite阵的不同特征值的特征向量相互正交**。证明：设$(1) Az_1=\lambda_1z_1, (2) Az_2=\lambda_2z_2, \lambda_1 \ne \lambda_2$， 在(1)两边同乘以$z_2^H$得：$(3)z_2^HAz_1=z_2^H\lambda_1z_1 \Rightarrow (4)z_2^HA^Hz_1=(Az_2)^Hz_1=\overline{\lambda_2}z_2^Hz_1=\lambda_2z_2^Hz_1$，由$(3)(4)\Rightarrow \lambda_1z_2^Hz_1=\lambda_2z_2^Hz_1\Rightarrow (\lambda_1-\lambda_2)z_2^Hz_1=0$，因为$\lambda_1\ne \lambda_2$得：$z_2^Hz_1=0$。

### 酉unitary矩阵

酉矩阵是正交阵的复数类比。$U_{n\times n}$是酉矩阵$\Leftrightarrow$ $\forall z\in C^n, ||Uz||=||z||$，证明：$U^HU=I_n\Rightarrow |U z|^2=z^HU^HUz = z^Hz=|z|^2\Rightarrow |Uz|=|z|\Rightarrow |\lambda|=1$ 。得出与实数矩阵类似的**性质1**：酉矩阵乘以任何向量不改变它的模长。**性质2**：$U$是酉矩阵，则$U$的特征值模长为1。 例：$u=\begin{pmatrix}{1\over \sqrt{2}}&-\frac{1}{\sqrt{6}}&\frac{1-i\sqrt{3}}{2\sqrt{3}}\\\frac{1}{\sqrt{2}}&\frac{1}{\sqrt{6}}&{-1+i\sqrt{3}\over 2\sqrt{3}}\\0&{1+i\sqrt{3}\over \sqrt{6}}&{1\over \sqrt{3}}\end{pmatrix}$ ，$|det U|=\prod{|\lambda_i|}=1$ (行列式的长度等于特征值长度的乘积)。

而实数的正交阵，也有类似的性质。下面证明正交阵不同特征值对应的特征向量相互正交：

因为$Q$正交阵,$Q^TQ=E,|Q|=1=λ_1λ_2\ldotsλ_n$,设$λ_1,λ_2$为$Q$的两个不同的特征值,$ξ_1,ξ_2$为对应的特征向量$ (1)Qξ_1=λ_1ξ_1, (2)Qξ_2=λ_2ξ_2,(3)(ξ_2)^T Q^T=λ_2(ξ_2)^T \Rightarrow (3)(1)\Rightarrow ξ_2^TQ^TQξ_1=λ_1λ_2ξ_2^Tξ_1\Rightarrow  \\(λ_1λ_2-1)ξ_2^Tξ_1=0$
而$|λ_1|=|λ_2|=1,λ_1≠λ_2$,得$ξ2^Tξ1=0,因此ξ_2,ξ_1$正交。

### 复正规阵

酉阵和Hermite矩阵均为复正规矩阵，即：$A^HA=AA^H$。 酉相似：设$A,B$是；两$n$阶复矩阵，若存在酉矩阵$U$，使得$A=U^HBU$，则$A$和$B$是**酉相似**（联系实数矩阵的正交相似）。**定理**：设$A$复正规阵，则

1.  向量$u$是$A$的关于$\lambda$的特征向量$\Leftrightarrow u$是$A^H$的关于$\bar{\lambda}$的特征向量。证明：
    设$Au=\lambda u\Rightarrow (A-\lambda I)u=0$令$B=A-\lambda I\Rightarrow ||B^Hu||^2=u^HBB^Hu=u^HB^HBu=||Bu||^2=0$，因为$||B^Hu||^2=0\Rightarrow B^Hu=0, (A-\lambda I)^H=B^H\Rightarrow  (A^H-\bar{\lambda}I)u=0\Rightarrow A^Hu=\bar{\lambda}u$
2.  不同特征值的特征向量正交。证明与Hermite矩阵一样。

**定理(Schur)**：**任意一个复矩阵$A$酉相似于一个上三角阵**。即：$\exists\ U\in $ unitary  matrix,$\forall\ A\in$ complex matrix, $U^H=U^{-1}, U^HAU=\begin{pmatrix}\lambda_1&\*&\* \\0&\ddots&\*\\0&0&\lambda_n\end{pmatrix} \Rightarrow$**任意一个复正规阵酉相似于对角阵**，特别地，酉相似于$\begin{pmatrix}1\\&1\\&&\ddots\\&&&1\end{pmatrix}$, $U^HAU=diag(\lambda_1,\ldots,\lambda_n)\Rightarrow AU=\lambda U$。

一个实矩阵$A$是正规的$\Leftrightarrow A^TA=AA^T$。例如，$A$是正交阵或者$A$是对称（反对称）矩阵。

如果$A$是正规的，那么存在正交阵$\Omega$使得：

$\Omega^TA\Omega=\begin{pmatrix}\begin{pmatrix}a_1&b_1 \\ -b_1&a_1\end{pmatrix}\\&\ddots\\&&\begin{pmatrix}a_s&b_s \\ -b_s&a_s\end{pmatrix}\\&&&\lambda_{2s+1}\\&&&&\ddots\\&&&&&\lambda_n\end{pmatrix}$，即**实正规阵正交相似于分块对角阵**。 

对于复正规阵酉相似对角阵$U^HAU=diag(\lambda_1,\ldots,\lambda_n)\Rightarrow AU=\lambda U$，这里如果把$U$的列向量写成$u_k=\beta+i\gamma,\ \ k\in [1,n],\ \beta,\gamma \in R_n$，例如：$\begin{pmatrix}1+i\\1-i\end{pmatrix}=\begin{pmatrix}1\\1\end{pmatrix}+i\begin{pmatrix}1 \\ -1\end{pmatrix}$。

$Au_k=\lambda_ku_k\Rightarrow A(\beta+i\gamma)=\lambda_k(\beta+i\gamma)$，令$\lambda_k=a+ib$，得：$A\beta=a\beta-b\gamma, A\gamma=b\beta+a\gamma\Rightarrow$
$A(\beta, \gamma)=(\beta,\gamma)\begin{pmatrix}a&b \\ -b&a\end{pmatrix}$ ，所以$\Omega$的实际上是由$U$的特征向量的实部和虚部组成的这样一个形式。 $\Omega$是一个正交阵，那$\beta$和$\gamma$是不是正交的？它们的长度相等嘛？不然无法保证$\Omega$是一个正交阵。 结论：设$A$是$n$解实正交阵。若$\lambda=a+ib(b\ne 0)$是$A$的特征值，$x=x_1+ix_2,\ x_1,x_2\in R_n$是对应的特征向量，则$||x_1||=||x_2||$，且$x_1,x_2$是相互正交的。

证明：如果$\lambda=a+ib$ 是$A$的特征值，那么$\lambda=a-ib$ 也是$A$的特征值。因为$A$实正交阵，所以对$Ax=\lambda x$取两边共轭得：$\overline{Ax}=A\bar{x}=\bar{\lambda}\bar{x}$。那么得到$\lambda,\bar{\lambda}$都是$A$的特征值，由于正交阵不同特征值对应的特征向量正交，所以${\bar{x}}^Hx=0, x=x_1+ix_2, \bar{x}=x_1-ix_2\Rightarrow ||x_1||=||x_2||, x_1^Tx_2=0$。 

例2：证明：$\begin{pmatrix}cos\theta&-sin\theta\\sin\theta&cos\theta\end{pmatrix}​$和$\begin{pmatrix}e^{i\theta}&0\\0&e^{-i\theta}\end{pmatrix}​$ 酉相似。$U={1\over \sqrt{2}}\begin{pmatrix}i&1\\1&i\end{pmatrix}​$

例3：设$A$是Hermite阵，则$I+iA$是非奇异的。由于A的特征值是实数，那么$I+iA$特征值的是$\lambda i+1$不可能是0，行列式就不可能是0，因此是非奇异的。如果A是Hermite阵，那么$U=(I-iA)(I+iA)^{-1}$是酉阵，验证$U^H=(I-iA)^{-1}(I+iA)=(I+iA)(I-iA)^{-1}$（注：分块是相同的矩阵是可交换即变成分块对角阵），这个是用来**通过实对称阵或Hermite阵构造酉矩阵**。

### 离散傅里叶变换DFT

回忆若$f(x), f'(x)$是piecewise连续的且$f(x+L)=f(x)$， 则$f(x)=a_0+\sum(a_ncos({2\pi nx\over L})+b_nsin({2\pi nx \over L})), a_n={2\over L}\int_{0}^{L}f(x)cos{2\pi nx \over L}dx,\ b_n={2\over L}\int_{0}^{L}f(x)sin{2\pi nx \over L}dx$， 令$V=\{f(x)|f(x)\text{如上条件}\}\rightarrow R^{\infty}$ 
$f(x)\rightarrow (a_0, a_1, b_1, a_2, b_2,\ldots)$ 
这是一个线性映射，$(a_0, a_1,b_1,\ldots)$是$f(x)$的逆傅里叶变换。 **当通过$f(x)$求系数$a_i,b_i,\ldots$即傅里叶变换，当通过系数$a_i,b_i,\ldots$求$f(x)$即逆傅里叶变换**。

由前文分析得到傅里叶级数的复形式是$F=\sum\limits_{k=-\infty}^{\infty}c_ke^{ikx}, c_k={1\over 2\pi}\int_{-\pi}^{\pi}f(x)e^{-ikx}dx$，通过变量代换：$x={2\pi \over L}t$ 得：$c_k={1\over L}\int_{-{L\over 2}}^{L\over 2}f(t)e^{-i{2\pi k\over L}t}dt, f(t)=\sum\limits_{k=-\infty}^{+\infty}c_ke^{-i{2\pi k\over L}t}$
令$n=k$，则得到**新的傅里叶级数复数形式**：$f(t)=\sum\limits_{n=-\infty}^{+\infty}c_ne^{-i{2\pi n\over L}t}, c_n={1\over L}\int_{-{L\over 2}}^{L\over 2}f(t)e^{-i{2\pi n\over L}t}dt\quad (1)$
令$\omega_n={2\pi n\over L}$得到**傅里叶级数的频率形式**：$\hat{f}(\omega)=\int_{-\infty}^{+\infty}f(t)e^{i\omega_nt}dt\quad (2)$

对(1)(2)进行离散化：
$f(t_j)=\sum\limits_{k=-\infty}^{+\infty}c_ke^{-i{2\pi k\over L}t_j}$令 $\ t_j={jL\over N}$则得到：$f(t_j)\approx \sum\limits_{k=0}^{N-1}c_ke^{i{2\pi kj\over N}}, c_k={1\over L}\int_{-{L\over 2}}^{L\over 2}f(t_j)e^{i{2\pi kj\over N}}dt_j\quad (1*)$，然后再设置$A_j=f(t_j)，a_k=c_k$得到：$f(t)\rightarrow (A_0,A_1,\cdots, A_{N-1}), (c_k)\rightarrow (a_0,a_1,\cdots, a_{N-1})$。

由上可举N=4的例子：
$A_0=f(t_0)=a_{0}e^{i2\pi 00\over 4}+a_1e^{i2\pi 10\over 4}+a_2e^{i2\pi 20\over 4}+a_3e^{i2\pi 30\over 4}=a_{0}+a_1+a_2+a_3=1a_{0}+1a_1+1a_2+1a_3$
$A_1=f(t_1)=a_{0}e^{i2\pi 01\over 4}+a_1e^{i2\pi 11\over 4}+a_2e^{i2\pi 21\over 4}+a_3e^{i2\pi 31\over 4}= a_{0}+ia_1-a_2-ia_3=1a_{0}+ia_1+i^2a_2+i^3a_3$ 
$A_2=f(t_2)=a_{0}e^{i2\pi 02\over 4}+a_1e^{i2\pi 12\over 4}+a_2e^{i2\pi22\over 4}+a_3e^{i2\pi 32\over 4} = a_{0}-a_1+a_2-a_3 = 1a_{0}+i^2a_1+i^4a_2+i^6a_3$
$A_3=f(t_3)=a_{0}e^{i2\pi 03\over 4}+a_1e^{i2\pi 13\over 4}+a_2e^{i2\pi 23\over 4}+a_3e^{i2\pi 33\over 4}=a_{0}-ia_1-a_2+ia_3=1a_{0}+i^3a_1+i^6a_2+i^9a_3$
写成矩阵形式：
$\begin{pmatrix}A_0\\A_1\\A_2\\A_3\end{pmatrix}=\begin{pmatrix}1&1&1&1\\1&i&i^2&i^3\\1&i^2&i^4&i^6\\1&i^3&i^6&i^9\end{pmatrix}\begin{pmatrix}a_0\\a_1\\a_2\\a_3\end{pmatrix}$
设$F=\begin{pmatrix}1&1&1&1\\1&i&i^2&i^3\\1&i^2&i^4&i^6\\1&i^3&i^6&i^9\end{pmatrix}$，令s表示第s行，t表示第t列，则F的第s行第t列元素为$F_{s,t}=i^{(s-1)(t-1)}$，其实上文中的记号j刚好可以视为行数，k刚好表示列数。

一般地，$\begin{pmatrix}A_0\\A_1\\\vdots\\A_{N-1}\end{pmatrix}=F\begin{pmatrix}a_0\\a_1\\\vdots\\a_{N-1}\end{pmatrix}$，$F_{j, k}=e^{i{2\pi jk\over N}}$令$\omega_N=e^{i{2\pi\over N}}\Rightarrow F_{j,k}=\omega^{jk}_{N}=F_{j,k}$。F称为傅里叶矩阵，F的各列相互正交且F对称(但注意：不是Hermite矩阵)，这个矩阵跟范德蒙德行列式很像。如果令$\omega_N=e^{i{2\pi\over N}}\Rightarrow F_{s,t}=\omega^{st}_{N}=F_{t,s}$那么F表示成$F=\begin{pmatrix}1&1&1&1\\1&\omega&\omega^2&\omega^3\\1&\omega^2&\omega^4&\omega^6\\1&\omega^3&\omega^6&\omega^9\end{pmatrix}$。 

对于给定的$\begin{pmatrix}A_0\\A_1\\\vdots\\A_{N-1}\end{pmatrix}$，求$\begin{pmatrix}a_0\\a_1\\\vdots\\a_{N-1}\end{pmatrix}=F^{-1}\begin{pmatrix}A_0\\A_1\\\vdots\\A_{N-1}\end{pmatrix}$，$F^{-1}={1\over N}\overline{F}$，需要$N^2$次乘法，$N(N-1)$次加法（忽略除以N的除法），计算量$=O(N^2)$。

注记：实际上由前文可得$\begin{pmatrix}a_0\\a_1\\\vdots\\a_{N-1}\end{pmatrix}=\begin{pmatrix}c_0\\c_1\\\vdots\\c_{N-1}\end{pmatrix}$，因此是向量$\begin{pmatrix}A_0\\A_1\\\vdots\\A_{N-1}\end{pmatrix}$关于某个正交向量基的投影长度，即坐标分量。$(a_0, a_1,b_1,\ldots)$是$f(x)$关于$\{1,cosx,sinx,\dots\}$的坐标。

### 快速傅里叶变换FFT

快速傅里叶变换减少了$DFT$的计算量到$O(Nlog_2^N)$

| $N$  |  $N^2$  | $Nlog_2^N$ | FFT efficiency |
| :--: | :-----: | :--------: | :------------: |
| 256  |  65536  |    1024    |      64:1      |
| 512  | 262144  |    2304    |     114:1      |
| 1024 | 1048576 |    5120    |     205:1      |

注：$\lim\limits_{N\rightarrow +\infty}{log_2^N\over N}=0$



解释算法：$N=4，\begin{pmatrix}a_0\\a_1\\a_2\\a_3\end{pmatrix}={1\over 4}\begin{pmatrix}1&1&1&1\\1&-i&-1&i\\1&-1&1&-1\\1&i&-1&-i\end{pmatrix}\begin{pmatrix}A_0\\A_1\\A_2\\A_3\end{pmatrix}\quad i^4=1$

$\begin{equation}4a_0=(A_0+A_2)+(A_1+A_3)\\4a_1=(A_0-A_2)-i(A_1-A_3)\\4a_2=(A_0+A_2)-(A_1+A_3)\\4a_3=(A_0-A_2)+i(A_1-A_3)\end{equation}$ 

注意：求$a_2$的时候，可以把在求$a_0$过程中的两个括号的值重新利用，求$a_3$的时候，可以把在求$a_1$过程中的两个括号的值重新利用。

引入记号：

![1st_notations_of_FFT](http://q83p23d9i.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-12/1.png)

将$A_0, A_1, A_2, A_3$重新排序$A_0,A_2,A_1,A_3$使用记号，则

![2nd_notations_of_FFT](http://q83p23d9i.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-12/2.png)

$FFT$算法将$DFT$算法分成$log_2^N$段，每一段有${N\over 2}$个butterfly operation。

举例：$N=8$，第一步将$A_0,A_1,\ldots,A_7$重新排序。原则：考虑$0,1,\ldots,7$的二进制，设$j$的二进制数的反转为$n_j$。若$j<n_j$，则交换$Aj$和$A_{n_j}$。例如1的二进制数为${001}_2$,反转为${100}_2=4, 1<4$，交换$A_1$和$A_4$。

排序后为：$A_0,A_4,A_2,A_6,A_1,A_5,A_3,A_7$（奇偶分离）
![1st_example_of_reordering_of_FFT](http://q83p23d9i.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-12/3.png)

奇偶分离的原因：$\begin{pmatrix}a_0\\a_1\\\vdots\\a_{N-1}\end{pmatrix}=({1\over N}\overline{F})\begin{pmatrix}A_0\\A_1\\\vdots\\A_{N-1}\end{pmatrix}$，
令$p(x)=A_0+A_1x+\cdots+A_{N-1}x^{N-1}=p_e(x^2)+xp_o(x^2),p_e=A_0+A_2x^2+\cdots\quad p_o=A_1+A_3x^2+\cdots$

注解：e代表even，o代表odd，则$a_j={1\over N}(1,\overline{\omega}_N^j,\overline{\omega}_{N}^{2j},\ldots)\begin{pmatrix}A_0\\\vdots\\A_{N-1}\end{pmatrix}={1\over N}p(\overline{\omega}_N^j)={1\over N}[p_e(\overline{\omega}_N^{2j})+\overline{\omega}_N^{j}p_o(\overline{\omega}_N^{2j})], j=0,1,\cdots,{N\over 2}-1$

$a_{N\over 2+j}={1\over  N}[p_e(\overline{\omega}_N^{2({N\over 2}+j)})+\overline{\omega}_N^{N\over 2+j}p_o(\overline{\omega}_N^{2({N\over 2+j})})],j=0,1,\cdots,{N\over 2}-1​$

再由于：$\omega_N=e^{i{2\pi\over N}}\Rightarrow\overline{\omega}_N^{2j}=\overline{\omega}_{N\over 2}^j, \overline{\omega}_N^{N\over 2+j}=-\overline{\omega}_N^j, \overline{\omega}_N^{N+2j}=\overline{\omega}_{N\over 2}^j$

所以：$\cases{a_j={1\over N}[p_e(\overline{\omega}_{N\over 2}^{j})+\overline{\omega}_N^{j}p_o(\overline{\omega}_{N\over 2}^{j})],j=0,1,\cdots,{N\over 2}-1\\a_{N\over 2+j}={1\over  N}[p_e(\overline{\omega}_{N\over 2}^{j})-\overline{\omega}_N^{j}p_o(\overline{\omega}_{N\over 2}^{j})],j=0,1,\cdots,{N\over 2}-1}$

所以：$a_j={1\over N}p(\overline{\omega}_N^j)$，再令$b_j=p_e(\overline{\omega}_{N\over 2}^{j}), b'_j=p_o(\overline{\omega}_{N\over 2}^{j})$，那么：$\cases{a_j = {1\over N}[ b_j+\overline{\omega}_N^{j}b'_j],j=0,1,\cdots,{N\over 2}-1\\a_{N\over 2+j} = {1\over N}[b_j-\overline{\omega}_N^{j}b'_j] ,j=0,1,\cdots,{N\over 2}-1}$，那么这又是一个butterfly operation:
![3rd_butterfly_operation_of_FFT](http://q83p23d9i.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-12/4.png)

可以重复利用以上原理对$b_j,b'_j$讨论，$b_j=p_e(\overline{\omega}_{N\over 2}^{j}),j=0,1,\cdots,{N\over 2}-1$，令$c_j=p_{ee}(\overline{\omega}_{N\over 4}^{j}), c'_j=p_{eo}(\overline{\omega}_{N\over 4}^{j})$，那么：$\cases{b_j = {1\over N}[c_j+\overline{\omega}_{N\over 2}^{j}c'_j],j=0,1,\cdots,{N\over 4}-1\\b_{N\over 4+j} = {1\over N}[c_j-\overline{\omega}_{N\over 2}^{j}c'_j],j=0,1,\cdots,{N\over 4}-1}$，那么这又是一个butterfly operation:

![4th_butterfly_operation_of_FFT](http://q83p23d9i.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-12/5.png)

不停的划分下去，即：$FFT$算法将$DFT$算法分成$log_2^N$段，每一段有${N\over 2}$个butterfly operation。

举例：

![](http://q83p23d9i.bkt.clouddn.com/gitpage/tsinghua_linear_algebra/2-12/6.png)
