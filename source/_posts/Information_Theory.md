---
title: Information Theory in Machine Learning
mathjax: true
mathjax2: true
categories: 中文
tags: [Information Theory]
date: 2018-03-28
comments: true
copyright: false
toc: true
---
**本文会随着工作学习持续不定期更新**

## 自信息[^1]

在[信息论](https://zh.wikipedia.org/wiki/%E4%BF%A1%E6%81%AF%E8%AE%BA)中，**自信息**（英语：self-information），由[克劳德·香农](https://zh.wikipedia.org/wiki/%E5%85%8B%E5%8B%9E%E5%BE%B7%C2%B7%E5%A4%8F%E8%BE%B2)提出，是与概率空间中的单一事件或离散随机变量的值相关的信息量的量度。它用信息的单位表示，例如 [bit](https://zh.wikipedia.org/wiki/%E4%BD%8D%E5%85%83)、[nat](https://zh.wikipedia.org/wiki/%E7%BA%B3%E7%89%B9)或是hart，使用哪个单位取决于在计算中使用的对数的底。**自信息**的期望值就是信息论中的[熵](https://zh.wikipedia.org/wiki/%E7%86%B5_(%E4%BF%A1%E6%81%AF%E8%AE%BA))，它反映了随机变量采样时的平均不确定程度。

由定义，当信息被拥有它的实体传递给接收它的实体时，仅当接收实体不知道信息的先验知识时信息才得到传递。如果接收实体事先知道了消息的内容，这条消息所传递的信息量就是0。只有当接收实体对消息对先验知识少于100%时，消息才真正传递信息。

因此，一个随机产生的事件 $\omega _{n}$ 所包含的自信息数量，只与事件发生的机率相关。事件发生的机率越低，在事件真的发生时，接收到的信息中，包含的自信息越大。

$$
{I} (\omega _{n})=f({P} (\omega _{n}))
$$


如果 $P{(\omega _{n})=1}$ ，那么 $I(\omega _{n})=0$。如果 $P (\omega _{n})<1$ ，那么 $I (\omega _{n})>0$ 。

此外，根据定义，自信息的量度是非负的而且是可加的。如果事件 $C$ 是两个[独立](https://zh.wikipedia.org/wiki/%E7%B5%B1%E8%A8%88%E7%8D%A8%E7%AB%8B%E6%80%A7)事件 $A$ 和 $B$ 的**交集**，那么宣告 $C$ 发生的信息量就等于分别宣告事件 $A$ 和事件的信息量的 $B$ **和**：

![{\displaystyle \operatorname {I} (C)=\operatorname {I} (A\cap B)=\operatorname {I} (A)+\operatorname {I} (B)}](https://wikimedia.org/api/rest_v1/media/math/render/svg/274722c35d47831844370de1d84367d6454c4964)

因为  $A$ 和  $B$ 是独立事件，所以 $C$ 的概率为

![{\displaystyle \operatorname {P} (C)=\operatorname {P} (A\cap B)=\operatorname {P} (A)\cdot \operatorname {P} (B)}](https://wikimedia.org/api/rest_v1/media/math/render/svg/3d16944a4dca14289025fdf821cd7b06a12efea2)

应用函数 $f(\cdot )$ 会得到

 ![{\displaystyle {\begin{aligned}\operatorname {I} (C)&=\operatorname {I} (A)+\operatorname {I} (B)\\f(\operatorname {P} (C))&=f(\operatorname {P} (A))+f(\operatorname {P} (B))\\&=f{\big (}\operatorname {P} (A)\cdot \operatorname {P} (B){\big )}\\\end{aligned}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/c8dcb2fb5f00e3711c19a911677d1b77a9a98c76)

所以函数 $f(\cdot )$ 有性质

![{\displaystyle f(x\cdot y)=f(x)+f(y)}](https://wikimedia.org/api/rest_v1/media/math/render/svg/923ed53f8261763c67a088c20230edcef546958f)

而对数函数正好有这个性质，不同的底的对数函数之间的区别只差一个常数

![{\displaystyle f(x)=K\log(x)}](https://wikimedia.org/api/rest_v1/media/math/render/svg/25eed90aee3296309aa14d6abf8c3f1dc62c0323)

由于事件的概率总是在0和1之间，而信息量必须是非负的，所以 $K<0$ 。考虑到这些性质，假设事件 $\omega _{n}$ 发生的机率是  $P(\omega _{n})$ ，自信息 $I(\omega _{n})$ 的定义就是:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/fb594603cfa40c04d95fdfed024d72337d9a57aa)

事件 $\omega _{n}$ 的概率越小, 它发生后的自信息量越大。

此定义符合上述条件。在上面的定义中，没有指定的对数的基底：如果以 2 为底，单位是[bit](https://zh.wikipedia.org/wiki/%E4%BD%8D%E5%85%83)。当使用以 e 为底的对数时，单位将是 nat。对于基底为 10 的对数，单位是 hart。

**信息量的大小不同于信息作用的大小，这不是同一概念。信息量只表明不确定性的减少程度，至于对接收者来说，所获得的信息可能事关重大，也可能无足轻重，这是信息作用的大小**。

## 信息熵[^2]

### 熵的计算

如果有一枚理想的硬币，其出现正面和反面的机会相等，则抛硬币事件的熵等于其能够达到的最大值。我们无法知道下一个硬币抛掷的结果是什么，因此每一次抛硬币都是不可预测的。因此，使用一枚正常硬币进行若干次抛掷，这个事件的熵是一[比特](https://zh.wikipedia.org/wiki/%E4%BD%8D%E5%85%83)，因为结果不外乎两个——正面或者反面，可以表示为`0, 1`编码，而且两个结果彼此之间相互独立。若进行`n`次[独立实验](https://zh.wikipedia.org/wiki/%E7%B5%B1%E8%A8%88%E7%8D%A8%E7%AB%8B%E6%80%A7)，则熵为`n`，因为可以用长度为`n`的[比特流](https://zh.wikipedia.org/w/index.php?title=%E6%AF%94%E7%89%B9%E6%B5%81&action=edit&redlink=1)表示。[[1\]](https://zh.wikipedia.org/wiki/%E7%86%B5_(%E4%BF%A1%E6%81%AF%E8%AE%BA)#cite_note-crypto-1)但是如果一枚硬币的两面完全相同，那个这个系列抛硬币事件的熵等于零，因为结果能被准确预测。现实世界里，我们收集到的数据的熵介于上面两种情况之间。

另一个稍微复杂的例子是假设一个[随机变量](https://zh.wikipedia.org/wiki/%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F)`X`，取三种可能值 $\begin{matrix}x_{1},x_{2},x_{3}\end{matrix}$，概率分别为 $\begin{matrix}{\frac  {1}{2}},{\frac  {1}{4}},{\frac  {1}{4}}\end{matrix}$，那么编码平均比特长度是：$\begin{matrix}{\frac  {1}{2}}\times 1+{\frac  {1}{4}}\times 2+{\frac  {1}{4}}\times 2={\frac  {3}{2}}\end{matrix}$。其熵为3/2。

因此熵实际是对随机变量的比特量和顺次发生概率相乘再总和的[数学期望](https://zh.wikipedia.org/wiki/%E6%95%B0%E5%AD%A6%E6%9C%9F%E6%9C%9B)。

### 定义

依据Boltzmann's H-theorem，香农把[随机变量](https://zh.wikipedia.org/wiki/%E9%9A%8F%E6%9C%BA%E5%8F%98%E9%87%8F)*X*的熵值 Η（希腊字母[Eta](https://zh.wikipedia.org/wiki/Eta)）定义如下，其值域为${x_1, \ldots,  x_n}$ ：

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/ad96d27b76b011b9cb7c41665e3e25edf04b3515)

其中，P为*X*的[概率质量函数](https://zh.wikipedia.org/wiki/%E6%A9%9F%E7%8E%87%E8%B3%AA%E9%87%8F%E5%87%BD%E6%95%B8)（probability mass function），E为[期望](https://zh.wikipedia.org/wiki/%E6%9C%9F%E6%9C%9B)函数，而I(*X*)是*X*的信息量（又称为[自信息](https://zh.wikipedia.org/wiki/%E8%B3%87%E8%A8%8A%E6%9C%AC%E9%AB%94)）。I(*X*)本身是个随机变数。

当取自有限的样本时，熵的公式可以表示为：

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/67841ec4b4f7e6ab658842ef2f53add46a2debbd)

在这里*b*是[对数](https://zh.wikipedia.org/wiki/%E5%B0%8D%E6%95%B8)所使用的[底](https://zh.wikipedia.org/wiki/%E5%BA%95%E6%95%B0_(%E5%AF%B9%E6%95%B0))，通常是2,自然常数[e](https://zh.wikipedia.org/wiki/E_(%E6%95%B0%E5%AD%A6%E5%B8%B8%E6%95%B0))，或是10。当*b* = 2，熵的单位是[bit](https://zh.wikipedia.org/wiki/%E4%BD%8D%E5%85%83)；当*b* = e，熵的单位是[nat](https://zh.wikipedia.org/wiki/%E5%A5%88%E7%89%B9_(%E5%8D%95%E4%BD%8D))；而当*b* = 10,熵的单位是Hart。

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/f995e60d396ef16491b788a068d1e4cc54ba8fa5)

$p_i$ = 0时，对于一些 i 值，对应的被加数 $0 log_b 0$ 的值将会是0，这与[极限](https://zh.wikipedia.org/wiki/%E5%87%BD%E6%95%B8%E6%A5%B5%E9%99%90)一致。

## 信息论[^3]

### 2.2.1 熵

 **熵（entropy）** 是信息论的基本概念。如果X是一个离散型随机变量， 取值空间为R， 其概率分布为p（x） ＝P（X＝x） ， x∈R。 那么， X的熵H（X） 定义为式（2-22） ：

$$
H(X)=-\sum\limits_{x\in \mathbf{R}}\mathcal{p(x)log_{2}p(x)} \tag{2-22}
$$

其中， 约定0log0＝0。 H（X） 可以写为H（p） 。 由于在公式（2-22） 中对数以2为底， 该公式定义的熵的单位为二进制位（比特） 。 通常将 $\mathcal{log_2p(x)}$ 简写成 $\mathcal{log\ p(x)}$ 。

熵又称为**自信息（self-information）** ， 可以视为<u>描述一个随机变量的不确定性的数量</u> 。 它表示信源X每发一个符号（不论发什么符号） 所提供的平均信息量［姜丹， 2001］ 。 一个随机变量的熵越大， 它的不确定性越大， 那么， 正确估计其值的可能性就越小。 越不确定的随机变量越需要大的信息量用以确定其值。

例2-3 假设a, b, c, d, e, f 6个字符在某一简单的语言中随机出现，每个字符出现的概率分别为： 1/8， 1/4， 1/8， 1/4， 1/8和1/8。 那么， 每个字符的熵为

$$
\begin{align}
H(P) &= \sum\limits_{x\in\{a,b,c,d,e,f\}}P(x)log P(x) \\
&= - \left[ 4\times\frac{1}{8}+2\times\frac{1}{4}log\frac{1}{4} \right] \\
&= 2\frac{1}{2}
\end{align}
$$

这个结果表明， 我们可以设计一种编码， 传输一个字符平均只需要2.5个比特：

{% raw %}
$$
\text{字符：  a   b     c   d    e      f} \\ \text{编码：100 00 101 01 110 111}
$$
{% endraw %}

在只掌握关于未知分布的部分知识的情况下， 符合已知知识的概率分布可能有多个， 但使熵值最大的概率分布最真实地反映了事件的分布情况， 因为熵定义了随机变量的不确定性， 当熵最大时， 随机变量最不确定， 最难准确地预测其行为。 也就是说， 在已知部分知识的前提下，关于未知分布最合理的推断应该是符合已知知识最不确定或最大随机的推断。 最大熵概念被广泛地应用于自然语言处理中， 通常的做法是， 根据已知样本设计特征函数， 假设存在 $k$ 个特征函数 $f_i(i＝1,2, \ldots, k)$ ，它们都在建模过程中对输出有影响， 那么， 所建立的模型应满足所有这些特征的约束， 即所建立的模型p应该属于这k个特征函数约束下所产生的所有模型的集合C。 使熵H（p） 值最大的模型用来推断某种语言现象存在的可能性， 或者作为进行某种处理操作的可靠性依据， 即：

$$
\hat{p}=\underset{p\in C}{\arg max}H(p)
$$

### 2.2.2 联合熵和条件熵

如果X， Y是一对离散型随机变量X， Y～p（x, y） ， X， Y的**联合熵（joint entropy）** H（X， Y） 定义为

$$
H(X,Y) = -\sum\limits_{x\in X}\sum\limits_{y\in Y}p(x, y)log\ p(x) \tag{2-23}
$$

联合熵实际上就是<u>描述一对随机变量平均所需要的信息量</u>。给定随机变量X的情况下， 随机变量Y的**条件熵（conditional entropy）** 由式（2-24） 定义：

$$
\begin{align}
H(Y|X) &= \sum\limits_{x\in X}p(x)H(Y|X = x) \\
&= \sum\limits_{x\in X}p(x) \left[ -\sum\limits_{y\in Y} p(y|x)log\ p(y|x) \right] \\
&= -\sum\limits_{x\in X}\sum\limits_{y\in Y}p(x,y)log\ p(y|x) \tag{2-24}
\end{align}
$$

将式（2-23） 中的联合概率log p(x, y) 展开，
$$
\begin{align}
H(X, Y) &= -\sum\limits_{x\in X}\sum\limits_{y\in Y}p(x, y)\mathrm{log}[p(x)p(y|x)] \\
&= -\sum\limits_{x\in X}\sum\limits_{y\in Y}p(x, y)\mathrm{log}[p(x) + p(y|x)] \\
&= -\sum\limits_{x\in X}\sum\limits_{y\in Y}p(x, y)\mathrm{log}\ p(x) - \sum\limits_{x\in X}\sum\limits_{y\in Y}p(x, y)\mathrm{log}\ p(y|x) \\
&= -\sum\limits_{x\in X}p(x)\mathrm{log}\ p(x) - \sum\limits_{x\in X}\sum\limits_{y\in Y}p(x, y)\mathrm{log}\ p(y|x) \\
&= H(X) + H(Y|X) \tag{2-25}
\end{align}
$$
可得我们称式（2-25） 为**熵的连锁规则（chain rule for entropy）** 。 推广到一般情况， 有
$$
H(X_1, X_2, \ldots, Xn) ＝ H(X_1) ＋ H(X_2|X_1) + \ldots + H(X_n|X_1, \cdots, X_{n-1})
$$

例2-4 假设某一种语言的字符有元音和辅音两类， 其中， 元音随机变量V＝{a, i, u}， 辅音随机变量C＝{p, t, k}。 如果该语言的所有单词都由辅音-元音（consonant-vowel, C-V） 音节序列组成， 其联合概率分布P（C， V） 如表2-1所示。

![](http://q9kvrafcq.bkt.clouddn.com/gitpages/Information_Theory/2-1_table.png)

根据表2-1中的联合概率， 我们不难算出p， t， k， a， i， u这6个字符的边缘概率分别为： 1/8， 3/4， 1/8， 1/2， 1/4， 1/4。 但需要注意的是， 这些边缘概率是基于音节的， 每个字符的概率是其基于音节的边缘概率的1/2， 因此， p， t， k， a， i， u 6个字符中的每个字符的概率值实际上为： 1/16， 3/8， 1/16， 1/4， 1/8， 1/8。
现在来求联合熵。 计算联合熵的方法有多种， 以下采用的是连锁规则方法。
$$
\begin{align}
H(C) &= -\sum\limits_{c=p,t,k}p(c)log\ p(c) = -2\times\frac{1}{8}\times log\ \frac{1}{8}-\frac{3}{4}\times log\ \frac{3}{4}\\
&= \frac{9}{4} - \frac{3}{4}log\ 3\approx 1.061 (\text{比特})
\end{align}
$$
根据表2-1给出的联合概率和边缘概率， 容易计算出条件概率。 例如，
$$
\begin{align}
p(\mathrm{a|p}) =& \frac{p(\mathrm{p,a})}{p(\mathrm{p})} = \frac{1}{16}\times\frac{8}{1}=\frac{1}{2} \\
p(\mathrm{i|p}) =& \frac{p(\mathrm{p,i})}{p(\mathrm{p})} = \frac{1}{16}\times\frac{8}{1}=\frac{1}{2} \\
p(\mathrm{u|p}) =& \frac{p(\mathrm{p,u})}{p(\mathrm{p})} = 0
\end{align}
$$
为了简化起见， 我们将条件熵 $\sum\limits_{c=p,t,k}H(V|c) = - \sum\limits_{c=p,t,k}p(V|c)\mathrm{log_2}p(V|c)$ 记为  $H(\frac{1}{2}, \frac{1}{2}, 0)$ 。 其他的情况类似。 因此， 我们得到如下条件熵：
$$
\begin{align}
H(V|C) &= \sum\limits_{c=p,t,k}P(C=c)H(V|C=c) \\
&= \frac{1}{8}H(\frac{1}{2}, \frac{1}{2}, 0) + \frac{3}{4}H(\frac{1}{2}, \frac{1}{4}, \frac{1}{4}) + \frac{1}{8}H(\frac{1}{2}, 0, \frac{1}{2}) \\
&= \frac{11}{8} = 1.375(\text{比特})
\end{align}
$$
因此，
$$
\begin{align}
H(C, V) &= H(C) + H(V|C) \\
&= \frac{9}{4} - \frac{3}{4}log3 + \frac{11}{8} \approx 2.44(\text{比特})
\end{align}
$$
一般地， 对于一条长度为n的信息， 每一个字符或字的熵为
$$
H_{rate}=\frac{1}{n}H(X_{1n}) = -\frac{1}{n}\sum\limits_{x_{1n}}p(x_{1n})\mathrm{log}\,p(x_{1n}) \tag{2-26}
$$
这个数值称为**熵率（entropy rate）** 。 其中， 变量 $X_{1n}$ 表示随机变量序列 $(X_1, X_2, \ldots, X_n),  x_{1n}＝(x_1, x_2, \ldots, x_n)$ 。 以后我们采用类似的符号标记。

如果假定一种语言是由一系列符号组成的随机过程， $L＝(X_i)$ ，例如， 某报纸的一批语料， 那么， 我们可以定义这种语言L的熵作为其随机过程的熵率， 即
$$
H_{rate}(L) = \lim_{n\rightarrow\infty}{1\over n}H(X_1, X_2, \cdots, X_n) \tag{2-27}
$$
我们之所以把语言L的熵率看作语言样本熵率的极限， 因为理论上样本可以无限长。

### 2.2.3 互信息

根据熵的连锁规则， 有
$$
H(X, Y) ＝H(X) ＋H(Y|X) ＝H(Y) ＋H(X|Y)
$$
因此，
$$
H(X) - H(X|Y) ＝H(Y) - H(Y|X)
$$
这个差叫做X和Y的**互信息（mutual information, MI）** ， 记作I（X； Y） 。或者定义为： 如果（X， Y） ～p（x, y） ， 则X， Y之间的互信息 I（X；Y） ＝H（X）- H（X|Y） 。

<u>I(X; Y) 反映的是在知道了Y的值以后X的不确定性的减少量。 可以理解为Y的值透露了多少关于X的信息量</u>。互信息和熵之间的关系可以用图2-1表示。

![](http://q9kvrafcq.bkt.clouddn.com/gitpages/Information_Theory/2-1_figure.png)

如果将定义中的H（X） 和H（X|Y） 展开， 可得
$$
\begin{align}
I(X;Y) &= H(X)-H(X|Y) \\
&= H(X) + H(Y) - H(X,Y) \\
&= \sum\limits_{x}p(x)\mathrm{log}\frac{1}{p(x)} + \sum\limits_{y}\mathrm{log}\frac{1}{p(y)} + \sum\limits_{x,y}p(x,y)\mathrm{log}\,p(x,y) \\
&= \sum\limits_{x, y}p(x,y)\mathrm{log}\frac{p(x,y)}{p(x)p(y)} \tag{2-28}
\end{align}
$$


由于H（X|X） ＝0， 因此，
$$
H(X) ＝H(X) -H(X|X) ＝I(X;X)
$$
这一方面<u>说明了为什么熵又称为自信息， 另一方面说明了两个完全相互依赖的变量之间的互信息并不是一个常量， 而是取决于它们的熵</u>。

实际上， <u>互信息体现了两变量之间的依赖程度： 如果  $I(X; Y) \gg 0$ ， 表明X和Y是高度相关的； 如果 $I(X; Y) = 0$ ， 表明X和Y是相互独立的； 如果 $I(X; Y) \ll 0$ ， 表明Y的出现不但未使X的不确定性减小， 反而增大了 X的不确定性， 常是不利的</u> 。 平均互信息量是非负的。同样， 我们可以推导出**条件互信息和互信息的连锁规则**：
$$
I(X;Y|Z) = I((X;Y)|Z) = H(X|Z) - H(X|Y,Z) \tag{2-29}
$$

$$
\begin{align}
I(X_{1n};Y) &= I(X_1,Y) + \cdots + I(X_n;Y|X_1, \cdots , X_{n-1}) \\
&= \sum\limits_{i=1}I(X_i;Y|X_1,\cdots,X_{i-1}) \tag{2-30}
\end{align}
$$

互信息在词汇聚类（word clustering） 、 汉语自动分词、 词义消歧等问题的研究中具有重要用途。

### 2.2.4 相对熵

**相对熵（relative entropy）** 又称**Kullback-Leibler差异（KullbackLeibler divergence）** ， 或简称**KL距离**， 是<u>衡量相同事件空间里两个概率分布相对差距的测度</u>。 两个概率分布p（x） 和q（x） 的相对熵定义为
$$
D(p\|q) = \sum\limits_{x\in X}p(x)log\frac{p(x)}{q(x)} \tag{2-31}
$$
该定义中约定 $0\mathrm{log}(0/q) ＝0, p\mathrm{log}(p/0) ＝\infty$ 。 表示成期望值为
$$
D(p\|q) = E_p\left(\mathrm{log}\frac{p(x)}{q(x)}\right) \tag{2-32}
$$
显然， <u>当两个随机分布完全相同时， 即p＝q， 其相对熵为0。 当两个随机分布的差别增加时， 其相对熵期望值也增大</u>。

互信息实际上就是衡量一个联合分布与独立性差距多大的测度：
$$
I(X;Y)=D(p(x,y)\|p(x)p(y)) \tag{2-33}
$$
证明：
$$
\begin{align}
I(X,Y)&=H(X)-H(X|Y) \\
&=-\sum\limits_{x\in X}p(x)\mathrm{log}\,p(x)+\sum\limits_{x\in X}\sum\limits_{y\in Y}p(x,y)\mathrm{log}\,p(x|y) \\
&= \sum\limits_{x\in X}\sum\limits_{y\in Y}p(x,y)\mathrm{log}\frac{p(x|y)}{p(x)} \\
&= \sum\limits_{x\in X}\sum\limits_{y\in Y}p(x,y)\mathrm{log}\frac{p(x,y)}{p(x)p(y)} \\
&= D(p(x,y)\|p(x)p(y))
\end{align}
$$
同样， 我们也可以推导出条件相对熵和相对熵的连锁规则：
$$
D(p(y|x)\|q(y|x)) =\sum\limits_{x}p(x)\sum\limits_{y}p(y|x)\mathrm{log}\frac{p(y|x)}{q(y|x)} \tag{2-34}
$$

$$
D(p(x,y)\|q(x,y))=D(p(x)\|q(x))+D(p(y|x)\|q(y|x)) \tag{2-35}
$$

### 2.2.5 交叉熵

根据前面熵的定义， 知道熵是一个不确定性的测度， 也就是说， 我们对于某件事情知道得越多， 那么， 熵就越小， 因而对于试验的结果我们越不感到意外。 <u>交叉熵的概念就是用来衡量估计模型与真实概率分布之间差异情况的</u>。如果一个随机变量X～p（x） ， q（x） 为用于近似p（x） 的概率分布， 那么， 随机变量X和模型q之间的**交叉熵（cross entropy）** 定义为
$$
\begin{align}
H(X,q) &= H(X)+D(p\|q) \\
&=-\sum\limits_{x}p(x)\mathrm{log}q(x) \\
&= E_p\left(\mathrm{log}\frac{1}{q(x)}\right) \tag{2-36}
\end{align}
$$
由此， 可以定义语言 $L＝(X_i) \sim p(x)$ 与其模型q的交叉熵为
$$
H(L,q)=-\lim_{n\rightarrow \infty}\frac{1}{n}\sum\limits_{x_1^n}\mathrm{x_1^n} \tag{2-37}
$$
其中， $X_1^n＝x_1, x_2, \ldots , x_n$ ，为 L 的语句， $p(X_1^n)$ 为 L 中 $X_1^n$ 的概率，$q(X_1^n)$ 为模型 q 对 $X_1^n$ 的概率估计。 至此， 仍然无法计算这个语言的交叉熵， 因为我们并不知道真实概率 $p(X_1^n)$ ， 不过可以假设这种语言是“理想”的， 即n趋于无穷大时， 其全部“单词”的概率和为1。 也就是说， 根据信息论的定理： 假定语言L是稳态（stationary） 遍历的（ergodic） 随机过程， L与其模型q的交叉熵计算公式就变为
$$
H(L,q)=-\lim_{n\rightarrow\infty}\frac{1}{n}\mathrm{log}q(x_1^n) \tag{2-38}
$$
由此， 可以根据模型q和一个含有大量数据的L的样本来计算交叉熵。 在设计模型q时， 目的是使交叉熵最小， 从而使模型最接近真实的概率分布p（x） 。 一般地， 在n足够大时我们近似地采用如下计算方法：
$$
H(L, q) \approx -\frac{1}{n}\mathrm{log}q(x_1^n) \tag{2-39}
$$
<u>交叉熵与模型在测试语料中分配给每个单词的平均概率所表达的含义正好相反， 模型的交叉熵越小， 模型的表现越好。</u>

## 引用

[^1]: [wkipedia 自信息](https://zh.wikipedia.org/wiki/%E8%87%AA%E4%BF%A1%E6%81%AF)
[^2]: [wikipedia 信息熵](https://zh.wikipedia.org/wiki/%E7%86%B5_(%E4%BF%A1%E6%81%AF%E8%AE%BA))
[^3]: [统计自然语言处理，第2版，宗庆成](https://book.douban.com/subject/25746399/)

