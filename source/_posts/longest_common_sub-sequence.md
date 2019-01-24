---
title: 动态规划-最长公共子序列
mathjax: true
mathjax2: true
categories: 中文
date: 2014-06-01 22:16:00
tags: [Basic Algorithm]
toc: true
copyright: false
---

## 算法总体思想

动态规划（Dynamic Programming）是通过组合子问题的解而解决整个问题的。分治是指将问题划分成一些独立的子问题，递归地求解各子问题，然后合并子问题的解而得到原始问题的解，与此不同，动态规划适用于子问题不是独立的情况，也就是各个子问题包含公共的子问题。在这种情况下，采用分治法会做许多不必要的工作，即重复地求解公共地子问题。动态规划算法对每个子问题只求解一次，将其结果保存在一张表中，从而避免每次遇到各个子问题时重新计算答案。

## 动态规划算法的基本要素

### 最优子结构

-   矩阵连乘计算次序问题的最优解包含着其子问题的最优解。这种性质称为最优子结构性质。
-   在分析问题的最优子结构性质时，所用的方法具有普遍性：首先假设由问题的最优解导出的子问题的解不是最优的，然后再设法说明在这个假设下可构造出比原问题最优解更好的解，从而导致矛盾。
-   利用问题的最优子结构性质，以自底向上的方式递归地从子问题的最优解逐步构造出整个问题的最优解。最优子结构是问题能用动态规划算法求解的前提。

注意：同一个问题可以有多种方式刻划它的最优子结构，有些表示方法的求解速度更快（空间占用小，问题的维度低）

### 重叠子问题

-   递归算法求解问题时，每次产生的子问题并不总是新问题，有些子问题被反复计算多次。这种性质称为子问题的重叠性质。
-   动态规划算法，对每一个子问题只解一次，而后将其解保存在一个表格中，当再次需要解此子问题时，只是简单地用常数时间查看一下结果。
-   通常不同的子问题个数随问题的大小呈多项式增长。因此用动态规划算法只需要多项式时间，从而获得较高的解题效率。 

## 问题举例

### 最长公共子序列(LCS)

-   若给定序列$X=\{x_1,x_2,…,x_m\}$，则另一序列$Z=\{z_1,z_2,…,z_k\}$，是 $X$ 的子序列是指存在一个严格递增下标序列$\{i_1,i_2,…,i_k\}$使得对于所有$j=1,2,…,k$ 有：$z_j=x_{i}$。例如，序列 $Z=\{B，C，D，B\}$ 是序列 $X=\{A，B，C，B，D，A，B\}$ 的子序列，相应的递增下标序列为$\{2，3，5，7\}$。
-   给定2个序列 $X$ 和 $Y$，当另一序列 $Z$ 既是 $X$ 的子序列又是 $Y$ 的子序列时，称 $Z$ 是序列 $X$ 和 $Y$ 的公共子序列。
-   给定2个序列$X=\{x_1,x_2,…,x_m\}$和 $Y=\{y_1,y_2,…,y_n\}$，找出 $X$ 和 $Y$ 的最长公共子序列。

### 最长公共子序列的结构(LCS)

设序列$X=\{x_1,x_2,…,x_m\}$和 $Y=\{y_1,y_2,…,y_n\}$的最长公共子序列为 $Z=\{z_1,z_2,…,z_k\}$ ，则

1.  若$x_m=y_n$，则$z_k=x_m=y_n$，且 $z_{k-1}$ 是 $\{x_1,\ldots, x_{m-1}\}$ 和 $\{y_1, \ldots, y_{n-1}\}$ 的最长公共子序列。
2.  若 $x_m≠y_n$ 且 $z_k≠x_m, z_k=y_n$，则 $Z$ 是 $\{x_1,\ldots, x_{m-1}\}$ 和 $Y$ 的最长公共子序列。
3.  若 $x_m≠y_n$且 $z_k≠y_n, z_k=x_m$，则 $Z$ 是 $X$ 和  $\{y_1, \ldots, y_{n-1}\}$ 的最长公共子序列。

由此可见，2个序列的最长公共子序列包含了这2个序列的前缀的最长公共子序列。因此，最长公共子序列问题具有最优子结构性质。 

## LCS时间复杂度

求解LCS问题，不能使用暴力搜索方法。一个长度为n的序列拥有 2的n次方个子序列，它的时间复杂度是指数阶，而且还是两个序列求最长公共子序列。

### 子问题的递归结构

由最长公共子序列问题的最优子结构性质建立子问题最优值的递归关系。**用$c[i][j]$记录序列和的最长公共子序列的长度。** 其中，$X[i]=\{x_1,x_2,…,x_i\}$；$Y[j]=\{y_1,y_2,…,y_j\}$。当 $i=0$ 或 $j=0$ 时，空序列是 $X[i]$ 和 $Y[j]$ 的最长公共子序列。故此时 $c[i][j]=0$ 。其他情况下，由最优子结构性质可建立递归关系如下：

$c[i][j]=\cases{0,\quad i=0,j=0 \\ c[i-1][j-1]+1\quad i,j>0;x_i=y_j \\ max\{c[i][j-1],c[i-1][j]\}\quad i,j>0;x_i\ne y_j}$

## 计算最优值（伪代码）

```c++
AlgorithmlcsLength(x,y,b)
mßx.length-1;
nßy.length-1;
c[i][0]=0;
c[0][i]=0;
for(int i= 1; i<= m;i++)
    for(int j = 1; j <= n; j++)
      if(x[i]==y[j])
          c[i][j]=c[i-1][j-1]+1;
          b[i][j]=1;
      else if(c[i-1][j]>=c[i][j-1])
          c[i][j]=c[i-1][j];
          b[i][j]=2;
      else
          c[i][j]=c[i][j-1];
          b[i][j]=3;
```

## 源代码实现（测试通过）

```c++
#include<iostream>
#define LEN_ARR_A  20
#define LEN_ARR_B  12

using namespace std;

/*
 *  brief     : calculate longest common substring of two string and 
 *              mark  path of getting the longest common substring  
 *	parameter : lenComStr : length of common substring
 *              solutionPath : mark of path of solution
 *
 *  note：      dynamic programming :caculate longest common substring between X(n)
 *              and Y(m-1),likey to caculate longest common substring between X(n) 
 *              and Y(m-1) or between X(n-1) and Y(m). X(0) and Y(0) are not referenced!
 *
 *
 *	return :    null
 */
template<class Type>
void LCSLength(size_t lenStrA, size_t lenStrB, Type *strA, Type *strB, size_t ** lenComStr, size_t ** solutionPath)
{
	for(size_t i = 0; i < lenStrA; ++i)
		lenComStr[i][0] = 0;
	for(size_t i = 0; i < lenStrB; ++i)
		lenComStr[0][i] = 0;
	for(size_t i = 1; i < lenStrA; ++i)
		for(size_t j = 1; j < lenStrB; ++j){
			if(strA[i] == strB[j]){
				lenComStr[i][j] = lenComStr[i -1][j - 1] + 1; 
				//global solution depends on local solution
				solutionPath[i][j] = 1;               
				//mark  path of getting the longest common substring 
			}
			else if(lenComStr[i - 1][j] >= lenComStr[i][j - 1]){
				lenComStr[i][j] = lenComStr[i - 1][j];  
				//global solution depends on local solution
				solutionPath[i][j] = 2;
			}else{
				lenComStr[i][j] = lenComStr[i][j - 1];  
				//global solution depends on local solution
				solutionPath[i][j] = 3;
			}
		}
}
/*
 *  brief     : output longest common substring of two string  
 *              
 *	parameter : i is beginning index of char array
 *              j is end index of char array
 *              solutionPath is the path of solution
 *
 *  note：      value of solutionPath[i][j] has 3 states  
 *
 *	return :    null
 */
 template<class Type>
void LCS(size_t i, size_t j, Type * strA, size_t ** solutionPath)
{
	if(i == -1 || j == -1)	return;
	if(solutionPath[i][j] == 1){
		LCS(i - 1, j - 1, strA, solutionPath); 
		cout << strA[i];
	}else if(solutionPath[i][j] == 2)
			LCS(i - 1, j, strA, solutionPath);
	else{
			LCS(i, j - 1, strA, solutionPath);
	}
}

int main()
{
	char strA[LEN_ARR_A] = { '0','B', 'D', 'C', 'A', 'B', 'A'};
	char strB[LEN_ARR_B] = { '0', 'A', 'B', 'C', 'B', 'D', 'A', 'B'};

	size_t ** lenComStr = new size_t*[LEN_ARR_A];
	size_t ** solutionPath = new size_t*[LEN_ARR_A];
	for(size_t i = 0; i < LEN_ARR_A; ++i)	{
		lenComStr[i] = new size_t[LEN_ARR_B];
		solutionPath[i] = new size_t[LEN_ARR_B];
	}
	LCSLength(LEN_ARR_A, LEN_ARR_B, strA, strB, lenComStr, solutionPath);
	LCS(LEN_ARR_A - 1, LEN_ARR_B - 1, strA, solutionPath);
	return 0;
}
```
![img](http://pltr89sz6.bkt.clouddn.com/gitpage/dynamic-programming/1.png)
