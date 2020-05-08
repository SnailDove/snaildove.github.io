---
title: positive definite and least squares
mathjax: true
mathjax2: true
categories: English
date: 2017-08-01 22:16:00
tags: [linear_algebra]
toc: true
---


## positive definite

When **a symmetric matrix $A$ has one of these five properties, it has them all** and $A$ is positive definite:

>1. all **n eigenvalue** are positive.
>2. all **n principal minors(n upper left determinants)** are positive.
>3. all **n pivots are positive**.
>4. $x^{T}Ax$ is positive except when $x = 0$ (this is usually the definition of positive definiteness and the **energy-based** definition).
>5. $A$ equals $R^{T}R$ for a matrix $R$ with **independent columns**.

Let us prove the fifth rule. If $A = R^{T}R$, then

$$
\begin{eqnarray} 
x^{T}Ax&=&x^{T}R^{T}Rx \nonumber\\
&=&(x^{T}R^{T})Rx \nonumber\\
&=&(Rx)^{T}Rx \nonumber\\
&=&\|Rx\| \nonumber\\
&\ge&0 \nonumber
\end{eqnarray}
$$

And the columns of $R$ are also independent, so $\|Rx\|=x^{T}Ax>0$, except when $x$=0 and thus $A$ is positive definite.

## $A^{T}A$

$A_{m\times n}$ is almost certainly not symmetric, but $A^{T}A$ is square (n by n) and symmetric. We can easily get the following equations through left multiplying $A^{T}A$ by $x^{T}$ and right multiplying $A^{T}A$ by $x$:

$$
\begin{eqnarray}
x^{T}A{^TA}x&=&x^{T}(A{^TA})x\nonumber\\
&=&(x^{T}A^{T})Ax\nonumber\\
&=&(Ax)^{T}(Ax)\nonumber\\
&=&\|Ax\|\nonumber\\
&\ge&0\nonumber
\end{eqnarray}
$$

If $A_{m\times\,n}$ has rank $n$ (independent columns), then except when $x = 0$, $Ax=\|Ax\|=x^{T}(A{^TA})x>0$ and thus $A^{T}A$ is positive definite. And vice versus.

Besides, $A^{T}A$ is invertible only if $A$ has rank $n$ (independent columns). To prove this, we assume $Ax=0$, then:

$$
\begin{eqnarray} 
Ax&=&0\nonumber\\
(Ax)^{T}(Ax)&=&0\nonumber\\
(x^{T}A{^T})(Ax)&=&0\nonumber\\
x^{T}A{^T}(Ax)&=&x^{T}0\nonumber\\
(A{^TA})x&=&0\nonumber
\end{eqnarray}
$$

From the above equations, we know solutions of $Ax=0$ are also solutions of  $(A{^TA})x=0$. Because $A_{m\times\,n}$ has a full set of column rank (independent columns),  $Ax=0$ only has a zero solution as well as $(A{^T}A)x=0$. Furthermore, if $A{^T}A$ is invertible, then $A_{m\times\,n}$ has rank $n$ (independent columns). We also notice that if $A$ is square and invertible, then  $A{^T}A$ is invertible. 

Overall, **if all columns of $A_{m\times\,n}$ are mutual independent, then $(A{^T}A)$ is invertible and positive definite as well, and vice versus.**

## least square
We have learned that **least square** comes from **projection** :
$$b-p=e\Rightarrow\,A^{T}(b-A\hat{x})=0\Rightarrow\,A^{T}A\hat{x}=A^{T}b$$
Consequently, only if $A^{T}A$ is invertible, then we can use linear regression to find approximate solutions $\hat{x}=(A^{T}A)^{-1}A^{T}b$ to unsolvable systems of linear equations. 

According to the reasoning before, we know as long as all columns of $A_{m\times\,n}$ are mutual independent, then $A{^T}A$ is invertible. At the same time we ought to notice that the columns of $A$ are guaranteed to be independent if they are orthoganal and even orthonormal. 

In another prospective, if $A^{T}A$ is positive definite, then $A_{m\times\,n}$ has rank $n$ (independent columns) and thus $A^{T}A$ is invertible.

**Overall, if $A^{T}A$ is positive definite or invertible, then we can find approximate solutions of least square**.
