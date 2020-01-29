---
title:  Trees describe the sample space 
mathjax: true
mathjax2: true
categories: English
tags: [probability]
date: 2017-08-15 20:16:00
commets: true
toc: true
---

This note comes from [Introduction to Probability, 2nd Edition](http://www.athenasc.com/probbook.html)

## Example 1.9 Rada Detection 

If an aircraft is present in a certain area, a radar detects it and generates an alarm signal with probability 0.99. If an aircraf is not present. the radar generates a (false) alarm, with probability 0.10. We assume that an aircraft is present with probability 0.05. What is the probability of no aircraf presence and a false alarm? What is the probability of aircraf presence and no detection?

$A$ sequential representation of the experiment is appropriate here, as shown in Fig. 1.9. Let $A$ and $B$ be the events $A = \{an\ aircraft\ is\ present\}$, $B = \{the\ radar\ generates\ an\ alarm\} $, and consider also their complements $A^c=\{an\ aircraft\ is\ not present\}$$，$$B^c=\{the\ radar\ does\ not\ generate\ an\ alarm\}$。

The given probabilities are recorded along the corresponding branches of the tree describing the sample space, as shown in Fig. 1.8. Each event of interest corresponds to a leaf of the tree and its probability is equal to the product of the probabilities associated with the branches in a path from the root to the corresponding leaf. The desired probabilities of false alarm and missed detection are

$$P(false\ alarm) = P(A^c ∩ B) = P(A^c)P(B | A^c) = 0.95 · 0.10 = 0.095$$，
$$P(missed\ detection) = P(A ∩ B^c) = P(A)P(B^c | A) = 0.05 · 0.01 = 0.0005$$. 

Extending the preceding example, we have a general rule for calculating various probabilities in conjunction with a tree-based sequential description of an experiment. In particular:

-   (a) We set up the tree so that an event of interest is associated with a leaf. We view the occurrence of the event as a sequence of steps, namely, the traversals of the branches along the path from the root to the leaf.
-   (b) We record the conditional probabilities associated with the branches of the tree.
-   (c) We obtain the probability of a leaf by multiplying the probabilities recorded along the corresponding path of the tree. 

![base64_of_Sequential_description_of_the_sample_space_for_the_radar_detection](http://q4vftizgw.bkt.clouddn.com/gitpage/introduction-to-probability/trees_describe_the_sample_space/1.png)

## multiplication rule

In mathematical terms, we are dealing with an event A which occurs if and only if each one of several events $A_1, . . . , A_n$ has occurred, i.e., $A = A_1 ∩ A_2 ∩ · · · ∩ A_n$. The occurrence of $A$ is viewed as an occurrence of $A_1$, followed by the occurrence of $A_2$, then of $A_3$, etc, and it is visualized as a path on the tree with $n$ branches, corresponding to the events $A_1, . . . , A_n$. The probability of $A$ is given by the following rule (see also Fig. 1.9). 

![base64_of_multiplication_rule](http://q4vftizgw.bkt.clouddn.com/gitpage/introduction-to-probability/trees_describe_the_sample_space/2.png)

The multiplication rule can be verified by writing
$$P(\cap^n_{i=1} A_i)=P(A_1)\frac{P(A_1\cap A_2)}{P(A_1)}\frac{P(A_1\cap A_2\cap A_3)}{P(A_1\cap A_2)}\cdots\frac{P(\cap_{i=1}^n A_i)}{P(\cap^{n-1}_{i=1} A_i)}$$,

and by using the definition of conditional probability to rewrite the right-hand side above as

$$P(A_1)P(A_2|A_1)P(A_3|A_1\cap A_2)\cdots P(A_N|\cap^{n-1}_{i=1} A_i)$$.

![base64_of_visualization_of_multiplication_rule](http://q4vftizgw.bkt.clouddn.com/gitpage/introduction-to-probability/trees_describe_the_sample_space/3.png)

**The intersection event $A = A_1∩A_2∩· · ·∩A_n$ is associated with a path on the tree of a sequential description of the experiment. We associate the branches of this path with the events $A_1, . . . , A_n$, and we record next to the branches the corresponding conditional probabilities.**

The final node of the path corresponds to the intersection event $A$, and its probability is obtained by multiplying the conditional probabilities recorded along the branches of the path 

$$P(A_1\cap A_2\cap\cdots\cap A_3)=P(A_1)P(A_2|A_1)\cdots P(A_n|A_1\cap A_2\cdots \cap A_{n-1}).$$

**Note that any intermediate node along the path also corresponds to some intersection event and its probability is obtained by multiplying the corresponding conditional probabilities up to that node.** For example, the event $A_1 ∩ A_2 ∩ A_3$ corresponds to the node shown in the figure, and its probability is 

$$P(A_1\cap A_2\cap A_3)=P(A_1)P(A_2|A_1)P(A_3|A_1\cap A_2).$$

For the case of just two events, A1 and A2, the multiplication rule is simply the definition of conditional probability. 


