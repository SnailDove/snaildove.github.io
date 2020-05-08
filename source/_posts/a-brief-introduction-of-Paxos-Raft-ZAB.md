---
title: CAP定理图示与Raft各种场景演示
date: 2019-12-11
copyright: true
categories: 中文
tags: [distributed-system]
mathjax: true
mathjax2: true
toc: true
top: 12
---

本文主要着重于CAP定理和raft各种场景演示


## CAP 定理

根据加州大学伯克利分校计算机科学家Eric Brewer说法，该定理于1998年秋季首次出现。该定理于1999年作为CAP原理发表，并由Brewer在<a style="color:#0879e3" href="https://en.wikipedia.org/wiki/Symposium_on_Principles_of_Distributed_Computing)作为[猜想](https://en.wikipedia.org/wiki/Conjecture">2000 年的分布式原理研讨会上</a>提出<a style="color:#0879e3" href="https://en.wikipedia.org/wiki/Symposium_on_Principles_of_Distributed_Computing">计算</a>（PODC）。2002年，麻省理工学院的塞斯·吉尔伯特（<a style="color:#0879e3" href="https://en.wikipedia.org/w/index.php?title=Seth_Gilbert&action=edit&redlink=1">Seth Gilbert</a> ) 和 南希·林奇（<a style="color:#0879e3" href="https://en.wikipedia.org/wiki/Nancy_Lynch">Nancy Lynch</a>）发表了布鲁尔猜想的正式证明，使之成为一个<a style="color:#0879e3" href="http://en.wikipedia.org/wiki/CAP_theorem">定理</a>。

<img src="https://stph.scenari-community.org/bdd/nos1/res/cap-theorem_1.png" alt="11122" style="zoom:65%;" />

<a style="color:#0879e3" href="http://en.wikipedia.org/wiki/CAP_theorem">CAP定理</a>指出分布式计算机系统不可能同时提供以下三个保证（来自<a style="color:#0879e3" href="https://en.wikipedia.org/wiki/CAP_theorem"> wiki : CAP Theorum</a>）：

- **Consistency**: Every read receives the most recent write or an error

    一致性：每次读取都会收到最新的写入或错误

- **Availability**: Every request receives a (non-error) response, without the guarantee that it contains the most recent write

    可用性：每个请求都会收到一个（非错误）响应，但不能保证它包含最新的写入

- **Partition tolerance**: The system continues to operate despite an arbitrary number of messages being dropped (or delayed) by the network between nodes

    分区容忍：尽管节点之间的网络丢弃（或延迟）了任意数量的消息，但系统仍继续运行

CAP定理证明详细查看：<a style="color:#0879e3" href="http://lpd.epfl.ch/sgilbert/pubs/BrewersConjecture-SigAct.pdf">Gilbert和Lynch的论文</a>，以下翻译自<a style="color:#0879e3" href="https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/">图解CAP定理</a>：

#### 分布式系统

让我们考虑一个非常简单的分布式系统。我们的系统由两个服务器 $G_1$ 和 $G_2$ 组成。这两个服务器都跟踪相同的变量 $v_0$，其初始值为 $v_0$。$G_1$ 和 $G_2$ 可以相互通信，也可以与外部客户端通信。这是我们的系统的布局。

![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap1.svg)

客户端可以请求从任何服务器进行写入和读取。服务器收到请求后，将执行所需的任何计算，然后响应客户端。例如，这是写的样子。

![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap2.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap3.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap4.svg)



这就是读取的情况。

![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap5.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap6.svg)

现在我们已经建立了系统，接下来让我们回顾一下对于系统一致性，可用性和分区容错性意味着什么。

#### 一致性

> 在写操作完成之后开始的任何读操作必须返回该值，或者以后的写操作的结果 —— Gilbert，Lynch

在一致的系统中，客户端将值写入任何服务器并获得响应后，它希望从其读取的任何服务器取回该值（或更新的值）。

这是一个**不一致的**系统的示例。

![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap7.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap8.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap9.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap10.svg)![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap11.svg)

我们的客户端写 $v_1$ 至 $G_1$ 和  $G_1$ 确认，但是当它从 $G_2$ 读取时 $G_2$，它将获取旧的数据：$v_0$。

另一方面，这是一个**一致的**系统的示例。

![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap12.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap13.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap14.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap15.svg)![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap16.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap17.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap18.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap19.svg)

在这个系统中，在向客户端发送确认之前 $G_1$ 将其值复制到 $G_2$ 。因此，当客户端从 $G_2$读取 ，它获取 $v$ 的最新值： $v_1$ 。

#### 可用性

> 系统中非故障节点收到的每个请求都必须得到响应 —— Gilbert，Lynch

在可用的系统中，如果我们的客户端向服务器发送请求并且服务器没有崩溃，则服务器最终必须响应客户端。不允许服务器忽略客户端的请求。

#### 分区容错性

> 网络将被允许任意丢失从一个节点发送到另一节点的许多消息  —— Gilbert，Lynch

这意味着 $G_1$ 和 $G_2$ 互相发送的任何消息能被删除。如果所有消息都被丢弃，那么我们的系统将如下所示（注：初始值为 $v_0$ ）。

![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap20.svg)

为了达到分区容错性，我们的系统必须能够在任意网络分区下正常运行。

#### 证明

现在我们已经了解了一致性，可用性和分区容错性的概念，我们可以证明一个系统不能同时拥有这三个。

对于这个矛盾，假设确实存在一个一致，可用且分区容错的系统。我们要做的第一件事是对系统进行分区。看起来像这样。

![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap21.svg)

接下来，我们有客户端要求 $v_1$ 写入 $G_1$。由于我们的系统可用，因此 $G_1$ 必须响应。由于网络是分区的，因此 $G_1$ 无法将其数据复制到 $G_2$。Gilbert 和 Lynch 将此执行阶段称为 $\alpha_1$。

><center><strong>译者附</strong></center>
>原文：”Since the network is partitioned, however, $G_1$ cannot replicate its data to $G_2$.”
>
>网络**被**分区，比如：中美海底电缆火山喷发断掉了，淘宝电缆被施工方不小心挖掉了，这时候，对于前者美国的所有服务器和中国的所有服务器都是对外可用的，国家内的服务器节点都是互通的，但是中美之间的服务器是不通的，虽然开始阶段所有各个服务器节点都是互通的，那么这时候就发生了**网络分区**。对于后者，国内的情况也同理，各个大区之间也有可能发生网络分区，举例：华东，华北，华南，西北，西南等等。

![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap22.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap23.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap24.svg)

接下来，我们让客户端向 $G_2$ 发出读取请求。同样，由于我们的系统可用，因此 $G_2$ 必须响应。由于网络是分区的，因此 $G_2$ 无法从 $G_1$ 更新其值。返回 $v_0$。Gilbert 和 Lynch 将此执行阶段称为 $\alpha_2$。

![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap25.svg) ![img](https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/assets/cap26.svg)

客户端已经将 $v_1$ 写至 $G_1$ 之后，$G_2$ 返回 $v_0$ 给客户端。这是不一致的。

我们假设存在一个一致的，可用的，分区容错的系统，但是我们只是展示了存在任何此类系统执行的情况，其中该系统的行为不一致。因此，不存在这样的系统。

><center><strong>译者附</strong></center>
>由于发生了网络分区，此时只能在可用性和一致性之间做一个取舍，比如上文提到的海底电缆断掉，那么为了一致性，停掉中国或者美国服务器节点提供的对外服务，这时候可能所有原本请求国内的服务器节点都得转向去请求美国的服务器节点，但是这时候就降低了可用性，比如：请求的延迟。那么另外一种情况，保持中美服务器节点继续对外服务，那么可用性没有变化，但是就破坏了一致性，因为网络分区以后中美内部服务器节点行为不一致，这样给系统留下的影响（数据，及其间接衍生物：比如算法迭代更新等）是不一样的。

##  什么是一致性

- 弱一致性

    - 最终一致性（无法实时获取最新更新的数据，但是一段时间过后，数据是一致的）

        1. DNS(Domain Name System)
        
            <img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/a-brief-introduction-of-Paxos-Raft-ZAB/1573824877788.png" alt="1572487" style="zoom:80%;" />
        
        2. Gossip(Cassandra的通信协议) 
        
        3.  优先保证AP（Availability, Partition tolerance）的 CouchDB，Cassandra，DynamoDB，Riak
    
- 强一致性

    - 同步
    - Paxos
    - Raft（multi-paxos）
    - ZAB（multi-paxos） 

## 定义问题

- 数据不能存在单个节点上，防止单点故障。
- 分布式系统对fault tolerance的一般解决方案是state machine replication。
- 本文主题是 state machine replication的共识（consensus）算法。paxos其实是一个共识算法。
- 系统的最终一致性，不仅需要达成共识，还会取决于客户端（client）的行为，后文将详细说明。

### 一致性模拟

## 强一致性算法

### 主从同步复制

1. 只有主节点（master）接受客户端（client）请求。
2. 由主节点复制日志到多个从节点（slave）。
3. 主节点（master）等待直到所有从节点（slave）返回成功，才能向客户端返回写成功。

**缺点**：

任意一个节点失败（master或者某一个slave阻塞）都将导致整个集群不可用，虽然保证了强一致性（Strong Consistency），但是却大大降低了可用性（Weak Availability）。

### 多数派

每次写都保证写入大于 N/2 个节点，每次读都保证大于 N/2 个节点中读，总共 N 个节点。

主要是为了解决主从同步复制中“所有”节点都得处于正常运转。**缺点**：在并发环境下，无法保证系统正确性，顺序非常重要。例如以下场景：

![1573827965034](http://q9kvrafcq.bkt.clouddn.com/distributed-system/a-brief-introduction-of-Paxos-Raft-ZAB/1573827965034.png)

### Paxos

Paxos算法是 Lesile Lamport（Latex发明者）提出的一种基于消息传递的分布式一致性算法，于1998年在《The Part-Time Parliament》论文中首次公开，使其获得2013年图灵奖。

最初的描述使用希腊的一个小岛Paxos作为比喻，描述了Paxos小岛中通过决议的流程，并以此命名这个算法，但是这个描述理解起来比较有挑战性。

为描述Paxos算法，Lamport虚拟了一个叫做Paxos的希腊小岛，这个小岛按照议会民主制的政治模式制定法律，但是没有人愿意将自己的全部时间精力放在这种事上。所以无论是议员，议长或者传递纸条的服务员都不能承诺别人需要时一定会出现，也无法承诺批准决议或者传递消息的时间。

### Basic Paxos

#### <font color="#3399cc">角色介绍（roles）</font>

- client：系统外部角色，请求发起者。像**民众**。
- proposer：**接受client请求，向集群提出提案**（proposal）。并在冲突发生时，起到冲突调节的作用。像议员，替民众提出提案。
- acceptor (voter)：**提案的投票和接收者**，只有在达到法定人数（Quorum，一般即为 majority 多数派）时，提案者提出的提案才会最终被接受。像国会。
- learner：提议接收者，backup，备份，对集群一致性没什么影响。像**记录员**。

#### <font color="#3399cc">2阶段（phases）的步骤</font>

在这个部分先大致过一遍流程，细节后文说明，形象化描述先把流程走通。

1. Phase 1a: Prepare

    proposer 提出一个提案，编号为N，此 N 大于这个proposer之前提出的提案编号。向所有 acceptor 请求接受。

2. Phase 1b: Promise

    如果 N 大于acceptor之前接受的任何提案编号则接受，否则认为此提案是已经提出过的旧提案，直接拒绝。如果 promise 阶段达到了 quorum（法定人数），proposer 这一步成功否则失败。

3. Phase 2a: Accept

    promise 阶段成功以后，proposer 进一步发出accept请求，此请求包含提案编号 （N），以及提案内容（V）。

4. Phase 2b: Accepted

    如果此acceptor在此期间没有收到任何编号大于 N 的提案否则忽略，且接受的acceptor达到法定人数，则接受此提案内容。因此在这个阶段，编号N的提案也可能失效。

所以由以上可知：跟现实生活中不太一样的地方在于acceptor（voter）不在乎提案内容，在乎提案编号。

#### <font color="#3399cc">图示流程</font>

**Basic Paxos when an Acceptor fails**

在下图中，有1个client，1个proposer，3个acceptor（即法定人数为3）和2个 learner（由2条垂直线表示）。该图表示第一轮成功的情况（即网络中没有进程失败）。

```txt
Client   Proposer      Acceptor     Learner
   |         |          |  |  |       |  |
   X-------->|          |  |  |       |  |  Request
   |         X--------->|->|->|       |  |  Prepare(1)
   |         |<---------X--X--X       |  |  Promise(1,{Va,Vb,Vc})
   |         X--------->|->|->|       |  |  Accept!(1,V)
   |         |<---------X--X--X------>|->|  Accepted(1,V)
   |<---------------------------------X--X  Response
   |         |          |  |  |       |  |
```

这里 V 是最后的 {Va, Vb, Vc}.

**Basic Paxos when an Acceptor fails**

```txt
Client   Proposer      Acceptor     Learner
   |         |          |  |  |       |  |
   X-------->|          |  |  |       |  |  Request
   |         X--------->|->|->|       |  |  Prepare(1)
   |         |          |  |  !       |  |  !! FAIL !!
   |         |<---------X--X          |  |  Promise(1,{Va, Vb, null})
   |         X--------->|->|          |  |  Accept!(1,V)
   |         |<---------X--X--------->|->|  Accepted(1,V)
   |<---------------------------------X--X  Response
   |         |          |  |          |  |
```

**Basic Paxos when a Proposer fails**

在这种情况下，proposer 在提出值之后但在达成协议之前失败。具体来说，它在Accept消息的中间失败，因此只有一个Acceptor接收到该值。此时由新的proposer（即图中的 NEW LEADER，选举出来的，怎么选举看后面详细分析）。请注意，在这种情况下有2轮（轮从上到下垂直进行）。

```txt
Client  Proposer        Acceptor     Learner
   |      |             |  |  |       |  |
   X----->|             |  |  |       |  |  Request
   |      X------------>|->|->|       |  |  Prepare(1)
   |      |<------------X--X--X       |  |  Promise(1,{Va, Vb, Vc})
   |      |             |  |  |       |  |
   |      |             |  |  |       |  |  !! Leader fails during broadcast !!
   |      X------------>|  |  |       |  |  Accept!(1,V)
   |      !             |  |  |       |  |
   |         |          |  |  |       |  |  !! NEW LEADER !!
   |         X--------->|->|->|       |  |  Prepare(2)
   |         |<---------X--X--X       |  |  Promise(2,{V, null, null})
   |         X--------->|->|->|       |  |  Accept!(2,V)
   |         |<---------X--X--X------>|->|  Accepted(2,V)
   |<---------------------------------X--X  Response
   |         |          |  |  |       |  |
```

新的提案人重新提出此前失败的提案，但是此时提案编号已经增大。

**Basic Paxos when a redundant learner fails**

在以下情况下，（冗余的）学习者之一失败，但是Basic Paxos协议仍然成功。

```txt
Client Proposer         Acceptor     Learner
   |         |          |  |  |       |  |
   X-------->|          |  |  |       |  |  Request
   |         X--------->|->|->|       |  |  Prepare(1)
   |         |<---------X--X--X       |  |  Promise(1,{Va,Vb,Vc})
   |         X--------->|->|->|       |  |  Accept!(1,V)
   |         |<---------X--X--X------>|->|  Accepted(1,V)
   |         |          |  |  |       |  !  !! FAIL !!
   |<---------------------------------X     Response
   |         |          |  |  |       |
```

**Basic Paxos when multiple Proposers conflict**

潜在问题：多个proposer竞争地提出各自提案，比如一个proposer，假设叫 Mike，提出提案的时候，正在处理第二个阶段却被另一个叫 Tom 的 proposer，打断（即这个提案失效了），因为 Tom 提出更大提案编号的提案，然后被打断的 Mike 重新提出提案，这时刚好也打断了 Tom 提出的提案，而这时候 Tom 的提案刚好也进行到了第二阶段，然后循环反复。这个现象称为：活锁（liveness）或 dueling（竞争）

```txt
Client  Proposer       Acceptor     Learner
   |      |             |  |  |       |  |
   X----->|             |  |  |       |  |  Request
   |      X------------>|->|->|       |  |  Prepare(1)
   |      |<------------X--X--X       |  |  Promise(1,{null,null,null})
   |      !             |  |  |       |  |  !! LEADER FAILS
   |         |          |  |  |       |  |  !! NEW LEADER (knows last number was 1)
   |         X--------->|->|->|       |  |  Prepare(2)
   |         |<---------X--X--X       |  |  Promise(2,{null,null,null})
   |      |  |          |  |  |       |  |  !! OLD LEADER recovers
   |      |  |          |  |  |       |  |  !! OLD LEADER tries 2, denied
   |      X------------>|->|->|       |  |  Prepare(2)
   |      |<------------X--X--X       |  |  Nack(2)
   |      |  |          |  |  |       |  |  !! OLD LEADER tries 3
   |      X------------>|->|->|       |  |  Prepare(3)
   |      |<------------X--X--X       |  |  Promise(3,{null,null,null})
   |      |  |          |  |  |       |  |  !! NEW LEADER proposes, denied
   |      |  X--------->|->|->|       |  |  Accept!(2,Va)
   |      |  |<---------X--X--X       |  |  Nack(3)
   |      |  |          |  |  |       |  |  !! NEW LEADER tries 4
   |      |  X--------->|->|->|       |  |  Prepare(4)
   |      |  |<---------X--X--X       |  |  Promise(4,{null,null,null})
   |      |  |          |  |  |       |  |  !! OLD LEADER proposes, denied
   |      X------------>|->|->|       |  |  Accept!(3,Vb)
   |      |<------------X--X--X       |  |  Nack(4)
   |      |  |          |  |  |       |  |  ... and so on ...
```

### Multi Paxos

Basic Paxos 除了活锁（liveness），还有2轮RPC效率低下且难以实现的问题。

#### <font color="#3399cc">Leader</font>

这是新的概念，是“**唯一**”的proposer，所有请求都要经过此 leader。

**Multi-Paxos without failures**

```txt
Client   Proposer      Acceptor     Learner
   |         |          |  |  |       |  | --- First Request ---
   X-------->|          |  |  |       |  |  Request
   |         X--------->|->|->|       |  |  Prepare(N)
   |         |<---------X--X--X       |  |  Promise(N,I,{Va,Vb,Vc})
   |         X--------->|->|->|       |  |  Accept!(N,I,V)   where V = last of (Va, Vb, Vc)
   |         |<---------X--X--X------>|->|  Accepted(N,I,V)
   |<---------------------------------X--X  Response
   |         |          |  |  |       |  |
```

- N 表示竞选出来的第 N 任 leader，
- I 表示第 I 个提案

在这种情况下，新的提案过来，由于使用相同的且唯一的 leader，因此Basic Paxos中包含“Prepare”和“Promise”子阶段的阶段一都可以被跳过。

```txt
Client   Proposer       Acceptor     Learner
   |         |          |  |  |       |  |  --- Following Requests ---
   X-------->|          |  |  |       |  |  Request
   |         X--------->|->|->|       |  |  Accept!(N,I+1,W)
   |         |<---------X--X--X------>|->|  Accepted(N,I+1,W)
   |<---------------------------------X--X  Response
   |         |          |  |  |       |  |
```

其实Basic Paxos中 prepare 和 promise 阶段可以认为是多个proposer在申请相应编号的提案权，所以会出现活锁（liveness），而在 Multi-Paxos 中由于只有唯一的被竞选出来的leader有提案权，所以就可以省去了阶段一。

**角色精简的 Multi-Paxos **

Multi-Paxos的常见部署包括将 proposer，acceptor 和  learner 的角色精简为为“server”。因此，最后只有“client”和“server”。

```txt
Client      Servers
   |         |  |  | --- First Request ---
   X-------->|  |  |  Request
   |         X->|->|  Prepare(N)
   |         |<-X--X  Promise(N, I, {Va, Vb})
   |         X->|->|  Accept!(N, I, Vn)
   |         X<>X<>X  Accepted(N, I)
   |<--------X  |  |  Response
   |         |  |  |
```

**角色精简且 leader 稳定的 Multi-Paxos**

因此后面来的提案，就可以简化流程了。

```txt
Client      Servers
   X-------->|  |  |  Request
   |         X->|->|  Accept!(N,I+1,W)
   |         X<>X<>X  Accepted(N,I+1)
   |<--------X  |  |  Response
   |         |  |  |
```

这时候，可以明显看出，只要 leader 稳定，没有经常竞选 leader，那么服务器之间的请求（RPC：远程过程调用）减少了，相应效率也提高了。

### Fast Paxos

### Raft

#### <font color="#3399cc">3个子问题</font>

1. Leader election
2. Log Replication
3. Safety

#### <font color="#3399cc">重新定义角色</font>

任意一个节点可以在不同时期扮演一下三个角色中的一个，因此这里的角色理解可以为状态（state）：

1. Leader
2. Follower
3. Candidate

#### <font color="#3399cc">原理的动画解释</font>

初始时，集群中所有节点，都是 follower 状态。

如果 follower 没有收到 leader 的来信，那么他们可以成为 candidate，怎么成为candidate后文会说明。

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/a-brief-introduction-of-Paxos-Raft-ZAB/1573898788730.png" style="zoom:68%;" />

**说明：**

1. term：表示任期，表示节点处在第几任的leader管辖下）
2. Vote Count : 投票计数。 

然后，candidate 从其他节点请求投票，其他节点会用投票进行回复。

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/a-brief-introduction-of-Paxos-Raft-ZAB/LeaderElection.gif" alt="1" style="zoom:68%;" />

如果 candidate 从多数（majority）节点中获得选票，它将成为 leader。这个过程称为**领导人选举（Leader election）**。下面是这个过程的细节：

**在Raft中，有两个超时设置可控制选举**。首先是<u>选举超时（election timeout）</u>。选举超时是指 follower 成为 candidate 之前所等待的时间。选举超时被随机分配在150毫秒至300毫秒之间。选举超时后，follower 将成为 candidate，开始新的选举任期（term），对其进行投票（ballot），然后将“请求投票”消息发送给其他节点。如果接收节点在此期限内尚未投票，则它将为 candidate 投票，节点将重置其选举超时。一旦 candidate  获得多数票（majority），便成为 leader。leader 开始向其 follower 发送“添加条目”消息。这些消息将按<u>心跳超时（heartbeat timeout）</u>指定的时间间隔发送，然后 follower 响应每个追加条目消息。此选举任期将持续到 follower 停止接收心跳并成为 candidate 为止。

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/a-brief-introduction-of-Paxos-Raft-ZAB/LeaderElectionDetails.gif" alt="1" style="zoom:68%;" />

让我们停止 leader 并观察再次选举，节点B现在是第2届的 leader。需要多数表决才能保证每个任期只能选举一名leader。如果两个节点同时成为 candidate，则“可能”会发生投票表决的分裂。

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/a-brief-introduction-of-Paxos-Raft-ZAB/LeaderRe-election.gif" alt="1" style="zoom:68%;"/>

让我们看一个**投票表决的分裂**的例子。两个节点都开始以相同的任期进行选举，并且每个节点都已经选举超时，并且先获得一个follower。现在，每个 candidate 都有2票，并且在这个任期中将无法获得更多选票。**节点将等待新的选举，然后重试。**节点A在第5届中获得了多数选票，因此成为 leader。

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/a-brief-introduction-of-Paxos-Raft-ZAB/LeaderElectionSplit.gif" alt="1" style="zoom:68%;"/>

系统的所有更改现在都通过领导者。每次更改都将添加为节点日志中的条目。该日志条目当前未提交（uncommitted），因此不会更新节点的值。

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/a-brief-introduction-of-Paxos-Raft-ZAB/LogReplication-uncommited.gif" alt="1" style="zoom:68%;"/>

要提交条目，节点首先将其复制到 follower 节点上， 然后 leader 等待，直到大多数（majority）节点都写入了条目，在这期间leader不断发送给follower心跳包，一方面确定集群中各节点是否存活，另一方面也可以知道follower是否写入了条目；现在，该条目已提交到 leader 节点上，并且节点状态为“ 5”；leader 节点然后通知 follower 节点该条目已提交（commited）；现在，集群已就系统状态达成共识（consensus），然后最后再响应给客户端表示成功。

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/a-brief-introduction-of-Paxos-Raft-ZAB/LogReplication.gif" alt="1" style="zoom:68%;" />

此过程称为**日志复制（Log Replication）**。

**面对网络分区**，日志甚至可以保持一致。让我们添加一个分区以将A＆B与C，D＆E分开。由于有了这个分区，我们现在拥有两个术语不同的 leader。让我们添加另一个客户端，并尝试更新两个 leader。一个客户端将尝试将节点B的值设置为“ 3”。节点B无法复制为多数，因此其日志条目保持未提交状态。另一个客户端将尝试将节点E的值设置为“ 8”。这将成功，因为它可以复制到大多数。现在，让我们修复网络分区。节点B将看到更高的任期（term）并退出。节点A和B都将回滚其未提交的条目并匹配新 leader 的日志。现在，我们的日志在整个集群中是一致的。

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/a-brief-introduction-of-Paxos-Raft-ZAB/NetworkPartitionRecovery.gif" alt="" style="zoom:68%;" />

这里验证上文 CAP定理中的保证了强一致性和分区容忍，但是分区时，底下两个节点形成的集群(即一个分区)不是多数派(quorum)不具备可用性，但是整个集群依然可用。但是因为 “SET 3” 被删除了，没有写到集群中，所以并没有完全正确虽然保证了一致性。此处跟前文我注明的”译者附""一致：网络分区下，只能选择C A中的一者，这里实验选择了一致性，牺牲了B节点作为另外一个leader的可用性，即降低了可用性：所有客户端只能通过节点成功完成服务。

#### <font color="#3399cc">场景测试</font>

由以上可知保证一致性并不能代表完全正确。接下来又一个例子将会说明，并到 https://raft.github.io/ 网站进行验证。

假设集群共有5个节点，Client 写请求，leader向follower同步日志，此时急集群中有3个节点失败，2个结点存活，对于客户端得到结果有3种情况。

1. unknown（Timeout）

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/a-brief-introduction-of-Paxos-Raft-ZAB/ClientUnkown.gif" alt="" style="zoom:68%;" />

2. （超时后）成功

    client 发送给 S5 的请求刚开始，由于只有2个节点存活，因此S4，S5的第二条日志只是虚线（表示未提交），client 此时并不知道，是否请求成功。随后其他节点相继恢复服务，同步了S5的日志，最终client第二条日志写入成功。

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/a-brief-introduction-of-Paxos-Raft-ZAB/ClientSuccess.gif" alt="" style="zoom:68%;" />

3. （超时后）失败

    client 发送给 S4 的请求刚开始，由于只有2个节点存活，因此S4，S5的索引为2的日志（即第二条内容为2的日志）只是虚线（表示未提交），client 此时并不知道，是否请求成功。此时如果S4，S5停止服务，但是S1，S2，S3恢复服务并且S3为leader，有client向S3发送了另一条请求，而后S4，S5恢复服务，那么此时将抹掉S4，S5的第二条日志。索引为2的日志内容统一为：4，即依然保证集群一致性。

    <img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/a-brief-introduction-of-Paxos-Raft-ZAB/ClientFailue.gif" alt="" style="zoom:68%;" />

对于client请求响应是unkown（Timeout），客户端可以跟集群配合来增强可用性，比如：重试机制，但同时带来了副作用——可能重复写入（在超时后成功却重试了）。

### ZAB

基本上与raft一样。不同点在于名词叫法上不同：ZAB将某一个leader的周期称为 epoch 而不是 term，实现上的不同：raft日志是连续的，心跳方向为leader->follower，而ZAB相反。

## 相关项目实现

1. Zookeeper（ZAB的实现）

2. etcd（raft的实现）

## 巨人的肩膀

1. <u><a style="color:#0879e3" href="https://en.wikipedia.org/wiki/CAP_theorem">wiki CAP theorem</a></u>
2. <u><a style="color:#0879e3" href="https://mwhittaker.github.io/blog/an_illustrated_proof_of_the_cap_theorem/">an_illustrated_proof_of_the_cap_theorem</a></u>
3. <u><a style="color:#0879e3" href="https://teddyma.gitbooks.io/learncassandra/content/about/the_cap_theorem.html">The CAP Theorem | Learn Cassandra</a></u>
4.  <u><a style="color:#0879e3" href="http://lpd.epfl.ch/sgilbert/pubs/BrewersConjecture-SigAct.pdf">Gilbert and Lynch's specification and proof of the CAP Theorem</a></u>
5. <u><a style="color:#0879e3" href="https://thesecretlivesofdata.com/raft/">Raft : Understandable Distributed Consensus</a></u>
6. <u><a style="color:#0879e3" href="https://raft.github.io/">https://raft.github.io/</a></u>

7. 
