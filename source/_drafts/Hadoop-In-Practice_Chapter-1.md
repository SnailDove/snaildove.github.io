---
title: 翻译 Hadoop In Practice, 2nd, Chapter1-Hadoop in a heartbeat
date: 2019-07-01
copyright: true
categories: 中文, english
tags: [Improving Deep Neural Networks, deep learning]
mathjax: true
mathjax2: true
toc: true
---
**注：《Hadoop硬实战》 第二版 无中文版**

![](https://img1.doubanio.com/view/subject/l/public/s27755759.jpg)

# <p align="right"><font color="#9a161a">Chapter 1 Hadoop in a heartbeat</font></right>

<font color="#9a161a">***This chapter covers 本章所涉及内容***</font>

- Examining how the core Hadoop system works

    查看核心Hadoop系统如何工作

- Understanding the Hadoop ecosystem

    理解Hadoop生态系统

- Running a MapReduce job 

    运行一个MapReduce作业

We live in the age of big data, where the data volumes we need to work with on a day-to-day basis have  <font color="#32cd32">outgrown</font>  the storage and <font color="#32cd32">processing capabilities</font>  of a single host. Big data <font color="#32cd32">brings with</font>  it two fundamental <font color="#32cd32">challenges</font> : how to store and work with voluminous data sizes, and more important, how to understand data and turn it into <font color="#32cd32">a competitive advantage</font>.

我们生活在大数据时代，我们需要在日常工作中处理的数据量超过了单个主机的存储和处理能力。**大数据带来了两个基本挑战：如何存储和处理大量数据**，更重要的是，如何理解数据并将其转化为竞争优势。

Hadoop <font color="Green">fills a gap in the market</font>  by effectively storing and providing computational capabilities for substantial amounts of data. It’s a distributed system made up of a distributed filesystem, and it offers a way to parallelize and execute programs on a cluster of machines (see figure 1.1). You’ve most likely come across Hadoop because it’s been adopted by technology giants like Yahoo!, Facebook, and Twitter to  <font color="#32cd32">address</font> their big data <font color="#32cd32">needs</font>, and it’s <font color="#32cd32">making inroads</font> across all <font color="#32cd32">industrial sectors</font>. Because you’ve come to this book to get some practical experience with Hadoop and Java [^1], I’ll start with a brief overview and then show you how to install Hadoop and run a MapReduce job. By the end of this chapter, you’ll have <font color="#32cd32">had a basic refresher</font> on <font color="#32cd32">the nuts and bolts of</font> Hadoop, which will allow you to move on to the more challenging aspects of working with it.

Hadoop通过有效存储和提供大量数据的计算功能填补了市场空白。它是一个由分布式文件系统组成的分布式系统，它提供了一种在一组机器上并行化和执行程序的方法（见图1.1）。你最有可能“遇到”Hadoop，因为雅虎，Facebook和Twitter等技术巨头已经采用它来满足他们的大数据需求，并且正在进军所有工业领域。因为您已经阅读本书以获得Hadoop和Java[^1-]的一些实践经验，所以我将从简要概述开始，然后向您展示如何安装Hadoop并运行MapReduce作业。到本章结束时，您将对Hadoop的基本内容进行基本的复习，这将使您能够进入使用它的更具挑战性的方面。

![1563191555818](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563191555818.png)

Let’s get started with a detailed overview. 

## <font  color="#9a1647">1.1 What is Hadoop?</font>

Hadoop is a platform that provides both distributed storage and computational capabilities. Hadoop <font color="#32cd32">was first conceived to fix</font> a scalability <font color="#32cd32">issue</font> that existed in Nutch [^2],  an open source <font color="32cd32">crawler</font> and search engine. At the time, Google had published papers that described its novel distributed filesystem, the Google File System (GFS), and MapReduce, a computational framework for parallel processing. The successful implementation of these papers’ concepts in Nutch resulted in it being split into two separate projects, the second of which became Hadoop, a first-class Apache project. 

**Hadoop是一个提供分布式存储和分布式计算功能的平台**。 Hadoop最初是为了解决Nutch [^2-]中存在的可伸缩性问题，Nutch 是一个开源搜寻器和搜索引擎。当时，谷歌发表的论文描述了其新颖的分布式文件系统，谷歌文件系统（GFS）和MapReduce，这是一个并行处理的计算框架。在Nutch中成功实现这些论文的概念导致它被分成两个独立的项目，第二个项目成为Hadoop，一个一流的Apache项目。

In this section we’ll look at Hadoop <font color="#32cd32">from an architectural perspective</font>, examine how industry uses it, and consider some of its  <font color="#32cd32">weaknesses</font>. Once we’ve covered this background, we’ll look at how to install Hadoop and run a MapReduce job. 

**在本节中，我们将从架构的角度来看待Hadoop**，研究行业如何使用它，并考虑它的一些弱点。一旦我们了解了这个背景，我们将了解如何安装Hadoop并运行MapReduce作业。

Hadoop proper, as shown in figure 1.2, is a distributed master-slave architecture [^3] that consists of the following <font color="#32cd32">primary components</font>:

如图1.2所示，**Hadoop是一个分布式主从架构**[^3-]，由以下主要组件组成：

![1563191179664](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563191179664.png)

- Hadoop Distributed File System (HDFS) for data storage.

    用于数据存储的Hadoop分布式文件系统（HDFS）。

- Yet Another Resource Negotiator (YARN), introduced in Hadoop 2, a <font color="32cd32">general-purpose</font> scheduler and resource manager. Any YARN application can run on a Hadoop cluster.

    另一个资源协调器（YARN），在Hadoop 2中引入，是一个通用的调度器和资源管理器。任何YARN应用程序都可以在一个Hadoop集群上运行。

- MapReduce, a <font color="32cd32">batch-based</font> computational engine. In Hadoop 2, MapReduce is implemented as a YARN application. 

    MapReduce，一种基于批处理的计算引擎。**在Hadoop 2中，MapReduce是作为YARN应用程序实现**。

<font color="32cd32">Traits intrinsic to</font> Hadoop are data partitioning and parallel computation of large datasets. Its storage and computational capabilities scale  <font color="32cd32">with the addition of</font> hosts to a Hadoop cluster; clusters with hundreds of hosts can easily reach data volumes in the <font color="32cd32">petabytes</font>.

**Hadoop固有的特征是数据分区和大量数据集的并行计算**。它的存储和计算能力随着集群主机数量的增加增强； 拥有数百台主机的集群可以轻松达到PB级别的数据量。

In the first step in this section, we’ll examine the HDFS, YARN, and MapReduce  architectures. 

在本节的第一步中，我们将检查HDFS，YARN和MapReduce架构。

### <font color="#9a1647">1.1.1 Core Hadoop components</font> 

To understand Hadoop’s architecture we’ll start by looking at the basics of HDFS.
<font color="#9a1647">**HDFS**</font>
HDFS is the storage component of Hadoop. It’s a distributed filesystem that’s modeled after the Google File System (GFS) paper[^4]. HDFS is optimized for <font color="#32cd32"> high throughput</font> and works best when reading and writing large files (<font color="#32cd32">gigabytes</font> and larger). To support this throughput, HDFS uses unusually large (for a filesystem) block sizes and data <font color="#32cd32">locality</font> optimizations to reduce network input/output (I/O). <font color="#32cd32">Scalability and availability</font> are also <font color="#32cd32">key traits of</font> HDFS, achieved in part due to <font color="#32cd32">data replication and fault tolerance</font>. HDFS replicates files for a configured number of times, is tolerant of both software and hardware failure, and automatically re-replicates data blocks on nodes that have failed.

要了解Hadoop的架构，我们首先要看一下HDFS的基础知识。

<font color="#9a1647">**HDFS**</font>

**HDFS是Hadoop的存储组件**。它是一个以Google文件系统（GFS）论文[^4-]为模型的分布式文件系统。  HDFS针对高吞吐量进行了优化，在读取和写入大文件（千兆字节或更大）时效果最佳。为了支持这种吞吐量，HDFS使用异常大的（对于文件系统）块大小和**数据局部性**优化来减少网络输入/输出（I / O）。**可扩展性和可用性**也是HDFS的关键特性，这些实现一定程度上归因于**数据复制和容错**。 HDFS将文件按照配置的份数来复制，能够对软件和硬件故障进行容错，并自动在出现故障的节点上重新复制数据块。

![1563200122477](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563200122477.png)

Figure 1.3 shows a <font color="#32cd32">logical representation</font> of the components in HDFS: the NameNode and the DataNode. It also shows an application that’s using the Hadoop filesystem library to access HDFS. Hadoop 2 introduced two significant new features for HDFS—<font color="#32cd32">Federation and High Availability (HA)</font>:

- Federation allows HDFS <font color="#32cd32">metadata</font> to be shared across multiple NameNode hosts, which aides with HDFS scalability and also provides <font color="#32cd32">data isolation</font>, allowing different applications or teams to run their  own NameNodes without fear of <font color="#32cd32">impacting </font> other NameNodes on the same cluster.

- High Availability in HDFS removes the single point of failure that existed in Hadoop 1, wherein a NameNode disaster would result in a <font color="#32cd32">cluster outage</font> . HDFS HA also offers the ability for failover (the process by which a standby NameNode takes over work from a failed primary NameNode) to be automated.

图 1.3 显示了 HDFS 中组件的逻辑表示形式: **NameNode** 和 **DataNode**。它还显示了HDFS 的应用程序使用 Hadoop文件系统库访问HDFS 。Hadoop 2 引入了 HDFS 的两个重要新功能 — **联邦(Federation) 和高可用性  (HA: High Availability)**: 

- 联邦 (Federation) 允许在多个 NameNode **主机之间共享 HDFS 元数据**，这增强了HDFS的可扩展性，并提供**数据隔离**，允许不同的应用程序或团队运行自己的 NameNode，而不必担心影响同一群集上的其他 NameNode。
- HDFS 的 HA 移除在hadoop集群中出现故障的单个节点，在这种情况下一个 NameNode 灾难将导致集群停止运行。HDFS 的 HA 所提供的**故障转移**是自动实现的（备用的 NameNode 从发生故障的 primary NameNode  接管工作实现的）。 

Now that you have a bit of HDFS knowledge, it’s time to look at YARN, Hadoop’s scheduler.
<font color="#9a1647">**YARN**</font>
YARN is Hadoop’s distributed resource scheduler. YARN is new to Hadoop version 2 and was created to address challenges with the Hadoop 1 architecture:

- Deployments larger than 4,000 nodes encountered scalability issues, and adding additional nodes didn’t yield the expected linear scalability improvements.
- Only MapReduce workloads were supported, which meant it wasn’t suited to run execution models such as machine learning algorithms that often require <font color="#32cd32">iterative computations</font>.

现在,您已经具备了一些 HDFS 知识,是时候查看 YARN,Hadoop 的调度器了。

<font color="#9a1647">**YARN**</font>

**YARN 是 Hadoop 的分布式资源调度器**。YARN 是 Hadoop 版本 2 的新增产品，其创建是为了解决 Hadoop 1 体系结构的挑战：

- 部署超过 4,000 个节点时遇到可伸缩性问题，而添加其他节点不会带来预期的伸缩性方面的线性改善。
- 仅支持 MapReduce 的工作负载，这意味着它不适合运行执行例如机器学习算法这方面的模型，这些模型通常需要迭代计算。

For Hadoop 2 these problems were solved by extracting the scheduling function from MapReduce and reworking it into a generic application scheduler, called YARN. With this change, Hadoop clusters are no longer limited to running MapReduce workloads; YARN enables a new set of workloads to be natively supported on Hadoop, and it allows <font color="#32cd32">alternative processing models, such as graph processing and stream processing</font>, to <font color="#32cd32">coexist with</font> MapReduce. Chapters 2 and 10 cover YARN and how to write YARN applications.

对于 Hadoop 2，这些问题通过从 MapReduce 中提取调度函数并将其重新加工为成**通用应用程序调度器**来解决，这个调度器称为 **YARN** 。通过此更改，Hadoop 群集不再局限于运行 MapReduce 工作负载；YARN 使 Hadoop 上支持一组新的工作负载，并允许 alternative processing models **（替代处理模型，如图形处理和流处理）与 MapReduce 共存**。第 2 章和第 10 章介绍 YARN 以及如何编写 YARN 应用程序。

YARN’s architecture is simple because its primary role is to schedule and manage  resources in a Hadoop cluster. Figure 1.4 shows a logical representation of the core  components in YARN: the ResourceManager and the NodeManager. Also shown are the components specific to YARN applications, namely, the YARN application client,  the ApplicationMaster, and the container.

**YARN** 的体系结构很简单，因为它**的主要角色是在 Hadoop 群集中计划和管理资源**。图 1.4 显示了 **YARN 中核心组件的逻辑表示形式：资源管理器和节点管理器**。还显示了专用于 YARN 应用程序的组件，即 **YARN 应用程序客户端、应用程序主机和容器**。

To fully realize the dream of a generalized distributed platform, Hadoop 2 introduced another change—the ability to allocate containers in various configurations.

为了充分实现通用分布式平台的梦想，Hadoop 2 引入了另一个变化——以各种配置分配容器的能力。

![1563208202848](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563208202848.png)

Hadoop 1 had the notion of “slots,” which were a fixed number of map and reduce processes that were allowed to run on a single node. This was wasteful in terms of cluster utilization and resulted in <font color="#32cd32">underutilized resources</font> during MapReduce operations, and it also <font color="#32cd32">imposed memory limits for</font> map and reduce tasks. With YARN, each container requested by an ApplicationMaster can have disparate memory and CPU traits, and this gives YARN applications <font color="#32cd32">full control over</font> the resources they need to <font color="#32cd32">fulfill their work</font>. 

**Hadoop 1 的概念是 "slots" （插槽），即固定数量的 map（映射）和 reduce（规约）进程运行在单个节点上。就群集利用而言是浪费的，导致在 MapReduce 操作期间出现未充足利用的资源，并且还对映射和任务施加了内存限制。使用 YARN，ApplicationMaster （应用程序管家）请求的每个容器都可以具有不同的内存和 CPU 特性，这使 YARN 应用程序能够完全控制完成其工作所需的资源**。

You’ll work with YARN in more detail in chapters 2 and 10, where you’ll learn how YARN works and how to write a YARN application. Next up is an examination of MapReduce, Hadoop’s computation engine.

您将在第 2 章和第 10 章中更详细地使用 YARN，您将了解 YARN 的工作原理以及如何编写 YARN 应用程序。接下来是考察MapReduce，Hadoop的计算引擎。

<font color="#9a1647">**MAPREDUCE**</font>
MapReduce is a batch-based, distributed computing framework modeled after Google’s paper on MapReduce[^5]. It allows you to parallelize work over a large amount of raw data, such as combining web logs with relational data from an OLTP database to model how users interact with your website. This type of work, which could take days or longer using conventional serial programming techniques, can be reduced to minutes using MapReduce on a Hadoop cluster. 

MapReduce 是一个基于批处理的分布式计算框架，是以基于MapReduce[^5-]的Google论文为原型。它允许您并行处理大量原始数据的工作，例如将 Web 日志与 OLTP 数据库中的关系数据相结合，这可以对用户与网站之间的互动进行建模。使用传统的串行编程技术，这种类型的工作可能需要数天或更长时间，使用 Hadoop 群集上的 MapReduce 可将工作缩短到几分钟。

The MapReduce model simplifies parallel processing by abstracting away the complexities <font color="#32cd32">involved in</font> working with distributed systems, such as <font color="#32cd32">computational parallelization, work distribution</font>, and dealing with unreliable hardware and software. With this abstraction, MapReduce allows the programmer to focus on <font color="#32cd32">addressing business needs</font> rather than <font color="#32cd32">getting tangled up in</font> distributed system complications. 

MapReduce 模型通过抽象出处理分布式系统所涉及的复杂性（如计算并行化、工作分配和处理不可靠的硬件和软件）来简化并行处理。通过这样的抽象工作，MapReduce 允许程序员专注于解决业务需求,而不是卷入分布式系统的复杂性。

MapReduce decomposes work submitted by a client into small parallelized map and reduce tasks, as shown in figure 1.5. The map and reduce constructs used in

MapReduce 将客户端提交的工作分解为小规模并行的 map（映射）和 reduce（规约） 任务，如图 1.5 所示。

![1563211136573](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563211136573.png)



**MapReduce are borrowed from those found in the Lisp functional programming language, and they use a shared-nothing model to remove any parallel execution interdependencies that could add unwanted synchronization points or state sharing**[^6].

MapReduce 是从 Lisp 函数式编程语言中借鉴的，它们使用无共享模型来删除任何并发执行的相互依赖性，这些相互依赖可能添加不需要的同步点或状态的共享[^6-]。

The role of the programmer is to define map and reduce functions where the map function outputs key/value tuples, which are processed by reduce functions to produce the final output. Figure 1.6 shows a <font color="#32cd32">pseudocode</font> definition of a map function with regard to its input and output.

程序员的作用是定义 map（映射）和 reduce（规约）函数，map 函数输出（即映射函数的返回值） key/value tuples （键/值的元组），这些元组被 reduce 函数处理后（即规约函数将映射函数的返回值当做自己函数的参数值来输入）生成最终输出（即函数返回值）。图 1.6 显示了 map 函数的输入（即函数参数）和输出（即函数返回值）的伪代码定义。

![1563212447011](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563212447011.png)

The power of MapReduce occurs between the map output and the reduce input in **the shuffle and sort phases**, as shown in figure 1.7.

MapReduce 的能量出现在 map 的输出 （即映射函数的返回值）和 reduce 的输入（规约函数的参数值）之间的随机和排序阶段中，如图 1.7 所示。

![1563214253628](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563214253628.png)

Figure 1.8 shows a pseudocode definition of a reduce function. 

图 1.8 显示了 reduce 规约函数的伪代码定义。

![1563211934312](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563211934312.png)

With the advent of YARN in Hadoop 2, MapReduce has been rewritten as a YARN application and is now referred to as MapReduce 2 (or MRv2). <font color="#32cd32">From</font> a developer’s <font color="#32cd32">perspective</font>, MapReduce in Hadoop 2 works in much the same way it did in Hadoop 1, and code written for Hadoop 1 will execute without code changes on version 2[^7]. There are changes to the physical architecture and internal <font color="#32cd32">plumbing</font> in MRv2 that are examined in more detail in chapter 2.

随着 YARN 在 Hadoop 2 中的出现，MapReduce 已被重写为 YARN 应用程序，现在称为 MapReduce 2(或 MRv2)。从开发人员的角度来看，Hadoop 2 中的 MapReduce 的工作方式与 Hadoop 1 中的工作方式大致相同,为 Hadoop 1 编写的代码不用更改能在 Hadoop 2[^7-]上执行。MRv2 中的物理体系结构和内部管道有一些变化，第 2 章将对此进行更详细的考察。

With some Hadoop basics under your belt, it’s time to take a look at the Hadoop ecosystem and the projects that are covered in this book. 

有了一些 Hadoop 基础知识,是时候看看 Hadoop 生态系统和本书中介绍的项目了。

### <font color="#9a1647">1.1.2 The Hadoop ecosystem</font>

The Hadoop ecosystem is diverse and <font color="#32cd32">grows by the day</font>. It’s impossible to keep track of all of the various projects that interact with Hadoop in some form. In this book the focus is on the tools that are currently receiving the greatest adoption by users, as shown in figure 1.9. 

Hadoop 生态系统是多样化的，并且一天比一天增长。不可能跟踪以某种形式与 Hadoop 交互的所有各种项目。在本书中，重点是目前用户接受最多的工具，如图 1.9 所示。

![1563252992685](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563252992685.png)

MapReduce and YARN are not for the faint of heart, which means the goal for many of these Hadoop-related projects is to increase the accessibility of Hadoop to programmers and nonprogrammers. I’ll cover many of the technologies listed in figure 1.9 in this book and describe them in detail within their respective chapters. In addition, the appendix includes descriptions and installation instructions for technologies that are covered in this book. 

MapReduce 和 YARN 并非针对胆小鬼（意译：没有编程经验的人），这意味着许多与 Hadoop 相关的项目的目标是增加程序员和非程序员对 Hadoop 的可访问性。我将介绍本书图 1.9 中列出的许多技术，并在各自的章节中详细介绍这些技术。此外，附录还包括本书中介绍的技术的说明和安装说明。

> <font color="#4d4d4d">**Coverage of the Hadoop ecosystem in this book**</font> The Hadoop ecosystem <font color="green">grows by the day</font>, and there are often multiple tools with <font color="#32cd32">overlapping features</font> and benefits. The goal of this book is to provide practical techniques that cover the core Hadoop technologies, as well as select ecosystem technologies that are <font color="#32cd32">ubiquitous</font> and essential to Hadoop.

> <font color="#4d4d4d">**本书对Hadoop 生态系统知识的涉猎程度**</font> Hadoop 生态系统与日俱增，并且通常有多个具有重叠功能和优势的工具。本书的目标是提供涵盖 Hadoop 核心技术的实用技术，以及选择对 Hadoop 而言无处不在且至关重要的生态系统技术。

Let’s look at the hardware requirements for your cluster.

 让我们看一下群集的硬件要求。

### <font color="#9a1647">1.1.3 Hardware requirements</font>

The term *commodity hardware* is often used to describe Hadoop hardware requirements. It’s true that Hadoop can run on any old servers you can <font color="#32cd32">dig up</font>, but you’ll still want your cluster to perform well, and you don’t want to <font color="#32cd32">swamp</font> your operations department <font color="#32cd32">with</font> diagnosing and fixing hardware issues. Therefore, *commodity* refers to mid-level rack servers with dual sockets, as much error-correcting RAM as is affordable, and <font color="#32cd32">SATA drives</font> optimized for RAID storage. Using RAID on the DataNode filesystems used to store HDFS content is strongly discouraged because HDFS already has replication and error-checking built in; on the NameNode, RAID is strongly recommended for additional security[^8]. 

术语 "commodity hardware（商用的硬件）" 通常用于描述 Hadoop 硬件要求。确实，Hadoop 可以在任何可以挖掘的旧服务器上运行，但您仍希望集群运行良好，并且不希望您的运营部门疲于应付诊断和修复硬件问题 。因此，commodity 涉及到：具有双插槽带有尽可能多的能够纠错的RAM的的中级机架服务器是经济实惠的，以及对 RAID 存储进行了优化的SATA 驱动器。强烈建议不要在用于存储 HDFS 内容的 DataNode 文件系统上使用 RAID，因为 HDFS 已经内置了复制和错误检查功能。在 NameNode 上，强烈建议 RAID 以提供额外的安全性[^8-]。

From a network topology perspective with regard to switches and firewalls, all of the master and slave nodes must be able to open connections to each other. For small clusters, all the hosts would run 1 GB network cards connected to a single, good-quality switch. For larger clusters, look at 10 GB top-of-rack switches that have at least multiple 1 GB uplinks to dual-central switches. Client nodes also need to be able
to talk to all of the master and slave nodes, but if necessary, that access can be from behind a firewall that permits connection establishment only from the client side. 

从交换机和防火墙的网络拓扑角度来看，所有主节点和从节点都必须能够相互打开连接。对于小型集群，所有主机都将运行连接到单个高质量交换机的 1 GB 网卡。对于较大的集群，请查看至少具有多个 1 GB 上行链路到具有双中心的交换机的 10 GB 机架顶部交换机。客户端节点还需要能够与所有主节点和从节点通信，但如有必要，这些访问可以来自防火墙后面，这可以只允许由客户端建立的连接。

After reviewing Hadoop from a software and hardware perspective, you’ve likely developed a good idea of who might benefit from using it. Once you start working with Hadoop, you’ll need to pick a distribution to use, which is the next topic. 

从软件和硬件角度查看 Hadoop 后，您可能已经了解了谁可能从使用它中获益。开始使用 Hadoop 后，您需要选择要使用的分布式，这是下一个主题。

### <font color="#9a1647">1.1.4 Hadoop distributions</font>

Hadoop is an Apache open source project, and regular releases of the software are available for download directly from the Apache project’s website (http://hadoop.apache.org/releases.html#Download). You can either download and install Hadoop from the website or use a quickstart virtual machine from a commercial distribution, which is usually a great starting point if you’re new to Hadoop and want to quickly get it up and running.

Hadoop 是一个 Apache 开源项目，该软件的常规版本可以直接从 Apache 项目的网站 (http://hadoop.apache.org/releases.html#Download) 下载。你可以从网站下载并安装Hadoop，也可以从商业发行版使用快速启动的虚拟机，如果你是Hadoop的新人，并且想快速启动并运行它，这通常是一个很好的起点。

After you’ve <font color="green">whet your appetite</font> with Hadoop and have committed to using it in production, the next question that you’ll need to answer is which distribution to use. You can continue to use the vanilla Hadoop distribution, but you’ll have to build the <font color="green">in-house expertise</font> to manage your clusters. This is not a trivial task and is usually only successful in organizations that are comfortable with having dedicated Hadoop DevOps engineers running and managing their clusters.

你激发了对Hadoop的兴趣并承诺在生产中使用它之后，您需要回答的下一个问题是使用哪个发行版。您可以继续使用 vanilla Hadoop distribution，但您必须建立内部专业知识来管理群集。这不是一项微不足道的任务，通常只有在拥有专注于运行和管理集群的Hadoop DevOps 工程师的组织中取得成功。

Alternatively, you can turn to a commercial distribution of Hadoop, which will give you <font color="green">the added benefits of</font> enterprise administration software, a support team to consult when planning your clusters or to help you out when things go bump in the night, and the possibility of a rapid fix for software issues that you encounter. Of course, none of this comes for free (or for cheap!), but if you’re running mission-critical services on Hadoop and don’t have a dedicated team to support your infrastructure and services, then going with a commercial Hadoop distribution is prudent. 

或者，您可以求助于 Hadoop 的商业分布式（版本），这将为您提供企业管理软件的额外优势、在规划集群时可以咨询的支持团队，或在夜晚发生碰撞时帮助您解决问题，以及可能快速修复您遇到的软件问题。当然，这些都不是免费的（或便宜的!），但是如果你在 Hadoop 上运行关键的服务，并且没有专门的团队来支持您的基础设施和服务，那么使用商业 Hadoop 分布式（版本）是明智谨慎的。

> <font color="#4d4d4d">**Picking the distribution that’s right for you**</font> It’s highly recommended that you engage with the major vendors to gain an understanding of which distribution suits your needs from a feature, support, and cost perspective. Remember that each vendor will highlight their advantages and at the same time expose the disadvantages of their competitors, so talking to two or more vendors will give you a more realistic sense of what the distributions offer. Make sure you download and test the distributions and validate that they integrate and work within your existing software and hardware stacks.

> <font color="#4d4d4d">**选择适合您的发行版**</font> 强烈建议您与主要供应商合作，从功能，支持和成本角度了解哪种发行版符合您的需求。请记住，每个供应商都会突出其优势，同时暴露其竞争对手的劣势，因此与两个或更多供应商交谈将使您更准确地了解分布式提供的内容。确保下载并测试分布式并验证它们是否在现有软件和硬件堆栈中集成和工作

There are a number of distributions to choose from, and in this section I’ll briefly summarize each distribution and highlight some of its advantages.

有许多发行版可供选择，在本节中，我将简要总结每个发行版并强调它的一些优点。

<font color="#9a1647">**APACHE**</font>
Apache is the organization that maintains the core Hadoop code and distribution, and because all the code is open source, you can crack open your favorite IDE and browse the source code to understand how things work under the hood. Historically the challenge with the Apache distributions has been that support is limited to the goodwill of the open source community, and there’s no guarantee that your issue will be investigated and fixed. Having said that, the Hadoop community is a very supportive one, and  responses to problems are usually rapid, even if the actual fixes will likely take longer than you may be able to afford. 

**Apache** 是维护核心 Hadoop 代码和分发的组织，并且由于所有代码都是开源的，因此您可以打开您喜欢的IDE并浏览源代码以了解工作原理。从历史上看，Apache 发行版面临的挑战是支持仅限于开源社区的善意，并且无法保证您的问题将被调查和修复。话虽如此，Hadoop 社区是一个非常支持的社区，对问题的回答通常很快，即使实际的修复可能需要的时间比你能负担的时间还要长。

The Apache Hadoop distribution has become more <font color="#32cd32">compelling</font> now that administration has been simplified with the advent of Apache Ambari, which provides a GUI to help with <font color="#32cd32">provisioning</font> and managing your cluster. As useful as Ambari is, though, it’s worth comparing it against offerings from <font color="#32cd32">the commercial vendors</font>, as the commercial tooling is typically more sophisticated. 

随着 **Apache Ambari** 的出现简化了管理，Apache Hadoop distribution 变得更加引人注目，Apache Ambari提供了一个图形界面来帮助配置和管理集群。然而，与Ambari一样有用的是，将其与商业供应商的产品进行比较是值得的，因为商业工具通常更复杂。

<font color="#9a1647">**CLOUDERA**</font>

Cloudera is the most tenured Hadoop distribution, and it employs a large number of Hadoop (and Hadoop ecosystem) committers. Doug Cutting, who along with Mike Caferella originally created Hadoop, is the chief architect at Cloudera. In aggregate, this means that <font color="#32cd32">bug fixes</font> and <font color="#32cd32">feature requests</font> have a better chance of being addressed in Cloudera compared to Hadoop distributions with fewer committers. 

Cloudera 是任期最多的 Hadoop 分布式版本，它雇佣了大量 Hadoop (和 Hadoop 生态系统) 提交者。Doug Cutting，他与 Hadoop的原始创建者 Mike Caferella 是Cloudera的首席架构师。总体而言，这意味着与提交者较少的 Hadoop 分布式版本相比，Bug 修复和功能需求在 Cloudera 中更有可能得到解决。

Beyond maintaining and supporting Hadoop, Cloudera has been innovating in the Hadoop space by developing projects that <font color="#32cd32">address areas</font> where Hadoop has been weak. A prime example of this is Impala, which offers a SQL-on-Hadoop system, similar to Hive but focusing on <font color="#32cd32">a near-real-time user experience</font>, as opposed to Hive, which has traditionally been a high-latency system. There are numerous other projects that Cloudera has been working on: <font color="#32cd32">highlights</font> include Flume, a log collection and distribution system; Sqoop, for moving relational data in and out of Hadoop; and Cloudera Search, which offers near-real-time <font color="#32cd32">search indexing.</font> 

除了维护和支持 Hadoop 之外，Cloudera 还一直在 Hadoop 领域进行创新，开发项目，解决 Hadoop 一直很薄弱的领域。**Impala** 就是一个典型的例子，它提供了一个与 Hive 类似的  建立在 hadoop 之上的 SQL 系统，但侧重于近乎实时的用户体验，与 Hive相反，后者传统上是一个高延迟系统。Cloudera 一直在进行许多其他项目：亮点包括 Flume、日志收集和分发系统；**Sqoop**，用于在 Hadoop 中移动关系数据；和 **Cloudera Search**，提供近乎实时的搜索索引。

<font color="#9a1647">**HORTONWORKS**</font>

Hortonworks is also made up of a large number of Hadoop committers, and it offers the same advantages as Cloudera in terms of the ability to quickly address problems and feature requests in core Hadoop and its ecosystem projects. 

Hortonworks 也由大量 Hadoop 提交者组成，在快速解决核心 Hadoop 及其生态系统项目中的问题和功能需求的能力方面，它具有与 Cloudera 相同的优势。

From an innovation perspective, Hortonworks has taken a slightly different approach than **Cloudera**. An example is Hive: Cloudera’s approach was to develop a whole new SQL-on-Hadoop system, but Hortonworks has instead looked at innovating inside of Hive to <font color="#32cd32">remove its high-latency shackles</font> and add new capabilities such as support for ACID. Hortonworks is also the main driver behind the next-generation YARN platform, which is a key strategic piece keeping Hadoop relevant. Similarly, Hortonworks has used Apache Ambari for its administration tooling rather than developing an in-house <font color="#32cd32">proprietary</font> administration tool, which is the path taken by the other distributions. Hortonworks’ focus on developing and expanding the Apache ecosystem tooling has a direct benefit to the community, as it makes its tools available to all users without the need for support contracts. 

从创新的角度来看，Hortonworks  采取了与 Cloudera 略有不同的方法。一个例子是 Hive: **Cloudera 的方法是开发一个全新的 SQL-on-Hadoop 系统，但 Hortonworks 转而考虑在 Hive 内部进行创新，以消除其高延迟的束缚，并添加新的功能，如支持 ACID**。Hortonworks 也是下一代 YARN 平台背后的主要驱动力，该平台是保持 Hadoop 相关的关键战略部分。同样，Hortonworks 也使用 Apache Ambari 作为管理工具，而不是开发内部专有管理工具，这是其他发行机构所走的道路。Hortonworks 专注于开发和扩大 Apache 生态系统工具，这对社区有直接的好处，因为它将其工具提供给所有用户，而无需支持合同。

<font color="#9a1647">**MAPR**</font>
MapR has fewer Hadoop committers on its team than the other distributions discussed here, so its ability to fix and <font color="#32cd32">shape Hadoop’s future</font> is potentially more bounded than its <font color="green">peers</font>.

MapR 团队中的 Hadoop 提交者比此处讨论的其他分布式要少，因此其修复和塑造 Hadoop 未来的能力可能比其对同行更有限制。

From an innovation perspective, MapR has taken a decidedly different approach to Hadoop support  <font color="#32cd32">compared to its peers</font>. From the start it decided that HDFS wasn’t an enterprise-ready filesystem, and instead developed its own <font color="#32cd32">proprietary</font> filesystem, which offers compelling features such as POSIX compliance (offering random-write support and atomic operations), High Availability, NFS mounting, data mirroring, and snapshots. Some of these features have been introduced into Hadoop 2, but MapR has offered them from the start, and, as a result, one can expect that these features are robust. 

从创新的角度来看,MapR 对 Hadoop 支持采取了与同行截然不同的方法。从一开始，它就决定 HDFS 不是企业就绪的文件系统，而是开发了自己的专有文件系统，它提供了引人注目的功能，如 遵从POSIX标准 (提供随机写入支持和原子操作)、高可用性、NFS 挂载、数据镜像和快照。其中一些功能已引入 Hadoop 2，但 MapR 从一开始就提供了这些功能,因此,可以预期这些功能是可靠的。

As part of the evaluation criteria, it should be noted that parts of the MapR stack, such as its filesystem and its HBase offering, are closed source and <font color="#32cd32">proprietary</font>. This affects the ability of your engineers to browse, fix, and contribute patches back to the community. <font color="#32cd32">In contrast</font>, most of Cloudera’s and Hortonworks’ stacks are open source, especially Hortonworks’, which is unique <font color="#32cd32">in that</font> the entire stack, including the management platform, is open source.

作为评估标准的一部分，应该注意的是MapR 技术栈的某些部分（如其文件系统和 HBase 产品）是闭源和专有的。这会影响工程师浏览程序、修补程序以及向社区提供补丁的能力。相比之下，Cloudera 和 Hortonworks 的大部分技术栈都是开源的，尤其是 Hortonworks ，这独一无二的，因为整个技术栈（包括管理平台）都是开源的。

MapR's <font color="green">notable highlights</font> include being made available in Amazon’s cloud as an alternative to Amazon’s own Elastic MapReduce and being integrated with Google’s Compute Cloud.

MapR 的显著亮点包括在亚马逊的云中作为亚马逊自己的弹性 MapReduce 的替代方案，并与谷歌的计算云集成。

I’ve just <font color="green">scratched the surface of</font> the advantages that the various Hadoop distributions offer; your next steps will likely be to contact the <font color="green">vendors</font> and start playing with the distributions yourself. 

刚刚浅尝辄止了各种 Hadoop 分布式版本提供的优势；您的后续步骤可能是联系供应商，并开始使用分布式系统。

Next, let’s take a look at companies currently using Hadoop, and in what capacity they’re using it .

 接下来,我们来看看当前使用 Hadoop 的公司，以及他们以何种容量使用它。

### <font color="#9a1647">1.1.5 Who’s using Hadoop?</font>

Hadoop has a high level of <font color="#32cd32">penetration</font> in high-tech companies, and it’s starting to <font color="#32cd32">make inroads in</font> a broad range of sectors, including the enterprise (Booz Allen Hamilton, J.P. Morgan), government (NSA), and health care. Facebook uses Hadoop, Hive, and HBase for data warehousing and real-time application serving[^9].

Hadoop在高科技公司中具有很高的占有率，它开始进军广泛的领域，包括企业（Booz Allen Hamilton，J.P.Morgan）、政府（NSA）和医疗保健。Facebook使用Hadoop、Hive和HBase进行数据仓库和实时应用程序服务。

Facebook’s data warehousing clusters are petabytes in size with thousands of nodes, and they use separate HBase-driven, real-time clusters for messaging and <font color="#32cd32">real-time analytics</font>. 

Facebook的数据仓库集群规模为千兆字节，拥有数千个节点，它们使用独立的 HBase 驱动的实时集群进行消息传递和实时分析。

Yahoo! uses Hadoop for <font color="#32cd32">data analytics</font>, machine learning, <font color="green">search ranking</font>, email antispam, ad optimization, ETL[^10], and more. Combined, it has over 40,000 servers running Hadoop with 170 PB of storage. Yahoo! is also running the first <font color="green">large-scale</font> YARN deployments with clusters of up to 4,000 nodes[^11]. 

雅虎！使用Hadoop进行数据分析、机器学习、搜索排名、反垃圾邮件、广告优化、ETL[^10-]等。它有超过40000台运行 Hadoop 的服务器，存储容量为170 PB。雅虎！也在运行第一个大规模的 YARN 部署，这个集群达到4000个节点[^11-]。

Twitter is a major big data innovator, and it has made notable contributions to Hadoop with projects such as Scalding, a Scala API for Cascading; Summingbird, a component that can be used to implement parts of Nathan Marz’s lambda architecture; and various other gems such as Bijection, Algebird, and Elephant Bird. eBay, Samsung, Rackspace, J.P. Morgan, Groupon, LinkedIn, AOL, Spotify, and StumbleUpon are some other organizations that are also heavily invested in Hadoop. Microsoft has collaborated with Hortonworks to ensure that Hadoop works on its platform.

Twitter是一个主要的大数据创新者，它为Hadoop做出了显著的贡献，其项目包括scalding、用于级联的**scala API**、SummingBird（可用于实现Nathan Marz lambda架构的一部分的组件）以及其他各种各样的精品（组件），如 Bijection、Algebird，还有Elephant Bird。eBay、Samsung、Rackspace、J.P.Morgan、Groupon、LinkedIn、AOL、Spotify和StumbleUpon是其他一些在Hadoop上投入巨大的组织。微软已经与HortonWorks合作，以确保Hadoop在其平台上工作。

Google, in its MapReduce paper, indicated that it uses Caffeine[^12], its version of MapReduce, to create its web index from <font color="#32cd32">crawl data</font>. Google also highlights applications of MapReduce to include activities such as a distributed grep, URL access frequency (from log data), and a term-vector algorithm, which determines popular keywords for a host. 

谷歌在其MapReduce论文中指出，它使用其MapReduce版本的Caffeine[^12-]，从爬行数据中创建其网络索引。Google还强调了MapReduce的应用，包括分布式的grep、URL访问频率（从日志数据中获取）和基于词向量的算法（用于确定主机的常用关键字）。

The number of organizations that use Hadoop <font color="#32cd32">grows by the day</font>, and if you work at a Fortune 500 company you almost certainly use a Hadoop cluster in some capacity. It’s clear that as Hadoop continues to mature, its adoption will continue to grow. 

使用Hadoop的组织数量与日俱增，如果您在财富500强公司工作，您几乎肯定会使用某种容量的Hadoop集群。很明显，随着Hadoop的不断成熟，它的采用将继续增长。

As with all technologies, a key part to being able to work effectively with Hadoop is to understand its shortcomings and design and architect your solutions to mitigate these as much as possible. 

**与所有技术一样，能够有效地与Hadoop合作的关键部分是了解其缺点，设计并构建解决方案，以尽可能减轻这些缺点。**

### <font color="#9a1647">1.1.6 Hadoop limitations</font>

High availability and security often rank among the top concerns cited with Hadoop. Many of these concerns have been addressed in Hadoop 2; let’s take a closer look at some of its <font color="#32cd32">weaknesses</font> as of <font color="#32cd32">release 2.2.0</font>.

高可用性和安全性通常是Hadoop提到的首要问题之一。其中许多问题已经在Hadoop2中得到了解决；让我们仔细看看它在2.2.0版中的一些弱点。

Enterprise organizations using Hadoop 1 and earlier had concerns with the lack of high availability and security. In Hadoop 1, all of the master processes are <font color="#32cd32">single points of failure</font>, which means that a failure in the master process causes <font color="#32cd32">an outage</font>. In Hadoop 2, HDFS now has high availability support, and the re-architecture of MapReduce with YARN has <font color="#32cd32">removed the single point of failure</font>. Security is another area that has had its wrinkles, and it’s <font color="#32cd32">receiving focus</font>.

**使用Hadoop1和更早版本的企业组织担心缺乏高可用性和安全性**。在Hadoop 1中，所有主进程都是单点故障，这意味着主进程中的故障会导致停机。在**Hadoop2中，HDFS现在支持高可用性**，并且用 YARN 重新构建MapReduce的体系结构消除了单点故障。安全是另一个有麻烦（原文：wrinkles 皱纹，此处为意译）的领域，它正受到关注。

<font color="#9a1647">**HIGH AVAILABILITY**</font>

High availability is often <font color="#32cd32">mandated</font> in enterprise organizations that have high uptime SLA requirements to ensure that systems are always on, even in the event of a node going down due to planned or unplanned circumstances. Prior to Hadoop 2, the master HDFS process could only run on a single node, resulting in single points of failure[^13]. Hadoop 2 brings NameNode High Availability (HA) support, which means that multiple NameNodes for the same Hadoop cluster can be running. <font color="#32cd32">With the current design</font>, one of the NameNodes is active and the other NameNode is designated as <font color="#32cd32">a standby process</font>. In the event that the active NameNode experiences a planned or unplanned outage, the standby NameNode will take over as the active NameNode, which is a process called <font color="#32cd32">failover</font>. This failover can <font color="#32cd32">be configured to be automatic</font>, <font color="#32cd32">negating the need for human intervention</font>. The fact that a NameNode failover occurred <font color="#32cd32">is transparent to</font> Hadoop clients.

高可用性通常是在长时间运行且具有高 SLA（Service Level Agreements ）要求的企业组织中强制要求的，以确保系统始终处于运行状态，即使在节点因计划内或计划外情况而停机的情况下也是如此。在Hadoop 2之前，主HDFS进程只能在单个节点上运行，从而导致单点故障[^13-]。 **Hadoop 2 带来了对 Namenode 高可用性（HA）的支持，这意味着同一 Hadoop 集群的多个 Namenode 可以运行。在当前设计中，一个 Namenode 处于激活状态，另一个 Namenode 被指定为备用进程。如果处于激活状态的 Namenode 遇到计划内或计划外停机，备用 Namenode 将接管被激活的 Namenode，这是一个称为故障转移的过程。这种故障转移可以配置为自动的，不需要人工干预。发生 Namenode 故障转移对Hadoop客户端是透明的。**

The MapReduce master process (the JobTracker) doesn’t have HA support in Hadoop 2, but now that each MapReduce job has its own JobTracker process (a separate YARN ApplicationMaster), HA support is arguably <font color="green">less important</font>. HA support in the YARN master process (the ResourceManager) is important, however, and development <font color="green">is currently underway</font> to add this feature to Hadoop[^14]. 

**在 Hadoop2 中，MapReduce 主进程（JobTracker）没有 HA 支持，但是现在每个 MapReduce 作业都有自己的 JobTracker 进程（单独的 YARN 应用程序主进程），HA 支持就不那么重要了。但是，YARN 主进程（ResourceManager）中的 HA 支持非常重要，目前正在开发将此功能添加到 Hadoop[^14-]中**。

<font color="#9a1647">**MULTIPLE DATACENTERS**</font>

Multiple datacenter support is <font color="green">another key feature</font> that’s increasingly expected in enterprise software, as it offers strong data protection and locality properties due to data being replicated across multiple datacenters. Apache Hadoop, and most of its <font color="green">commercial distributions</font>, has never <font color="green">had support for</font> multiple datacenters, which poses challenges for organizations that have software running in multiple datacenters. WAN-disco is currently the only solution available for Hadoop multidatacenter support.

**多数据中心支持**是另一个企业软件中越来越需要的关键功能，因为它**提供了强大的数据保护和位置属性**，因为数据在多个数据中心之间被复制。Apache Hadoop及其大多数商业发行版从未支持多个数据中心，这给在多个数据中心运行软件的组织带来了挑战。WAN-disco 是目前唯一可用于支持Hadoop多数据中心的解决方案。

<font color="#9a1647">**SECURITY**</font>

Hadoop does offer a security model, but by default it’s disabled. With the security model disabled, the only security feature that exists in Hadoop is HDFS file- and directory-level ownership and permissions. But it’s easy for <font color="#32cd32">malicious users</font> to <font color="#32cd32">subvert</font> and <font color="#32cd32">assume other users’ identities</font>. By default, all other Hadoop services are wide open, allowing any user to perform any kind of operation, such as killing another user’s MapReduce jobs.

**Hadoop确实提供了一个安全模型，但是默认情况下它是被禁用的。禁用安全模型后，Hadoop中唯一存在的安全特性是HDFS文件级和目录级的所有权和权限。但是恶意用户很容易破坏和假冒其他用户的身份。默认情况下，所有其他Hadoop服务都是完全开放的，允许任何用户执行任何类型的操作，例如杀死其他用户的MapReduce作业。**

Hadoop can be configured to run with Kerberos, a network authentication protocol, which requires Hadoop <font color="#32cd32">daemons</font> to authenticate clients, both users and other Hadoop components. Kerberos can <font color="#32cd32">be integrated with</font> an organization’s existing Active Directory and therefore offers a <font color="#32cd32">single-sign-on</font> experience for users. Care needs to be taken when enabling Kerberos, as any Hadoop tool that wishes to interact with your cluster will need to support Kerberos.

**Hadoop可以配置成使用 Kerberos（网络身份验证协议）运行，该协议要求Hadoop守护进程对客户机、用户和其他Hadoop组件进行身份验证。Kerberos 可以与组织的现有 Active Directory 集成，因此为用户提供单一登录体验。启用 kerberos 时需要小心，因为任何希望与集群交互的Hadoop工具都需要支持 kerberos。**

Wire-level encryption can be configured in Hadoop 2 and allows data crossing the network (both HDFS transport[^15] and MapReduce shuffle data[^16]) to be encrypted. Encryption of data at rest (data stored by HDFS on disk) is currently missing in Hadoop.

可以在Hadoop2中配置 Wire-level（线级）加密，并允许加密穿过网络的数据（HDFS传输[^15-]和MapReduce Shuffle数据[^16-]）。Hadoop中当前缺少静态数据加密（HDFS在磁盘上存储的数据）。

Let’s examine <font color="#32cd32">the limitations of</font> some of the individual systems. 

让我们来研究一些单独系统的局限性。

<font color="#9a1647">**HDFS**</font>

The weakness of HDFS is mainly its lack of high availability (in Hadoop 1.x and earlier), its inefficient handling of small files[^17], and its lack of transparent compression. HDFS doesn’t support random writes into files (only appends are supported), and it’s generally designed to support <font color="#32cd32">high-throughput sequential reads and writes</font> over large files.

**HDFS的缺点主要是缺乏高可用性（在Hadoop1.x和更早版本中）、处理小文件的效率低下[^17-]、以及缺乏透明的压缩。HDFS不支持随机写入文件（只支持增添操作），它通常设计成支持大文件的高吞吐量顺序读写。**

<font color="#9a1647">**MAPREDUCE**</font>

MapReduce is a batch-based architecture, which means it doesn’t lend itself to use cases that need real-time data access. Tasks that require global synchronization or sharing of mutable data <font color="#32cd32">aren’t a good fit for</font> MapReduce, because it’s <font color="#32cd32">a shared-nothing architecture</font>, which can <font color="#32cd32">pose challenges for</font> some algorithms.

**MapReduce是一种基于批处理的体系结构，这意味着它不适合于需要实时数据访问的使用案例。需要全局同步或共享可变数据的任务不适合MapReduce，因为它是一种无共享架构，这可能会对某些算法构成挑战。**

<font color="#9a1647">**VERSION INCOMPATIBILITIES**</font>

The Hadoop 2 release brought with it some headaches <font color="#32cd32">with regard to</font> MapReduce API runtime compatibility, especially in the `org.hadoop.mapreduce` package. These problems often result in <font color="#32cd32">runtime issues</font> with code that’s compiled against Hadoop 1 (and earlier). The solution is usually to recompile against Hadoop 2, or to consider a technique outlined in chapter 2 that introduces a compatibility library to target both Hadoop versions without the need to recompile code. 

Hadoop2版本带来了一些与MapReduce API运行时**兼容性相关的难题**，特别是在`org.hadoop.mapReduce`包中。这些问题通常会导致针对Hadoop1（及更早版本）编译的代码出现运行时问题。解决方案通常是针对Hadoop2重新编译，或者考虑第2章中概述的一种技术，该技术**引入了一个兼容性库来针对两个Hadoop版本，而不需要重新编译代码**。

Other challenges with Hive and Hadoop also exist, where Hive may need to be recompiled to work with versions of Hadoop other than the one it was built against. Pig has had compatibility issues, too. For example, the Pig 0.8 release didn’t work with Hadoop 0.20.203, and <font color="#32cd32">manual intervention</font> was required to work around this problem. This is one of the advantages of using a Hadoop distribution other than Apache, as these compatibility problems have been fixed. If using the vanilla Apache distributions is desired, <font color="#32cd32">it’s worth taking a look at</font> Bigtop (http://bigtop.apache.org/), an Apache open source automated build and compliance system. It includes all of the major Hadoop ecosystem components and runs a number of integration tests to ensure they all work <font color="#32cd32">in conjunction with</font> each other.

**Hive和Hadoop还存在其他挑战，在这些挑战中，可能需要重新编译Hive才能与Hadoop的版本一起工作，而不是它所针对的版本。Pig 也有兼容性问题。**例如，PIG 0.8版本不能与Hadoop 0.20.203一起工作，需要手动干预来解决这个问题。**这是使用Hadoop发行版而不是Apache的优点之一，因为这些兼容性问题已经得到了解决。如果需要使用普通的Apache发行版，那么值得一看Bigtop（http://bigtop.apache.org/），它是一个Apache开源的自动化构建和遵从（规范）的系统。它包含了所有主要的Hadoop生态系统组件，并运行了许多集成测试，以确保它们都能相互协作。**

After <font color="#32cd32">tackling</font> Hadoop’s architecture and its <font color="#32cd32">weaknesses</font>, you’re probably ready to <font color="#32cd32">roll up your sleeves</font> and get hands-on with Hadoop, so let’s look at running the first example in this book. 

在解决了Hadoop的体系结构及其弱点之后，您可能已经准备好卷起袖子上手Hadoop了，所以让我们来看看（怎么）运行本书中的第一个例子。

## <font color="#9a1647">1.2 Getting your hands dirty with MapReduce</font>
This section shows you how to run a MapReduce job on your host. 

> <font color="#4d4d4d">**Installing Hadoop and building the examples**</font> To run the code example in this section, you’ll need to follow the instructions in the appendix, which explain how to install Hadoop and download and run the examples bundled with this book. 

本节向您展示如何在主机上运行MapReduce作业。

> <font color="#4d4d4d">**安装Hadoop并构建示例**</font> 要运行本节中的代码示例，您需要遵循附录中的说明，其中解释了如何安装Hadoop以及下载和运行与本书捆绑的示例。

Let’s say you want to build <font color="#32cd32">an inverted index</font>. MapReduce would be a good choice for this task because it can create indexes <font color="#32cd32">in parallel</font> (a common MapReduce use case). Your input is a number of text files, and your output is a list of tuples, where each tuple is a word and a list of files that contain the word. Using standard processing techniques, this would require you to find a mechanism to join all the words together. A naive approach would be to perform this join in memory, but you might run out of memory if you have large numbers of unique keys. You could use  <font color="#32cd32">an intermediary datastore</font>, such as a database, but that would be inefficient.

假设您想要构建一个反向索引。对于这个任务来说，MapReduce  是一个很好的选择，因为它可以并行创建索引（一个常见的MapReduce 使用案例）。您的输入是许多文本文件，输出是一个元组列表，其中每个元组是一个单词，以及包含该单词的文件列表。使用标准的处理技术，这将需要您找到一种机制来将所有单词连接在一起。一种幼稚的方法是在内存中执行这个连接，但是如果您有大量的单键，那么可能会耗尽内存。您可以使用中间数据存储，例如数据库，但这样做效率很低。

A better approach would be to tokenize each line and produce <font color="#32cd32">an intermediary file</font> containing a word per line. Each of these intermediary files could then be sorted. The final step would be to open all the sorted intermediary files and call a function for each unique word. This is what MapReduce does, <font color="#32cd32">albeit in a distributed fashion</font>.

更好的方法是对每一行进行分词，并生成一个中间文件，其中每行包含一个单词。然后可以对这些中间文件中的每一个进行排序。最后一步是打开所有经过排序的中间文件，并为每个惟一的单词调用一个函数。这就是MapReduce所做的，尽管是分布式的。

Figure 1.10 walks you through an example of a simple inverted index in MapReduce. Let’s start by defining your mapper. Your reducers need to be able to generate a line for each word in your input, so your map output key should be each word in the input files so that MapReduce can join them all together. The value for each key will be the containing filename, which is your document ID. 

图1.10为您介绍了MapReduce中一个简单的反向索引示例。让我们从定义 mapper 映射器开始。reducer 规约器需要能够为输入中的每个单词生成一行，因此映射输出键应该是输入文件中的每个单词，以便MapReduce可以将它们连接在一起。每个键的值将是包含文件名，即您的文档ID。

![1563298602928](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563298602928.png)

This is the mapper code: 

这是 mapper 映射器的代码：

![1563298706037](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563298706037.png)

The goal of this reducer is to create an output line for each word and a list of the document IDs in which the word appears. The MapReduce framework will take care of calling the reducer once per unique key outputted by the mappers, along with a list of document IDs. All you need to do in the reducer is combine all the document IDs together and output them once in the reducer, as you can see in the following code:

这个reducer（规约器）的目标是为每个单词创建一个输出行，并在其中显示单词的文档ID列表。MapReduce  框架将负责为映射器输出的每个唯一键调用reducer一次，并附带一个文档ID列表。在Reducer中需要做的就是将所有文档ID组合在一起，并在Reducer中输出它们一次，如下面的代码所示：

![1563298912555](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563298912555.png)

The last step is to write the driver code that will set all the necessary properties to configure the MapReduce job to run. You need to let the framework know what classes should be used for the map and reduce functions, and also let it know where the input and output data is located. By default, MapReduce assumes you’re working with text; if you’re working with more complex text structures, or altogether different datastorage technologies, you’ll need to tell MapReduce how it should read and write from these data sources and sinks. The following source shows the full driver code[^18]:

最后一步是编写驱动程序的代码，该代码将设置所有必要的属性，以配置要运行的 MapReduce 作业。您需要让框架知道应该为 map 和 reduce 函数使用哪些类，还需要让它知道输入和输出数据的位置。默认情况下，MapReduce  假定您使用的是文本；如果您使用的是更复杂的文本结构，或者完全不同的数据存储技术，则需要告诉 MapReduce  应该如何从这些数据源和 sinks （接收器）中读写。**以下源代码显示完整的驱动程序代码[^18-]：

![1563299335243](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563299335243.png)

Let’s see how this code works. First, you need to create two simple input files in HDFS:

让我们看看这个代码是如何工作的。首先，您需要在HDFS中创建两个简单的输入文件：

![1563299457350](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563299457350.png)

Next, run the MapReduce code. You’ll use a shell script to run it, supplying the two input files as arguments, along with the job output directory:

接下来，运行 MapReduce  代码。您将使用一个shell脚本来运行它，并将两个输入文件作为参数，以及作业输出目录：

![1563299584559](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563299584559.png)

> <font color="#4d4d4d">**Executing code examples in the book**</font>  The appendix contains instructions for downloading and installing the binaries and code that accompany this book. Most of the examples are launched via the hip script, which is located inside the bin directory. For convenience, it’s recommended that you add the book’s bin directory to your path so that you can copy-paste all the example commands as is. The appendix has instructions on how to set up your environment. 

When your job completes, you can examine HDFS for the job output files and view their contents:

> <font color="#4d4d4d">**在书中执行代码示例**</font>。附录包含下载和安装本书附带的二进制文件和代码的说明。大多数示例都是通过 hip 脚本启动的，该脚本位于 bin 目录中。为了方便起见，建议您将书籍的 bin 目录添加到路径中，以便您可以复制粘贴所有示例命令。附录中有关于如何设置环境的说明。

作业完成后，可以检查 HDFS 中的作业输出文件并查看其内容：

![1563299813709](C:\Users\ruito\Desktop\hadoop-in-action_charpter-1\1563299813709.png)

This completes your <font color="green">whirlwind tour of</font> how to run Hadoop.

这就完成了如何运行Hadoop的旋风之旅。

## <font  color="#9a1647">1.3 Chapter summary</font>

Hadoop is a distributed system designed to process, generate, and store large datasets. Its MapReduce implementation provides you with <font color="#32cd32">a fault-tolerant mechanism</font> for largescale data analysis of <font color="#32cd32">heterogeneous structured and unstructured data sources</font>, and YARN now supports multi tenant disparate applications on the same Hadoop cluster.

Hadoop是一个分布式系统，设计用于处理、生成和存储大型数据集。它的 MapReduce  实现为异构结构化和非结构化数据源的大规模数据分析提供了一种容错机制，而yarn现在支持同一Hadoop集群上的多用户（原文：tenant）的不同应用程序。

In this chapter, we examined Hadoop from both functional and physical architectural standpoints. You also installed Hadoop and ran a MapReduce job. 

在本章中，我们从功能和物理架构的角度来研究 Hadoop。您还安装了 Hadoop 并运行了一个 MapReduce 作业。

The remainder of this book <font color="#32cd32">is dedicated to</font> presenting <font color="#32cd32">real-world</font> techniques for solving common problems you’ll encounter when working with Hadoop. You’ll be introduced to <font color="#32cd32">a broad spectrum of</font> subject areas, starting with YARN, HDFS and MapReduce, and Hive. You’ll also look at data-analysis techniques and explore technologies such as Mahout and Rhipe. 

**本书的其余部分致力于展示现实世界中解决 Hadoop 工作时遇到的常见问题的技术。我们将向您介绍广泛的主题领域，从 yarn、hdfs、mapreduce 和 hive 开始。您还将了解数据分析技术并探索诸如 Mahout 和Rhipe 等技术。**

In chapter 2,  <font color="#32cd32">the first stop</font> on your journey, you’ll discover YARN, which <font color="#32cd32">heralds</font> a new era for Hadoop, one that transforms Hadoop into a distributed processing kernel.  <font color="#32cd32">Without further ado</font>, let’s get started.

在第2章，旅程的第一站，你会发现 YARN ，这预示着Hadoop的新时代，一个将Hadoop转化为分布式处理内核的时代。不用再多费吹灰之力，我们开始吧。

---

[^1]: To benefit from this book, you should have some practical experience with Hadoop and understand the basic concepts of MapReduce and HDFS (covered in Manning’s Hadoop in Action by Chuck Lam, 2010). Further, you should have an intermediate-level knowledge of Java—Effective Java, 2nd Edition by Joshua Bloch (Addison-Wesley, 2008) is an excellent resource on this topic.
[^1-]: 要从本书中受益，您应该具备Hadoop的一些实践经验并理解MapReduce和HDFS的基本概念（由Chuck Lam撰写的Manning的Hadoop in Action，2010）。此外，您应该具有Java的中级知识——《Effective Java, 2nd Edition》 Joshua Bloch（Addison-Wesley，2008）是这个主题的优秀资源。
[^2]: The Nutch project, and by extension Hadoop, was led by Doug Cutting and Mike Cafarella.
[^2-]: Nutch项目，以及Hadoop，由 Doug Cutting 和 Mike Cafarella 领导。
[^3]: A model of communication where one process, called the master, has control over one or more other processes, called slaves.
[^3-]: 通信模型，其中一个进程（称为 master，术语个人翻译：主进程）控制一个或多个其他进程（称为 slave， 术语个人翻译：从进程）。
[^4]: See “The Google File System‚” http://research.google.com/archive/gfs.html.
[^4-]: 查看 [“The Google File System”](http://research.google.com/archive/gfs.html)
[^5]: See “MapReduce: Simplified Data Processing on Large Clusters,” http://research.google.com/archive/mapreduce.html.
[^5-]: 查看 [“MapReduce: Simplified Data Processing on Large Clusters”](http://research.google.com/archive/mapreduce.html)
[^6]: A shared-nothing architecture is a distributed computing concept that represents the notion that each node is independent and self-sufficient.
[^6-]: 无共享体系结构是一个分布式计算的概念，它表示每个节点都是独立的和自给自足的。
[^7]: Some code may require recompilation against Hadoop 2 binaries to work with MRv2; see chapter 2 for more details.
[^7-]: 某些代码可能需要针对 Hadoop 2 二进制文件重新编译才能使用 MRv2;有关详细信息,请参阅第 2 章。
[^8]:  HDFS uses disks to durably store metadata about the filesystem.
[^9]: See Dhruba Borthakur, “Looking at the code behind our three uses of Apache Hadoop” on Facebook at http://mng.bz/4cMc. Facebook has also developed its own SQL-on-Hadoop tool called Presto and is migrating away from Hive (see Martin Traverso, “Presto: Interacting with petabytes of data at Facebook,” http://mng.bz/p0Xz).
[^9-]: 参见Facebook上的 dhruba borthakur，["Looking at the code behind our three uses of Apache Hadoop"](http://mng.bz/4cmc)。Facebook还开发了自己的 SQL on Hadoop 工具Presto，并正在从Hive迁移（参见Martin Traverso，["Presto:Interaction with Petabytes of Data at Facebook"](http://mng.bz/p0xz)）。
[^10]: Extract, transform, and load (ETL) is the process by which data is extracted from outside sources, transformed to fit the project’s needs, and loaded into the target data sink. ETL is a common process in data warehousing.
[^10-]: 提取、转换和加载（ETL）是从外部源中提取数据、转换以满足项目需求并加载到目标数据接收器中的过程。ETL是数据仓库中的一个常见过程。
[^11]: There are more details on YARN and its use at Yahoo! in “Apache Hadoop YARN: Yet Another Resource Negotiator” by Vinod Kumar Vavilapalli et al., www.cs.cmu.edu/~garth/15719/papers/yarn.pdf
[^11-]: 关于 YARN 及其在雅虎的应用，在 Vinod Kumar Vavilapalli 等人写的 [“Apache Hadoop Yarn:yet another resource negotiator”](www.cs.cmu.edu/~garth/15719/papers/yarn.pdf) 中还有更多的细节。
[^12]: In 2010 Google moved to a real-time indexing system called Caffeine; see “Our new search index: Caffeine” on the Google blog (June 8, 2010), http://googleblog.blogspot.com/2010/06/our-new-search-indexcaffeine.html.
[^12-]: 在2010年，谷歌转向了一个名为Caffeine的实时索引系统；参见谷歌在2010年6月8号的博客：["Our new search index: Caffeine”](http://googleblog.blogspot.com/2010/06/our-new-search-indexagfe.html)
[^13]: In reality, the HDFS single point of failure may not be terribly significant; see “NameNode HA” by Suresh Srinivas and Aaron T. Myers, http://goo.gl/1iSab.
[^13-]: 事实上，HDFS单点故障可能不太重要；请参阅Suresh Srinivas和Aaron T.Myers的 ["namenode ha"](http://goo.gl/1isab)
[^14]: For additional details on YARN HA support, see the JIRA ticket titled “ResourceManager (RM) High-Availability (HA),” https://issues.apache.org/jira/browse/YARN-149.
[^14-]: 或有关yarn-ha支持的其他详细信息，请参阅 JIRA 标签，标题为[“ResourceManager（RM）High Availability（HA）”](https://issues.apache.org/jira/browse/yarn-149)
[^15]: See the JIRA ticket titled “Add support for encrypting the DataTransferProtocol” at https://issues.apache.org/jira/browse/HDFS-3637.
[^15-]: 请参阅标题为“添加对加密数据传输协议的支持”的 JIRA 标签，网址为https://issues.apache.org/jira/browse/hdfs-3637。
[^16]: See the JIRA ticket titled “Add support for encrypted shuffle” at https://issues.apache.org/jira/browse/MAPREDUCE-4417.
[^16-]: 请参阅标题为“添加加密随机播放支持”的 JIRA 标签，网址为https://issues.apache.org/jira/browse/mapreduce-4417。
[^17]: Although HDFS Federation in Hadoop 2 has introduced a way for multiple NameNodes to share file metadata, the fact remains that metadata is stored in memory.
[^17-]: 尽管Hadoop2中的HDFS Federation引入了一种多个NameNodes 享文件元数据的方法，但事实仍然是元数据存储在内存中。
[^18]: GitHub source: https://github.com/alexholmes/hiped2/blob/master/src/main/java/hip/ch1/InvertedIndexJob.java.
[^18-]: GitHub 源码: https://github.com/alexholmes/hiped2/blob/master/src/main/java/hip/ch1/InvertedIndexJob.java.
