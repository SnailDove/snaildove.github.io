---
title: 了解Hadoop YARN架构 
mathjax: true
mathjax2: true
categories: 中文
date: 2019-02-01
tags: [Hadoop YARN, Distributed System]
commets: true
toc: true
---

## 前言

本文是对 [官网：Architecture of Apache Hadoop YARN](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/YARN.html) 的翻译，记录本篇的时候，版本为 2.9.2。本文简单过一下，深入了解可以看看列出的参考资料。

## 正文

The fundamental idea of YARN is to split up the functionalities of resource management and job scheduling/monitoring into separate daemons. The idea is to have a global ResourceManager (RM) and per-application ApplicationMaster (AM). An application is either a single job or a DAG of jobs.

YARN 的基本思想是将资源管理和作业调度/监视功能拆分为单独的守护进程。其想法是拥有一个全局资源管理器（ResourceManager: RM）和每个应用程序的应用程序主控器（ApplicationMaster：AM）。应用程序可以是单个作业，也可以是一个DAG作业。

The ResourceManager and the NodeManager form the data-computation framework. The ResourceManager is <font color="#32cd32">the ultimate authority</font> that arbitrates resources among all the applications in the system. The NodeManager is the per-machine framework agent who is responsible for containers, monitoring their resource usage (cpu, memory, disk, network) and reporting the same to the ResourceManager/Scheduler.

ResourceManager 和 NodeManager 构成了数据计算框架。ResourceManager 是在系统中所有应用程序之间仲裁资源的最高权威（机构）。NodeManager 是每台计算机框架的代理，它负责容器、监视其资源使用情况（CPU、内存、磁盘、网络），并向资源管理器（**ResourceManager**）/调度程序（**Scheduler**）报告这些情况。

The per-application ApplicationMaster is, in effect, a framework specific library and is tasked with negotiating resources from the ResourceManager and working with the NodeManager(s) to execute and monitor the tasks.

实际上，每个应用程序的 ApplicationMaster  是一个特定于框架的库，它的任务是与ResourceManager协商资源，并与节点管理器一起执行和监视任务。

![MapReduce NextGen Architecture](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/yarn_architecture.gif)

The ResourceManager has two main components: Scheduler and ApplicationsManager.

ResourceManager有两个主要组件：调度器（**Scheduler** ）和应用程序管理器（**ApplicationsManager**）。

1. The Scheduler is responsible for allocating resources to the various running applications subject to familiar constraints of capacities, queues etc. The Scheduler is pure scheduler in the sense that it performs no monitoring or tracking of status for the application. Also, it offers no guarantees about restarting failed tasks either due to application failure or hardware failures. The Scheduler performs its scheduling function based on the resource requirements of the applications; it does so based on the abstract notion of a resource Container which incorporates elements such as memory, cpu, disk, network etc.

    调度器（**Scheduler**）负责根据熟悉的容量、队列等限制将资源分配给各种正在运行的应用程序。调度器是纯粹的调度程序，在某种意义上，它不执行对应用程序状态的监视或跟踪。此外，它不能保证由于应用程序故障或硬件故障而重新启动失败的任务。调度器根据应用程序的资源需求来执行其调度功能；它是基于资源容器的抽象概念来执行的，该资源容器包含内存、CPU、磁盘、网络等元素。

    The **Scheduler** has <font color="#32cd32">a pluggable policy</font> which is responsible for partitioning the cluster resources among the various queues, applications etc. The current schedulers such as the [CapacityScheduler](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/CapacityScheduler.html) and the [FairScheduler](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/FairScheduler.html) would be some examples of <font color="#32cd32">plug-ins</font>.

    调度器有一个可插拔的策略，负责在不同的队列、应用程序等之间划分集群资源。当前的调度器（如[CapacityScheduler](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/CapacityScheduler.html) 和 [FairScheduler](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/FairScheduler.html)）是插件的一些示例。

2. The ApplicationsManager is responsible for accepting job-submissions, negotiating the first container for executing the application specific ApplicationMaster and provides the service for restarting the ApplicationMaster container on failure. The per-application ApplicationMaster has the responsibility of negotiating appropriate resource containers from the Scheduler, tracking their status and monitoring for progress.

    **ApplicationsManager**负责接受作业提交、协商用于执行某一个具体应用程序的ApplicationMaster的第一个容器，并提供在失败时重新启动ApplicationMaster容器的服务。每个应用程序ApplicationMaster负责与Scheduler协商合适的资源容器，跟踪其状态并监视进度。

MapReduce in hadoop-2.x maintains API compatibility with previous stable release (hadoop-1.x). This means that all MapReduce jobs should still run unchanged on top of YARN with just a recompile.

**hadoop-2.x中的 MapReduce 与以前的稳定版本（hadoop-1.x）保持API兼容性**。这意味着，所有 MapReduce  作业都应该在 YARN 上保持不变，只需重新编译即可。

YARN supports the notion of resource reservation via the [ReservationSystem](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/ReservationSystem.html), a component that allows users to specify a profile of resources over-time and temporal constraints (e.g., deadlines), and reserve resources to ensure the predictable execution of important jobs.The ReservationSystem tracks resources over-time, performs admission control for reservations, and dynamically instruct the underlying scheduler to ensure that the reservation is fullfilled.

YARN 支持通过 **ReservationSystem** 保存资源的概念，[ReservationSystem](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/ReservationSystem.html) 是一个组件，允许用户指定一个临时和时间限制的资源配置文件（例如截止日期），并保存资源以确保重要作业的可预测执行。ReservationSystem 跟踪随着时间的推移，资源将对预订执行许可控制，并动态指示基础调度器以确保预订得到执行。

In order to scale YARN beyond few thousands nodes, YARN supports the notion of Federation via the [YARN Federation](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/Federation.html) feature. Federation allows to transparently wire together multiple yarn (sub-)clusters, and make them appear as a single massive cluster. This can be used to achieve larger scale, and/or to allow multiple independent clusters to be used together for very large jobs, or for tenants who have capacity across all of them.

**为了将YARN扩展到数千个节点之外，YARN 通过 [YARN Federation](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/Federation.html) 特性支持 Federation 的概念。**联邦（Federation）允许透明地将多个YARN（子）集群连接在一起，并使它们看起来像一个单独的大集群。这可以用于实现更大的规模，和/或允许将多个独立的集群一起用于非常大的工作，或用于具有所有能力使用所有集群的租户（tennant: 用户。租户）。

<font color="red">YARN Federation 详细参考的另一篇读记：[Hadoop YARN Federation](https://snaildove.github.io/2019/05/14/HadoopYARNFederation)</font>

## 参考资料

1. Hadoop In Action, 2nd Edition
2. Hadoop: The Definitive Guide, 4th Edition
3. Apache Hadoop YARN-Moving beyond MapReduce and Batch Processing with Apache Hadoop 2
4. [YARN Architecture 笔记小结 - 葛尧的文章 - 知乎](https://zhuanlan.zhihu.com/p/31810137)
5. [YARN Architecture 笔记二 - 葛尧的文章 - 知乎](https://zhuanlan.zhihu.com/p/34884574)
6. [Hadoop技术内幕：深入解析YARN架构设计与实现原理](https://www.amazon.cn/dp/B00IHSW3A2/ref=sxbs_sxwds-stvp?__mk_zh_CN=%E4%BA%9A%E9%A9%AC%E9%80%8A%E7%BD%91%E7%AB%99&keywords=Hadoop%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF&pd_rd_i=B00IHSW3A2&pd_rd_r=ccb0a97e-0268-40b1-bbb4-713bcbc48a40&pd_rd_w=nvgph&pd_rd_wg=QUvST&pf_rd_p=4e9fa468-0f23-4b50-b356-19e116e59ff6&pf_rd_r=70T8YKXF8BHDHP3XPPHE&qid=1563728392&s=gateway)
