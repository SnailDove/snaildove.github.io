---
title: 翻译 Hadoop In Practice, 2nd, Chapter2.1-Introduction to YARN
date: 2019-07-01
copyright: true
categories: 中文, english
tags: [Improving Deep Neural Networks, deep learning]
mathjax: true
mathjax2: true
toc: true
---

**注：《Hadoop硬实战》 第二版 无中文版**

<img src="https://img1.doubanio.com/view/subject/l/public/s27755759.jpg" alt="1122" style="zoom:68%;" />

# <p align="right"><font color="#9a161a">Chapter 2 Introduction to YARN</font></p>

<font color="#9a161a">*This chapter covers 本章所涉及的内容*</font>

- Understanding how YARN works

    理解YARN如何工作

- How MapReduce works as a YARN application

    MapReduce如何作为一个YARN程序

- A look at other YARN applications 

    YARN应用程序一览

Imagine buying your first car, which upon delivery has <font color="#32cd32">a steering wheel</font> that <font color="#32cd32">doesn’t function</font> and <font color="#32cd32">brakes</font> that don’t work. Oh, and it only drives <font color="#32cd32">in first gear</font>. No speeding <font color="#32cd32">on winding back roads</font> for you! That empty, sad feeling is familiar to those of us who want to run some cool new tech such as graph or real-time data processing with Hadoop 1[^1], only to be reminded that our powerful Hadoop clusters were good for one thing, and one thing only: MapReduce. 

想象一下，你买的第一辆车，交付时有一个方向盘不起作用，刹车不起作用。哦，而且它只开一档。不要在弯路上超速！对于那些想用 Hadoop1[^1-]，运行一些很酷的新技术（如图形或实时数据处理）的人来说，这种空虚、悲伤的感觉是很熟悉的，只是要提醒他们，我们强大的 Hadoop 集群只对一件事有好处，而且只有一件事 MapReduce。

Luckily for us the Hadoop committers <font color="#32cd32">took</font> these and other constraints <font color="#32cd32">to heart</font> and <font color="#32cd32">dreamt up a vision</font> that would <font color="#32cd32">metamorphose</font> Hadoop above and beyond MapReduce. YARN is the realization of this dream, and it’s an exciting new development that transitions Hadoop into a distributed computing kernel that can support any type of workload[^2]. This opens up the types of applications that can be run on Hadoop to efficiently support computing models for machine learning, graph processing, and other generalized computing projects (such as Tez), which are discussed later in this chapter.

幸运的是，Hadoop的代码提交者（原文：commiter）把这些和其他的限制放在心上，并梦想着一个能使Hadoop在MapReduce之上和之外彻底改变的愿景。YARN 是这个梦想的实现，它是一个令人兴奋的新发展，它将Hadoop转变成一个可以支持任何类型工作负载的分布式计算内核[^2-]。这打开了可以在Hadoop上运行的应用程序类型（的大门），从而有效地支持机器学习的计算模型、图形处理和其他通用计算项目（如Tez），将在本章后面讨论。

The <font color="#32cd32">upshot</font> of all this is that you can now run MapReduce, Storm, and HBase all on a single Hadoop cluster. This allows for exciting new possibilities, not only in computational multi-tenancy, but also in the ability to efficiently share data between applications. Because YARN is a new technology, we’ll <font color="#32cd32">kick off</font> this chapter with a look at how YARN works, followed by a section that covers how to interact with YARN from the command line and the UI. <font color="#32cd32">Combined</font>, these sections will <font color="#32cd32">give you a good grasp of</font> what YARN is and how to use it.

所有这些的结果是，您<font style="background:yellow;">现在可以在单个 Hadoop 集群上运行 MapReduce， Storm， 和 HBase。这就允许了令人兴奋的新可能性，不仅在计算多用户（原文：tenant，有些人翻译为“多租户”）中，而且在应用程序之间高效共享数据的能力中</font>。因 YARN 是一项新技术，所以我们<font style="background:yellow;">将在本章开始介绍yarn的工作原理，接下来是一节，介绍如何与来自命令行和用户界面的 YARN 进行交互。结合起来，这些部分会让你很好地了解什么是 YARN  以及如何使用它</font>。

Once you <font color="#32cd32">have a good handle on</font> how YARN works, you’ll see how MapReduce has been rewritten to be a YARN application (titled MapReduce 2, or MRv2), and look at some of the architectural and systems changes that occurred in MapReduce to make this happen. This will help you better understand how to work with MapReduce in Hadoop 2 and <font color="#32cd32">give you some background into</font> why some aspects of MapReduce changed in version 2. 

一旦您对 YARN  的工作方式有了很好的了解，您将看到 MapReduce 是如何被重写为 YARN  应用程序的（标题为MapReduce 2或mrv2），并查看 MapReduce 中发生的一些架构和系统更改，以实现这一点。这将帮助您更好地了解如何在 Hadoop 2 中使用 MapReduce，并为您提供一些背景：为什么在版本2中 MapReduce 的某些方面发生了变化。

> <font color="#4d4d4d">**YARN development**</font> If you’re looking for details on how to write YARN applications, feel free to skip to chapter 10. But if you’re new to YARN, I recommend you read this chapter before you move on to chapter 10. In the final section of this chapter, you’ll examine several YARN applications and their practical uses.
>
> <font color="#4d4d4d">**YARN 发展**</font> 如果您正在寻找有关如何编写 YARN 应用程序的详细信息，请随意跳到第10章。但是，如果你是新手，我建议你在进入第10章之前先阅读这一章。在本章的最后一节中，您将研究几种 YARN 应用及其实际用途。

Let’s get things started with an overview of YARN. 

让我们从 YARN 的概述开始。

##  <font  color="#9a1647">2.1 YARN overview</font>

With Hadoop 1 and older versions, you were limited to only running MapReduce jobs. This was great if the type of work you were performing fit well into the MapReduce processing model, but it was <font color="#32cd32">restrictive</font> for those wanting to perform graph processing, iterative computing, or any other type of work.

对于Hadoop1 和旧版本，您只能运行 MapReduce 作业。如果您所执行的工作类型与 MapReduce 处理模型非常匹配，那么这是非常好的，但是对于那些希望执行图形处理、迭代计算或任何其他类型的工作的人来说，这是有限制的。

In Hadoop 2 the scheduling pieces of MapReduce were externalized and reworked into a new component called YARN, which is short for *Yet Another Resource Negotiator*. YARN is agnostic to the type of work you do on Hadoop—all that it requires is that applications that wish to operate on Hadoop are implemented as YARN applications. As a result, MapReduce is now a YARN application. The old and new Hadoop stacks can be seen in figure 2.1.

<font style="background:yellow;">在Hadoop2中，MapReduce 的调度部分被分离到外部，并重新生成一个新的组件，称为YARN ，它是另一个资源协商者（*Yet Another Resource Negotiator*）的缩写</font>。YARN 与您在 Hadoop 上所做的工作类型无关，它只需要将希望在 Hadoop 上操作的应用程序实现为  YARN 应用程序。因此，MapReduce 现在是一种 YARN 应用程序。旧的和新的 Hadoop 技术栈如图2.1所示。

There are multiple benefits to this architectural change, which you’ll examine in the next section. 

这种体系结构更改有多种好处，您将在下一节中对其进行研究。

![1563353879104](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1563353879104.png)

### <font color="#9a1647">2.1.1 Why YARN?</font>

We’ve touched on how YARN enables work other than MapReduce to be performed on Hadoop, but let’s expand on that and also look at additional advantages that YARN brings to the table.

我们已经讨论了 YARN 如何使 Hadoop 上执行 MapReduce 以外的工作，但是让我们进一步讨论一下 YARN  带来的其他优势。

MapReduce is a powerful distributed framework and programming model that allows batch-based parallelized work to be performed on a cluster of multiple nodes. Despite being very efficient at what it does, though, MapReduce has some disadvantages; principally that it’s batch-based, and as a result isn’t suited to real-time or even near-real-time data processing. Historically this has meant that processing models such as graph, iterative, and real-time data processing are not a natural fit for MapReduce[^3].

MapReduce 是一个强大的分布式框架和编程模型，允许在多个节点的集群上执行基于批处理的并行工作。<font style="background:yellow;">尽管MapReduce 的工作效率非常高，但它也有一些缺点，主要是它是基于批处理的，因此不适合于实时甚至接近实时的数据处理。从历史上看，这意味着图形、迭代和实时数据处理等处理模型并不适合MapReduce [^3-]</font>。

<font color="#32cd32">The bottom line</font> is that Hadoop version 1 <font color="#32cd32">restricts</font> you <font color="#32cd32">from</font> running exciting new processing frameworks. YARN changes all of this by taking over <font color="#32cd32">the scheduling portions of</font> MapReduce, and nothing else. At its core, YARN is a distributed scheduler and <font color="#32cd32">is responsible for</font> two activities:

最后结果是 Hadoop 版本1 限制您运行令人兴奋的新处理框架。<font style="background:yellow;">YARN 通过接管 MapReduce 的调度部分而改变了所有这些，而其他什么都没有。核心是 Yarn 是一个分布式调度程序，负责两个活动</font>：

- Responding to a client’s request to create a container—A container is in essence a process, with a contract governing the physical resources that it’s permitted to use.

    <font style="background:yellow;">响应客户创建容器的请求——容器本质上是一个进程，它具有一个经授权去控制物理资源的协议（约定，原文：contract）</font>。

- Monitoring containers that are running, and terminating them if needed—Containers can be terminated if a YARN scheduler wants to <font color="#32cd32">free up resources</font> so that containers from other applications can run, or if a container is using more than its allocated resources. 

    <font style="background:yellow;">监视正在运行的容器，如果需要，可以终止容器，如果 YARN 调度程序希望释放资源以便容器可以从其他应用程序运行，或者如果容器使用的资源超过了其分配的资源，则可以终止容器</font>。

HBase is an exception; it uses HDFS for storage but doesn’t use MapReduce for the processing engine. 

HBase 是一个例外；它使用 HDFS 进行存储，但不使用 MapReduce 作为处理引擎。

Table 2.1 compares MapReduce 1 and YARN (in Hadoop versions 1 and 2) to show why YARN is such a revolutionary jump.

表2.1比较了MapReduce 1 和 YARN（在 Hadoop 版本1和2中），以说明为什么 YARN 是如此革命性的跳跃。

<center><strong><font color="#b38000">表 2.1</font> MapReduce 1 vs. YARN</strong></center>
| Capability      | MapReduce 1                                                  | YARN                                                         |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Execution model<br />执行模型 | Only MapReduce is supported on Hadoop 1, limiting the types of activities you can perform to batch-based flows that fit within the confines of the MapReduce processing model.<br />Hadoop 1仅支持MapReduce，将您可以执行的活动类型限制为适合MapReduce处理模型范围的基于批处理的流。 | YARN <font color="#32cd32">places no restrictions on</font> the type of work that can be executed in Hadoop; you pick which execution engines you need (whether it’s real-time processing with Spark, graph processing with Giraph, or MapReduce batch processing), and they can all be executing in parallel on the same cluster.<br />YARN 对 Hadoop 中可以执行的工作类型没有任何限制；您可以选择所需的执行引擎（无论是Spark 实时处理、Giraph 图形处理还是 MapReduce 批量处理），它们都可以在同一集群上并行执行。 |
| Concurrent processes<br />并发进程 | MapReduce had the notion of “slots,” which were node-specific static configurations that determined the maximum number of map and reduce processes that could run concurrently on each node. Based on where in the lifecycle a MapReduce application was, this would often lead to underutilized clusters.<br />MapReduce 有“slots”的概念，slots是专用于节点的静态配置，它决定了可以在每个节点上并发运行的map和reduce进程的最大数量。根据 MapReduce 应用程序在生命周期中的位置，这通常会导致未充分利用的集群。 | YARN allows for more fluid resource allocation, and the number of processes is limited only by the configured maximum amount of memory and CPU for each node.<br />YARN 允许更多的动态（原文：fluid）资源分配，并且进程的数量仅受每个节点配置的最大内存和CPU数量的限制。 |
| Memory limits<br />内存限制 | Slots in Hadoop 1 also had <font color="#32cd32">a maximum limit</font>, so typically Hadoop 1 clusters were provisioned such that the number of slots multiplied by the maximum configured memory for each slot was less than the available RAM. This often resulted in smaller than desired maximum slot memory sizes, which <font color="#32cd32">impeded your ability</font> to run memory intensive jobs.[^a] Another <font color="#32cd32">drawback of</font> MRv1 was that it was more difficult for <font color="#32cd32">memory-intensive and IO intensive jobs</font> to coexist on the same cluster or machines. Either you had more slots to <font color="#32cd32">boost the I/O jobs</font>, or fewer slots but more RAM for RAM jobs. Once again, the static nature of these slots made it a challenge to tune clusters for mixed workloads.<br />Hadoop1中的插槽也有一个最大限制，因此通常会设置Hadoop1集群使插槽数乘以每个插槽的最大配置内存小于可用RAM。这通常会导致小于所需的最大插槽内存大小，这会妨碍您运行内存密集型作业的能力。[^a-]MRV1的另一个缺点是内存密集型和IO密集型作业更难在同一个集群或机器上共存。要么您有更多的插槽来增加I/O作业，要么插槽更少，但RAM作业的内存更多。再一次，这些槽的静态特性使调整集群以适应混合工作负载成为一个挑战。 | YARN allows applications to request resources of <font color="#32cd32">varying memory sizes</font>. YARN has minimum and maximum memory limits, but because the number of slots <font color="#32cd32">is no longer fixed</font>, the maximum values can be much larger to support memory-intensive workloads. YARN therefore provides a much more <font color="#32cd32">dynamic scheduling model</font> that doesn’t limit the number of processes or the amount of RAM requested by a process.<br />YARN 允许应用程序请求不同内存大小的资源。YARN 具有最小和最大内存限制，但由于插槽数量不再固定，因此最大值可以大得多，以支持内存密集型工作负载。因此，YARN 提供了一个更为动态的调度模型，它不限制进程的数量或进程请求的RAM的数量。 |
| Scalability<br />可扩展性 | There were concurrency issues with the Job Tracker, which limited the number of nodes in a Hadoop cluster to 3,000–4,000 nodes.<br />Job Tracker（作业跟踪器）存在并发问题，它将Hadoop集群中的节点数限制为3000–4000个节点。 | By separating out the scheduling parts of MapReduce into YARN and making it light weight by <font color="#32cd32">delegating fault tolerance to</font> YARN applications, YARN can scale to much larger numbers than prior versions of Hadoop.[^b]<br />通过将MapReduce的调度部分分离成 YARN，并通过将容错委托给 YARN 应用程序，使其重量减轻，YARN 可以比Hadoop的早期版本扩展到更大的数量。[^b-] |
| Execution<br />执行 | Only a single version of MapReduce could be supported on a cluster at a time. This was problematic in large multi-tenant environments where product teams that wanted to upgrade to newer versions of MapReduce had to convince all the other users. This typically resulted in <font color="#32cd32">huge coordination and integration efforts</font> and made such upgrades huge infrastructure projects.<br />一次只能在集群上支持单一版本的MapReduce。在大型多用户（原文：tenant）环境中，这是一个问题，在这种环境中，想要升级到较新版本的 MapReduce 的产品团队必须说服所有其他用户。这通常会导致大量的协调和整合工作，并对大型基础设施项目进行升级。 | MapReduce is no longer at the core of Hadoop, and is now a YARN application running in user space. This means that you can now run different versions of MapReduce on the same cluster at the same time. This is <font color="#32cd32">a huge productivity gain</font> in large multi-tenant environments, and it allows you to organizationally <font color="#32cd32">decouple product teams and roadmaps</font>.<br />MapReduce 不再是 Hadoop 的核心，现在是一个在用户空间中运行的应用程序。这意味着您现在可以在同一集群上同时运行不同版本的 MapReduce 。在大型多用户（原文：tenant）环境中，这是一个巨大的生产力增长，它允许您在组织上分离产品团队和路线图。 |

Now that you know about the key benefits of YARN, it’s time to look at the main components in YARN and examine their roles.

现在你已经了解了 YARN 的主要优点，现在是时候看看 YARN 中的主要成分并研究它们的作用了。

### <font color="#9a1647">2.1.2 YARN concepts and components</font>

YARN comprises a framework that’s responsible for resource scheduling and monitoring, and applications that execute application-specific logic in a cluster. Let’s examine YARN concepts and components in more detail, starting with the YARN framework components.

YARN 包括一个负责资源调度和监控的框架，以及在集群中执行特定应用程序逻辑的应用程序。让我们从 YARN 框架组件开始，更详细地研究 YARN 概念和组件。

<font color="#9a1647">**YARN FRAMEWORK**</font>

The YARN framework performs one primary function, which is to schedule resources (containers in YARN parlance) in a cluster. Applications in a cluster talk to the YARN framework, asking for application-specific containers to be allocated, and the YARN framework evaluates these requests and attempts to fulfill them. An important part of the YARN scheduling also includes monitoring currently executing containers. There are two reasons that container monitoring is important: Once a container has completed, the scheduler can then use freed-up capacity to schedule more work. Additionally, each container has a contract that specifies the system resources that it’s allowed to use, and in cases where containers <font color="#32cd32">overstep</font> these <font color="#32cd32">bounds</font>, the scheduler can terminate the container to avoid <font color="#32cd32">rogue</font> containers impacting other applications.

<font style="background:yellow;">YARN 框架执行一个主要函数，即在集群中调度资源（YARN 术语中的容器：container）。集群中的应用程序与YARN 框架对话，请求分配特定应用程序的容器，YARN 框架评估这些请求并尝试实现它们。YARN 调度的一个重要部分还包括监控当前正在执行的容器。容器监视很重要有两个原因：一旦容器完成，调度程序就可以使用释放的容量来安排更多的工作。此外，每个容器都有一个协议（约定，原文：contract），指定允许它使用的系统资源，如果容器超出了这些限制，调度程序可以终止容器，以避免恶意容器影响其他应用程序</font>。

The YARN framework was intentionally designed to be as simple as possible; <font color="#32cd32">as such</font>, it doesn’t know or care about the type of applications that are running. Nor does it care about keeping any historical information about what has executed on the cluster. These design decisions are the primary reasons that YARN can <font color="#32cd32">scale beyond the levels of</font> MapReduce.

<font style="background:yellow;">YARN 框架被有意设计为尽可能简单；因此，它不知道或不关心正在运行的应用程序类型。它也不关心保存关于集群上执行的操作的任何历史信息。这些设计决策是 YARN 能够超越 MapReduce 水平的主要原因</font>。

There are two primary components that comprise the YARN framework—the ResourceManager and the NodeManager—which are seen in figure 2.2. 

图2.2显示了<font style="background:yellow;">YARN 有两个主要的组件，即资源管理器ResourceManager 和节点管理器NodeManager</font>。

![1563375735085](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1563375735085.png)

- **ResourceManager**—A Hadoop cluster has a single ResourceManager (RM) for the entire cluster. The ResourceManager is the YARN master process, and its <font color="#32cd32">sole function</font> is to <font color="#32cd32">arbitrate</font> resources on a Hadoop cluster. It responds to client requests to create containers, and a scheduler determines when and where a container can be created according to scheduler-specific <font color="#32cd32">multi-tenancy rules</font> that govern who can create containers where and when. Just like with Hadoop 1, the scheduler part of the ResourceManager is <font color="#32cd32">pluggable</font>, which means that you can pick the scheduler that works best for your environment. The actual creation of containers <font color="#32cd32">is delegated to</font> the NodeManager.

    ResourceManager——Hadoop集群在整个集群中只有一个ResourceManager（RM）。ResourceManager是YARN主进程，其唯一功能是仲裁Hadoop集群上的资源。它响应客户创建容器的请求，调度程序（scheduler ）根据特定于调度程序的 multi-tenancy 规则确定何时何地可以创建容器，该规则控制谁可以在何时何地创建容器。就像Hadoop 1一样，ResourceManager的调度程序（scheduler ）部分是可插拔（pluggable）的，这意味着您可以选择最适合您的环境的调度程序。容器的实际创建委托给NodeManager。

- **NodeManager**—The NodeManager is the slave process that runs on every node in a cluster. Its job is to create, monitor, and kill containers. It <font color="#32cd32">services requests from</font> the ResourceManager and ApplicationMaster to create containers, and it reports on the status of the containers to the ResourceManager. The ResourceManager uses the data contained in these status messages to <font color="#32cd32">make scheduling decisions</font> for new container requests. 

    NodeManager——节点管理器是在集群中的每个节点上运行的从属进程 （slave process）。它的工作是创建、监视和杀死容器。它服务来自 ResourceManager 和 ApplicationMaster 的请求去创建容器，并向 ResourceManager 报告容器的状态。ResourceManager 使用这些状态消息中包含的数据为新的容器请求制定调度决策。

In non-HA mode, only a single instance of the ResourceManager exists.[^4]

在非 HA (High Available ) 模式下，只有一个 ResourceManager 实例存在[^4-]。

The YARN framework exists to manage applications, so let’s take a look at what components a YARN application is composed of. 

YARN 框架是用来管理应用程序的，所以让我们来看一下 YARN 应用程序由哪些组件组成。

<font color="#9a1647">**YARN APPLICATIONS**</font>

A YARN application implements a specific function that runs on Hadoop. MapReduce is an example of a YARN application, as are projects such as Hoya, which allows multiple <font color="#32cd32">HBase instances</font> to run on a single cluster, and storm-yarn, which allows Storm to run inside a Hadoop cluster. You’ll see more details on these projects and other YARN applications later in this chapter.

YARN 应用程序实现了在 Hadoop 上运行的特定功能。MapReduce 是一个 YARN 应用程序的例子，Hoya 等项目允许多个 HBASE 实例在一个集群上运行，storm-yarn 允许 Storm 在 Hadoop 集群内运行。在本章后面，您将看到关于这些项目和其他 YARN 应用程序的更多详细信息。

A YARN application involves three components—the client, the ApplicationMaster (AM), and the container, which can be seen in figure 2.3.

YARN 应用程序包括三个组件：client （客户端）、ApplicationMaster（AM：应用程序的主管或者应用程序负责人或者应用程序的掌控者）和 Container（容器），如图2.3所示。

![1563376245066](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1563376245066.png)

Launching a new YARN application starts with a YARN client communicating with the ResourceManager to create a new YARN ApplicationMaster instance. Part of this process involves the YARN client informing the ResourceManager of the ApplicationMaster’s physical resource requirements.

启动新的YARN应用程序时，首先是YARN客户端与ResourceManager通信，以创建新的YARN ApplicationMaster实例。此过程的一部分涉及YARN客户端将ApplicationMaster的物理资源要求通知给ResourceManager。

The ApplicationMaster is the master process of a YARN application. It doesn’t perform any application-specific work, as these functions are delegated to the containers. Instead, it’s responsible for managing the application-specific containers: asking the ResourceManager of its intent to create containers and then liaising with the NodeManager to actually perform the container creation.

<font style="background:yellow;">**ApplicationMaster**是YARN应用程序的主进程。它不会执行任何特定于（具体某一个）应用程序的工作，因为这些功能被委托给了容器。相反，它负责去管理特定于应用程序的容器：询问ResourceManager创建容器的意图，然后与NodeManager联络以实际执行容器创建</font>。

As part of this process, the ApplicationMaster must specify the resources that each container requires in terms of which host should launch the container and what the container’s memory and CPU requirements are.[^5] The ability of the ResourceManager to schedule work based on exact resource requirements is a key to YARN’s flexibility, and it enables hosts to run a mix of containers, as highlighted in figure 2.4.

<font style="background:yellow;">作为此过程的一部分，ApplicationMaster 必须指定每个 Container 所需的资源、哪些主机应该启动 Container以及 Container 的内存和 CPU 要求是什么[^5-]。ResourceManager 能够根据准确的资源需求来安排工作，这是 YARN 灵活性的一个关键，它使主机能够运行各种不同的 Container ，如图2.4所示</font>。

![1563377543777](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1563377543777.png)

The ApplicationMaster is also responsible for the specific fault-tolerance behavior of the application. It receives status messages from the ResourceManager when its containers fail, and it can decide to take action based on these events (by asking the ResourceManager to create a new container), or to ignore these events.[^6]  

ApplicationMaster 还负责应用程序相应的容错行为。当其容器发生故障时，它从 ResourceManager 接收状态消息，并可以决定根据这些事件（通过请求 ResourceManager 创建新容器）采取操作，或者忽略这些事件。[^6-]

A container is an application-specific process that’s created by a NodeManager on behalf of an ApplicationMaster. The ApplicationManager itself is also a container, created by the ResourceManager. A container created by an ApplicationManager can be <font color="#32cd32">an arbitrary process</font>—for example, a container process could simply be a Linux command such as awk, a Python application, or any process that can <font color="#32cd32">be launched by</font> the operating system. This is the power of YARN—the ability to launch and manage any process across any node in a Hadoop cluster. 

Container 是由NodeManager代表ApplicationMaster创建的特定于（某一个具体）应用程序的进程。ApplicationManager本身也是由ResourceManager创建的容器。由ApplicationManager创建的 Container 可以是任意进程——例如，Container 进程可以简单地是Linux命令（例如awk）、Python应用程序或操作系统可以启动的任何进程。这就是YARN的强大功能——它能够跨Hadoop集群中的任何节点启动和管理任何进程。

<font color="#32cd32">By this point</font>, you should have a high-level understanding of the YARN components and what they do. Next we’ll look at common YARN configurables.

至此，您应该对 YARN 组件及其作用有一个高层次的了解。接下来，我们将讨论常见的 YARN 配置。

### <font color="#9a1647">2.1.3 YARN configuration</font>

YARN brings with it a whole slew of configurations for various components, such as the UI, remote procedure calls (RPCs), the scheduler, and more.[^7] In this section, you’ll learn how you can quickly access your running cluster’s configuration.

yarn为各种组件提供了一系列配置，例如UI、远程过程调用（RPCs）、调度程序等[^7-]。  在本节中，您将了解如何快速访问正在运行的集群的配置。

#### <font color="#939598">**TECHNIQUE 1**</font> <font color="#9a161a">**Determining the configuration of your cluster**</font> 

<font color="#939598">**技术1**</font> <font color="#9a161a">**确定集群的配置**</font>  

Figuring out the configuration for a running Hadoop cluster can be a nuisance—it often requires looking at several configuration files, including the default configuration files, to determine the value for the property you’re interested in. In this technique, you’ll see how to <font color="#32cd32">sidestep the hoops</font> you normally need to jump through, and instead focus on how to <font color="#32cd32">expediently</font> get at the configuration of a running Hadoop cluster. 

确定正在运行的Hadoop集群的配置可能很麻烦，通常需要查看几个配置文件，包括默认配置文件，以确定您感兴趣的属性的值。在这项技术中，您将看到如何避开通常需要跳过的环节，而将重点放在如何便捷高效地获得正在运行的Hadoop集群的配置上。

<font color="#9a1647">**Problem**</font>

​	You want to access the configuration of a running Hadoop cluster.

​	您希望访问正在运行的Hadoop集群的配置。

<font color="#9a1647">**Solution**</font>

​	View the configuration using the ResourceManager UI.

​	使用 ResourceManager UI 查看配置。

<font color="#9a1647">**Discussion**</font>

​	The ResourceManager UI shows the configuration for your Hadoop cluster; figure 2.5 shows how you can navigate to this information. 

​	ResourceManager UI 显示了Hadoop集群的配置；图2.5显示了如何导航到此信息。

​	<font color="#32cd32">What’s useful about this feature is that</font> the UI shows not only a property value, but also which file it originated from. If the value wasn’t defined in a <component>-site.xml file, then it’ll show the default value and the default filename.

​	<font style="background:yellow;">这个特性的有用之处在于，用户界面不仅显示一个属性值，还显示它来自哪个文件。如果该值不是在 `<component>-site.xml` 文件中定义的，那么它将显示默认值和默认文件名</font>。

​	Another useful feature of this UI is that it’ll show you the configuration from <font color="#32cd32">multiple files</font>, including the core, HDFS, YARN, and MapReduce files.

​	<font style="background:yellow;">这个用户界面的另一个有用特性是，它将显示多个文件中的配置</font>，包括 core，HDFS，YARN 和 MapReduce  文件。

​	The configuration for an individual Hadoop slave node can be navigated to in the same way from the NodeManager UI. This is most helpful when working with Hadoop clusters that consist of <font color="#32cd32">heterogeneous nodes</font>, where you often have <font color="#32cd32">varying</font> configurations that cater to differing hardware resources.

​	<font style="background:yellow;">单个 Hadoop 从节点（slave node ）的配置可以从 NodeManager UI 以相同的方式导航到。在处理由异构节点组成的 Hadoop 集群时，这一点非常有用，因为在这些节点中，常常有不同的配置来满足不同的硬件资源</font>。

​	<font color="#32cd32">By this point</font>, you should have a high-level understanding of the YARN components, what they do, and how to configure them for your cluster. The next step is to actually see YARN in action by using the command line and the UI. 

​	至此，您应该对 YARN 组件、它们的作用以及如何为集群配置它们有了较高层次的理解。下一步是通过使用命令行和用户界面实际查看 YARN 的运行情况。

![1563380899779](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1563380899779.png)

### <font color="#9a1647">2.1.4 Interacting with YARN</font>

<font color="#32cd32">Out of the box</font>, Hadoop 2 is bundled with two YARN applications—MapReduce 2 and DistributedShell. You’ll learn more about MapReduce 2 later in this chapter, but for now, you can <font color="#32cd32">get your toes wet</font> by taking a look at a simpler example of a YARN application: the DistributedShell. You’ll see how to run your first YARN application and where to go to examine the logs.

开箱即用，Hadoop 2 与两个 YARN 的应用程序： MapReduce 2 和 DistributedShell 捆绑在一起。在本章后面，您将了解更多关于 MapReduce 2 的内容，但现在，您可以通过查看一个更简单的 YARN 应用程序示例（DistributedShell）开始学习。您将看到如何运行您的第一个 YARN 应用程序以及到哪里检查日志。

If you don’t know the configured values for your cluster, you have two options:

如果不知道集群的配置值，则有两个选项：

- Examine the contents of yarn-site.xml to view the property values. If an entry doesn’t exist, the default value will be in effect.[^8]

    检查 yarn-site.xml 的内容以查看属性值。如果条目不存在，则默认值将生效。[^8-]

- Even better, use the ResourceManager UI, which gives you more detailed information on the running configuration, including what the default values are and if they’re in effect.

    更好的是，使用 ResourceManager UI，它可以提供有关运行配置的更详细信息，包括默认值是什么以及它们是否有效。

Let’s now take a look at how to quickly view the YARN configuration for a running Hadoop cluster. 

现在让我们来看看如何快速查看正在运行的Hadoop集群的 YARN 配置。

#### <font color="#939598">**TECHNIQUE 2**</font> <font color="#9a161a">**Running a command on your YARN cluster**</font>

Running a command on your cluster is a good first step when you start working with a new YARN cluster. It’s the “hello world” in YARN, if you will.

在集群上运行命令是开始使用新的 YARN 集群的第一步。如果你愿意的话，这就是 YARN 中的 “hello world”。

- <font color="#9a1647">**Problem**</font>
    You want to run a Linux command on a node in your Hadoop cluster.

    您希望在Hadoop集群中的节点上运行Linux命令。
    
- <font color="#9a1647">**Solution**</font>
    Use the DistributedShell example application bundled with Hadoop.
    
    使用与 Hadoop 捆绑在一起的 DistributedShell  示例应用程序。
    
- <font color="#9a1647">**Discussion**</font>

    YARN is bundled with the DistributedShell application, which serves two primary purposes—it’s a reference YARN application that’s also a handy utility for running a command in parallel across your Hadoop cluster. Start by issuing a Linux find command in a single container: 

    YARN 与 DistributedShell  应用程序捆绑在一起，它有两个主要用途。它是一个参考 YARN 应用程序，也是在 Hadoop 集群中并行运行命令的一个方便实用程序。首先在单个容器中发出 `linux find` 命令：

    ![1563383979049](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1563383979049.png)

    If all is well with your cluster, then executing the preceding command will result in the following log message: 

    如果您的集群一切正常，那么执行前面的命令将导致以下日志消息：

    `INFO distributedshell.Client: Application completed successfully `

There are various other logging statements that you’ll see in the command’s output prior to this line, but you’ll notice that none of them contain the actual results of your find command. This is because the DistributedShell ApplicationMaster launches the find command in a separate container, and the standard output (and standard error) of the find command is redirected to the log output directory of the container. To see the output of your command, you need to get access to that directory. That, as it happens, is covered in the next technique ! 

<font style="background:yellow;">在这行之前，您将在命令的输出中看到其他各种日志记录语句，但您会注意到它们都不包含 `find` 命令的实际结果。这是因为 DistributedShell ApplicationMaster 在单独的容器中启动`find`命令，`find` 命令的标准输出（和标准错误）被重定向到容器的日志输出目录。要查看命令的输出，需要访问该目录</font>。这一点，正如它所发生的，将在下一个技术中涉及到！

#### <font color="#939598">**TECHNIQUE 3**</font> <font color="#9a161a">**Accessing container logs**</font>

Turning to the log files is the most common first step one takes when trying to diagnose an application that behaved in an unexpected way, or to simply understand more about the application. In this technique, you’ll learn how to access these application log files.

当试图调试一个以意外方式运行的应用程序，或者只是为了了解更多关于该应用程序的信息时，转向日志文件是最常见的第一步。在这种技术中，您将学习如何访问这些应用程序日志文件。

- Problem

    You want to access container log files.

    要访问容器日志文件。

- Solution

    Use YARN’s UI and the command line to access the logs.

    使用 YARN 的用户界面和命令行访问日志。

- Discussion

  Each container that runs in YARN has its own output directory, where the standard output, standard error, and any other output files are written. Figure 2.6 shows the location of the output directory on a slave node, including the data retention details for the logs. Access to container logs is not as simple as it should be—let’s take a look at how you can use the CLI and the UIs to access logs. 
  
  在 YARN 中运行的每个容器都有自己的输出目录，其中写入标准输出、标准错误和任何其他输出文件。图2.6显示了从节点（slave node）上输出目录的位置，包括保留日志的数据详细信息。对容器日志的访问并不像应该的那么简单，让我们看看如何使用 CLI （Command Line Interface）和 UIs 来访问日志。
  
  ![1563420846180](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1563420846180.png)

<font color="#9a161a">***Accessing container logs using the YARN command line***</font>
YARN comes with a command-line interface (CLI) for accessing YARN application logs. To use the CLI, you need to know the ID of your application.

YARN 附带一个命令行界面（CLI），用于访问 YARN 应用程序的日志。要使用 CLI，您需要知道应用程序的 ID。

> How do I find the application ID? Most YARN clients will display the application ID in their output and logs. For example, the DistributedShell command that you executed in the previous technique echoed the application ID to standard output: 
>
> 如何找到应用程序ID？大多数 YARN 客户端将在其输出和日志中显示应用程序ID。例如，在上文执行的 DistributedShell  命令将应用程序ID回送到标准输出：
>
> ```shell
> $ hadoop o.a.h.y.a.d.Client ...
> ...
> INFO impl.YarnClientImpl:
> Submitted application application_1388257115348_0008 to
> ResourceManager at /0.0.0.0:8032
> ...
> ```
>
> Alternatively, you can use the CLI (using `yarn application -list`) or the ResourceManager UI to browse and find your application ID. 
>
> 或者，您可以使用 CLI（即：`yarn application -list`）或 ResourceManager UI 来浏览和查找应用程序ID。

If you attempt to use the CLI when the application is still running, you’ll be presented with the following error message:

如果在应用程序仍在运行时尝试使用 CLI，将显示以下错误信息：

```shell
$ yarn logs -applicationId application_1398974791337_0070
Application has not completed. Logs are only available after
an application completes
```
The message tells it all—the CLI is only useful once an application has completed. You’ll need to use the UI to access the container logs when the application is running, which we’ll cover shortly. Once the application has completed, you may see the following output if you attempt to run the command again:

返回的信息表明：只有在应用程序完成后，CLI 才有用。当应用程序运行时，您需要使用 UI 来访问容器日志，稍后我们将介绍这一点。应用程序完成后，如果再次尝试运行该命令，可能会看到以下输出：

```shell
$ yarn logs -applicationId application_1400286711208_0001
Logs not available at /tmp/.../application_1400286711208_0001
Log aggregation has not completed or is not enabled.
```
Basically, the YARN CLI only works if the application has completed and log aggregation is enabled. Log aggregation is covered in the next technique. If you enable log aggregation, the CLI will give you the logs for all the containers in your application, as you can see in the next example:

基本上，**只有在应用程序完成并且启用了日志聚合时，YARN CLI 才起作用**。下一个技术将介绍日志聚合。如果启用日志聚合，则 CLI 将为您提供应用程序中所有容器的日志，如下例所示：

```shell
$ yarn logs -applicationId application_1400287920505_0002

client.RMProxy: Connecting to ResourceManager at /0.0.0.0:8032
Container: container_1400287920505_0002_01_000002
on localhost.localdomain_57276
=================================================
LogType: stderr
LogLength: 0
Log Contents:
LogType: stdout
34 CHAPTER 2 Introduction to YARN
LogLength: 1355
Log Contents:
/tmp
default_container_executor.sh
/launch_container.sh
/.launch_container.sh.crc
/.default_container_executor.sh.crc
/.container_tokens.crc
/AppMaster.jar
/container_tokens
Container: container_1400287920505_0002_01_000001
on localhost.localdomain_57276
=================================================
LogType: AppMaster.stderr
LogLength: 17170
Log Contents:
distributedshell.ApplicationMaster: Initializing ApplicationMaster
...
LogType: AppMaster.stdout
LogLength: 8458
Log Contents:
System env: key=TERM, val=xterm-256color
...
```

The preceding output shows the contents of the logs of the DistributedShell example that you ran in the previous technique. There are two containers in the output—one for the find command that was executed, and the other for the ApplicationMaster, which is also executed within a container.

前面的输出显示了在前面介绍的技术中运行的 DistributedShell  示例的日志内容。输出中有两个容器，一个用于执行的 `find` 命令，另一个用于 ApplicationMaster，后者也在容器中执行。

<font color="#9a161a">***Accessing logs using the YARN UIs***</font>
YARN provides access to the ApplicationMaster logs via the ResourceManager UI. On a <font color="#32cd32">pseudo-distributed</font> setup, point your browser at http://localhost:8088/cluster. If you’re working with a multi-node Hadoop cluster, point your browser at http://$yarn.resourcemanager.webapp.address/cluster. Click on the application you’re interested in, and then select the Logs link as shown in figure 2.7.

**通过 ResourceManager UI，可以访问 ApplicationMaster 日志。在伪分布式设置中，将浏览器指向 http://localhost:8088/cluster。如果您使用的是多节点 Hadoop 集群，请将浏览器指向 <font color="#4183c4"><u>http://$yarn.resourcemanager.webapp.address/cluster</u></font>。单击您感兴趣的应用程序，然后选择logs链接**，如图2.7所示。

![1563469184095](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1563469184095.png)

Great, but how do you access the logs for containers other than the ApplicationMaster? Unfortunately, things get a little <font color="#32cd32">murky</font> here. The ResourceManager doesn’t <font color="#32cd32">keep track of</font> a YARN application’s containers, so it can’t provide you with a way to list and navigate to the container logs. Therefore, <font color="#32cd32">the onus is on</font> individual YARN applications <font color="#32cd32">to</font> provide their users with a way to access container logs.

很好，但是如何访问 ApplicationMaster 以外的容器的日志？不幸的是，这里的情况有点混乱。ResourceManager  不跟踪 YARN 应用程序的容器，因此它无法为您提供列出和导航到容器日志的方法。因此，责任在于单个 YARN 应用程序为其用户提供访问容器日志的方法。

> **Hey, ResourceManager, what are my container IDs?** In order to keep the ResourceManager lightweight, it doesn’t keep track of the container IDs for an application. As a result, the ResourceManager UI only provides a way to access the ApplicationMaster logs for an application.
>
> **嘿，资源管理器，我的容器ID是什么？** 为了保持 ResourceManager 的轻量级，它不跟踪应用程序的容器ID。因此，ResourceManager UI仅提供访问应用程序的ApplicationMaster日志的方法。

<font color="#32cd32">Case in point</font> is the DistributedShell application. It’s a simple application that doesn’t provide an ApplicationMaster UI or keep track of the containers that it’s launched. 

以 DistributedShell  应用程序为例。这是一个简单的应用程序，它不提供 ApplicationMaster UI 或跟踪它启动的容器。

Therefore, there’s no easy way to view the container logs other than by using the approach presented earlier: using the CLI.

因此，除了使用前面介绍的方法（使用 CLI ）之外，查看容器日志是不容易的。

Luckily, the MapReduce YARN application provides an ApplicationMaster UI that you can use to access the container (the map and reduce task) logs, as well as a JobHistory UI that can be used to access logs after a MapReduce job has completed. When you run a MapReduce job, the ResourceManager UI gives you a link to the MapReduce ApplicationMaster UI, as shown in figure 2.8, which you can use to access the map and reduce logs (much like the JobTracker in MapReduce 1)

**幸运的是，MapReduce YARN 应用程序提供了一个 ApplicationMaster  用户界面，您可以使用它来访问容器（map和reduce任务）日志，以及一个 JobHistory 用户界面，在 MapReduce  作业完成后可以使用它来访问日志。运行 MapReduce 作业时，ResourceManager UI 会向您提供指向 MapReduce ApplicationMaster UI 的链接**，如图2.8所示，您**可以使用它来访问 map 和 reduce 日志**（与 MapReduce 1 中的 JobTracker 非常相似）。

![1563469935354](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1563469935354.png)

If your YARN application provides some way for you to identify container IDs and the hosts that they execute on, you can either access the container logs using the NodeManager UI or you can use a shell to ssh to the slave node that executed a container. 

如果您的 YARN 应用程序为您提供了一些方法来标识容器 ID 和它们执行的主机，那么您可以使用 NodeManager UI访问容器日志，也可以使用 shell 对执行容器的从属节点（slave node）进行 ssh。

The NodeManager URL for accessing a container’s logs is <font color="#4183c4"><u>http://<nodemanagerhost>:8042/node/containerlogs/<container-id>/<username></u></font> . Alternatively, you can ssh to the NodeManager host and access the container logs directory at $yarn.nodemanager.log-dirs/<application-id>/<container-id>.

访问容器日志的节点管理器 URL 是 <font color="#4183c4"><u>http://<nodemanagerhost>:8042/node/containerlogs/<container-id>/<username></u></font>。或者，您可以ssh到nodemanager主机并访问 <font color="#4183c4"><u>$yarn.nodemanager.log dirs/<application id>/<container id></u></font> 处的container logs目录。

Really, the best advice I can give here is that you should enable log aggregation, which will allow you to use the CLI, HDFS, and UIs, such as the MapReduce ApplicationMaster and JobHistory, to access application logs. Keep reading for details on how to do this.

实际上，我在这里给出的**最佳建议是，您应该启用日志聚合，这将允许您使用 CLI、HDFS 和 UIs（如 MapReduce ApplicationMaster  和 JobHistory）访问应用程序日志**。继续阅读以了解如何执行此操作的详细信息。

<font color="#939598">**TECHNIQUE 4**</font> <font color="#9a161a">**Aggregating container log files **</font>

Log aggregation is a feature that was missing from Hadoop 1, making it challenging to archive and access task logs. Luckily Hadoop 2 has this feature baked-in, and you have a number of ways to access aggregated log files. In this technique you’ll learn how to configure your cluster to <font color="#32cd32">archive log files</font> for long-term storage and access.

日志聚合是 Hadoop 1 中缺少的一个功能，这使得归档和访问任务日志变得很困难。幸运的是，Hadoop 2 已经有了这个特性，并且您有许多方法可以访问聚合的日志文件。在这项技术中，您**将学习如何配置集群来归档日志文件以供长期存储和访问。**

- Problem
  You want to aggregate container log files to HDFS and manage their <font color="#32cd32">retention policies</font>.

  您希望将容器日志文件聚合到 HDFS 并管理它们的保留策略。

- Solution
  Use YARN’s built-in log aggregation capabilities.

  使用 YARN 的内置日志聚合功能。

- Discussion
  In Hadoop 1 your logs <font color="#32cd32">were stowed locally</font> on each slave node, with the JobTracker and TaskTracker being the only mechanisms for getting access to these logs. This was cumbersome and didn’t easily support programmatic access to them. In addition, log files would often disappear due to aggressive log-retention policies that existed to prevent local disks on slave nodes from filling up.

  在 Hadoop 1 中，日志存储在每个从节点（slave node）上的本地位置，JobTracker 和 TaskTracker 是访问这些日志的唯一机制。这很麻烦，而且不容易支持对它们的程序访问。此外，由于存在大量的日志保留策略，为了防止从节点上的本地磁盘被填满，日志文件通常会舍去一些。

  Log aggregation in Hadoop 2 is therefore <font color="#32cd32">a welcome feature</font>, and if enabled, it copies container log files into a Hadoop filesystem (such as HDFS) after a YARN application has completed. By default, this behavior is disabled, and you need to set `yarn.log-aggregation-enable` to true to enable this feature. Figure 2.9 shows the data flow for container log files. 
  
  **因此，Hadoop 2 中的日志聚合是一个受欢迎的特性，如果启用了这个特性，它会在一个yarn应用程序完成后将容器日志文件复制到 Hadoop 文件系统（如 hdfs）中。默认情况下，此行为被禁用，您需要将 `yarn.log-aggregation-enable` 设置为true才能启用此功能**。图2.9显示了容器日志文件的数据流。
  
  ![1563508312457](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1563508312457.png)

Now that you know how log aggregation works, let’s take a look at how you can access aggregated logs. 

既然您了解了日志聚合的工作原理，那么让我们来看看如何访问聚合日志。

<font color="#9a161a">***Accessing log files using the CLI*** </font>

With your application ID in hand (see technique 3 for details on how to get it), you can use the command line to fetch all the logs and write them to the console:

有了应用程序ID（有关如何获取它的详细信息，请参阅 technique 3 ），您可以使用命令行获取所有日志并将其写入控制台：

> **Enabling log aggregation** If the preceding yarn logs command yields the following output, then it’s likely that you don’t have YARN log aggregation enabled: 
>
> 启用日志聚合如果前面的 `yarn logs` 命令生成以下输出，则可能没有启用yarn logs聚合：
>
> `Log aggregation has not completed or is not enabled `

This will <font color="#32cd32">dump out</font> all the logs for all the containers for the YARN application. The output for each container is delimited with <font color="#32cd32">a header</font> indicating the container ID, followed by details on each file in the container’s output directory. For example, if you ran a DistributedShell command that executed ls -l, then the output of the yarn logs command would yield something like the following: 

这将输出所有用于 YARN 应用的容器的日志。每个容器的输出用一个指示容器 ID 的开头分隔，后面是容器输出目录中每个文件的详细信息。例如，如果运行执行 `ls-l` 的 DistributedShell 命令，那么 `yarn logs` 命令的输出将生成如下内容：

```shell
Container: container_1388248867335_0003_01_000002 on localhost
==============================================================
LogType: stderr
LogLength: 0
Log Contents:
LogType: stdoutLogLength: 268
Log Contents:
total 32
-rw-r--r-- 1 aholmes 12:29 container_tokens
-rwx------ 1 aholmes 12:29 default_container_executor.sh
-rwx------ 1 aholmes launch_container.sh
drwx--x--- 2 aholmes tmp
Container: container_1388248867335_0003_01_000001 on localhost
==============================================================
LogType: AppMaster.stderr
(the remainder of the ApplicationMaster logs removed for brevity)
```

The stdout file contains the directory listing of the ls process’s current directory, which is a container-specific working directory.

标准输出（stdout）文件包含 `ls` 进程当前目录的目录列表，该目录是相应容器的工作目录。

<font color="#9a161a">***Accessing aggregated logs via the UI***</font>

Fully featured YARN applications such as MapReduce provide an ApplicationMaster UI that can be used to access container logs. Similarly, the JobHistory UI can also access aggregated logs. 

全功能 YARN 应用程序（如MapReduce）提供了一个 ApplicationMaster UI，可用于访问容器日志。同样，JobHistory 用户界面也可以访问聚合日志。

> **UI aggregated log rendering** If log aggregation is enabled, you’ll need to update yarn-site.xml and set `yarn.log.server.url` to point at the job history server so that the ResourceManager UI can render the logs.
>
> 如果启用了日志聚合，则需要更新 yarn-site.xml 并将 `yarn.log.server.url` 设置为指向作业历史服务器，以便资源管理器UI可以呈现日志。

<font color="#9a161a">***Accessing log files in HDFS***</font>

By default, aggregated log files go into the following directory in HDFS:
`/tmp/logs/${user}/logs/application_<appid>`
The directory prefix can be configured via the `yarn.nodemanager.remote-app-log-dir` property; similarly, the path name after the username (“logs” in the previous example, which is the default) can be customized via yarn.nodemanager.remote-app-log-dir-suffix.

默认情况下，聚合日志文件将进入HDFS中的以下目录：

`/tmp/logs/${user}/logs/application_<appid>`

可以通过 `yarn.nodemanager.remote-app-log-dir` 属性配置目录前缀；同样，可以通过`yarn.nodemanager.remote-app-log-dir-suffix` 自定义用户名后面的路径名（“上一个示例中的日志”，这是默认值）。

<font color="#9a161a">***Differences between log files in local filesystem and HDFS***</font> 

As you saw earlier, each container results in two log files in the local filesystem: one for standard output and another for standard error. As part of the aggregation process, all the files for a given node are concatenated together into a node-specific log. For example, if you had five containers running across three nodes, you’d end up with three log files in HDFS.

如前所述，每个容器在本地文件系统中生成两个日志文件：一个用于标准输出，另一个用于标准错误。作为聚合过程的一部分，给定节点的所有文件都连接到一个（专门用于存储日志）节点的日志中。例如，如果您有五个容器跨三个节点运行，那么最终会在 HDFS 中得到三个日志文件。

<font color="#9a161a">***Compression***</font>

Compression of aggregated logs is disabled by default, but you can enable it by setting the value of `yarn.nodemanager.log-aggregation.compression-type` to either `lzo` or `gzip` depending on your compression requirements. As of Hadoop 2.2, these are the only two compression codecs supported.

聚合日志的压缩在默认情况下是没有启动的，但您可以根据压缩要求将 `yarn.nodemanager.log-aggregation.compression-type` 的值设置为 `lzo` 或 `gzip` 来启用它。从 Hadoop 2.2 开始，这是唯一支持的两个压缩编解码器。

<font color="#9a161a">***Log retention***</font>

When log aggregation is turned off, the container log files on the local host are retained for `yarn.nodemanager.log.retain-seconds` seconds, the default being 10,800 (3 hours). 

关闭日志聚合后，本地主机上的容器日志文件将保留为 `yarn.nodemanager.log.retain-seconds` 秒，默认值为10800（3小时）。

When log aggregation is turned on, the `yarn.nodemanager.log.retain-seconds` configurable is ignored, and instead the local container log files are deleted as soon as they are copied into HDFS. But all is not lost if you want to retain them on the local filesystem—simply set `yarn.nodemanager.delete.debug-delay-sec` to a value that you want to keep the files around for. Note that this applies not only to the log files but also to all other metadata associated with the container (such as JAR files).

**当启用日志聚合时，可以忽略可配置的 `yarn.nodemanager.log.retain-seconds`，而是在本地容器日志文件复制到 HDFS 后立即将其删除。但是，如果您想将它们保留在本地文件系统上，只需将`yarn.nodemanager.delete.debug-delay-sec` 设置为您想要保留这些文件的时间，就不会丢失所有内容。请注意，这不仅适用于日志文件，还适用于与容器关联的所有其他元数据（如JAR文件）。**

The data retention for the files in HDFS is configured via a different setting, `yarn.log-aggregation.retain-seconds`. 

HDFS中文件的数据保留是通过另一个设置（`yarn.log-aggregation.retain-seconds`）配置的。

<font color="#9a161a">***NameNode considerations***</font>

At scale, you may want to consider <font color="#32cd32">an aggressive log retention setting</font> so that you don’t <font color="#32cd32">overwhelm the NameNode with</font> all the log file metadata. The NameNode keeps the metadata in memory, and on a large active cluster, the number of log files can quickly overwhelm the NameNode. 

在大规模的时候，您可能需要考虑一个激进的日志保留设置，这样您就不会用所有日志文件元数据压倒名称节点。 NameNode 将元数据保存在内存中，并且在一个大型活动集群上，日志文件的数量可以迅速覆盖 NameNode。

>**Real-life example of NameNode impact** Take a look at Bobby Evans’ “Our Experience Running YARN at Scale” (http://www.slideshare.net/Hadoop_Summit/evans-june27-230pmroom210c) for a real-life example of how Yahoo! <font color="#32cd32">utilized</font> 30% of their NameNode <font color="#32cd32">with</font> seven days’ worth of aggregated logs. 
>
>Namenode 的影响在现实生活中的示例请看 Bobby Evans 的“Our Experience Running YARN at Scale”（http://www.slideshare.net/hadoop-summit/evans-june27-230pmroom210c），了解雅虎如何做出实例！利用其名称节点的30%做7天的聚合日志。

<font color="#9a161a">***Alternative solutions***</font>

The solution highlighted in this technique is useful for getting your logs into HDFS, but if you will need to  <font color="#32cd32">organize any log mining or visualization activities</font> yourself, there are other options available such as Hunk, which supports aggregating logs from both Hadoop 1 and 2 and providing first-class query, visualization, and monitoring features, just like regular Splunk. You could also set up a query and visualization pipeline using tools such as Logstash, ElasticSearch, and Kibana if you want to own the log management process. Other tools such as Loggly are worth investigating.

**此技术中突显了对于将日志导入 HDFS 的解决方案很有用，但是如果您需要自己组织任何日志挖掘或可视化活动，则还有其他可用选项，例如 Hunk，它支持从 Hadoop 1 和 2 聚合日志，并提供了一级查询，可视化和监视功能，就像普通的 Splunk 一样。如果您想拥有日志管理流程，还可以使用 Logstash， ElasticSearch，和 Kibana 等工具设置查询和可视化管道。其他工具如 Loggly  值得研究。**

For now, this  <font color="#32cd32">concludes</font> our hands-on look at YARN. That’s not the end of the story, however. Section 2.2 looks at how MapReduce works as a YARN application, and later in chapter 10, you’ll learn how to write your own YARN applications.

现在，这就结束了我们对 YARN 的亲身体验。然而，这并不是故事的结尾。第2.2节介绍了 MapReduce 作为 YARN 应用程序的工作原理，在第10章的后面，您将学习如何编写自己的 YARN 应用程序。

### <font color="#9a1647">2.1.5 YARN challenges</font>

There are  <font color="#32cd32">some gotchas</font> to be aware of with YARN:
使用 YARN ，有一些问题需要注意：

- YARN currently isn’t designed to work well with long-running processes. This has created challenges for projects such as Impala and Tez that would benefit from such a feature. Work <font color="#32cd32">is currently underway to</font> bring this feature to YARN, and it’s being tracked in a JIRA ticket titled “Roll up for long-lived services in YARN,” https://issues.apache.org/jira/browse/YARN-896.

    YARN 目前设计不适合长时间运行的进程。这给 Impala 和 Tez 等项目带来了挑战，这些项目将从这一功能中获益。目前正在努力将此功能引入到 YARN 中，并在一张名为“Roll up for long-lived services in YARN” 的JIRA 标签中进行跟踪，https://issues.apache.org/jira/browse/yarn-896。

- Writing YARN applications is quite complex, as you’re required to implement container management and fault tolerance. This may require some complex ApplicationMaster and container-state management so that upon failure the work can continue from some previous well-known state. There are several frameworks whose goal is to simplify development—refer to chapter 10 for more details. 

    编写 YARN 应用程序非常复杂，因为需要实现容器管理和容错。这可能需要一些复杂的 ApplicationMaster  和容器状态管理，以便在失败时可以从之前运行的某个已知状态继续工作。有几个框架的目标是简化开发，请参阅第10章了解更多详细信息。

- *Gang scheduling, which is the ability to rapidly launch a large number of containers in parallel, is currently not supported.* This is another feature that projects such as Impala and Hamster (OpenMPI) would require for native YARN integration. The  Hadoop committers are currently working on adding support for gang scheduling, which is being tracked in the JIRA ticket titled “Support gang scheduling in the AM RM protocol,” https://issues.apache.org/jira/browse/YARN-624.

    目前不支持快速并行启动大量容器的成群调度。这是另一个项目，如 Impala 和 Hamster（OpenMPI）将需要本地 YARN 集成的功能。Hadoop 代码提交者目前正致力于添加对成群调度的支持，这一支持在名为“Support gang scheduling in the AM RM protocol”的 JIRA 标签中得到跟踪，https://issues.apache.org/jira/browse/yarn-624。

So far we’ve focused on the capabilities of the core YARN system. Let’s move on to look at how MapReduce works as a YARN application.

到目前为止，我们关注的是核心 YARN 系统的性能。接下来我们来看看 MapReduce 是如何作为一个 YARN 应用程序工作的。

### <font color="#9a1647">YARN and MapReduce </font>

In Hadoop 1, MapReduce was the only way to process your data natively in Hadoop. YARN was created so that Hadoop clusters could run any type of work, and its only requirement was that applications adhere to the YARN specification. This meant MapReduce had to become a YARN application and required the Hadoop developers to rewrite key parts of MapReduce.

在Hadoop 1中，MapReduce是在Hadoop中本地处理数据的唯一方法。创建YARN的目的是使Hadoop集群可以运行任何类型的工作，它的唯一要求是应用程序必须遵守YARN规范。这意味着MapReduce必须成为YARN应用程序，并且需要Hadoop开发人员重写MapReduce的关键部分。

Given that MapReduce had to go through some open-heart surgery to get it working as a YARN application, the goal of this section is to demystify how MapReduce works in Hadoop 2. You’ll see how MapReduce 2 executes in a Hadoop cluster, and you’ll also get to look at configuration changes and backward compatibility with MapReduce 1. Toward the end of this section, you’ll learn how to run and monitor jobs, and you’ll see how small jobs can be quickly executed.

鉴于MapReduce必须经过一些坦率的手术才能使其作为YARN应用程序运行，本节的目标是揭开MapReduce在Hadoop 2中的工作方式的神秘性。您将看到MapReduce 2如何在Hadoop集群中执行，以及您还将了解配置更改以及与MapReduce 1的向后兼容性。到本节结束时，您将学习如何运行和监视作业，并了解如何快速执行小型作业。

There’s a lot to go over, so let’s take MapReduce into the lab and see what’s going on under the covers.

有很多事情要做，所以让我们将MapReduce带入实验室，看看幕后发生了什么。



---

[^1]: While you can do graph processing in Hadoop 1, it’s not a native fit, which means you’re either incurring the inefficiencies of multiple disk barriers between each iteration on your graph, or hacking around in MapReduce to avoid such barriers.
[^1-]: 虽然您可以在Hadoop1中进行图形处理，但它不是本地的，这意味着您要么在图形的每次迭代之间产生多个磁盘屏障的低效性，要么在MapReduce中进行破解避免此类屏障。
[^2]: Prior to YARN, Hadoop only supported MapReduce for computational work.
[^2-]: 在 YARN 之前，Hadoop只支持MapReduce进行计算工作。
[^3]: HBase is an exception; it uses HDFS for storage but doesn’t use MapReduce for the processing engine.
[^3-]: HBase是一个例外；它使用HDFS进行存储，但不使用MapReduce进行处理。
[^a]: This limitation in MapReduce was especially painful for those running machine-learning tasks using tools such as Mahout, as they often required large amounts of RAM for processing—amounts often larger than the maximum configured slot size in MapReduce.
[^a-]: 对于那些使用诸如 Mahout 之类的工具运行机器学习任务的人来说，他在 MapReduce 中的限制尤其痛苦，因为他们通常需要大量的RAM来处理，数量通常大于 MapReduce 中配置的最大插槽大小。
[^b]: The goal of YARN is to be able to scale to 10,000 nodes; scaling beyond that number could result int he ResourceManager becoming a bottleneck, as it’s a single process.
[^b-]: YARN 目标是能够扩展到10000个节点；超过这个数量的扩展可能会导致 ResourceManager 成为一个瓶颈，因为它是一个单独的过程。
[^4]: As of the time of writing, YARN ResourceManager HA is still actively being developed, and its progress can be followed on a JIRA ticket titled “ResourceManager (RM) High-Availability (HA), ” https://issues.apache.org/
[^4-]:截至撰写之时，YARN resourcemanager HA 仍在积极开发中，其进展可在一张名为“resourcemanager（RM）high availability（HA）”的JIRA标签上跟踪，https://issues.apache.org/jira/browse/yarn-149。
[^5]: Future versions of Hadoop may allow network, disk, and GPU requirements to be specified.
[^5-]: Hadoop的未来版本可能允许指定网络、磁盘和GPU需求
[^6]: Containers can fail for a variety of reasons, including a node going down, a container being killed by YARN to allow another application’s container to be launched, or YARN killing a container when the container exceeds its configured physical/virtual memory.
[^6-]: Container可能由于多种原因而失败，包括节点故障、Container 被 YARN 杀死以允许启动另一个应用程序的 Container，或者 Container 超过其配置的物理或虚拟内存时 YARN 杀死 Container。
[^7]: Details on the default YARN configurations can be seen at http://hadoop.apache.org/docs/r2.2.0/hadoopyarn/hadoop-yarn-common/yarn-default.xml.
[^7-]: 有关默认 YARN 置的详细信息，请访问http://hadoop.apache.org/docs/r2.2.0/hadoop yarn/hadoop-yarn-common/yarn-default.xml。
[^8]: Visit the following URL for YARN default values: http://hadoop.apache.org/docs/r2.2.0/hadoop-yarn/ hadoop-yarn-common/yarn-default.xml.
[^8-]:访问以下URL获取yarn默认值：http://hadoop.apache.org/docs/r2.2.0/hadoop-yarn/hadoop-yarn-common/yarn-default.xml。



