---
title: Paper Google Bigtable 翻译与总结
date: 2019-12-31
copyright: true
categories: English,中文
tags: [distributed system,papers,google]
mathjax: false
mathjax2: false
toc: true
top: 12
---

## 前言

1. 第一部分主要是论文的翻译与旁注：按照论文原文结构一步步翻译
2. 第二部分主要是BigTable思想总结：BigTable论文相比GFS、MapReduce两篇复杂，行文并不流畅（可能本渣渣太弱），文中甚至没有总体结构说明和一些难点解释（例如：BigTable中出现的且在后来众多优秀的开源组件（例如：LevelDB, RocksDB）中常用的SSTable文件索引格式：LSM 都没有详细说明），因此在总结处弥补这方面的说明。

## 论文翻译与旁注

### Abstract

Bigtable is a distributed storage system for managing structured data, which is designed to scale to a very large scale: petabytes of data in thousands of commercial servers.  Many Google projects store data in Bigtable, including web indexing, Google Earth, and Google Finance.  These applications place very different requirements on Bigtable in terms of data size (from URL to web page to satellite imagery) and latency requirements (from back-end batch processing to real-time data services).  Despite the varied requirements, Bigtable has successfully provided a flexible, high-performance solution for all of these Google products.  In this article, we describe the simple data model provided by Bigtable, which provides customers with dynamic control over data layout and format, and describes the design and implementation of Bigtable.

Bigtable是用于管理结构化数据的分布式存储系统，该系统旨在扩展到非常大的规模：数千个商用服务器中的PB级数据。  Google的许多项目都将数据存储在Bigtable中，包括网络索引，Google Earth和Google Finance。 这些应用程序在数据大小（从URL到网页到卫星图像）和延迟要求（从后端批量处理到实时数据服务）方面都对Bigtable提出了截然不同的要求。尽管需求千差万别，Bigtable已成功为所有这些Google产品提供了一种灵活的高性能解决方案。 在本文中，我们描述了Bigtable提供的简单数据模型，该模型为客户提供了对数据布局和格式的动态控制，并描述了Bigtable的设计和实现。

### 1 Introduction

Introduction Over the last two and a half years we have designed, implemented, and deployed a distributed storage system for managing structured data at Google called Bigtable. Bigtable is designed to reliably scale to petabytes of data and thousands of machines. Bigtable has achieved several goals: wide applicability, scalability, high performance, and high availability. Bigtable is used by more than sixty Google products and projects, including Google Analytics, Google Finance, Orkut, Personalized Search, Writely, and Google Earth. These products use Bigtable for a variety of demanding workloads, which range from throughput-oriented batch-processing jobs to latency-sensitive serving of data to end users. The Bigtable clusters used by these products span a wide range of configurations, from a handful to thousands of servers, and store up to several hundred terabytes of data.

简介在过去的两年半中，我们在Google上设计，实施和部署了一个分布式存储系统来管理结构化数据，称为Bigtable。  Bigtable旨在可靠地扩展到PB级数据和数千台计算机。**Bigtable实现了多个目标：广泛的适用性，可伸缩性，高性能和高可用性**。 Bigtable被60多个Google产品和项目所使用，包括Google Analytics（分析），Google Finance，Orkut，个性化搜索，Writely和Google Earth。 这些产品将Bigtable用于各种要求高的工作负载，从面向吞吐量的批处理作业到对延迟敏感的终端用户所享受的数据服务。 这些产品使用的Bigtable集群涵盖了多种配置，从少量服务器到数千个服务器，最多可存储数百TB的数据。

In many ways, Bigtable resembles a database: it shares many implementation strategies with databases. Parallel databases<a href=" #[14]">[14]</a> and main-memory databases<a href=" #[13]">[13]</a> have achieved scalability and high performance, but Bigtable provides a different interface than such systems. Bigtable does not support a full relational data model; instead, it provides clients with a simple data model that supports dynamic control over data layout and format, and allows clients to reason about the <u>locality properties</u> of the data <u>represented in</u> the underlying storage. Data is indexed using row and column names that can be arbitrary strings. Bigtable also treats data as <u>uninterpreted strings</u>, although clients often serialize various forms of structured and semi-structured data into these strings. Clients can control the locality of their data through careful choices in their schemas. Finally, Bigtable schema parameters let clients dynamically control whether to <u>serve</u> data <u>out of memory</u> or <u>from disk</u>.

在许多方面，Bigtable类似于数据库：它与数据库共享许多实现策略。 并行数据库<a href=" #[14]">[14]</a>和主内存数据库<a href=" #[13]">[13]</a>已经实现了可伸缩性和高性能，但是Bigtable提供了与此类系统不同的接口。  **Bigtable不支持完整的关系数据模型； 相反，它为客户端提供了一个简单的数据模型，该模型支持对数据布局和格式的动态控制，并允许客户端推理存储[^1]在底层的数据的位置属性[^16]。 可以使用任意字符串的行和列名称为数据建立索引。 尽管客户端经常将各种形式的结构化和半结构化数据序列化为这些字符串，但Bigtable还将数据视为未解析[^2]的字符串。客户端可以通过在模式中进行仔细选择来控制其数据的位置。 最后，Bigtable模式参数可让客户端动态控制是从磁盘还是从内存获得数据[^3]**。

Section 2 describes the data model in more detail, and Section 3 provides an overview of the client API. Section 4 briefly describes the underlying Google infrastructure on which Bigtable depends. Section 5 describes the fundamentals of the Bigtable implementation, and Section 6 describes some of the refinements that we made to improve Bigtable’s performance. Section 7 provides measurements of Bigtable’s performance. We describe several examples of how Bigtable is used at Google in Section 8, and discuss some lessons we learned in designing and supporting Bigtable in Section 9. Finally, Section 10 describes related work, and Section 11 presents our conclusions.

第2节将更详细地描述数据模型，第3节将概述客户端API，第4节简要介绍了Bigtable所依赖的基础Google架构。 第5节介绍了Bigtable实现的基础知识，第6节介绍了我们为提高Bigtable的性能所做的一些改进。 第7节提供了Bigtable性能的衡量标准。 在第8节中，我们描述了如何在Google中使用Bigtable的几个示例，并在第9节中，讨论了我们在设计和支持Bigtable方面学到的一些教训。最后，第10节描述了相关工作，第11节介绍了我们的结论。

### 2 Data Model

A Bigtable is a sparse, distributed, persistent multidimensional sorted map. The map is indexed by a row key, column key, and a timestamp; each value in the map is an uninterpreted array of bytes. 

**一个BigTable是一个稀疏的、分布的、永久的多维有序的映射表（map）。我们采用行键（row key）、列键（column key）和时间戳（timestamp）对映射表（map）进行索引。映射表（map）中的每个值都是未经解析的字节数组**。

(row:string, column string, time:int64)→string

We settled on this data model after examining a variety of potential uses of a Bigtable-like system. As one concrete example that drove some of our design decisions, suppose we want to keep a copy of a large collection of web pages and related information that could be used by many different projects; let us call this particular table the Webtable. In Webtable, we would use URLs as row keys, various aspects of web pages as column names, and store the contents of the web pages in the contents: column under the timestamps when they were fetched, as illustrated in Figure 1. 

在研究了类似Bigtable的系统的各种潜在用途之后，我们选择了此数据模型。 作为推动我们某些设计决策的一个具体示例，假设我们想要保留大量网页和相关信息的副本，这些副本可以由许多不同的项目使用。 让我们将此特定表称为Webtable。 在Webtable中，我们将使用URL作为行键，将网页的各个方面用作列名，并将网页的内容存储在 contents: 列，这些列在时间戳（获取时的时间）底下[^17]，如图1所示。

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/1.png" alt="1122" style="zoom:50%;" />

<p><i><center>图1 存储了网页数据的Webtable的一个片段</center></br>行名称是反转的URL，contents列家族包含了网页内容，anchor列家族包含了任何引用这个页面的anchor文本。CNN的主页被Sports Illustrated和MY-look主页同时引用，因此，我们的行包含了名称为 ”anchor:cnnsi.com” 和 ”anchor:my.look.ca” 的列。每个anchor单元格都只有一个版本，contents列有三个版本，分别对应于时间戳t3，t5和t6。</i></p>
#### Rows

The row keys in a table are arbitrary strings (currently up to 64KB in size, although 10-100 bytes is a typical size for most of our users). Every read or write of data under a single row key is atomic (regardless of the number of different columns being read or written in the row), a design decision that makes it easier for clients to <u>reason about</u> the system’s behavior in the presence of concurrent updates to the same row.

**表中的行键是任意字符串**（当前大小最大为64KB，尽管对于大多数用户而言，典型大小是10-100字节）。 **单个行键下的每次数据读取或写入都是原子性的**（无论该行中读取或写入的不同列的数量如何），该设计决策使客户端在出现并发更新到同一行时更容易推断系统行为。

Bigtable maintains data <u>in lexicographic order</u> by row key. The row range for a table is dynamically partitioned. Each row range is called a tablet, which is the unit of distribution and load balancing. As a result, reads of short row ranges are efficient and typically require communication with only a small number of machines. Clients can exploit this property by selecting their row keys so that they get good locality for their data accesses. For example, in Webtable, pages in the same domain are grouped together into contiguous rows by reversing the hostname components of the URLs. For example, we store data for maps.google.com/index.html under the key com.google.maps/index.html. Storing pages from the same domain near each other makes some host and domain analyses more efficient.

**Bigtable按行键的字典顺序维护数据。表的行区间是动态分区的。每个行区间称为一个**Tablet**，它是分配和负载平衡的单位**。结果，对行的小范围读取（reads of short row ranges，这里short修饰的名词是 ranges 还是 row ，最终根据下文的例子进行反推的）是高效的并且通常仅需要与少量机器通信。客户端可以通过选择行键来利用此属性，以便他们可以很好地进行数据访问。例如，在Webtable中，通过反转URL的主机名部分，可以将同一域中的页面分组为连续的行。例如我们将数据maps.google.com/index.html存储在键com.google.maps/index.html下。 将同一域中的页面彼此靠近存储可以使某些主机和域分析更加高效。

#### Column Families

Column keys are grouped into sets called column families, which form the basic unit of access control. All data stored in a column family is usually of the same type (we compress data in the same column family together). A column family must be created before data can be stored under any column key in that family; after a family has been created, any column key within the family can be used. It is our intent that the number of distinct column families in a table be small (in the hundreds at most), and that families rarely change during operation. In contrast, a table may have an unbounded number of columns.

**列键被分组成称为列族的集合，这些集合构成访问控制的基本单元**。 <u>列族中存储的所有数据通常都是同一类型（我们将同一列族中的数据压缩在一起）。 必须先创建一个列族，然后才能将数据存储在该族中的任何列键下。 创建族后，可以使用族中的任何列键。 我们的目的是使表中不同的列族的数量少（最多数百个），并且在操作过程中族很少改变。 相反，表可能具有无限数量的列。</u>

A column key is named using the following syntax: family:qualifier. Column family names must be printable, but qualifiers may be arbitrary strings. An example column family for the Webtable is language, which stores the language in which a web page was written. We use only one column key in the language family, and it stores each web page’s language ID. Another useful column family for this table is anchor; each column key in this family represents a single anchor, as shown in Figure 1. The qualifier is the name of the referring site; the cell contents is the link text.

列键使用以下语法命名：`family:qualifier`。列族（ column family）名称必须是可打印的，但限定词（qualifier）可以是任意字符串。 Webtable的一个示例列族是language，它存储编写网页所用的语言。我们在语言族中仅使用一个列键，并且它存储每个网页的语言ID。此表的另一个有用的列族是锚； 该族中的每个列键都代表一个锚，如图1所示。限定符是引用站点的名称。单元格内容是链接文本。

Access control and both disk and <u>memory accounting</u> are performed at the column-family level. In our Webtable example, these controls allow us to manage several different types of applications: some that add new base data, some that read the base data and create derived column families, and some that are only allowed to view existing data (and possibly not even to view all of the existing families for privacy reasons).

<u>访问控制以及磁盘和内存统计[^4] 均在列族层次执行</u>。 在我们的Webtable示例中，<u>这些控制（权限）使我们能够管理几种不同类型的应用程序：一些应用程序是添加新的基本数据，一些应用程序是读取基本数据并创建派生的列族，某些应用程序仅被许可查看现有数据（出于隐私原因，甚至可能不允许查看所有现有列族）</u>。

#### Timestamps

Each cell in a Bigtable can contain multiple versions of the same data; these versions are indexed by timestamp. Bigtable timestamps are 64-bit integers. They can be assigned by Bigtable, in which case they represent “real time” in microseconds, or be explicitly assigned by client applications. Applications that need to avoid collisions must generate unique timestamps themselves. Different versions of a cell are stored <u>in decreasing timestamp order</u>, so that the most recent versions can be read first.

**Bigtable中的每个单元格可以包含同一数据的多个版本；这些版本通过时间戳索引**。 <u>Bigtable时间戳是64位整数。它们可以由Bigtable分配，在这种情况下，它们以微秒为单位表示“真实时间”，也可以由客户端应用程序明确分配。需要避免冲突的应用程序必须自己生成唯一的时间戳。单元格的不同版本以时间戳的降序[^5]存储，因此可以首先读取最新版本</u>。

To make the management of versioned data less onerous, we support two per-column-family settings that tell Bigtable to garbage-collect cell versions automatically. The client can specify either that only the last n versions of a cell be kept, or that only new-enough versions be kept (e.g., only keep values that were written in the last seven days).

<u>为了减少版本化数据的管理工作，我们支持每个列族的两个设置，这些设置告诉Bigtable自动垃圾回收单元格版本。 客户端可以指定仅保留单元格的最后n个版本，或者仅保留足够新的版本（例如，仅保留最近7天写入的值）</u>。

In our Webtable example, we set the timestamps of the crawled pages stored in the contents: column to the times at which these page versions were actually crawled. The garbage-collection mechanism described above lets us keep only the most recent three versions of every page.

在我们的Webtable示例中，我们将 content: 列中存储的爬虫网页的时间戳设置为实际爬虫这些页面版本的时间。 上述的垃圾收集机制使我们仅保留每个页面的最新三个版本。

### 3 API

The Bigtable API provides functions for creating and deleting tables and column families. It also provides functions for changing cluster, table, and column family metadata, such as access control rights.

Bigtable API提供了用于创建和删除表和列族的功能。它还提供了用于更改集群，表和列族元数据的功能，例如访问控制权限。


```c++
// Open the table
Table *T = OpenOrDie("/bigtable/web/webtable");
// Write a new anchor and delete an old anchor
RowMutation r1(T, "com.cnn.www");
r1.Set("anchor:www.c-span.org", "CNN");
r1.Delete("anchor:www.abc.com");
Operation op;
Apply(&op, &r1);
```

<center><i>图 2 写入到Bigtable</i></center>
Client applications can write or delete values in Bigtable, look up values from individual rows, or iterate over a subset of the data in a table. Figure 2 shows C++ code that uses <u>a RowMutation abstraction</u> to perform a series of updates. (Irrelevant details were elided to keep the example short.) The call to Apply performs an atomic mutation to the Webtable: it adds one anchor to www.cnn.com and deletes a different anchor.

客户端应用程序可以在Bigtable中写入或删除值，可以从各个行中查找值，也可以遍历表中的数据子集。图2显示了使用`RowMutation`抽象（对象）[^6]来执行一系列更新的 C++ 代码。（省略了详细信息，以使示例简短）对`Apply`的调用对`Webtable`进行了原子修改：它将一个锚点添加到 www.cnn.com 并删除另一个锚点。

```c++
Scanner scanner(T);
ScanStream *stream;
stream = scanner.FetchColumnFamily("anchor");
stream->SetReturnAllVersions();
scanner.Lookup("com.cnn.www");
for (; !stream->Done(); stream->Next()) {
    printf("%s %s %lld %s\n",
    scanner.RowName(),
    stream->ColumnName(),
    stream->MicroTimestamp(),
    stream->Value());
}
```

<center><i>图3: 从Bigtable读取数据</i></center>
Figure 3 shows C++ code that uses <u>a Scanner abstraction</u> to iterate over all anchors in a particular row. Clients can iterate over multiple column families, and there are several mechanisms for limiting the rows, columns, and timestamps produced by a scan. For example, we could restrict the scan above to only produce anchors whose columns match the regular expression `anchor:*.cnn.com`, or to only produce anchors whose timestamps fall within ten days of the current time.

图3显示了使用`Scanner`抽象（对象） [^7] 对特定行中的所有锚点进行迭代的C ++代码。客户端可以迭代多个列族，并且有几种机制可以限制扫描产生的行，列和时间戳。例如，我们可以将上面的扫描限制为仅生成其列与正则表达式 `anchor:*.cnn.com` 匹配的锚，或者仅生成其时间戳在当前时间的十天内之内的锚。

Bigtable supports several other features that allow the user to manipulate data in more complex ways. First, Bigtable supports single-row transactions, which can be used to perform atomic read-modify-write sequences on data stored under a single row key. Bigtable does not currently support general transactions across row keys, although it provides an interface for batching writes across row keys at the clients. Second, Bigtable allows cells to be used as integer counters. Finally, Bigtable supports the execution of client-supplied scripts in the address spaces of the servers. The scripts are written in a language developed at Google for processing data called Sawzall<a href=" #[28]">[28]</a>. At the moment, our Sawzall-based API does not allow client scripts to write back into Bigtable, but it does allow various forms of data transformation, filtering based on arbitrary expressions, and summarization via a variety of operators.

Bigtable支持其他几种功能，这些功能允许用户以更复杂的方式操作数据。 **首先，Bigtable支持单行事务（single-row transaction）**，该事务可用于对存储在单个行键下的数据执行原子的 “读-修改-写”（read-modify-write） 序列。  Bigtable目前不支持跨行键的常规事务，尽管它提供了用于在客户端跨行键批处理写入的接口。 **其次，Bigtable允许将单元格[^18]用作整数计数器**。 **最后，Bigtable支持在服务器的地址空间中执行客户端提供的脚本。 这些脚本是用Google开发的一种用于处理数据的语言（称为Sawzall<a href=" #[28]">[28]</a>）编写的**。 目前，我们基于Sawzall的API不允许客户端脚本写回到Bigtable，但允许多种形式的数据转换，基于任意表达式的过滤以及通过各种运算符的汇总。

Bigtable can be used with MapReduce<a href=" #[12]">[12]</a>, a framework for running large-scale parallel computations developed at Google. We have written a set of wrappers that allow a Bigtable to be used both as an input source and as an output target for MapReduce jobs.

Bigtable可与MapReduce<a href=" #[12]">[12]</a>结合使用，MapReduce是一种由Google开发的用于运行大规模并行计算的框架。 我们编写了一组包装器（wrappers），这些包装器允许Bigtable用作MapReduce作业的输入源和输出目标。

### 4 Building Blocks

Bigtable is built on several other pieces of Google infrastructure. Bigtable uses the distributed Google File System (GFS) <a href=" #[17]">[17]</a> to store log and data files. A Bigtable cluster typically operates in <u>a shared pool of machines</u> that run a wide variety of other distributed applications, and Bigtable processes often share the same machines with processes from other applications. Bigtable depends on a cluster management system for scheduling jobs, managing resources on shared machines, dealing with machine failures, and monitoring machine status.

**Bigtable建立在Google其他几个基础架构之上。 Bigtable使用分布式Google文件系统（GFS）<a href=" #[17]">[17]</a>存储日志和数据文件。 Bigtable集群通常运行在与多种其他分布式应用程序共享的服务器池[^10]中，并且Bigtable进程通常与其他应用程序的进程共享同一台计算机。 Bigtable依靠集群管理系统来调度作业、来管理共享计算机上的资源、来处理计算机故障以及监视计算机状态**。

The Google SSTable file format is used internally to store Bigtable data. An SSTable provides a persistent, ordered immutable map from keys to values, where both keys and values are arbitrary byte strings. Operations are provided to look up the value associated with a specified key, and to iterate over all key/value pairs in a specified key range. Internally, each SSTable contains a sequence of blocks (typically each block is 64KB in size, but this is configurable). A block index (stored at the end of the SSTable) is used to locate blocks; the index is loaded into memory when the SSTable is opened. A lookup can be performed with a single disk seek: we first find <u>the appropriate block</u> by performing a binary search in the in-memory index, and then reading <u>the appropriate block</u> from disk. Optionally, an SSTable can be completely mapped into memory, which allows us to perform lookups and scans <u>without touching disk</u>.

**Google SSTable文件格式在内部用于存储Bigtable数据**。 <u>SSTable提供了从键到值都可以持久化、有序的、不可变的映射表（map），其中键和值都是任意字节字符串。提供操作以查找与指定键相关联的值，并遍历指定键范围内的所有键/值对。在内部，每个SSTable包含一系列块（通常每个块的大小为64KB，但这是可配置的）。块的索引（存储在SSTable的末尾）用于定位块。当打开SSTable时，索引将加载到内存中。可以使用单次磁盘寻址（ disk seek）执行一次查找：我们首先对内存中的索引执行二分搜索来找到对应的块索引[^11]，然后从磁盘读取相应[^8]的块。可选项是可以将一个SSTable全部映射到内存中，这使我们无需与磁盘进行io[^12]即可执行查找和扫描</u>。 

Bigtable relies on a highly-available and persistent distributed lock service called Chubby <a href=" #[8]">[8]</a>. A Chubby service consists of five active replicas, one of which is elected to be the master and <u>actively serve requests</u>. <u>The service is live</u> when a majority of the replicas are running and can communicate with each other. Chubby uses the Paxos algorithm <a href=" #[9]">[9]</a><a href=" #[23]">[23]</a> to keep its replicas consistent in the face of failure. Chubby provides a namespace that consists of directories and small files. Each directory or file can be used as a lock, and reads and writes to a file are atomic. The Chubby client library provides consistent caching of Chubby files. Each Chubby client maintains a session with a Chubby service. A client’s session expires if it is unable to <u>renew its session lease</u> within the lease expiration time. When a client’s session expires, it loses any locks and open handles. Chubby clients can also register callbacks on Chubby files and directories for notification of changes or session expiration.

**Bigtable依赖一个高可用且持久的分布式锁定服务，称为Chubby<a href=" #[8]">[8]</a>**。<u>Chubby服务由五个活动副本组成，其中一个活动副本被选为主副本，并积极响应请求[^13]。当大部分副本处于运行状态并且能够彼此通信时，这个服务是可用的</u>[^9]。**Chubby使用Paxos算法 <a href=" #[9]">[9]</a><a href=" #[23]">[23]</a> 应对失败时如何保持其副本的一致性**。 <u>Chubby提供了一个由目录和小文件组成的命名空间。每个目录或文件都可以用作锁，并且对文件的读写是原子的。 Chubby客户端函数库提供一致的Chubby文件缓存。每个Chubby客户端都维护一个Chubby服务会话（session）。如果客户端的会话（session）无法在租约（lease）到期时间内续签（renew）其会话租约（session lease），则该会话将过期。客户端会话（session）期满后，它将丢失所有锁以及已打开的文件句柄（handle）。Chubby客户端也可以在Chubby文件和目录上注册回调函数（callback），以通知（出现）变化或会话（session）到期。</u>

Bigtable uses Chubby for a variety of tasks: to ensure that there is at most one active master at any time; to store the bootstrap location of Bigtable data (see Section 5.1); to discover tablet servers and finalize tablet server deaths (see Section 5.2); to store Bigtable schema information (the column family information for each table); and to store access control lists. If Chubby becomes unavailable for an extended period of time, Bigtable becomes unavailable. We recently measured this effect in 14 Bigtable clusters spanning 11 Chubby instances. The average percentage of Bigtable server hours during which some data stored in Bigtable was not available due to Chubby unavailability (caused by either Chubby outages or network issues) was 0.0047%. The percentage for the single cluster that was most affected by Chubby unavailability was 0.0326%.

**Bigtable使用Chubby来完成各种任务：**

1. 确保任何时候最多一个活跃的master（active master）；
2. 存储Bigtable数据的引导位置（bootstrap location）（请参阅第5.1节）；
3. 发现 Tablet  服务器并确定 Tablet  服务器的死机（请参阅第5.2节）；
4. 存储Bigtable模式（schema）信息（每个表的列族信息）；
5. 存储用于访问控制的信息而组成的列表；

如果Chubby长时间不可用，则Bigtable将不可用。我们最近在跨越11个Chubby实例的14个Bigtable集群中测量了这种影响。由于Chubby不可用（由于Chubby中断或网络问题所致）导致存储在Bigtable服务器上的一些数据无法访问的时间平均占比为0.0047％。受Chubby不可用性影响最大的单个集群上面的数据无法访问的时间占比为0.0326％。

### 5 Implementation

The Bigtable implementation has three major components: a library that is linked into every client, one master server, and many tablet servers. Tablet servers can be dynamically added (or removed) from a cluster to accomodate changes in workloads.

**Bigtable实现具有三个主要组件：一个链接到每个客户端的函数库，一个主服务器（master server）和许多Tablet服务器。可以从集群中动态添加（或删除）Tablet服务器，以适应工作负载的变化**。

The master is responsible for assigning tablets to tablet servers, detecting the addition and expiration of tablet servers, balancing tablet-server load, and garbage collection of files in GFS. In addition, it handles schema changes such as table and column family creations.

**主服务器（master）负责将Tablet分配给Tablet服务器，检测Tablet服务器的添加和到期，平衡Tablet服务器的负载以及GFS中文件的垃圾回收。此外，它还处理模式（schema）的变化，例如创建表和列族**。

Each tablet server manages a set of tablets (typically we have somewhere between ten to a thousand tablets per tablet server). The tablet server handles read and write requests to the tablets that it has loaded, and also splits tablets that have grown too large.

**每个Tablet服务器管理一组Tablet（通常每个Tablet服务器有十到一千个Tablet）。Tablet服务器处理对已加载的Tablet的读写请求，并且还会切分太大的Tablet**。


As with many single-master distributed storage systems <a  href="#[17]">[17]</a><a href=" #[21]">[21]</a>, client data does not move through the master: clients communicate directly with tablet servers for reads and writes. Because Bigtable clients do not rely on the master for tablet location information, most clients never communicate with the master. As a result, the master is lightly loaded in practice.

**与许多单个主服务器（single-master）的分布式存储系统<a  href="#[17]">[17]</a><a href=" #[21]">[21]</a>一样，客户端数据不会传输到主服务器（master）：客户端直接与Tablet服务器通信以进行读取和写入数据。由于Bigtable客户端不依赖主服务器（master）获取Tablet的位置信息，所以大多数客户端从不与主服务器（master）通信**。结果，在实践中主服务器（master）是低负载的。

A Bigtable cluster stores a number of tables. Each table consists of a set of tablets, and each tablet contains all data associated with a row range. Initially, each table consists of just one tablet. As a table grows, it is automatically split into multiple tablets, each approximately 100-200 MB in size by default.

**Bigtable集群存储许多表。每个表由一组Tablet组成，并且每个Tablet包含了关联一个行区间的所有数据。最初，每个表格仅包含一个Tablet。随着表的增长，它会自动切分成多个Tablet，默认情况下每个Tablet的大小约为100-200 MB**。

#### 5.1 Tablet Location

We use <u>a three-level hierarchy</u> analogous to that of a B+ tree <a href=" #[10]">[10]</a> to store tablet location information (Figure 4). 

 **我们使用类似于B+树<a href=" #[10]">[10]</a>的三级层次结构来存储Tablet位置信息（图4）**。

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/1577620233604.png" style="zoom:60%;" />

The first level is a file stored in Chubby that contains the location of the root tablet. The root tablet contains the location of all tablets in a special METADATA table. Each METADATA tablet contains the location of a set of user tablets. The root tablet is just the first tablet in the METADATA table, but is treated specially—it is never split—to ensure that the tablet location hierarchy has no more than three levels.

第一级是存储在Chubby中的文件，它包含Root Tablet的位置。 Root Tablet 包含特殊的 METADATA table 中所有Tablet的位置。 每个METADATA Tablet都包含一组 User Tablets 的位置。 Root Tablet只是METADATA table中的第一个Tablet，但经过特殊处理（从不切分），以确保Tablet位置层次结构不超过三个级别。

The METADATA table stores the location of a tablet under a row key that is an encoding of the tablet’s table identifier and its end row. Each METADATA row stores approximately 1KB of data in memory. With a modest limit of 128 MB METADATA tablets, our three-level location scheme is sufficient to address $2^{34}$ tablets (or $2^{61}$ bytes in 128 MB tablets).

METADATA table 存储了某个行键下的Tablet的位置信息，该行键是Tablet表标识符及其最后一行的编码。 每个METADATA行在内存中存储大约1KB的数据。 由于 METADATA Tablet 的 128 MB 这个不大的限制，我们的三级定位方案足以处理 $2^{34}$  个Tablet（或128 MB Tablet中的 $2^{61}$ 字节）。

> <center><strong>译者附</strong></center>
> - 第一级：**Chubby 中的一个文件**
> - 第二级：**METADATA tables**（第一个 METADATA table 比较特殊，所以在图中单独画出，但它其实和其他 METADATA table 都属于第二级，即 METADATA tables = 图示中的1st METADATA Tablet (Root Tablet) + Other METADATA Tablets）
> - 第三级：**User Tables**
>
> METADATA 是一个特殊的 Tablet，其中的第一个 Tablet 称为 Root Tablet 。Root Tablet 和 METADATA 内其他 Tablet 不同之处在于：它永远不会分裂，这样就可以保证 Tablet location 层级不会超过三层。
>
> **三级间的关系**：
>
> - Chubby 中的文件保存了 Root Tablet 的位置
> - Root Tablet 保存了 METADATA Tablet 内所有其他 Tablet 的位置
> - 每个 METADATA Tablet（Root Tablet 除外）保存了一组 User Tables 的位置
>
> METADATA 的每行数据在内存中大约占 1KB。而 METADATA Tablet 的大小限制在 128MB，这种三级位置方案就可以存储高达  128MB = $2^{17} \* $  1KB，即每个 METADATA Tablet 可以指向 $2^{17}$ 个 User Table，每个 User Table 同样是 128MB 的大小话，就有 $2^{17} \* 2^{17} = 2^{34}$ 个 Tablet 。 如果每个 Tablet 128 MB 大小，那总数据量就高达 128MB = $2^{27}$ Byte， $2^{34} \* 2^{27} = 2^{61}$ Byte，即2000PB

The client library caches tablet locations. If the client does not know the location of a tablet, or if it discovers that cached location information is incorrect, then it <u>recursively moves up the tablet location hierarchy</u>. If the client’s cache is empty, the location algorithm requires three network round-trips, including one read from Chubby. If the client’s cache is stale, the location algorithm could take up to six round-trips, because <u>stale cache entries</u> are only discovered upon misses (assuming that METADATA tablets do not move very frequently). Although tablet locations are stored in memory, so no GFS accesses are required, we further reduce this cost in the common case by having the client library prefetch tablet locations: it reads the metadata for more than one tablet whenever it reads the METADATA table.

客户端库缓存Tablet的位置信息。 如果客户端不知道Tablet的位置，或者发现缓存的位置信息不正确，则它将在Tablet位置层级中向上递归[^14]（查找想要的位置信息）。 如果客户的缓存为空，则定位算法需要进行三次网络往返，包括从Chubby中读取一次。 如果客户的缓存过时，则定位算法最多可能需要进行六次往返，因为过时的缓存项仅在未命中时才被发现（假设METADATA Tablet的移动频率不高）。 尽管Tablet位置存储在内存中，所以不需要GFS访问，但在常见情况下，我们通过让客户端库预取Tablet位置来进一步降低了此成本：每当读取METADATA表时，它都会读取一个以上Tablet的元数据。

We also store <u>secondary information</u> in the METADATA table, including a log of all events <u>pertaining to</u> each tablet (such as when a server begins serving it). This information is helpful for debugging and performance analysis.

我们还将辅助信息存储在METADATA表中，包括与每个Tablet有关的所有事件的日志（例如服务器何时开始为其服务）。 此信息有助于调试和性能分析。

#### 5.2 Tablet Assignment

Each tablet is assigned to one tablet server at a time. The master keeps track of the set of live tablet servers, and the current assignment of tablets to tablet servers, including which tablets are unassigned. When a tablet is unassigned, and a tablet server with sufficient room for the tablet is available, the master assigns the tablet by sending a tablet load request to the tablet server.

每个Tablet每次分配到一个Tablet服务器。主服务器跟踪有效的Tablet服务器的集合[^15]以及Tablet到Tablet服务器的当前分配关系，包括未分配的Tablet。当Tablet未分配并且可用的Tablet服务器有足够的空间来容纳Tablet时，主服务器通过向Tablet服务器发送Tablet加载请求来分配Tablet。

Bigtable uses Chubby to keep track of tablet servers. When a tablet server starts, it creates, and acquires <u>an exclusive lock</u> on, a uniquely-named file in a specific Chubby directory. The master monitors this directory (the servers directory) to discover tablet servers. A tablet server stops serving its tablets if it loses its <u>exclusive lock</u>: e.g., due to a network partition that caused the server to lose its Chubby session. (Chubby provides an efficient mechanism that allows a tablet server to check whether it still holds its lock without incurring network traffic.) A tablet server will attempt to reacquire an exclusive lock on its file as long as the file still exists. If the file no longer exists, then the tablet server will never be able to serve again, so <u>it kills itself</u>. Whenever a tablet server terminates (e.g., because the cluster management system is removing the tablet server’s machine from the cluster), it attempts to release its lock so that the master will reassign its tablets more quickly.

**Bigtable使用Chubby来跟踪Tablet服务器**。<u>Tablet服务器启动后，将在特定的Chubby目录中创建一个命名唯一的文件并获这个文件的独占锁。主服务器监控此目录（服务器目录）以发现Tablet服务器。Tablet服务器如果丢失文件的独占锁，则会停止为其Tablet提供服务</u>：例如，由于网络分区导致服务器丢失了Chubby会话。（Chubby提供了一种高效的机制，可让Tablet服务器检查其是否仍然持有独占锁而不会引起网络通信）<u>只要该文件仍然存在，Tablet服务器将尝试重新获取对其文件的独占锁。如果该文件不再存在，则Tablet服务器将永远无法再次提供服务，因此它将自行终止。Tablet服务器终止时（例如，由于集群管理系统正在从集群中删除Tablet服务器的计算机），它将尝试释放它持有的锁，以便主机可以更快地重新分配这个Tablet服务器被分配到的Tablet。</u>

The master is responsible for detecting when a tablet server is no longer serving its tablets, and for reassigning those tablets as soon as possible. To detect when a tablet server is no longer serving its tablets, the master periodically asks each tablet server for the status of its lock. If a tablet server reports that it has lost its lock, or if the master was unable to reach a server during its last several attempts, the master attempts to acquire an exclusive lock on the server’s file. If the master is able to acquire the lock, then Chubby is live and the tablet server is either dead or having trouble reaching Chubby, so the master ensures that the tablet server can never serve again by deleting its server file. Once a server’s file has been deleted, the master can move all the tablets that were previously assigned to that server into the set of unassigned tablets. To ensure that a Bigtable cluster is not vulnerable to networking issues between the master and Chubby, the master kills itself if its Chubby session expires. However, as described above, master failures do not change the assignment of tablets to tablet servers.

主服务器（master ）负责检测Tablet服务器何时不再为其Tablet提供服务，并负责尽快重新分配这些Tablet。为了检测Tablet服务器何时不再为其Tablet提供服务，主服务器（master ）会定期向每个Tablet服务器询问其锁的状态。如果Tablet服务器报告其锁已丢失，或者主服务器（master ）在最后几次尝试期间都无法访问服务器，则主服务器（master ）将尝试获取Chubby所在的服务器的Chubby目录下的文件独占锁。如果主服务器（master ）能够获取锁，则Chubby处于存活的状态，以及如果Tablet服务器死机或者无法访问Chubby，那么主服务器（master ）通过删除Chubby所在的服务器的Chubby目录下的文件来确保Tablet服务器永远不会再次服务。删除Chubby所在的服务器的Chubby目录下的文件后，主服务器（master ）可以将以前分配给处于无效状态的Tablet服务器的所有Tablet移至未分配的Tablet集合中。为了确保Bigtable集群不会受到主服务器（master ）和Chubby之间的网络问题的影响，如果主服务器的Chubby会话到期，则主服务器会自行杀死。但是，如上所述，主服务器（master）设备故障不会更改Tablet到Tablet服务器的分配关系。

When a master is started by the cluster management system, it needs to discover the current tablet assignments before it can change them. The master executes the following steps at startup. (1) The master grabs a unique master lock in Chubby, which prevents concurrent master instantiations. (2) The master scans the servers directory in Chubby to find the live servers. (3) The master communicates with every live tablet server to discover what tablets are already assigned to each server. (4) The master scans the METADATA table to learn the set of tablets. Whenever this scan encounters a tablet that is not already assigned, the master adds the tablet to the set of unassigned tablets, which makesthe tablet eligible for tablet assignment.

当主服务器由集群管理系统启动时，它需要先发现当前的Tablet分配关系，然后才能更改它们。**主服务器在启动时执行以下步骤**。 

（1）主服务器在Chubby中获取唯一的主服务器锁，这可以防止并发的主服务器实例化。 
（2）主服务器扫描Chubby中的服务器目录以找到有效的Tablet服务器。
（3）主服务器与每个有效的Tablet服务器通信，以发现已分配给每个服务器的Tablet。 
（4）主服务器扫描METADATA table获知Tablet集合。每当此扫描遇到尚未分配的Tablet时，主服务器就会将该Tablet添加到未分配的Tablet集合中，这使该Tablet有资格进行Tablet分配。

One complication is that the scan of the METADATA table cannot happen until the METADATA tablets have been assigned. Therefore, before starting this scan (step 4), the master adds the root tablet to the set of unassigned tablets if an assignment for the root tablet was not discovered during step 3. This addition ensures that the root tablet will be assigned. Because the root tablet contains the names of all METADATA tablets, the master knows about all of them after it has scanned the root tablet.

一种复杂的情况是，在分配 METADATA Tablet 之前，无法进行 METADATA table 的扫描。因此，在开始此扫描（步骤4）之前，如果在步骤3中未找到针对Root Tablet的分配，则主服务器会将Root Tablet添加到未分配Tablet的集合中。此添加操作确保了将对Root Tablet进行分配。由于Root Tablet包含所有METADATA Tablet的名称，因此主服务器在扫描了Root Tablet之后便知道了所有这些名称。

><center><strong>译者附</strong></center>
>在扫描 METADATA Tablet 之前，必须保证 METADATA table 自己已经被分配出去了。因此，如果在步骤 3 中发现 Root Tablet 还没有被分配出去，那主服务器就要先将它放到 未分配 Tablet 集合，然后去执行步骤 4。 这样就保证了 Root Tablet 将会被分配出去。

The set of existing tablets only changes when a table is created or deleted, two existing tablets are merged to form one larger tablet, or an existing tablet is split into two smaller tablets. The master is able to keep track of these changes because it initiates all but the last. Tablet splits are treated specially since they are initiated by a tablet server. The tablet server commits the split by recording information for the new tablet in the METADATA table. When the split has committed, it notifies the master. In case the split notification is lost (either because the tablet server or the master died), the master detects the new tablet when it asks a tablet server to load the tablet that has now split. The tablet server will notify the master of the split, because the tablet entry it finds in the METADATA table will specify only a portion of the tablet that the master asked it to load.

现有的Tablet集合，只有在以下情形才会发生改变：

（1）当一个Tablet被创建或删除；

（2）对两个现有的Tablet进行合并得到一个更大的Tablet；

（3）一个现有的tablet被切分成两个较小的Tablet。

主服务器能够跟踪这些变化，因为它负责启动除最后一次以外的所有操作。Tablet切分操作是由Tablet服务器启动的，因此受到特殊对待。Tablet服务器通过在 METADATA table 中记录新Tablet的信息来提交切分操作。提交切分操作后，它将通知主服务器。万一切分事件通知丢失（由于Tablet服务器或主服务器死机），则主服务器在要求Tablet服务器加载现在已切分的Tablet时，会检测到新的Tablet。Tablet服务器会把切分操作通知主服务器，因为它在 METADATA table 中查到的Tablet条目将仅指定一部分的Tablet，而Tablet是主服务器要求Tablet服务器加载的。

><center><strong>译者附</strong></center>
>如果通知丢失（由于Tablet服务器或主服务器挂掉），主服务器会在它下次要求一个Tablet server 加载 Tablet 时发现。这个 Tablet 服务器会将这次切分事件通知给主服务器，因为“Tablet服务器通过在 METADATA table 中记录新Tablet的信息来提交切分操作。提交切分操作后，它将通知主服务器”。所以它在 METADATA table 中发现的 Tablet 项只覆盖主服务器要求它加载的 Tablet 的了一部分。

#### 5.3 Tablet Serving

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/1577705846192.png" alt="1122" style="zoom:50%;" />

The persistent state of a tablet is stored in GFS, as illustrated in Figure 5. Updates are committed to a commit log that stores redo records. Of these updates, the recently committed ones are stored in memory in a sorted buffer called a memtable; the older updates are stored in a sequence of SSTables. To recover a tablet, a tablet server reads its metadata from the METADATA table. This metadata contains the list of SSTables that comprise a tablet and a set of a redo points, which are pointers into any commit logs that may contain data for the tablet. The server reads the indices of the SSTables into memory and reconstructs the memtable by applying all of the updates that have committed since the redo points.

Tablet的持久化状态存储在GFS中，如图5所示。更新被提交（commit）到一个提交日志（commit log），这些日志存储着重做的记录（redo records）。在这些更新当中，最近提交的更新被存储到内存当中的一个被称为memtable的排序缓冲区，比较老的更新被存储在一系列SSTable中。为了恢复Tablet，Tablet服务器从 METADATA table 读取其元数据。该元数据包含SSTables列表，该SSTables包含一个Tablet和一个重做点（redo point）的集合 ，这些重做点（redo point）是指向任何可能包含该Tablet数据的提交日志的指针。服务器将SSTables的索引读入内存，并通过应用自重做点以来已提交的所有更新来重建memtable。

When a write operation arrives at a tablet server, the server checks that it is well-formed, and that the sender is authorized to perform the mutation. Authorization is performed by reading the list of permitted writers from a Chubby file (which is almost always a hit in the Chubby client cache). A valid mutation is written to the commit log. Group commit is used to improve the throughput of lots of small mutations <a href=" #[13]">[13]</a><a href="#[16]">[16]</a>. After the write has been committed, its contents are inserted into the memtable.

当**写操作**到达Tablet服务器时，服务器将检查其格式是否正确，以及发送方是否有权执行这个更改（mutation）。通过从Chubby文件中读取允许的作者列表来执行授权（这在Chubby客户端缓存中几乎总是命中）。有效的更改（mutation）将写入提交日志（commit log）。整组提交（group commit）用于提高许多小更改的吞吐量 <a href=" #[13]">[13]</a><a href="#[16]">[16]</a>。提交写入后，其内容将插入到memtable中。

When a read operation arrives at a tablet server, it is similarly checked for well-formedness and proper authorization. A valid read operation is executed on a merged view of the sequence of SSTables and the memtable. Since the SSTables and the memtable are lexicographically sorted data structures, the merged view can be formed efficiently. Incoming read and write operations can continue while tablets are split and merged.

当**读操作**到达Tablet服务器时，同样会检查其格式是否正确以及是否获得适当的授权。在SSTables和memtable序列的合并视图上执行有效的读取操作。由于SSTables和memtable是按字典顺序排序的数据结构，因此可以有效地形成合并视图。切分和合并Tablet时，传入的读写操作可以继续。

#### 5.4 Compactions

As write operations execute, the size of the memtable increases. When the memtable size reaches a threshold, the memtable is frozen, a new memtable is created, and the frozen memtable is converted to an SSTable and written to GFS. This minor compaction process has two goals: it shrinks the memory usage of the tablet server, and it reduces the amount of data that has to be read from the commit log during recovery if this server dies. Incoming read and write operations can continue while compactions occur.

随着写操作的执行，memtable的大小增加。 当memtable大小达到阈值时，该memtable被冻结，创建新的memtable，并将冻结的memtable转换为SSTable并写入GFS。 这个次要的压缩过程有两个目标：减少Tablet服务器的内存使用量，并且如果该服务器死机，那么在恢复期间，压缩将减少必须从提交日志中读取的数据量。 发生压缩时，传入的读取和写入操作可以继续。

Every minor compaction creates a new SSTable. If this behavior continued unchecked, read operations might need to merge updates from an arbitrary number of SSTables. Instead, we bound the number of such files by periodically executing a merging compaction in the background. A merging compaction reads the contents of a few SSTables and the memtable, and writes out a new SSTable. The input SSTables and memtable can be discarded as soon as the compaction has finished. A merging compaction that rewrites all SSTables into exactly one SSTable is called a major compaction.

每次**minor compaction**（小型压缩）都会创建一个新的SSTable。 如果此行为持续未经检查，则读操作可能需要合并任意数量的SSTables中的更新。 相反，我们通过在后台定期执行**merging compaction**（合并压缩）来限制此类文件的数量。 合并压缩读取一些SSTables和memtable的内容，并输出新的SSTable。 压缩完成后，可以立即丢弃输入的SSTables和memtable。 将所有SSTable重写为一个SSTable的合并压缩称为**major compaction**（大型压缩）。

SSTables produced by non-major compactions can contain special deletion entries that suppress deleted data in older SSTables that are still live. A major compaction, on the other hand, produces an SSTable that contains no deletion information or deleted data. Bigtable cycles through all of its tablets and regularly applies major compactions to them. These major compactions allow Bigtable to reclaim resources used by deleted data, and also allow it to ensure that deleted data disappears from the system in a timely fashion, which is important for services that store sensitive data.

<u>由 **non-major compaction**（非大型压缩）产生的SSTable可以包含特殊的删除条目（这里删除条目视为存储着：起到删除功能的指令，然而执行指令在：major compaction阶段），这些条目用于删除掉仍然存在于旧SSTable中逻辑上视为已删除的数据（逻辑上视为已删除的数据：客户端无法读取这些数据，即对客户端不可见，然而磁盘上这些数据还在。逻辑上已经不存在，物理上还存在）。 另一方面，major compaction（大型压缩）会产生一个SSTable，该表不包含删除信息或已删除的数据。  Bigtable会遍历其所有Tablet，并定期对其应用major compaction（大型压缩）。 这些major compaction（大型压缩）使Bigtable可以回收已删除数据所使用的资源，还可以确保Bigtable及时地从系统中删除已删除的数据，这对于存储敏感数据的服务很重要</u>。

### 6 Refinements

The implementation described in the previous section required a number of refinements to achieve the high performance, availability, and reliability required by our users. This section describes portions of the implementation in more detail in order to highlight these refinements.
上一节中描述的实现需要大量改进，以实现我们的用户所需的高性能，可用性和可靠性。 本节将更详细地描述实现的各个部分，以突出显示这些改进。

#### Locality groups

Clients can group multiple column families together into a locality group. A separate SSTable is generated for each locality group in each tablet. Segregating column families that are not typically accessed together into separate locality groups enables more efficient reads. For example, page metadata in Webtable (such as language and checksums) can be in one locality group, and the contents of the page can be in a different group: an application that wants to read the metadata does not need to read through all of the page contents.

<u>客户端可以将多个列族组合到一个 locality group 中。为每个Tablet中的每个位置组生成一个单独的SSTable。将通常不一起访问的列族隔离到单独的 locality group 中，可以提高读取效率</u>。例如，Webtable中的页面元数据（例如语言以及校验和）可以在一个 locality group 中，而页面的内容可以在另一个组中：想要读取元数据的应用程序不需要通读所有页面内容。


In addition, some useful tuning parameters can be specified on a per-locality group basis. For example, a locality group can be declared to be in-memory. SSTables for in-memory locality groups are loaded lazily into the memory of the tablet server. Once loaded, column families that belong to such locality groups can be read without accessing the disk. This feature is useful for small pieces of data that are accessed frequently: we use it internally for the location column family in the METADATA table.

<u>另外，可以在每个 locality group 的基础上指定一些有用的调整参数。例如，可以将一个 locality group 声明为内存中。内存中 locality group 的SSTable延迟加载到Tablet服务器的内存中。一旦加载后，无需访问磁盘即可读取属于此类 locality group 的列族。此功能对经常访问的小数据很有用：我们在内部将其用于METADATA表中的location列族</u>。

><center><strong>译者附</strong></center>
>主要是根据数据访问的局部性原理与在操作系统中内存页的缓存算法是同理。

#### Compression

Clients can control whether or not the SSTables for a locality group are compressed, and if so, which compression format is used. The user-specified compression format is applied to each SSTable block (whose size is controllable via a locality group specific tuning parameter). Although we lose some space by compressing each block separately, we benefit in that small portions of an SSTable can be read without decompressing the entire file. Many clients use a two-pass custom compression scheme. The first pass uses Bentley and McIlroy’s scheme <a href=" #[6]">[6]</a>, which compresses long common strings across a large window. The second pass uses a fast compression algorithm that looks for repetitions in a small 16 KB window of the data. Both compression passes are very fast—they encode at 100–200 MB/s, and decode at 400–1000 MB/s on modern machines. Even though we emphasized speed instead of space reduction when choosing our compression algorithms, this two-pass compression scheme does surprisingly well.

客户端可以控制是否压缩 locality group 的SSTable，以及如果压缩，则使用哪种压缩格式。<u>用户指定的压缩格式将应用于每个SSTable块（其大小可通过 locality group 的特定的调整参数来控制）。尽管我们通过分别压缩每个块而损失了一些空间，但我们的好处是因为：可以读取SSTable的一小部分而无需解压缩整个文件。许多客户端使用两阶段自定义压缩方案</u>。第一阶段使用Bentley和McIlroy的方案<a href=" #[6]">[6]</a>，该方案在一个大窗口中压缩长的公共字符串。第二阶段使用快速压缩算法，该算法在一个小的16 KB数据窗口中查找重复项。两种压缩过程都非常快——在现代机器上，它们的编码速度为100-200 MB/s，解码速度为 400-1000 MB/s。尽管在选择压缩算法时我们强调速度而不是减少空间，但这种两阶段压缩方案的效果出奇地好。


For example, in Webtable, we use this compression scheme to store Web page contents. In one experiment,we stored a large number of documents in a compressed locality group. For the purposes of the experiment, we limited ourselves to one version of each document instead of storing all versions available to us. The scheme achieved a 10-to-1 reduction in space. This is much better than typical Gzip reductions of 3-to-1 or 4-to-1 on HTML pages because of the way Webtable rows are laid out: all pages from a single host are stored close to each other. This allows the Bentley-McIlroy algorithm to identify large amounts of shared boilerplate in pages from the same host. Many applications, not just Webtable, choose their row names so that similar data ends up clustered, and therefore achieve very good compression ratios. Compression ratios get even better when we store multiple versions of the same value in Bigtable.

例如，在Webtable中，我们使用这种压缩方案来存储Web页面内容。在一个实验中，我们将大量文档存储在一个压缩的 locality group 中。为了进行实验，我们将自己限制为每个文档的一个版本，而不是存储所有可用的版本。该方案使空间减少了10比1。由于Webtable行的布局方式，这比HTML页面上通常的Gzip压缩（3比1或4比1）要好得多：<u>来自单个主机的所有页面都存储得彼此靠近。这使Bentley-McIlroy算法可以识别来自同一主机的页面中的大量共享样板。许多应用程序（不仅是Webtable）都选择其行名致使相似的数据最终聚集在一起，因此实现了很好的压缩率</u>。当我们在Bigtable中存储相同值的多个版本时，压缩率甚至会更高。

#### Caching for read performance

To improve read performance, tablet servers use two levels of caching. The Scan Cache is a higher-level cache that caches the key-value pairs returned by the SSTable interface to the tablet server code. The Block Cache is a lower-level cache that caches SSTables blocks that were read from GFS. The Scan Cache is most useful for applications that tend to read the same data repeatedly. The Block Cache is useful for applications that tend to read data that is close to the data they recently read (e.g.,  sequential reads, or random reads of different columns in the same locality group within a hot row).

为了提高读取性能，Tablet服务器使用两个级别的缓存。 **Scan Cache**是一个更高层次的缓存，它将SSTable接口返回的键值对缓存到Tablet服务器代码。 **Block Cache**是较低层次的缓存，它缓存从GFS读取的SSTables块。 <u>Scan Cache对于倾向于重复读取相同数据的应用程序最有用。 对于倾向于读取与其最近读取的数据接近的数据的应用程序（例如，顺序读取或对热点行中同一个 locality group 中不同列的随机读取），Block Cache非常有用。</u>

#### Bloom filters

As described in Section 5.3, a read operation has to read from all SSTables that make up the state of a tablet. If these SSTables are not in memory, we may end up doing many disk accesses. We reduce the number of accesses by allowing clients to specify that Bloom filters <a href=" #[7]">[7]</a> should be created for SSTables in a particular locality group. A Bloom filter allows us to ask whether an SSTable might contain any data for a specified row/column pair. For certain applications, a small amount of tablet server memory used for storing Bloom filters drastically reduces the number of disk seeks required for read operations. Our use of Bloom filters also implies that most lookups for non-existent rows or columns do not need to touch disk.

如第5.3节所述，读取操作必须从构成Tablet状态的所有SSTable中读取。如果这些SSTable不在内存中，我们可能最终会进行许多磁盘访问。通过允许客户端指定应为特定 locality group 中的SSTable创建Bloom过滤器<a href=" #[7]">[7]</a>，我们减少了访问次数。 布隆过滤器允许我们询问SSTable是否可以包含指定行/列对的任何数据。对于某些应用程序，用于存储布隆过滤器的少量Tablet服务器的内存会大大减少读取操作所需的磁盘搜寻次数。 我们对Bloom过滤器的使用还意味着对于不存在的行或列的大多数查找都不需要接触磁盘。

#### Commit-log implementation

If we kept the commit log for each tablet in a separate log file, a very large number of files would be written concurrently in GFS. Depending on the underlying file system implementation on each GFS server, these writes could cause a large number of disk seeks to write to the different physical log files. In addition, having separate log files per tablet also reduces the effectiveness of the group commit optimization, since groups would tend to be smaller. To fix these issues, we append mutations to a single commit log per tablet server, co-mingling mutations for different tablets in the same physical log file <a href=" #[18]">[18]</a><a href=" #[20]">[20]</a>.

如果我们将每个Tablet的提交日志保存在单独的日志文件中，则会在GFS中同时写入大量文件。根据每个GFS服务器上基础文件系统的实现，这些写操作可能导致大量磁盘搜索以写到不同的物理日志文件。此外，<u>每个Tablet使用单独的日志文件还会降低整组提交（ group commit ）优化的效率，因为组的规模往往较小。为了解决这些问题，我们将数据的变化记录（mutation）追加到每个Tablet服务器的单个提交日志中，将不同Tablet的变化记录（mutation）混合在同一物理日志文件中</u> <a href=" #[18]">[18]</a><a href=" #[20]">[20]</a>。

Using one log provides significant performance benefits during normal operation, but it complicates recovery. When a tablet server dies, the tablets that it served will be moved to a large number of other tablet servers: each server typically loads a small number of the original server’s tablets. To recover the state for a tablet, the new tablet server needs to reapply the mutations for that tablet from the commit log written by the original tablet server. However, the mutations for these tablets were co-mingled in the same physical log file. One approach would be for each new tablet server to read this full commit log file and apply just the entries needed for the tablets it needs to recover. However, under such a scheme, if 100 machines were each assigned a single tablet from a failed tablet server, then the log file would be read 100 times (once by each server).

在正常操作期间，使用一个日志可以显著提高性能，但是会使恢复变得复杂。当Tablet服务器死亡时，其所服务的Tablet将被移至大量其他Tablet服务器：每个服务器通常会加载少量原始服务器的Tablet。<u>要恢复Tablet的状态，新的Tablet服务器需要从原始Tablet服务器写入的提交日志中重新应用该Tablet的变化日志。但是，这些Tablet的变化日志被混合在同一物理日志文件中</u>。一种方法是让每个新的Tablet服务器读取此完整的提交日志文件，并仅应用其需要恢复的Tablet所需的条目。但是，在这种方案下，如果从故障的Tablet服务器中分别为100台计算机分配了一个Tablet，那么日志文件将被读取100次（每个服务器一次）。

We avoid duplicating log reads by first sorting the commit log entries in order of the keys
<htable; row name; log sequence number>. In the sorted output, all mutations for a particular tablet are contiguous and can therefore be read efficiently with one disk seek followed by a sequential read. To parallelize the sorting, we partition the log file into 64 MB segments, and sort each segment in parallel on different tablet servers. This sorting process is coordinated by the master and is initiated when a tablet server indicates that it needs to recover mutations from some commit log file.

我们<u>通过以 `(table; row name; log sequence number)` 为键对提交日志条目进行排序来避免重复的日志读取。在排序的输出中，特定Tablet的所有mutation（数据的变化）都是连续的，因此可以通过一个磁盘搜索有效读取，然后顺序读取。为了并行化排序，我们将日志文件划分为64 MB的分段，然后在不同的Tablet服务器上并行地对每个分段进行排序。</u>此排序过程由主服务器（master）协调，并在Tablet服务器指示需要从某些提交日志文件中恢复mutation（数据的更改）时启动。

Writing commit logs to GFS sometimes causes performance hiccups for a variety of reasons (e.g., a GFS server machine involved in the write crashes, or the network paths traversed to reach the particular set of three GFS  servers is suffering network congestion, or is heavily loaded). To protect mutations from GFS latency spikes, each tablet server actually has two log writing threads, each writing to its own log file; only one of these two threads is actively in use at a time. If writes to the active log file are performing poorly, the log file writing is switched to the other thread, and mutations that are in the commit log queue are written by the newly active log writing thread. Log entries contain sequence numbers to allow the recovery process to elide duplicated entries resulting from this log switching process.

<u>将提交日志写入GFS有时会由于各种原因而导致性能下降</u>（例如，写入时发生崩溃的GFS服务器计算机，或用来穿越以便到达特定的三个GFS服务器集的网络路径正遭受网络拥塞或负载过重） 。<u>为了保护变化免受GFS延迟高峰的影响，每个Tablet服务器实际上都有两个日志写入线程（一个是被激活也就是正在使用的线程，一个是备用线程），每个线程都写入自己的日志文件。一次仅积极使用这两个线程之一。如果对激活的（active 有些人翻译：活跃的）日志文件的写入性能不佳，则日志文件的写入将切换到另一个线程，并且提交日志队列中的数据变化记录将由新激活的日志写线程进行写入。日志条目包含序列号，以允许恢复过程清除此日志切换过程产生的重复条目</u>。

#### Speeding up tablet recovery

If the master moves a tablet from one tablet server to another, the source tablet server first does a minor compaction on that tablet. This compaction reduces recovery time by reducing the amount of uncompacted state in the tablet server’s commit log. After finishing this compaction, the tablet server stops serving the tablet. Before it actually unloads the tablet, the tablet server does another (usually very fast) minor compaction to eliminate any remaining uncompacted state in the tablet server’s log that arrived while the first minor compaction was being performed. After this second minor compaction is complete, the tablet can be loaded on another tablet server without requiring any recovery of log entries.

如果主服务器（master）将 Tablet 从一台 Tablet 服务器移动到另一台 Tablet 服务器，则源 Tablet 服务器首先对该 Tablet 进行 minor compaction（小型压缩）。 这种压缩通过减少 Tablet 服务器的提交日志中未压缩状态的数量来减少恢复时间。 完成这次压缩后，Tablet 服务器将停止为 Tablet 提供服务。 在实际卸载 Tablet 之前，Tablet 服务器会进行另一次（通常非常快） minor compaction（小型压缩）来消除执行第一次 minor compaction（小型压缩）时到达 Tablet 服务器的日志当中任何剩余的未压缩状态。 在完成第二次  minor compaction（小型压缩）后，可将 Tablet 加载到另一台 Tablet 服务器上，而无需恢复日志条目。

#### Exploiting immutability

Besides the SSTable caches, various other parts of the Bigtable system have been simplified by the fact that all of the SSTables that we generate are immutable. For example, we do not need any synchronization of accesses to the file system when reading from SSTables. As a result, concurrency control over rows can be implemented very efficiently. The only mutable data structure that is accessed by both reads and writes is the memtable. To reduce contention during reads of the memtable, we make each memtable row copy-on-write and allow reads and writes to proceed in parallel.

<u>除了SSTable缓存外，我们生成的所有SSTable都是不可变的，从而简化了Bigtable系统的其他各个部分</u>。例如，当从SSTables读取数据时，我们<u>不需要对文件系统的访问进行任何同步。结果，可以非常有效地实现对行的并发控制。读取和写入均访问的唯一可变数据结构是memtable。为了减少在读取memtable期间的竞争，我们使每个memtable的行使用写时复制的策略，并允许读取和写入并行进行</u>。

Since SSTables are immutable, the problem of permanently removing deleted data is transformed to garbage collecting obsolete SSTables. Each tablet’s SSTables are registered in the METADATA table. The master removes obsolete SSTables as <u>a mark-and-sweep garbage collection</u> <a href=" #[25]">[25]</a> over the set of SSTables, where the METADATA table contains the set of roots. Finally, the immutability of SSTables enables us to split tablets quickly. Instead of generating a new set of SSTables for each child tablet, we let the child tablets share the SSTables of the parent tablet. 

<u>由于SSTable是不可变的，因此永久删除已删除数据（前面讲过的发出删除指令，但未被执行的数据）的问题被转换为垃圾收集过期的SSTable</u>。每个Tablet的SSTables都注册在 METADATA table 中。主服务器（master）删除过时的SSTables作为SSTables集合上的标记再清除式的垃圾收集<a href=" #[25]">[25]</a>，其中 METADATA table 包含根集合（按照前文：METADATA table 记录了这些 SSTable 的对应的 tablet 的 root）。<u>最后，SSTables的不变性使我们能够快速拆分Tablet。我们不必为每个子 Tablet 生成一组新的SSTable，而是让子 Tablet 共享 Tablet 的SSTable。</u>

### 7 Performance Evaluation

We set up a Bigtable cluster with N tablet servers to measure the performance and scalability of Bigtable as N is varied. The tablet servers were configured to use 1 GB of memory and to write to a GFS cell consisting of 1786 machines with two 400 GB IDE hard drives each. N client machines generated the Bigtable load used for these tests. (We used the same number of clients as tablet servers to ensure that clients were never a bottleneck.) Each machine had two dual-core Opteron 2 GHz chips, enough physical memory to hold the working set of all running processes, and a single gigabit Ethernet link. The machines were arranged in a two-level tree-shaped switched network with approximately 100-200 Gbps of aggregate bandwidth available at the root. All of the machines were in the same hosting facility and therefore the round-trip time between any pair of machines was less than a millisecond. 

我们建立了一个由N台Tablet服务器组成的Bigtable集群，以随着 N 的变化来衡量Bigtable的性能和可扩展性。Tablet服务器配置为使用1 GB的内存，并写入由1786台计算机组成的GFS单元，每台计算机具有两个400 GB的IDE硬盘驱动器。 N个客户端计算机生成了用于这些测试的Bigtable负载。（我们使用与Tablet服务器相同数量的客户端，以确保客户端永远不会成为瓶颈）每台机器都具有两个双核Opteron 2 GHz芯片，足够的物理内存来容纳所有正在运行的进程的工作集以及一个 1Gbp/s 以太网链路。这些机器被安排在两级树形交换网络（two-level tree-shaped switched network）中，网络根节点大约有100-200 Gbps的总带宽。所有机器都位于同一托管设施中，因此任何两对机器之间的往返时间均不到一毫秒。

The tablet servers and master, test clients, and GFS servers all ran on the same set of machines. Every machine ran a GFS server. Some of the machines also ran either a tablet server, or a client process, or processes from other jobs that were using <u>the pool</u> at the same time as these experiments. 

Tablet服务器以及主服务器，测试客户端和GFS服务器都在同一组计算机上运行。每台机器都运行GFS服务器。其中一些机器还运行了Tablet服务器或客户端进程，或者运行了与这些实验同时使用这些机器池（根据本文第四节第一段推测 “the pool” 翻译为：机器池）的其他作业的进程。

R is the distinct number of Bigtable row keys involved in the test. R was chosen so that each benchmark read or wrote approximately 1 GB of data per tablet server.

R 是测试中涉及的Bigtable不重复行键的数量。选择R是为了使每个基准测试中每个Tablet服务器读取或写入大约1 GB的数据。


The sequential write benchmark used row keys with names 0 to R - 1. This space of row keys was partitioned into 10N equal-sized ranges. These ranges were assigned to the N clients by a central scheduler that as signed the next available range to a client as soon as the client finished processing the previous range assigned to it. This dynamic assignment helped mitigate the effects of performance variations caused by other processes running on the client machines. We wrote a single string under each row key. Each string was generated randomly and was therefore uncompressible. In addition, strings under different row key were distinct, so no cross-row compression was possible. The random write benchmark was similar except that the row key was hashed modulo R immediately before writing so that the write load was spread roughly uniformly across the entire row space for the entire duration of the benchmark.

**顺序写基准测试**使用名称为 0 到 R - 1 的行键。此行键空间被划分为 10N 个相等大小的区间。这些区间由中央调度程序分配给N个客户端，该中央调度程序在客户端完成对分配给它的先前区间的处理后立即将下一个可用区间分配给客户端。这种动态分配有助于减轻由客户端计算机上运行的其他进程引起的性能变化的影响。我们在每个行键下写了一个字符串。每个字符串都是随机生成的，因此不可压缩的。另外，不同行键下的字符串是不同的，因此不可能进行跨行压缩。**随机写基准测试**类似于顺序写基准测试，不同的是在写入之前立即对行密钥进行了模R哈希运算，以便在基准测试的整个期间，写入负载大致均匀地分布在整个行空间中。

The sequential read benchmark generated row keys in exactly the same way as the sequential write benchmark,  but instead of writing under the row key, it read the string stored under the row key (which was written by an earlier invocation of the sequential write benchmark). Similarly, the random read benchmark <u>shadowed</u> the operation of the random write benchmark.

**顺序读基准测试**产生的行密钥与顺序写入基准完全相同，但它不是在行密钥下写入，而是读取存储在行密钥下的字符串（该字符串是由顺序写基准测试的较早时候调用写入的） 。同样，**随机读基准测试**与随机写基准测试的操作一样。

The scan benchmark is similar to the sequential read benchmark, but uses support provided by the Bigtable API for scanning over all values in a row range. Using a scan reduces the number of RPCs executed by the benchmark since a single RPC fetches a large sequence of values from a tablet server.

**扫描基准测试**（scan benchmark）类似于顺序读基准测试，但是使用Bigtable API提供的支持来扫描行区间内的所有值。使用扫描减少了基准测试执行的RPC数量，因为单个RPC从Tablet服务器中提取了大量的值。

The random reads (mem) benchmark is similar to the random read benchmark, but the locality group that contains the benchmark data is marked as in-memory, and therefore the reads are satisfied from the tablet server’s memory instead of requiring a GFS read. For just this benchmark, we reduced the amount of data per tablet server from 1 GB to 100 MB so that it would <u>fit comfortably in the memory</u> available to the tablet server. 

**随机读（mem）基准测试**类似于**随机读基准测试**，但是包含基准数据的 locality group 被标记为内存中，因此可以从Tablet服务器的内存中读取数据，而无需进行GFS读取。<u>对于该基准测试，我们将每个Tablet服务器的数据量从1 GB减少到100 MB，以便可以合适地容纳在Tablet服务器可用的内存中</u>。

![1577707829591](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/1577707829591.png)

Figure 6 shows two views on the performance of our benchmarks when reading and writing 1000-byte values to Bigtable. The table shows the number of operations per second per tablet server; the graph shows the aggregate number of operations per second. 

图6显示了在向 Bigtable 读取和写入 1000MB/S 时基准测试性能的两个视图。该表显示了每台Tablet服务器每秒的操作数；该图显示了每秒的总操作数。

#### Single tablet-server performance

Let us first consider performance with just one tablet server. Random reads are slower than all other operations by an order of magnitude or more. Each random read involves the transfer of a 64 KB SSTable block over the network from GFS to a tablet server, out of which only a single 1000-byte value is used. The tablet server executes approximately 1200 reads per second, which translates into approximately 75 MB/s of data read from GFS. This bandwidth is enough to <u>saturate the tablet server CPUs</u> because of overheads in our networking stack, SSTable parsing, and Bigtable code, and is also almost enough to saturate the network links used in our system. Most Bigtable applications with this type of an access pattern reduce the block size to a smaller value, typically 8KB.

<u>首先让我们考虑一台Tablet服务器的性能。**随机读取**的速度比所有其他操作慢一个数量级或更多</u>。每次随机读取都涉及通过网络将64 KB SSTable块从GFS传输到Tablet服务器，其中仅使用一个1000字节的值。Tablet服务器每秒执行大约1200次读取，这意味着从GFS读取的数据大约为75 MB/s (1200 * 64 KB / 1024 = 75MB/s)。由于网络堆栈，SSTable解析和Bigtable代码的开销，该带宽足以使 Tablet 服务器的CPU饱和，也几乎足以使系统中使用的网络链路饱和。<u>具有这种访问模式的大多数Bigtable应用程序将块大小减小为一个较小的值</u>，通常为8KB。

Random reads from memory are much faster since each 1000-byte read is satisfied from the tablet server’s local memory without fetching a large 64 KB block from GFS.

从**内存中进行随机读取**的速度要快得多，因为每次从Tablet服务器的本地内存读取 1000B 即可满足需要，而无需从GFS提取大的 64 KB块。

Random and sequential writes perform better than random reads since each tablet server appends all incoming writes to a single commit log and uses group commit to stream these writes efficiently to GFS. There is no significant difference between the performance of random writes and sequential writes; in both cases, all writes to the tablet server are recorded in the same commit log. 

**随机和顺序写入**的性能要优于随机读取，<u>因为每个Tablet服务器会将所有传入的写入都追加到单个提交日志中，并使用整组提交（group commit）将这些写入高效地流式传输到GFS</u>。随机写入和顺序写入的性能之间没有显着差异。在这两种情况下，对Tablet服务器的所有写入都记录在同一提交日志中。

Sequential reads perform better than random reads since every 64 KB SSTable block that is fetched from GFS is stored into our block cache, where it is used to serve the next 64 read requests. 

**顺序读取**的性能要优于随机读取，因为从GFS提取的每个64 KB SSTable块都存储在我们的块缓存中，用于满足接下来的64个读取请求。

Scans are even faster since the tablet server can return a large number of values in response to a single client RPC, and therefore RPC overhead is amortized over a large number of values. 

由于Tablet服务器可以在响应单个客户端RPC时返回大量值，因此 **scan** 速度甚至更快，因此RPC开销将在大量值上摊销。

#### Scaling

Aggregate throughput increases dramatically, by over a factor of a hundred, as we increase the number of tablet servers in the system from 1 to 500. For example, the performance of random reads from memory increases by almost a factor of 300 as the number of tablet server increases by a factor of 500. This behavior occurs because the bottleneck on performance for this benchmark is the individual tablet server CPU. 

随着我们将系统中Tablet服务器的数量从1个增加到500个，总的吞吐量急剧增加了一百倍。例如，随着内存数量的增加，从内存中随机读取的性能几乎提高了300倍。Tablet服务器增加了500倍。之所以发生这种现象，是因为该基准测试的性能瓶颈是各个Tablet服务器CPU。


However, performance does not increase linearly. For most benchmarks, there is a significant drop in per-server throughput when going from 1 to 50 tablet servers. This drop is caused by imbalance in load in multiple server configurations, often due to other processes contending for CPU and network. Our load balancing algorithm attempts to deal with this imbalance, but cannot do a perfect job for two main reasons: rebalancing is throttled to reduce the number of tablet movements (a tablet is unavailable for a short time, typically less than one second, when it is moved), and the load generated by our benchmarks shifts around as the benchmark progresses. 

但是，性能不会线性增加。对于大多数基准测试，当从1台Tablet服务器增加到50台Tablet服务器时，每台服务器的吞吐量将大幅下降。这种下降是由于多个服务器配置中的负载不平衡而引起的，通常是由于其他争用CPU和网络的进程所致。我们的负载平衡算法试图解决这种不平衡问题，但由于两个主要原因而无法做到完美：限制重新平衡以减少Tablet的移动次数（Tablet在短时间内无法使用，通常少于一秒钟，移动），并且随着基准测试的进行，由基准测试产生的负载也会发生变化。

The random read benchmark shows the worst scaling (an increase in aggregate throughput by only a factor of 100 for a 500-fold increase in number of servers). This behavior occurs because (as explained above) we transfer one large 64KB block over the network for every 1000- byte read. This transfer saturates various shared 1 Gigabit links in our network and as a result, the per-server throughput drops significantly as we increase the number of machines. 

**随机读基准测试**显示最差的扩展性（服务器数量增加500倍时，总吞吐量仅增加100倍）。<u>发生这种现象的原因是（如上所述），每读取1000字节，我们就会通过网络传输一个 64 KB的大块。这种转移使我们网络中的各种共享 1 Gigabit 链路饱和，结果，随着计算机数量的增加，每服务器的吞吐量显着下降。</u>

### 8 Real Applications

<img src="http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/1577708726543.png" alt="1122" style="zoom:30%;" />

As of August 2006, there are 388 non-test Bigtable clusters running in various Google machine clusters, with a combined total of about 24,500 tablet servers. Table 1 shows a rough distribution of tablet servers per cluster. Many of these clusters are used for development purposes and therefore are idle for significant periods. One group of 14 busy clusters with 8069 total tablet servers saw an aggregate volume of more than 1.2 million requests per second, with incoming RPC traffic of about 741 MB/s and outgoing RPC traffic of about 16 GB/s.

截至2006年8月，在各种Google机器集群中运行着388个非测试版Bigtable集群，总共约有24,500台Tablet服务器。表1显示了每个集群的Tablet服务器的大致分布。这些集群中的许多集群都用于开发目的，因此在相当长的一段时间内都处于空闲状态。一组14个繁忙的集群（总共8069个Tablet服务器）每秒总计收到超过120万个请求，其中传入RPC流量约为 741 MB/s，传出RPC流量约为 16 GB/s。

![1122](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/1577708648568.png)

Table 2 provides some data about a few of the tables currently in use. Some tables store data that is served to users, whereas others store data for batch processing; the tables range widely in total size, average cell size, percentage of data served from memory, and complexity of the table schema. In the rest of this section, we briefly describe how three product teams use Bigtable.

表2提供了一些有关当前使用的表的数据。有些表存储提供给用户的数据，而另一些表则存储用于批处理的数据。这些表在总大小，平均单元大小，从内存提供的数据百分比以及表模式的复杂性方面分布的范围很广。在本节的其余部分，我们简要描述三个产品团队如何使用Bigtable。

#### 8.1 Google Analytics

Google Analytics (analytics.google.com) is a service that helps webmasters analyze traffic patterns at their web sites. It provides aggregate statistics, such as the number of unique visitors per day and the page views per URL per day, as well as site-tracking reports, such as the percentage of users that made a purchase, given that they earlier viewed a specific page. 

Google Analytics（分析）（analytics.google.com）是一项服务，可帮助网站管理员分析其网站上的流量模式。它提供了汇总统计信息，例如每天，身份不重复的访客数量和每个URL每天的页面浏览量，以及网站跟踪报告，例如在先前查看特定页面的情况下进行购买的用户所占的百分比。

To enable the service, webmasters embed a small JavaScript program in their web pages. This program is invoked whenever a page is visited. It records various information about the request in Google Analytics, such as a user identifier and information about the page being fetched. Google Analytics summarizes this data and makes it available to webmasters.

为了启用该服务，网站管理员将一个小的JavaScript程序嵌入其网页中。每当访问页面时都会调用此程序。它在Google Analytics（分析）中记录有关请求的各种信息，例如用户标识符和有关正在获取的页面的信息。 Google Analytics（分析）会汇总这些数据并将其提供给网站管理员。


We briefly describe two of the tables used by Google Analytics. The raw click table (˜200 TB) maintains a row for each end-user session. The row name is a tuple containing the website’s name and the time at which the session was created. This schema ensures that sessions that visit the same web site are contiguous, and that they are sorted chronologically. This table compresses to 14% of its original size.

我们简要介绍了Google Analytics（分析）使用的两个表格。**原始点击表**（约200 TB）为每个终端用户会话维护一行。行名称是一个元组，其中包含网站的名称和创建会话的时间。此模式可确保访问同一网站的会话是连续的，并且可以按时间顺序对其进行排序。该表压缩到其原始大小的14％。

The summary table (˜20 TB) contains various predefined summaries for each website. This table is generated from the raw click table by periodically scheduled MapReduce jobs. Each MapReduce job extracts recent session data from the raw click table. The overall system’s throughput is limited by the throughput of GFS. This table compresses to 29% of its original size.

**摘要表**（约20 TB）包含每个网站的各种预定义摘要。该表是通过定期计划的MapReduce作业从原始点击表生成的。<u>每个MapReduce作业都会从原始点击表中提取最近的会话数据</u>。整个系统的吞吐量受GFS吞吐量的限制。该表压缩到其原始大小的29％。

#### 8.2 Google Earth

Google operates a collection of services that provide users with access to high-resolution satellite imagery of the world’s surface, both through the web-based Google Maps interface (maps.google.com) and through the Google Earth (earth.google.com) custom client software. These products allow users to navigate across the world’s surface: they can pan, view, and annotate satellite imagery at many different levels of resolution. This system uses one table to preprocess data, and a different set of tables for serving client data.

Google提供了一系列服务，可通过基于Web的Google Maps界面（maps.google.com）和Google Earth（earth.google.com）自定义客户端软件向用户提供世界地面的高分辨率卫星图像。这些产品使用户可以在整个地球表面导航：他们可以以许多不同的分辨率摇动拍摄，查看和注释卫星图像。该系统使用一个表预处理数据，并使用一组不同的表来提供客户端数据。

The preprocessing pipeline uses one table to store raw imagery. During preprocessing, the imagery is cleaned and consolidated into final serving data. This table contains approximately 70 terabytes of data and therefore is served from disk. The images are efficiently compressed already, so Bigtable compression is disabled. 

预处理管道使用一张表存储原始图像。在预处理期间，图像将被清理并合并为最终投放数据。该表包含大约70 TB的数据，因此是从磁盘提供的。图像已被高效压缩，因此已禁用Bigtable压缩。

Each row in the imagery table corresponds to a single geographic segment. Rows are named to ensure that adjacent geographic segments are stored near each other. The table contains a column family to keep track of the sources of data for each segment. This column family has a large number of columns: essentially one for each raw data image. Since each segment is only built from a few images, this column family is very sparse. The preprocessing pipeline relies heavily on MapReduce over Bigtable to transform data. The overall system processes over 1 MB/sec of data per tablet server during some of these MapReduce jobs.

图像表格中的每一行都对应一个地理区域。 对行进行命名以确保相邻的地理段彼此相邻存储。 该表包含一个列族，以跟踪每个段的数据源。 该列族有大量列：基本上每个原始数据图像（raw data image）都有一列。 由于每个段仅由几个图像构成，因此该列族非常稀疏。 预处理管道严重依赖BigTable上的MapReduce来转换数据。 在其中的某些MapReduce作业中，整个系统每台Tablet服务器处理超过 1 MB/s 的数据。

The serving system uses one table to index data stored in GFS. This table is relatively small (˜500 GB), but it must serve tens of thousands of queries per second per datacenter with low latency. As a result, this table is hosted across hundreds of tablet servers and contains in-memory column families. 

服务系统使用一张表索引存储在GFS中的数据。该表相对较小（约500 GB），但每个数据中心每秒必须处理数万个查询，且延迟低。结果，该表托管在数百台Tablet服务器中，并包含内存列族。

#### 8.3 Personalized Search

Personalized Search (www.google.com/psearch) is an opt-in service that records user queries and clicks across a variety of Google properties such as web search, images, and news. Users can browse their search histories to revisit their old queries and clicks, and they can ask for personalized search results based on their historical Google usage patterns.

个性化搜索（www.google.com/psearch）是一项选择性服务，可记录用户对各种Google属性（例如网络搜索，图片和新闻）的查询和点击。用户可以浏览其搜索历史记录以重新访问其以前的查询和点击，还可以根据其Google的历史使用模式（pattern：模型，模式）来请求个性化搜索结果。

Personalized Search stores each user’s data in Bigtable. Each user has a unique userid and is assigned a row named by that userid. All user actions are stored in a table. A separate column family is reserved for each type of action (for example, there is a column family that stores all web queries). Each data element uses as its Bigtable timestamp the time at which the corresponding user action occurred. Personalized Search generates user profiles using a MapReduce over Bigtable. These user profiles are used to personalize live search results.

个性化搜索将每个用户的数据存储在Bigtable中。每个用户都有一个唯一的用户ID，并分配有一个由该用户ID命名的行。所有用户操作（action）都存储在一个表中。每种操作（action）类型都保留一个单独的列族（例如，有一个列系列存储所有Web查询）。每个数据元素都将发生相应用户操作的时间用作其Bigtable时间戳。个性化搜索使用BigTable上的MapReduce生成用户个人资料（user profiles）。这些用户个人资料用于个性化实时搜索结果。

The Personalized Search data is replicated across several Bigtable clusters to increase availability and to reduce latency due to distance from clients. The Personalized Search team originally built a client-side replication mechanism on top of Bigtable that ensured eventual consistency of all replicas. The current system now uses a replication subsystem that is built into the servers.

<u>个性化搜索数据可在多个Bigtable集群之间复制，以提高可用性并减少由于与客户端之间的距离而引起的延迟</u>。个性化搜索团队最初在Bigtable之上构建了一个客户端复制机制，以确保所有副本的最终一致性。现在，当前系统使用服务器内置的复制子系统。

The design of the Personalized Search storage system allows other groups to add new per-user information in their own columns, and the system is now used by many other Google properties that need to store per-user configuration options and settings. Sharing a table amongst many groups resulted in an unusually large number of column families. To help support sharing, we added a simple quota mechanism to Bigtable to limit the storage consumption by any particular client in shared tables; this mechanism provides some isolation between the various product groups using this system for per-user information storage.

个性化搜索存储系统的设计允许其他组在其自己的列中添加新的每个用户信息，并且该系统现在已由许多其他需要存储每个用户配置选项和设置的Google属性所使用。<u>在许多组之间共享一张表导致了异常多的列族</u>。为了帮助支持共享，我们在Bigtable中添加了一个简单的配额机制，以限制共享表中任何特定客户端的存储消耗。这种机制使用此系统为每个用户的信息存储提供了各种产品组之间的隔离。

### 9 Lessons

In the process of designing, implementing, maintaining, and supporting Bigtable, we gained useful experience and learned several interesting lessons.

在设计，实施，维护和支持Bigtable的过程中，我们获得了有益的经验并吸取了一些有趣的经验教训。

One lesson we learned is that large distributed systems are vulnerable to many types of failures, not just the standard network partitions and fail-stop failures assumed in many distributed protocols. For example, we have seen problems due to all of the following causes: memory and network corruption, large clock skew, hung machines, extended and asymmetric network partitions, bugs in other systems that we are using (Chubby for example), overflow of GFS quotas, and planned and unplanned hardware maintenance. As we have gained more experience with these problems, we have addressed them by changing various protocols. For example, we added check summing to our RPC mechanism. We also handled some problems by removing assumptions made by one part of the system about another part. For example, we stopped assuming a given Chubby operation could return only one of a fixed set of errors. 

我们** 吸取的教训是，大型分布式系统容易遭受多种类型的故障，而不仅仅是许多分布式协议中假定的标准网络分区和出错后停止服务（fail-stop failures）**。例如，由于以下所有原因，我们发现了问题：

1. 内存和网络损坏
2. 很大的时钟偏差（clock skew）
3. 停止响应的机器
4. 扩展
5. 非对称的网络分区
6. 我们正在使用的其他系统中的错误（例如，Chubby）
7. GFS溢出配额
8. 计划和计划外的硬件维护

随着我们在这些问题上获得更多经验，我们已通过更改各种协议来解决这些问题。例如，我们在RPC机制中添加了校验和。我们还通过消除系统某个部分对另一部分所做的假设来处理一些问题。例如，我们停止假设给定的Chubby操作只能返回一组固定错误中的一个。

><center><strong>总结</strong></center>
>非预期的故障源远比你想象中多

Another lesson we learned is that it is important to delay adding new features until it is clear how the new features will be used. For example, we initially planned to support general-purpose transactions in our API. Because we did not have an immediate use for them, however, we did not implement them. Now that we have many real applications running on Bigtable, we have been able to examine their actual needs, and have discovered that most applications require only single-row transactions. Where people have requested distributed transactions, the most important use is for maintaining secondary indices, and we plan to add a specialized mechanism to satisfy this need. The new mechanism will be less general than distributed transactions, but will be more efficient (especially for updates that span hundreds of rows or more) and will also interact better with our scheme for optimistic cross-data-center replication.

我们吸取的** 另一个教训是，重要的是延迟添加新特性直到明确如何使用新特性**。例如，我们最初计划在我们的API中支持通用事务（general-purpose transaction）。因为我们没有立即使用它们，所以我们没有实现它们。现在，我们在Bigtable上运行了许多真实的应用程序，我们已经能够检查它们的实际需求，并且发现大多数应用程序仅需要单行事务。当人们要求进行分布式交易时，最重要的用途是维护二级索引，我们计划添加一种专门的机制来满足这一需求。新机制将不如分布式事务通用，但效率更高（特别是对于跨越数百行或更多行的更新），并且还将与我们的乐观跨数据中心复制方案更好地交互。

><center><strong>总结</strong></center>
>避免过早添加使用场景不明确的新特性

A practical lesson that we learned from supporting Bigtable is the importance of proper system-level monitoring (i.e., monitoring both Bigtable itself, as well as the client processes using Bigtable). For example, we extended our RPC system so that for a sample of the RPCs, it keeps a detailed trace of the important actions done on behalf of that RPC. This feature has allowed us to detect and fix many problems such as lock contention on tablet data structures, slow writes to GFS while committing Bigtable mutations, and stuck accesses to the METADATA table when METADATA tablets are unavailable. Another example of useful monitoring is that every Bigtable cluster is registered in Chubby. This allows us to track down all clusters, discover how big they are, see which versions of our software they are running, how much traffic they are receiving, and whether or not there are any problems such as unexpectedly large latencies. 

我们从支持Bigtable中学到的**实践经验是正确进行系统级监视的重要性（即监视Bigtable本身以及使用Bigtable的客户端进程）**。例如，我们扩展了RPC系统，以便对于RPC的抽样，它可以详细记录代表该RPC进行的重要操作。此功能使我们能够检测并修复许多问题，例如Tablet数据结构上的锁争用，在提交Bigtable更改时缓慢写入GFS以及在 METADATA Tablet 不可用时卡住对 METADATA table 的访问。有用监视的另一个示例是，每个Bigtable集群都在Chubby中注册。这使我们能够跟踪所有集群，发现它们有多大，查看它们正在运行的软件版本，正在接收多少流量，以及是否存在诸如意外的长延迟之类的问题。

><center><strong>总结</strong></center>
>合理的系统级监控非常重要。

The most important lesson we learned is the value of simple designs. Given both the size of our system (about 100,000 lines of non-test code), as well as the fact that code evolves over time in unexpected ways, we have found that code and design clarity are of immense help in code maintenance and debugging. One example of this is our tablet-server membership protocol. Our first protocol was simple: the master periodically issued leases to tablet servers, and tablet servers killed themselves if their lease expired. Unfortunately, this protocol reduced availability significantly in the presence of network problems, and was also sensitive to master recovery time. We redesigned the protocol several times until we had a protocol that performed well. However, the resulting protocol was too complex and depended on the behavior of Chubby features that were seldom exercised by other applications. We discovered that we were spending an inordinate amount of time debugging obscure corner cases, not only in Bigtable code, but also in Chubby code. Eventually, we scrapped this protocol and moved to a newer simpler protocol that depends solely on widely-used Chubby features.

我们** 学到的最重要的一课是简单设计的价值**。考虑到系统的大小（大约100,000行非测试代码），以及代码会以意想不到的方式随时间变化的事实，我们** 发现代码和设计的清晰性对代码维护和调试有极大的帮助**。我们的Tablet服务器成员身份协议就是一个例子。我们的第一个协议很简单：主服务器定期向Tablet服务器发布租约，而Tablet服务器在租约到期时会自杀。不幸的是，该协议在存在网络问题的情况下大大降低了可用性，并且对主服务器的恢复时间也很敏感。我们多次对协议进行了重新设计，直到有了一个性能良好的协议。但是，最终的协议太复杂了，取决于其他应用程序很少使用的Chubby功能的行为。我们发现，不仅在Bigtable代码中，而且在Chubby代码中，我们花费大量时间调试晦涩难解的案例。最终，我们放弃了该协议，转而使用仅依赖于广泛使用的Chubby特性的更新的更简单协议。

><center><strong>总结</strong></center>
>保持设计的简洁

### 10 Related Work

The Boxwood project <a href=" #[24]">[24]</a> has components that overlap in some ways with Chubby, GFS, and Bigtable, since it provides for distributed agreement, locking, distributed chunk storage, and distributed B-tree storage. In each case where there is overlap, it appears that the Boxwood’s component is targeted at a somewhat lower level than the corresponding Google service. The Boxwood project’s goal is to provide infrastructure for building higher-level services such as file systems or databases, while the goal of Bigtable is to directly support client applications that wish to store data.

Boxwood项目<a href=" #[24]">[24]</a>具有与Chubby，GFS和Bigtable在某些方面重叠的组件，因为它提供了分布式协议，锁，分布式块存储和分布式B树存储。在每种情况下，如果出现重叠，则Boxwood的组件似乎定位在比相应Google服务更低的级别上。 Boxwood项目的目标是为构建高级服务（例如文件系统或数据库）提供基础结构，而Bigtable的目标是直接支持希望存储数据的客户端应用程序。

Many recent projects have tackled the problem of providing distributed storage or higher-level services over wide area networks, often at “Internet scale.” This includes work on distributed hash tables that began with projects such as CAN <a href=" #[29]">[29]</a>, Chord <a href=" #[32]">[32]</a>, Tapestry <a href=" #[37]">[37]</a>, and Pastry <a href=" #[30]">[30]</a>. These systems address concerns that do not arise for Bigtable, such as highly variable bandwidth, untrusted participants, or frequent reconfiguration; decentralized control and Byzantine fault tolerance are not Bigtable goals. 

许多最近的项目解决了通常在“ Internet规模”上通过广域网提供分布式存储或更高级别服务的问题。这包括以CAN <a href=" #[29]">[29]</a>，Chord <a href=" #[32]">[32]</a>，Tapestry <a href=" #[37]">[37]</a> 和 Pastry <a href=" #[30]">[30]</a> 等项目开头的分布式哈希表的工作。这些系统解决了Bigtable不会出现的问题，例如带宽可变，参与者不受信任或频繁重新配置。分散控制和拜占庭容错并不是Bigtable的目标。

In terms of the distributed data storage model that one might provide to application developers, we believe the key-value pair model provided by distributed B-trees or distributed hash tables is too limiting. Key-value pairs are a useful building block, but they should not be the only building block one provides to developers. The model we chose is richer than simple key-value pairs, and supports sparse semi-structured data. Nonetheless, it is still simple enough that it lends itself to a very efficient **flat-file** representation, and it is transparent enough (via locality groups) to allow our users to tune important behaviors of the system.

就可能会提供给应用程序开发人员的分布式数据存储模型而言，我们认为由分布式B树或分布式哈希表提供的键值对模型过于局限。键值对是一个有用的构建块，但它们不应成为一个唯一提供给开发人员的构建块。我们选择的模型比简单的键/值对丰富，并且支持稀疏的半结构化数据。尽管如此，它仍然非常简单，可以使其非常有效地使用 flate file 表示，并且它（通过locality group）足够透明以允许我们的用户调整系统的重要行为。

><center><strong>译者附</strong></center>
>**flat file**:  n. a file consisting of records of a single record type in which there is no embedded structure information that governs relationships between records. 
>
>由单一记录类型的记录组成的文件，其中没有控制记录之间关系的嵌入式结构信息。
>
>—— 《微软计算机词典》

Several database vendors have developed parallel databases that can store large volumes of data. Oracle’s Real Application Cluster database <a href=" #[27]">[27]</a> uses shared disks to store data (Bigtable uses GFS) and a distributed lock manager (Bigtable uses Chubby). IBM’s DB2 Parallel Edition <a href=" #[4]">[4]</a> is based on a shared-nothing <a href=" #[33]">[33]</a> architecture similar to Bigtable. Each DB2 server is responsible for a subset of the rows in a table which it stores in a local relational database. Both products provide a complete relational model with transactions. 

几个数据库供应商已经开发了可以存储大量数据的并行数据库。 Oracle 的 Real Application Cluster 数据库<a href=" #[27]">[27]</a>使用共享磁盘存储数据（Bigtable使用GFS）和分布式锁管理器（Bigtable使用Chubby）。 IBM的DB2并行版<a href=" #[4]">[4]</a>基于类似于Bigtable的无共享<a href=" #[33]">[33]</a>架构。每个DB2服务器负责存储在本地关系数据库中的表中行的子集。两种产品都提供了完整的交易关系模型。


Bigtable locality groups realize similar compression and disk read performance benefits observed for other systems that organize data on disk using column-based rather than row-based storage, including C-Store <a href=" #[1]">[1]</a><a href=" #[34]">[34]</a>  and commercial products such as Sybase IQ <a href="#[15]">[15]</a><a href="#[36]">[36]</a>, SenSage <a href=" #[31]">[31]</a>, KDB+ <a href=" #[22]">[22]</a>, and the ColumnBM storage layer in MonetDB/X100 <a href=" #[38]">[38]</a>. Another system that does vertical and horizontal data partioning into flat files and achieves good data compression ratios is AT&T’s Daytona database <a href=" #[19]">[19]</a>. Locality groups do not support CPUcache-level optimizations, such as those described by Ailamaki <a href=" #[2]">[2]</a>.

<u>对于使用基于列而不是基于行的存储在磁盘上组织数据的其他系统，Bigtable locality group 实现了类似的压缩和磁盘读取性能优势</u>，包括C-Store <a href=" #[1]">[1]</a><a href=" #[34]">[34]</a> 和Sybase IQ <a href="#[15]">[15]</a><a href="#[36]">[36]</a>，SenSage <a href=" #[31]">[31]</a>，KDB + <a href=" #[22]">[22]</a> 和MonetDB / X100 <a href=" #[38]">[38] </a>中的ColumnBM存储层。 AT＆T的Daytona数据库<a href=" #[19]">[19]</a>是将垂直和水平数据分成  flat file 并实现良好数据压缩率的另一个系统。 locality group 不支持 CPU 缓存级别的优化，例如 Ailamaki <a href=" #[2]">[2]</a> 所描述的那些。

The manner in which Bigtable uses memtables and SSTables to store updates to tablets is analogous to the way that the Log-Structured Merge Tree <a href=" #[26]">[26]</a> stores updates to index data. In both systems, sorted data is buffered in memory before being written to disk, and reads must merge data from memory and disk.

<u>Bigtable 使用 memtable 和 SSTables 将更新存储到Tablet的方式类似于 Log-Structured Merge Tree <a href=" #[26]">[26]</a> 存储更新到索引数据的方式。在这两个系统中，已排序的数据在写入磁盘之前都要先在内存中进行缓冲，并且读取操作必须合并内存和磁盘中的数据</u>。


C-Store and Bigtable share many characteristics: both systems use a shared-nothing architecture and have two different data structures, one for recent writes, and one for storing long-lived data, with a mechanism for moving data from one form to the other. The systems differ significantly in their API: C-Store behaves like a relational database, whereas Bigtable provides a lower level read and write interface and is designed to support many thousands of such operations per second per server. C-Store is also a “read-optimized relational DBMS”, whereas Bigtable provides good performance on both read-intensive and write-intensive applications.

<u>C-Store和Bigtable具有许多特征：这两个系统都使用无共享架构，并且具有两种不同的数据结构，一种用于最近的写入，一种用于存储长期存在的数据，其机制是将数据从一种形式转移到另一种形式</u>。这些系统的API显着不同：C-Store的行为类似于关系数据库，而Bigtable提供了较低级别的读写接口，并且旨在每服务器每秒支持数千个此类操作。 C-Store也是“读取优化的关系DBMS”，而Bigtable在读取密集型和写入密集型应用程序上均提供了良好的性能。

Bigtable’s load balancer has to solve some of the same kinds of load and memory balancing problems faced by shared-nothing databases (e.g., <a href=" #[11]">[11]</a><a href=" #[35]">[35]</a>). Our problem is somewhat simpler: (1) we do not consider the possibility of multiple copies of the same data, possibly in alternate forms due to views or indices; (2) we let the user tell us what data belongs in memory and what data should stay on disk, rather than trying to determine this dynamically; (3) we have no complex queries to execute or optimize.

<u>Bigtable的负载平衡器必须解决无共享数据库（例如 <a href=" #[11]">[11]</a><a href=" #[35]">[35]</a>）面临的某些相同类型的负载和内存平衡问题</u>。我们的问题稍微简单一些：

（1）我们不考虑同一数据的多个副本的可能性，这些副本可能由于视图或索引而以其他形式出现；

（2）让用户告诉我们哪些数据属于内存，哪些数据应保留在磁盘上，而不是试图动态地确定它； 

（3）我们没有执行或优化的复杂查询；

### 11 Conclusions

We have described Bigtable, a distributed system for storing structured data at Google. Bigtable clusters have been in production use since April 2005, and we spent roughly seven person-years on design and implementation before that date. As of August 2006, more than sixty projects are using Bigtable. Our users like the performance and high availability provided by the Bigtable implementation, and that they can scale the capacity of their clusters by simply adding more machines to the system as their resource demands change over time.

我们已经介绍了Bigtable，这是一个用于在Google存储结构化数据的分布式系统。自2005年4月以来，Bigtable集群已投入生产使用，在此日期之前，我们在设计和实施上花费了大约7人年的时间。截至2006年8月，超过60个项目正在使用Bigtable。我们的用户喜欢Bigtable实施提供的性能和高可用性，他们可以通过随资源需求随时间的变化向系统中添加更多计算机，从而扩展集群的容量。


Given the unusual interface to Bigtable, an interesting question is how difficult it has been for our users to adapt to using it. New users are sometimes uncertain of how to best use the Bigtable interface, particularly if they are accustomed to using relational databases that support <u>general-purpose transactions</u>. Nevertheless, the fact that many Google products successfully use Bigtable demonstrates that our design works well in practice. 

鉴于Bigtable具有非同寻常的界面，一个有趣的问题是，我们的用户适应使用它有多困难。新用户有时不确定如何最好地使用Bigtable接口，特别是如果他们习惯于使用支持通用事务的关系数据库时。不过，许多Google产品成功使用Bigtable的事实表明我们的设计在实践中效果很好。

We are in the process of implementing several additional Bigtable features, such as support for secondary indices and infrastructure for building cross-data-center replicated Bigtables with multiple master replicas. We have also begun deploying Bigtable as a service to product groups, so that individual groups do not need to maintain their own clusters. As our service clusters scale, we will need to deal with more resource-sharing issues within Bigtable itself <a href=" #[3]">[3]</a>, <a href=" #5">[5]</a>.

我们正在实现几个其他Bigtable功能，例如对二级索引的支持以及用于构建具有多个主副本的跨数据中心复制Bigtable的基础结构。我们也已开始将Bigtable作为服务部署到产品组，以便各个组不需要维护自己的集群。随着我们服务集群的扩展，我们将需要在Bigtable自身内部处理更多的资源共享问题 <a href=" #[3]">[3]</a>, <a href=" #5">[5]</a>。

Finally, we have found that there are significant advantages to building our own storage solution at Google. We have gotten a substantial amount of flexibility from designing our own data model for Bigtable. In addition, our control over Bigtable’s implementation, and the other Google infrastructure upon which Bigtable depends, means that we can remove bottlenecks and inefficiencies as they arise.

<u>最后，我们发现在Google建立自己的存储解决方案具有明显的优势。通过为Bigtable设计我们自己的数据模型，我们获得了很大的灵活性。此外，我们对Bigtable的实施以及Bigtable依赖的其他Google基础架构的控制权意味着我们可以消除瓶颈和效率低下的情况</u>。

### Acknowledgements

We thank the anonymous reviewers, David Nagle, and our shepherd Brad Calder, for their feedback on this paper. The Bigtable system has benefited greatly from the  feedback of our many users within Google. In addition,we thank the following people for their contributions to Bigtable: Dan Aguayo, Sameer Ajmani, Zhifeng Chen, Bill Coughran, Mike Epstein, Healfdene Goguen, Robert Griesemer, Jeremy Hylton, Josh Hyman, Alex Khesin, Joanna Kulik, Alberto Lerner, Sherry Listgarten, Mike Maloney, Eduardo Pinheiro, Kathy Polizzi, Frank Yellin, and Arthur Zwiegincew. 

我们感谢匿名审稿人David Nagle和我们的牧羊人Brad Calder对本文的反馈。 Bigtable系统得益于Google众多用户的反馈。 此外，我们感谢以下人员对Bigtable的贡献：Dan Aguayo，Sameer Ajmani，Zhifeng Chen，Bill Coughran，Mike Epstein，Healfdene Goguen，Robert Griesemer，Jeremy Hylton，Josh Hyman，Alex Khesin，Joanna Kulik，Alberto Lerner， Sherry Listgarten，Mike Maloney，Eduardo Pinheiro，Kathy Polizzi，Frank Yellin和Arthur Zwiegincew。

### References

<a name="[1]"></a>[1] ABADI, D. J., MADDEN, S. R., AND FERREIRA, M. C. Integrating compression and execution in column oriented database systems. Proc. of SIGMOD (2006).

<a name="[2]"></a>[2] AILAMAKI, A., DEWITT, D. J., HILL, M. D., AND SKOUNAKIS, M. Weaving relations for cache performance. In The VLDB Journal (2001), pp. 169-180. 

<a name="[3]"></a>[3] BANGA, G., DRUSCHEL, P., AND MOGUL, J. C. Resource containers: A new facility for resource management in server systems. In Proc. of the 3rd OSDI (Feb. 1999), pp. 45-58. 

<a name="[4]"></a>[4] BARU, C. K., FECTEAU, G., GOYAL, A., HSIAO, H., JHINGRAN, A., PADMANABHAN, S., COPELAND,G. P., AND WILSON, W. G. DB2 parallel edition. IBM Systems Journal 34, 2 (1995), 292-322. 

<a name="[5]"></a>[5] BAVIER, A., BOWMAN, M., CHUN, B., CULLER, D., KARLIN, S., PETERSON, L., ROSCOE, T., SPALINK, T., AND WAWRZONIAK, M. Operating system support for planetary-scale network services. In Proc. of the 1st NSDI(Mar. 2004), pp. 253-266. 

<a name="[6]"></a>[6] BENTLEY, J. L., AND MCILROY, M. D. Data compression using long common strings. In Data Compression Conference (1999), pp. 287-295.  

<a name="[7]"></a>[7] BLOOM, B. H. Space/time trade-offs in hash coding with allowable errors. CACM 13, 7 (1970), 422-426. 

<a name="[8]"></a>[8] BURROWS, M. The Chubby lock service for loosely coupled distributed systems. In Proc. of the 7th OSDI (Nov. 2006). 

<a name="[9]"></a>[9] CHANDRA, T., GRIESEMER, R., AND REDSTONE, J. Paxos made live ? An engineering perspective. In Proc. of PODC (2007). 
<a name="[10]"></a>[10] COMER, D. Ubiquitous B-tree. Computing Surveys 11, 2 (June 1979), 121-137. 

<a name="[11]"></a>[11] COPELAND, G. P., ALEXANDER, W., BOUGHTER, E. E., AND KELLER, T. W. Data placement in Bubba. In Proc. of SIGMOD (1988), pp. 99-108. 

<a name="[12]"></a>[12] DEAN, J., AND GHEMAWAT, S. MapReduce: Simplified data processing on large clusters. In Proc. of the 6th OSDI (Dec. 2004), pp. 137-150.

<a name="[13]"></a>[13] DEWITT, D., KATZ, R., OLKEN, F., SHAPIRO, L., STONEBRAKER, M., AND WOOD, D. Implementation techniques for main memory database systems. In Proc. of SIGMOD (June 1984), pp. 1-8.

<a name="[14]"></a>[14] DEWITT, D. J., AND GRAY, J. Parallel database systems: The future of high performance database systems. CACM 35, 6 (June 1992), 85-98.

<a name="[15]"></a>[15] FRENCH, C. D. One size ts all database architectures do not work for DSS. In Proc. of SIGMOD (May 1995), pp. 449-450.

<a name="[16]"></a>[16] GAWLICK, D., AND KINKADE, D. Varieties of concurrency control in IMS/VS fast path. Database Engineering Bulletin 8, 2 (1985), 3-10.

<a name="[17]"></a>[17] GHEMAWAT, S., GOBIOFF, H., AND LEUNG, S.-T. The Google file system. In Proc. of the 19th ACM SOSP (Dec.2003), pp. 29-43.

<a name="[18]"></a>[18] GRAY, J. Notes on database operating systems. In Operating Systems ? An Advanced Course, vol. 60 of Lecture Notes in Computer Science. Springer-Verlag, 1978.

<a name="[19]"></a>[19] GREER, R. Daytona and the fourth-generation language Cymbal. In Proc. of SIGMOD (1999), pp. 525-526. 

<a name="[20]"></a>[20] HAGMANN, R. Reimplementing the Cedar file system using logging and group commit. In Proc. of the 11th SOSP (Dec. 1987), pp. 155-162. 

<a name="[21]"></a>[21] HARTMAN, J. H., AND OUSTERHOUT, J. K. The Zebra striped network file system. In Proc. of the 14th SOSP(Asheville, NC, 1993), pp. 29-43.

<a name="[22]"></a>[22] KX.COM. kx.com/products/database.php. Product page. 

<a name="[23]"></a>[23] LAMPORT, L. The part-time parliament. ACM TOCS 16,2 (1998), 133-169. 

<a name="[24]"></a>[24] MACCORMICK, J., MURPHY, N., NAJORK, M., THEKKATH, C. A., AND ZHOU, L. Boxwood: Abstractions as the foundation for storage infrastructure. In Proc. of the 6th OSDI (Dec. 2004), pp. 105-120. 

<a name="[25]"></a>[25] MCCARTHY, J. Recursive functions of symbolic expressions and their computation by machine. CACM 3, 4 (Apr. 1960), 184-195. 

<a name="[26]"></a>[26] O’NEIL, P., CHENG, E., GAWLICK, D., AND O’NEIL, E. The log-structured merge-tree (LSM-tree). Acta Inf. 33, 4 (1996), 351-385. 

<a name="[27]"></a>[27] ORACLE.COM.   www.oracle.com/technology/products/database/clustering/index.html.    Product page.

<a name="[28]"></a>[28] PIKE, R., DORWARD, S., GRIESEMER, R., AND QUINLAN, S. Interpreting the data: Parallel analysis with Sawzall. Scientific Programming Journal 13, 4 (2005), 227-298.

<a name="[29]"></a>[29] RATNASAMY, S., FRANCIS, P., HANDLEY, M., KARP, R., AND SHENKER, S. A scalable content-addressable network. In Proc. of SIGCOMM (Aug. 2001), pp. 161-172.

<a name="[30]"></a>[30] ROWSTRON, A., AND DRUSCHEL, P. Pastry: Scalable, distributed object location and routing for largescale peer-to-peer systems. In Proc. of Middleware 2001(Nov. 2001), pp. 329-350. 

<a name="[31]"></a>[31] SENSAGE.COM. sensage.com/products-sensage.htm. Product page. 

<a name="[32]"></a>[32] STOICA, I., MORRIS, R., KARGER, D., KAASHOEK, M. F., AND BALAKRISHNAN, H. Chord: A scalable peer-to-peer lookup service for Internet applications. In Proc. of SIGCOMM (Aug. 2001), pp. 149-160. 

<a name="[33]"></a>[33] STONEBRAKER, M. The case for shared nothing. Database Engineering Bulletin 9, 1 (Mar. 1986), 4-9. 

<a name="[34]"></a>[34] STONEBRAKER,M., ABADI, D. J., BATKIN, A., CHEN, X., CHERNIACK, M., FERREIRA, M., LAU, E., LIN, A., MADDEN, S., O’NEIL, E., O’NEIL, P., RASIN, A., TRAN, N., AND ZDONIK, S. C-Store: A columnoriented DBMS. In Proc. of VLDB (Aug. 2005), pp. 553-564. 

<a name="[35]"></a>[35] STONEBRAKER, M., AOKI, P. M., DEVINE, R., LITWIN, W., AND OLSON, M. A. Mariposa: A new architecture for distributed data. In Proc. of the Tenth ICDE(1994), IEEE Computer Society, pp. 54-65. 

<a name="[36]"></a>[36] SYBASE.COM.   www.sybase.com/products/databaseservers/sybaseiq.   Product page. 

<a name="[37]"></a>[37] ZHAO, B. Y., KUBIATOWICZ, J., AND JOSEPH, A. D. Tapestry: An infrastructure for fault-tolerant wide-area location and routing. Tech. Rep. UCB/CSD-01-1141, CS Division, UC Berkeley, Apr. 2001. 

<a name="[38]"></a>[38] ZUKOWSKI, M., BONCZ, P. A., NES, N., AND HEMAN, S. MonetDB/X100 ?A DBMS in the CPU cache. IEEE Data Eng. Bull. 28, 2 (2005), 17-22. 

###  翻译参考：

1. [Google Bigtable (中文版)](http://dblab.xmu.edu.cn/post/google-bigtable/)
2. [BIGTABLE中文版论文](http://blog.bizcloudsoft.com/?p=292)
3. [[译] [论文] Bigtable: A Distributed Storage System for Structured Data (OSDI 2006)](https://arthurchiao.github.io/blog/google-bigtable-zh/)
4. [深入浅出BigTable](https://www.youtube.com/watch?v=r1bh90_8dsg)
5. [BigTable论文阅读](https://niceaz.com/2019/03/24/bigtable/)

## 论文总结

### BigTable 推演过程

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/Snipaste_2020-01-07_17-20-15.png)

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/Snipaste_2020-01-07_17-20-06.png)

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/Snipaste_2020-01-07_17-19-55.png)

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/Snipaste_2020-01-07_17-19-47.png)

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/Snipaste_2020-01-07_17-19-38.png)

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/Snipaste_2020-01-07_17-19-31.png)

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/Snipaste_2020-01-07_17-19-25.png)

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/Snipaste_2020-01-07_17-19-18.png)

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/Snipaste_2020-01-07_17-19-10.png)

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/Snipaste_2020-01-07_17-18-59.png)

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/Snipaste_2020-01-07_17-18-48.png)

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/Snipaste_2020-01-07_17-18-18.png)

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/Snipaste_2020-01-07_17-18-10.png)



### BigTable架构

![](http://q9kvrafcq.bkt.clouddn.com/distributed-system/Google-BigTable/bigtable_arch.png)

### SSTable

参考：

1. [Log Structured Merge Trees(LSM) 原理](https://www.open-open.com/lib/view/open1424916275249.html)
2. [Log Structured Merge-Trees(LSM) 论文翻译](http://duanple.com/?s=The+Log-Structured+Merge-Tree)
3. [SSTable 原理](https://niceaz.com/2018/11/27/sstable/)
4. [Leveled Compaction · facebook/rocksdb Wiki](https://github.com/facebook/rocksdb/wiki/Leveled-Compaction)

[^1]: 原文 ：reason about the locality properties of the data <font color="#32cd32">represented</font> in the underlying storage.
[^2]: 原文：<font color="#32cd32">uninterpreted</font> strings.
[^3]:  原文：whether to <font color="#32cd32">serve</font> data <font color="#32cd32">out of</font> memory or <font color="#32cd32">from</font> disk.
[^4]: 原文: and memory <font color="#32cd32">accounting</font>.
[^5]: 原文: in <font color="#32cd32">decreasing</font> timestamp order.
[^6]: 原文: Figure 2 shows C++ code that uses a <font color="#32cd32">RowMutation abstraction</font> to perform a series of updates.
[^7]: 原文: C++ code that uses a <font color="#32cd32">Scanner abstraction</font> to iterate over all anchors in a particular row.
[^8]: 原文: and then reading the <font color="#32cd32">appropriate</font> block from disk.
[^9]: 原文: The service is <font color="#32cd32">live</font>.
[^18]: 原文：<font color="#32cd32">cell</font>，这里指的是table中一个cell.
[^10]: 原文: A Bigtable cluster typically operates in <font color="#32cd32">a shared pool of machines</font>.
[^11]:原文: we first find <font color="#32cd32">the appropriate block</font> by performing a binary search <font color="#32cd32">in the in-memory index</font>.
[^12]: 原文: without touching disk.
[^13]: 原文: which is elected to be the master and <font color="#32cd32">actively serve requests</font>.
[^14]:  原文: it <font color="#32cd32">recursively moves up the tablet location hierarchy</font>.
[^15]: 原文: live tablet sever有些人翻译为：存活的Tablet服务器，我觉得不够贴切，因为机器只要还在运行，那么我们认为它是活着的（就像生物的生命一样），但是有木有效是看机器运行过程的自身资源符不符合相关服务的规定，这个是动态的，在特定时刻的一台机器对有些服务而言是有效的，对有其他服务言可能是无效的。当然有些人觉得也可以这样解释：在某个特定时刻，针对有些服务而言，某台机器是存活的，然而对其他服务而言是已经死了（down， death），但是这样解释更像是强行与 live 的意思靠拢。
[^16]: 位置属性可以这样理解，比如树状结构，具有相同前缀的数据的存放位置接近。在读取的时候，可以把这些数据一次读取出来，联想到数据的局部性原理。
[^17]: 获取该网页的时间戳作为标识：即按照获取时间不同，存储了多个版本的网页数据。
