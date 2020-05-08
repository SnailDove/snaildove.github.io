---
title: Kafka分区副本分配策略解析
mathjax: true
mathjax2: true
categories: 中文
date: 2019-10-02 20:16:00
tags: [linux]
toc: true
---

主要是根据自己的理解重新梳理了 `assignReplicasToBrokersRackUnaware()` 和 `assignReplicasToBrokersRackAware()` 两个方法的思路。

## 未指定机架的分配策略

主要介绍是未指定机架信息的分配策略，Kafka 版本是 2.0.0，具体实现为 `kafka.admin.AdminUtils.scala` 文件中的 `assignReplicasToBrokersRackUnaware()` 方法，该方法的内容如下：

```scala
private def assignReplicasToBrokersRackUnaware(nPartitions: Int,//分区数 
replicationFactor: Int,//副本因子 
brokerList: Seq[Int],//集群中broker列表
fixedStartIndex: Int,//起始索引，即第一个副本分配的位置，默认值为-1
startPartitionId: Int): Map[Int, Seq[Int]] = {//起始分区编号，默认值为-1
    // fixedStartIndex表示第一个副本分配的位置，默认为-1
    // startPartitionId表示起始分区编号，默认为-1
    // ret表示 <partition,Seq[replica所在brokerId]> 的关系
    val ret = mutable.MapInt, Seq[Int]//保存分配结果的集合
    val brokerArray = brokerList.toArray //brokerId的列表
    // 如果起始索引fixedStartIndex小于0，则根据broker列表长度随机生成一个，以此来保证是有效的brokerId
    val startIndex = if (fixedStartIndex >= 0) fixedStartIndex else rand.nextInt(brokerArray.length)
    // 确保起始分区号不小于0
    var currentPartitionId = math.max(0, startPartitionId)
    // 指定了副本的间隔，目的是为了更均匀地将副本分配到不同的broker上
    var nextReplicaShift = if (fixedStartIndex >= 0) fixedStartIndex else rand.nextInt(brokerArray.length)
    // 轮询所有分区，将每个分区的副本分配到不同的broker上
    for (_ <- 0 until nPartitions) {
        //只有分区编号大于0且刚好分区编号已经轮流一遍broker时才递增下一个副本的间隔
    	if (currentPartitionId > 0 && (currentPartitionId % brokerArray.length == 0))
      		nextReplicaShift += 1
        val firstReplicaIndex = (currentPartitionId + startIndex) % brokerArray.length
        val replicaBuffer = mutable.ArrayBuffer(brokerArray(firstReplicaIndex))
        // 保存该分区所有副本分配的broker集合
        for (j <- 0 until replicationFactor - 1)
        	// 为其余的副本分配broker
        	replicaBuffer += brokerArray(replicaIndex(firstReplicaIndex, nextReplicaShift, j, brokerArray.length))
        // 保存该分区所有副本的分配信息
        ret.put(currentPartitionId, replicaBuffer)
        // 继续为下一个分区分配副本
        currentPartitionId += 1
    }
    ret
}
```

该方法参数列表中的 `fixedStartlndex` 和 `startPartitionld` 值是从上游的方法 中调用传下来的，默认都是 -1 ，分别表示第一个副本分配的位置和起始分区编号。 `assignReplicasToBrokersRackUnaware()` 方法的核心是遍历每个分区 partition ， 然后从 `brokerArray(brokerld 的列表)`中选取 ` replicationFactor` 个 `brokerld` 分配给这个 partition 。

该方法首先创建一个可变的 Map 用来存放该方法将要返回的结果 ，即分区 partition 和分配副本的映射关系 。 由于 `fixedStartlndex` 为 -1​ ，所以 `startlndex` 是一个随机数，用来计算一个起始分配的 `brokerId`，同时又因为 `startPartitionld` 为 -1 ​， 所 以 `currentPartitionld` 的值为 ​0，可见默认情况下创建主题时总是从编号为 $0$ 的分区依次轮询进行分配 。

`nextReplicaShift` 表示下一次副本分配相对于前一次分配的位移量 ，从字面上理解有点绕口 。举个例子 ： 假设集群中有 3 个 broker 节点 ， 对应于代码 中的 brokerArray，创建的某个主题中有 3 个副本和 6 个分区，那么 首先从 `partitionld` ( partition 的编号） 为 0 的 分区开始进行分配，假设第一次计算（由`rand.nextlnt(brokerArray.length`）随机产生）得到的 `nextReplicaShift` 的值为 1，第一次随机产生的 `startlndex` 值为 2，那么 `partitionld` 为 0 的第一个副本的位置 （ 这里指的是
`brokerArray` 的数组下标 ） `firstReplicalndex =(currentPartitionld + startlndex)% brokerArray.Length=(0+2)%3=2` ，第二个副本的位置为 `replicalndex(firstReplicalndex, nextReplicaShift， j , brokerArray.length)= replicalndex(2, 1, 0, 3)`  = ?， 这里引 入了一个新 的方法 `replicalndex()` ， 不过这个方法很简单， 具体如下：

```scala
private def replicaIndex(firstReplicaIndex : Int, secondReplicaShift : Int,
replicaIndex : Int, nBrokers : Int) : Int = {
    val shift = 1 + (secondReplicaShift + replicaIndex ) % ( nBrokers - 1 )
    (firstReplicaIndex + shift) % nBrokers
}
```

**主要思想**：该方法是基于第一个副本分配的broker位置，再根据偏移量计算出后续副本被分配到的broker位置。

在计算副本分配位置的时候，第一个副本的位置已经在循环外面计算过了，并且放入数组中了。后续计算剩余副本的过程只计算了副本数 3-1=2​ 次。在副本为3的情况下，`replicaIndex` 的值只能是 0 和 1。`secondReplicaShift`的值只有在分区编号大于 0 改变且轮一遍所有broker这时才会递增，否则一直是 1。那么计算偏移量公式  `(secondReplicaShift + replicaIndex ) % (nBrokers - 1)`可以理解为：分区副本循环一轮broker的偏移量由 `secondReplicaShift` 控制，同一分区的副本偏移量由 `replicaIndex` 控制。对 broker 数量 (3 -1) 取余只能在 0 和 1 中，再加 1 的话，偏移量就只能是1和2，这样副本的偏移量不会等于 ​0​，也就分配的均匀了。

继续计算 `replicaIndex(2, 1, 0, 3)=(2+(1+(1+0)%(3-1)))%3 = 1​`。继续计算下一个副本的位置 `replicaIndex(2, 1, 1, 3)=(2+(1+(1+1)%(3-1)))%3=0`。所以 `partitionId` 为 0​ 的副本分配位置列表[2,1,0]。

给出最后的分配结果，buffer中的第一个为Leader副本，以此类推，得出所有。依次类推，更多的分配细节可以参考下面的示例， topic-test2 的分区分配策略和上面陈述的一致 ：

```shell
[root@nodel kafka2.11-2.0.0] bin/kafka-topics . sh --zookeeper localhost:2181/ 
kafka --create -topic topic-test2 -replication-factor 3 --partitions 6 

Created topic "topic- test2 ”

[root@nodel kafka2.ll-2.0.0] bin/kafka-topics.sh zookeeper localhost : 2181/
kafka -- describe -- topic topic-test2

Topic:topic-test2 PartitionCount:6 ReplicationFactor : 3 Configs:
Topic : topic-test2 Partition: 0 Leader: 2 Replicas: 2,0,1 Isr: 2 , 0 , 1
Topic : topic-test2 Partition: 1 Leader: 0 Replicas: 0,1,2 Isr: 0 , 1 , 2
Topic : topic-test2 Partition: 2 Leader: 1 Replicas: 1,2,0 Isr: 1 , 2 , 0
Topic : topic-test2 Partition: 3 Leader: 2 Replicas: 2,1,0 Isr: 2 , 1 , 0
Topic : topic-test2 Partition: 4 Leader: 0 Replicas: 0,2,1 Isr: 0 , 2 , 1
Topic : topic-test2 Partition: 5 Leader: 1 Replicas: 1,0,2 Isr: 1 , 0 , 2
```

我们无法预先获知 `startlndex` 和 `nextReplicaShi`食 的值，因为都是随机产生的。 `startIndex` 和
 `nextReplicaShift` 的值可以通过最终的分区分配方案来反推，比如上面的 topic-test2 ， 第一个分区（即 `partitionId=0` 的分区）的第一个副本为 2 ，那么可由 `2 = (0+startlndex) % 3` 推断出 `startIndex` 为 2。**之所以 `startlndex` 选择随机产生，是因为这样可以在多个主题的情况下尽可能地均匀分布分区副本，如果这里固定为一个特定值，那么每次的第一个副本都是在这个 broker 上，进而导致少数几个 broker 所分配到的分区副本过多而其余 broker 分配到的分区副本过少，最终导致负载不均衡**。尤其是某些主题的副本数和分区数都比较少，甚至都为 1 的情况下，所有的副本都落到了那个指定的 broker 上。**与此同时，在分配时位移量 `nextReplicaShit` 食也可以更好地使分区副本分配得更加均匀。** 

## 指定机架的分配策略

相比较而言，指定机架信息的分配策略比未指定机架信息的分配策略要稍微复杂一些，但主体思想并没相差很多，只是将机架信息作为附加的参考工页。假设目前有 3 个机架 rack1 、 rack2 和 rack3 , Kafka 集群中的 9 个 broker 点都部署在这 3 个机架之上，机架与 broker 节点的对照关系如下 ：

|  机架   |    副本   |
| :----: | :------: |
| rack1 | 0, 1,  2 |
| rack2 | 3, 4,  5 |
| rack3 | 6, 7,  8 |

如果不考虑机架信息，那么对照 `assignReplicasToBrokersRackUnaware()` 方法里的 `brokerArray` 变量的值为［0, 1, 2, 3, 4, 5 6, 7, 8］ 。 指定基架信息的 `assignReplicasToBrokersRackAware()` 方法里的 brokerArray 的值在这里就会被转换为 ［0 , 3, 6, 1, 4, 7, 2, 5, 8 ］ ，显而易见，这是轮询各个机架而产生的结果， 如此新的 `brokerArray `（确切地说是 `arrangedBrokerList`  ）中包含了简单的机架分配信息。 之后的步骤也和 `assignReplicasToBrokersRackUnaware()` 方法类似 ， 同样包含 `startlndex` 、`currentPartiionld` 、 `nextReplicaShit` 的概念，循环为每一个分区分配副本。**分配副本时 ，除了 处理第一个副本 ， 其余的也调用 `replicalndex()` 方法来获得一个 broker， 但这里 和 `assignReplicasToBrokersRackUnaware()` 不同的是 ， 这里不是简单地将这个 broker 添加到当前分区的副本列表之中 ， 还要经过一层筛选， 满足以下任意一个条件的 broker 不能被添加到当前分区的副本列表之中**：

- 如果此 broker 所在的机架中己经存在一个 broker 拥有该分区的副本，并且还有其他的机架中没有任何一个 broker 拥有该分区的副本。
- 如果此 broker 中己经拥有该分区的副本，并且还有其他 broker 中没有该分区的副本 。

这2个条件也是为了分区副本分配均匀所设的条件，容易理解。对照 `assignReplicasToBrokersRackAware` 源代码如下：

```scala
private def assignReplicasToBrokersRackAware(nPartitions: Int,
replicationFactor: Int,
brokerMetadatas: Seq[BrokerMetadata],
fixedStartIndex: Int,
startPartitionId: Int): Map[Int, Seq[Int]] = {
    val brokerRackMap = brokerMetadatas.collect { 
        case BrokerMetadata(id, Some(rack)) => id -> rack
    }.toMap
    // 统计机架个数
    val numRacks = brokerRackMap.values.toSet.size
    // 基于机架信息生成一个Broker列表，不同机架上的Broker交替出现
    val arrangedBrokerList = getRackAlternatedBrokerList(brokerRackMap)
    // 统计broker个数
    val numBrokers = arrangedBrokerList.size
    val ret = mutable.Map[Int, Seq[Int]]()
    val startIndex = if (fixedStartIndex >= 0) fixedStartIndex else rand.nextInt(arrangedBrokerList.size)
    var currentPartitionId = math.max(0, startPartitionId)
    var nextReplicaShift = if (fixedStartIndex >= 0) fixedStartIndex else rand.nextInt(arrangedBrokerList.size)
    for (_ <- 0 until nPartitions) {
		if (currentPartitionId > 0 && (currentPartitionId % arrangedBrokerList.size == 0))
        	nextReplicaShift += 1
        val firstReplicaIndex = (currentPartitionId + startIndex) % arrangedBrokerList.size
        // 第一个副本所在的broker即leader副本的broker
        val leader = arrangedBrokerList(firstReplicaIndex)
        // 每个分区的副本分配列表
        val replicaBuffer = mutable.ArrayBuffer(leader)
        // 每个分区中所分配的机架的列表集
        val racksWithReplicas = mutable.Set(brokerRackMap(leader))
        // 每个分区所分配的brokerId的列表集，和racksWithReplicas一起用来做一层筛选处理
        val brokersWithReplicas = mutable.Set(leader)
        var k = 0
      	for (_ <- 0 until replicationFactor - 1) {
            var done = false
            while (!done) {
                val broker = arrangedBrokerList(replicaIndex(firstReplicaIndex, nextReplicaShift * numRacks, k, arrangedBrokerList.size))
                val rack = brokerRackMap(broker)
                // Skip this broker if
                // 1. there is already a broker in the same rack that has assigned a replica AND there is one or more racks that do not have any replica, or
                // 2. the broker has already assigned a replica AND there is one or more brokers that do not have replica assigned
                if ((!racksWithReplicas.contains(rack) || racksWithReplicas.size == numRacks) && (!brokersWithReplicas.contains(broker) || brokersWithReplicas.size == numBrokers))    {
                    replicaBuffer += broker
                    racksWithReplicas += rack
                    brokersWithReplicas += broker
                    done = true
                }
                k += 1
            }
        }
        ret.put(currentPartitionId, replicaBuffer)
        currentPartitionId += 1
    }
    ret
}

private def replicaIndex(firstReplicaIndex : Int, secondReplicaShift : Int,
replicaIndex : Int, nBrokers : Int) : Int = {
    val shift = 1 + (secondReplicaShift + replicaIndex ) % ( nBrokers - 1 )
    (firstReplicaIndex + shift) % nBrokers
}
```

无论是带机架信息的策略还是不带机架信息的策略，上层调用方法 `AdminUtils.assignReplicasToBrokers()` 最后都是获得一个 `[Int, Seq[Int]]` 类型的副本分配列表。

当创建一个主题时，无论通过 katka-topics.sh 脚本，还是通过其他方式（ 比如 4.2 节中介绍的 `KatkaAdminClient` ） 创建主题时， 实质上是在 ZooKeeper 中的 /brokers/topics/{topic-name} 节点下创建与该主题对应的子节点并写入分区副本分配方案，并且在 /config/topics/{topic-name} 节点下创建与该主题对应的子节点并写入主题相关的配置信息（这个步骤可以省略不执行）。 而 Kafka 创建主题的实质性动作是交由控制器异步去完成的 ， 有关控制器的更多细节可 以参考 6.4 节的相关内容。 

知道了 kafka-topics .sh 脚本的实质之后 ， 我们 可以直接使用 ZooKeeper 的 客户端在 /brokers/topics 节点下创建相应的主题节点并写入预先设定好的分配方案 ， 这样就可以创建一个新的主题了。这种创建主题的方式还可 以绕过一些原本使用 kafka-topics.sh 脚本创建主题时的一些限制，比如分区的序号可以不用从 0 开始连续累加了 。 

## 参考资料

1. 《深入理解 Kafka：核心设计与实践原理》
2.  Kafka 源码
3. [Kafka解惑之topic创建（1）](https://www.jianshu.com/p/2641983427e3)
4. [Kafka解惑之topic创建（2）](https://www.jianshu.com/c/585da46f947c)
