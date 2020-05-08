## <font color="#9a161a"> 分区的管理</font>

### 优先副本的选举

分区使用多副本机制来提升可靠性，但**只有 leader 副本对外提供读写服务**，而 follower 副本只负责在内部进行消息的同步。如果一个分区的 leader 副本不可用，那么就意味着整个分区变得不可用 ，此时就需要 Kafka 从剩余的 follower 副本中挑选一个新的 leader 副本来继续对外提供服务。 

在创建主题的时候，该主题的分区及副本会尽可能**均匀地**分布到 Kafka 集群的各个 broker节点上，对应的 leader 副本的分配也比较均匀。 

可以将 leader 副本所在的 broker 节点叫作<font color="#32cd32"><strong>分区的 leader 节点</strong></font>，而 follower副本所在的 broker 节点叫作<font color="#32cd32"><strong>分区的 follower 节点</strong></font> 。 

---

<p><center><font color="#f08200" size=4><strong>场景需求</strong></font></center></p>
随着时间的更替， Kafka 集群的 broker 节点不可避免地会遇到岩机或崩溃的问题 ， 当分区的 leader 节点发生故障时 ，其中一个 follower 节点就会成为新的 leader 节点，这样就会导致集群的负载不均衡 ， 当原来的 leader 节点恢复之后重新加入集群时，它只能成为一个新的 follower 节点而不再对外提供服务 。 

---

为了能够有效地治理负载失衡的情况 ， Kafka 引入了 <font color="#32cd32"><strong>preferred replica</strong></font> （翻译：优先副本或首选副本） 的概念 。所谓的优先副本是指在 AR （Assigned Replicas）集合列表中的第一个副本 。**Kafka 要确保所有主题的优先副本在 Kafka 集群中均匀分布，这样就保证 了所有分区 的 leader 均衡分布 。 如果 leader 分布过于集中， 就会造成集群负载不均衡 。**

所谓的优先副本的选举是指通过一定的方式促使优先副本选举为 leader 副本，以此来促进集群的负载均衡 ， 这一行为也可以称为“<font color="#32cd32"><strong>分区平衡</strong></font>” 。需要注意的是 ， 分区平衡并不意味着 Kafka 集群的负载均衡，因为还要考虑集群中的分区分配是否均衡。更进一步，每个分区的 leader 副本的负载也是各不相同 的， 有些 leader 副本的负载很高。  

在 **Kafka 中可以提供分区自动平衡的功能**，与此对应的 broker 端参数是 `auto.leader.rebalance.enable` ，此参数的默认值为 true ，即默认情况下此功能是开启的。如果开启分区自动平衡的功能，则 Kafka 的控制器会启动一个定时任务，这个定时任务会轮询所有的 broker 节点，计算每个 broker 节点的分区不平衡率（ broker 中的不平衡率＝非优先副本的 leader 个数／分区总数）是否超过 `leader.imbalance.per.broker.percentage` 参数配置的比值，默认值为 10%，如果超过设定的比值则会自动执行优先副本的选举动作以求分区平衡。执行周期由参数 `leader.imbalance.check.interval.seconds` 控制，默认值为 300 秒，即 5 分钟 。

**不过在生产环境中不建议将 `auto.leader.rebalance.enable` 设置为默认的 true ， 因为这可能引起负面的性能问题，也有可能引起客户端一定时间的阻塞。因为执行的时间无法自主掌控，如果在关键时期（比如电商大促波峰期）执行关键任务的关卡上执行优先副本的自动选举操作，势必会有业务阻塞、频繁超时之类的风险 。 前面也分析过，分区及副本的均衡也不能完全确保集群整体的均衡，并且集群中 一定程度上的不均衡也是可以忍受的，为防止出现<font color ="red">关键时期“掉链子”</font>的行为，笔者建议还是将掌控权把控在自己的手中，可以针对此类相关的埋点指标设置相应的告警，在合适的时机执行合适的操作，而这个“合适的操作”就是指手动执行分区平衡。**

Kafka 中 `kafka-perferred-replica-election.sh` 脚本提供了对分区 leader 副本进行重新平衡的功能。优先副本的选举过程是一个安全的过程， Kafka 客户端可以自动感知分区 leader 副本的变更。 

```shell
[root@nodel kafka_2.11-2.0.OJ bin/kafka-preferred-replica-election.sh --zookeeper localhost:2181/kafka

Created preferred replica election path with topic-demo-3,consumer-offsets-22,topic-config-1,＿consumer-offsets-30,bigdata_monitor-12,__consumer_offsets8,__consumer-offsets-21,topic-create-0,__consumer_offsets-4,topic-demo-1,topic-partitions-1, consumer-offsets-27, consumer-offsets-7,consumer-offsets-9,consumer-offsets-46,(...省略若干）

[root@nodel kafka_2.ll-2.0.0]bin/kafka-topics.sh --zookeeper localhost:2181/kafka --describe --topic topic- partitions

Topic:topic-partitions PartitionCount: 3 ReplicationFactor : 3 Configs:
Topic : topic-partitions Partition: 0 Leader: 1 Replicas : 1, 2, 0 Isr: 1, 0, 2
Topic : topic-partitions Partition: 1 Leader: 2 Replicas : 2, 0, 1 Isr: 0, 1, 2
Topic : topic-partitions Partition: 2 Leader: 0 Replicas : 0, 1, 2 Isr: 0, 1, 2
```

**leader 副本的转移也是一项高成本的工作，如果集群中包含大量的分区，那么上面的这种使用方式有可能会失效。在优先副本的选举过程中 ，具体的元数据信息会被存入 ZooKeeper/admin/preferred-replica-election 节点，如果这些数据超过了 ZooKeeper 节点所允许的大小，那么选举就会失败**。默认情况下 ZooKeeper 所允许的节点数据大小为 1MB 。

**`kafka-perferred-replica-election.sh` 脚本中还提供了 path-to-json-file 参数来小批量地对部分分区执行优先副本的选举操作，这个 JSON 文件里保存需要执行优先副本选举的分区清单**。 

```shell
[root@nodel kafka2.11-2.0.0] cat election.json
{
	"partitions":[
        {
        	"partition": 0,
        	"topic": "topic-partitions"
        },
        {
            "partition": 1,
            "topic": "topic-partitions"
        },
        {
            "partition": 2,
            "topic": "topic-partitions"
        }
    ]
}

[root@nodel kafka2.11-2.0.0] bin/kafka-preferred-replica-election.sh --zookeeper localhost : 2181/kafka --path-to-json-file election.json

Created preferred replica election path with topic-partitions-0 , topic-partitions-1 ,topic-partitions-2

Successfully started preferred replica election for partitions Set (topic-partitions-0, topic-partitions-1, topic-partitions-2)

[root@nodel kafka 2.11-2.0.O] bin/kafka-topics.sh --zookeeper localhost: 2181/kafka --describe --topic topic-partitions

Topic : topic-partitions PartitionCount : 3 ReplicationFactor: 3 Configs:
Topic : topic partitions Partition: 0 Leader: 1 Replicas: 1, 2, 0 Isr: 1, 0 , 2
Topic : topic-partitions Partition: 1 Leader: 2 Replicas: 2, 0, 1 Isr: 0, 1 , 2
Topic : topic-partitions Partition: 2 Leader: 0 Replicas: 0, 1, 2 Isr: 0, 1 , 2
```

### 分区重分配

---

<p><center><font color="#f08200" size=4><strong>场景需求</strong></font></center></p>
- 当集群中的一个节点突然若机下线时，这个节点上的分区副本都已经处于功能失效的状态， Kafka 并不会将这些失效的分区副本自动地迁移到集群中剩余的可用 broker 节点上，如果放任不管，则不仅会影响整个集群的均衡负载，还会影响整体服务的可用性和可靠性 。 


- 当要对集群中的一个节点进行有计划的下线操作时，为了保证分区及副本的合理分配，我们也希望通过某种方式能够将该节点上的分区副本迁移到其他的可用节点上。 
- 当集群中新增 broker 节点时，只有新创建的主题分区才有可能被分配到这个节点上，而之前的主题分区并不会自动分配到新加入的节点中，因为在它们被创建时还没有这个新节点，这样新节点的负载和原先节点的负载之间严重不均衡 。

---

Kafka 提供了 `kafka-reassign-partitions.sh` 脚本来执行分区重分配的工作，它可以在集群扩容、 broker 节点失效的场景下对分区进行迁移 。`kafka-reassign-partitions.sh` 脚本的使用分为 3 个步骤 ： 

1. **首先创建需要一个包含主题清单的 JSON 文件**

    例如：一个3个broker的 Kafka 集群，因为某种原因下线 broker 1


```json
{
    "topics": [
    	{
    		"topic": "topic-reassign"
    	}
    ],
    "version": 1
}
```

2. **其次根据主题清单和 broker 节点清单生成一份重分配方案**

    ```shell
    [root@nodel kafka 2.11-2.0.O] bin/kafka-reassign-partitions.sh --zookeeper
    localhost:2181/kafka --generate --topics-to-move-json-file reassign.json
    --brker-list 0,2
    
    Current partition replica assignment
    {
    	"version": 1,
    	"partitions": 
    	[ 
            {
                "topic": "topic-reassign",
                "partition": 2,
                "replicas”: [2, 1],
                "log_dirs”:	["any", "any"]
            }, 
            {
                "topic": "topic-reassign", 
                "partition”: 1, 
                "replicas": [1, 0], 
                "log_dirs": ["any", "any"]
            },
            {
                "topic”: ”topic- reassign”,
                "partition": 3, 
                "replicas": [0, 1], 
                "log_dirs": ["any", "any"]
            }, 
            {
                "topic": "topic-reassign",
                "partition”: 0,
                "replicas": [0, 2],
                "log_dirs": ["any", "any"]
             }
         ]
    }
    
    Proposed partition reassignment configuration
    {
    	"version": 1, 
    	"partitions": [
    		{
    			"topic"："topic-reassign", 
    			"partition": 2, 
    			"replicas": [2, O], 
    			"log_dirs": ["any", "any"]
    		},
    		{
    			"topic"："topic-reassign",  
    			"partition": 1,
    			"replicas": [0, 2],
    			"log_dirs": ["any", "any"]
    		},
    		{ 
    			"topic"："topic-reassign", 
    			"partition": 3, 
    			"replicas": [0, 2],
    			"log_dirs": ["any", "any"]
    		}, 
    		{
    			"topic"："topic-reassign",  
    			"partition": 0, 
    			"replicas": [2, O], 
    			"log_dirs": ["any", "any"]
    		}
    	]
    }
    ```
    - `–generate` 是 kafka-reassign-partitions.sh 脚本中指令类型的参数，可以类比于 kafka-topics.sh 脚本中的 create 、 list 等，它用来生成一个重分配的候选方案。 
    - `–topic-to-move-json` 用来指定分区重分配对应的主题清单文件的路径，该清单文件的具体的格式可以归纳为 {”topics": [ {”topic＂:”foo”}, {”topic”:"fool”}],"version”: 1} 
    - `–broker-list` 用来指定所要分配的 broker 节点列表，比如示例中的 “0,2” 。

3. **最后根据这份方案执行具体的重分配动作。** 

    我们需要将第二个 JSON 内容保存在一个 JSON 文件中，假定这个文件的名称为 project.json 。 

    ```shell
    [root@nodel kafka2.11-2.0.0] bin/kafka-reassign-partitions.sh --zookeeper localhost : 2181/kafka --execute --reassignment-json-file project.json
    
    Current partition replica assignment
    { 
    	"version": 1, 
    	"partitionS": [
    		{
    			"topic": "topic-reassign",
    			"partition": 2, 
    			"replicas": [2, 1],
    			"log_dirs": ["any","any"]
			}, 
    		{
    			"topic": "topic-reassign ",
    			"partition": 1,
    			"replicas": [1, 0],
    			"log_dirs": ["any", "any"]
    		}, 
    		{ 
    			"topic": "topic-reassign",
    			"partition": 3,
    			"replicas": [0, 1],
    			"log_dirs": ["any", "any"]
    		}, 
    		{
    			"topic": "topic- reassign",
    			"partition": 0,
    			"replicas": [0, 2] , 
    			"log_dirs ": ["any"," any"]
    		}
    	]
    }
    
    Save this to use as the --reassignrnent-json-file option during rollback
    Successfully started reassignment of partitions.
    ```
    
    我们再次查看主题 topic-reassign 的具体信息 ：
    
    ```shell
    [root@nodel kafka2.11-2.0.0] bin/kafka-topics.sh --zookeeper localhost:2181/kafka --describe --topic topic-reassign
    
    Topic: topic-reassign PartitionCount: 4 ReplicationFactor: 2 Configs:
    Topic: topic-reassign Partition : 0 Leader: 0 Replicas : 2, 0 Isr : 0, 2
    Topic: topic-reassign Partition : 1 Leader: 0 Replicas : 0, 2 Isr : 0, 2
    Topic: topic-reassign Partition : 2 Leader: 2 Replicas : 2, 0 Isr : 2, 0
    Topic: topic-reassign Partition : 3 Leader: 0 Replicas : 0, 2 Isr : 0, 2 
    ```
    
    **除了让脚本自动生成候选方案，用户还可以自定义重分配方案，这样也就不需要执行第一步和第二步的操作了。**
    
    分区重分配的基本原理是先通过控制器为每个分区添加新副本（增加副本因子），新的副本将从分区的 leader 副本那里复制所有的数据。根据分区的大小不同 ， 复制过程可能需要花一些时间，因为数据是通过网络复制到新副本上的 。在复制完成之后，控制器将 旧副本从副本清单里移除（恢复为原先的副本因子数）。**注意在重分配的过程中要确保有足够的空间。**
    
    主题 topic-assign 中有 3 个 leader 副本在 broker 0 上，而只有 1 个 leader 副本在 broker 2 上， **这样负载就不均衡了。可以借助 kafka-perferred-replica-election.sh 脚本来执行一次优先副本的选举动作 。**
    
4. 验**证查看分区重分配的进度** 。 只需将上面的 execute 替换为 verify 即可， 具体示例如下：

   ```shell
   [root ＠口 odel kafka 2.11-2.0.0] bin/kafka-reassign-partitions.sh --zookeeper localhost:2181/kafka --verify --reassignment-json-file project.json
   
   Status of partition reassignment :
   Reassignment of partition topic-reassign-2 completed successfully
   Reassignment of partition topic-reassign-1 completed successfully
   Reassignment of partition topic-reassign-3 completed successfully
   Reassignment of partition topic-reassign-0 completed successfully
   ```

**在实际操作中，我们将降低重分配的粒度，分成多个小批次来执行，以此来将负面的影响降到最低，这一点和优先副本的选举有异曲同工之妙**。

**如果要将某个 broker 下线，那么在执行分区重分配动作之前最好先关闭或重启 broker。这样这个 broker 就不再是任何分区的 leader 节点了，它的分区就可以被分配给集群中的其他 broker。这样可以减少 broker 间的流量复制** ，以此提升重分配的性能，以及减少对集群的影响。 

### 复制限流

**分区重分配本质在于数据复制，先增加新的副本，然后进行数据同步，最后删除旧的副本来达到最终的目的**。数据复制会占用额外的资源，如果重分配的量太大必然会严重影响整体的性能，尤其是处于业务高峰期的时候。

---

<p><center><font color="#f08200" size=4><strong>场景需求</strong></font></center></p> 
如果集群中某个主题或某个分区的流量在某段时间内特别大，那么只靠减小粒度是不足以应对的，这时就需要有一个限流的机制，可以对副本间的复制流量加以限制来保证重分配期间整体服务不会受太大的影响 。

---

副本间的复制限流有两种实现方式： kafka-config.sh 脚本和 kafka-reassign-partitions.sh 脚本。 

#### kafka-config.sh 限流

kafka-config.sh 脚本主要以动态配置的方式来达到限流的目的，在 broker 级别有两个与复制限流相关的配置参数 ： `follower.replication.throttled.rate` 和 `leader.replication.throttled.rate` ，前者用于设置 follower 副本复制的速度，后者用于设置 leader 副本传输的速度，它们的单位都是 Bit/s。通常情况下，两者的配置值是相同的 。 

##### <font color="#0979e3" size=4><strong>broker端示例</strong></font>

下面的中将 broker 1 中的 leader 副本和 follower 副本的复制速度限制在 1024B/s 之 内，即 IKB/s :

```shell
[root@node l kafka-2.11-2.0.0] bin/kafka-config.sh --zookeeper localhost:2181/kafka --entity-type brokers --entity-name 1 --alter --add-config follower.replication.throttled.rate=l024, leader.replication.throttled.rate=l024

Completed Updating config for entity : brokers '1'． 
```

我们再来查看一下 broker 1 中刚刚添加的配置 ，参考如下：

```shell
[root@node l kafka-2.11-2.0.0] bin/kafka-config.sh --zookeeper localhost:2181/kafka --entity-type brokers --entity-name 1 --describe

Configs for brokers ’ 1 ’ are leader.replication.throttled.rate=1024, follower.replication.throttled.rate=l024
```

了解到变更配置时会在 ZooKeeper 中创建一个命名形式为 `/config/<entity-type>/<entity-name＞` 的节点 ，对于这里的示例而言 ，其节点就是 `/config/brokers/1` ，节点中相应的信息如下 ： 

```shell
[zk:localhost:2181/kafka(CONNECTED) 6] get /config/brokers/1

{ 
	"version":1, 
	"config" : 
	{ 
		"leader.replication.throttled.rate": "1024",
		"follower.replication.throttled.rate": "1024" 
	}
}
```

删除刚刚添加的配置也很简单：

```shell
[root@node 1 kafka-2.11-2.0.0] bin/kafka-config.sh --zookeeper localhost:2181/kafka --entity-type brokers --entity-name 1 --alter --delete-config follower.replication. throttled.rate, leader.replication.throttled.rate

Completed Updating config for entity : brokers '1'.
```

##### <font color="#0979e3" size=4><strong>主题层面示例</strong></font>

**在主题级别也有两个相关的参数来限制复制的速度** ： `leader.replication.throttled.replicas` 和`follower.replication.throttled.replicas` ，**它们分别用来配置被限制速度的主题所对应的 leader 副本列表和 follower 副本列表**。  

```shell
[root@node 1 kafka-2.11-2.0.0] bin/kafka-topics.sh --zookeeper localhost : 2181/kafka --describe --topic topic-throttle

Topic: topic-throttle PartitionCount: 3 ReplicationFactor : 2 Configs:
Topic: topic-throttle Partition: O Leader : 0 Replicas: 0, 1 Isr: 0, 1
Topic: topic-throttle Partition: l Leader : 1 Replicas: 1, 2 Isr: 1, 2
Topic: topic-throttle Partition: 2 Leader : 2 Replicas: 2, 0 Isr: 2, 0
```

在上面示例中，主题 topic-throttle 的三个分区所对应的 leader 节点分别为 0, 1, 2，即分区与 broker 的映射关系为 0:0,1:1, 2:2，而对应的 follower 节点分别为 1, 2, 0，相关的分区与代理的映射关系为 0:1, 1:2, 2:0，那么此主题的限流副本列表及具体的操作细节如下： 

```shell
[root@node1 kafka-2.11-2.0.0] bin/kafka-config.sh --zookeeper localhost:2181/kafka --entity-type topics --entity-name topic-throttle --alter --add-config leader.replication.throttled.replicas=[0:0,1:1,2:2], follower.replication.throttled .replicas=[O:1,1:2,2:0]

Completed Updating config for entity: topic 'topic-throttle'.
```

对应的 ZooKeeper 中的 `/config/topics/topic-throttle` 节点信息如下 ：

```json
{
	"version": 1, 
	"config":
	{
		"leader.replication.throttled.replicas": "0:0, 1:1, 2:2",
		"follower.replication.throttled.replicas"："0:1, 1:2, 2:0" 
	}
}
```

#### 限流的分区重分配

首先按步骤创建一个包含关于分区重分配的可行性方案的 project.json 文件，内容如下：

```json
{
	"version": 1,
	"partitions":
    [
        {
            "topic":"topic-throttle",
            "partit ion": 1, 
            "replicas": [2, 0], 
            "log_dirs": ["any","any" ]
        },
        { 
            "topic": "topic-throttle", 
            "partition": 0,
            "replicas": [0, 2], 
            "log_dirs": ["any", " any"] 
        }, 
        {
            "topic": "topic-throttle", 
            "partition": 2,
            "replicas": [0, 2],
            "log_dirs": ["any","any"]
        }
	]
} 
```

接下来按照可行性方案设置被限流的副本列表，首先看一下重分配前和分配后的分区副本布局对比，详细如下： 

| partition | 重分配前的 AR | 分自己后的预期 AR |
| --------- | ------------- | ----------------- |
| 0         | 0,1           | 0, 2              |
| 1         | 1,2           | 2, 0              |
| 2         | 2,0           | 2, 0              |

**如果分区重分配会引起某个分区 AR (Assigned Replicas) 集合的变更，那么这个分区中与 leader 有关的限制会应用于重分配前的所有副本，因为任何一个副本都可能是 leader，而与 follower 有关的限制会应用于所有移动的目的地**。

从概念上理解会比较抽象，这里不妨举个例子，对上面的布局对比而言，分区 0 重分配前的 AR 为［0 ,1］， 重分配后的 AR 为［0,2］，那么这里的目的地就是新增的 2。也就是说，对分区 0 而言， `leader.replication.throttled.replicas` 配置为 ［0:0, 0:1]，`follower.replication.throttled.replicas` 配置为［0:2] 。 同理，对于分区 1 而言，
`leader.replication.throttled.replicas` 配置为 ［1:1, 1:2] , `follower.replication.throttled.replicas` 配置为［1:0J 。分区 3 的 AR 集合没有发生任何变化，这里可以忽略。获取限流副本列表之后，我们就可以执行具体的操作了，详细如下： 

```shell
[root@nbde1 kafka-2.11-2.0.0] bin/kafka-configs.sh --zookeeper localhost:2181/kafka --entity-type topics --entity-name topic-throttle --alter --add-config leader.replication.throttled.replicas=[1:1, 1:2, 0:0, 0:1], follower.replication.throttled.replicas=[1:0, 0:2]

Completed Updating config for entity : topic 'topic-throttle'.
```

接下来再设置 broker 2 的复制速度为 10 B/s，这样在下面的操作中可以很方便地观察限流与不限流的不同：

```shell
[root@node1 kafka-2.11-2.0.0] bin/kafka-configs.sh --zookeeper localhost:2181/kafka --entity-type brokers --entity-name 2 --alter --add-config follower.replication.throttled.rate=10, leader.replication.throttled.rate=10

Completed Updating config for entity : brokers '2'.
```

之后我们再执行正常的分区重分配的操作，示例如下 ： 

```shell
[root@node1 kafka-2.11-2.0.0] bin/kafka-reassign-partitions.sh --zookeeper localhost:2181/kafka --execute reassignment-json-file project.json

Current partition replica assignment
{
	"version"： 1,
	"partitions": 
	[
		{ 
			"topic": "topic-throttle",
			"partition": 2,
			"replicas": [2, 0],
			"log_dirs": ["any", "any"]
        }, 
        { 
            "topic": "topic-throttle",
            "partition"： 1,
            "replicas": [1, 2],
            "log_dirs": ["any", "any"]
         },
         { 
         	"topic": "topic-throttle",
         	"partition": 0,
         	"replicas": [0, 1],
         	"log_dirs": ["any", "any"]
         }
    ]
}

Save this to use as the --reassignment-json-file option during rollback
Successfully started reassignment of partitions.
```

执行之后，可以查看执行的进度，示例如下： 

```shell
[root@node1 kafka-2.11-2.0.0] bin/kafka-reassign-partitions.sh --zookeeper localhost:2181/kafka --verify --reassignment-json-file project.json

Status of partition reassignment:
Reassignment of partition topic-throttle-1 completed successfully
Reassignment of partition topic-throttle-0 is still in progress
Reassignment of partition topic-throttle-2 completed successfully
```

可以看到分区 `topic-throttle-0` 还在同步过程中，因为我们之前设置了 broker 2 的复制速度为10B/s ，这样使同步变得缓慢，分区 `topic-throttle-0` 需要同步数据到位于 broker 2 的新增副本中。随着时间的推移，分区 `topic-throttle-0` 最终会变成“ completed successful”的状态。

**为了不影响 Kafka 本身的性能，往往对临时设置的一些限制性的配置在使用完后要及时删除，而 kafka-reassign-partitions.sh 脚本配合指令参数 verify 就可以实现这个功能**，在所有的分区都重分配完成之后执行查看进度的命令时会有如下的信息： 

```shell
[root@node1 kafka-2.11-2.0.0] bin/kafka-reassign-partitions.sh --zookeeper localhost:2181/kafka --verify --reassignment-json-file project.json

Status of partition reassignment:
Reassignment of partition topic-throttle-1 completed successfully
Reassignment of partition topic-throttle-0 completed successfully
Reassignment of partition topic-throttle-2 completed successfully
Throttle was removed.
```

**注意到最后一行信息“ Throttle was removed. ”，它提示了所有之前针对限流做的配置都已经被清除了**，读者可 以 自行查看一下相应的 ZooKeeper 节点中是否还有相关的配置。 

#### kafka-reassign-partitions.sh 限流

kafka-reassign-partitions.sh 脚本本身也提供了限流的功能，只需一个 throttle 参数即可，具体用法如下 ： 

```shell
[root@node1 kafka-2.11-2.0.0] bin/kafka-reassign-partitions.sh --zookeeper localhost:2181/kafka --execute --reassignment-json-file project.json --throttle 10

Current partition replica assignment
{
	"version"： 1,
	"partitions": 
	[
		{ 
			"topic": "topic-throttle",
			"partition": 2,
			"replicas": [2, 0],
			"log_dirs": ["any", "any"]
        }, 
        { 
            "topic": "topic-throttle",
            "partition"： 1,
            "replicas": [1, 2],
            "log_dirs": ["any", "any"]
         },
         { 
         	"topic": "topic-throttle",
         	"partition": 0,
         	"replicas": [0, 1],
         	"log_dirs": ["any", "any"]
         }
    ]
}

Save this to use as the --reassignment-json-file option during rollback
Warning : You must run Verify periodically, until the reassignment completes, to ensure the throttle is removed. You can also alter the throttle by rerunning the Execute command passing a new value.
The inter-broker throttle limit was set to 10 B/s
Successfully started reassignment of partitions.
```

**包含了明确的告警信息 ：需要周期性地执行查看进度的命令直到重分配完成，这样可以确保限流设置被移除**。

如果想在重分配期间修改限制来增加吞吐量，以便完成得更快， 则可以重新运行 `kafka-reassign-partitions.sh` 脚本的 execute 命令 ，使用相同的 reassignment-json-file ，示例如下： 

```shell
[root@node1 kafka-2.11-2.0.0] bin/kafka-reassign-partitions.sh --zookeeper localhost:2181/kafka --execute --reassignment-json-file project.json --throttle 1024
There is an existing assignment running.
```

注意：上面最后一行的提示，表明重分配期间的限流更改。

可以查看对应的 ZooKeeper 节点内容 ：

```shell
[zk:localhost:2181/kafka(CONNECTED) 30] get /config/topics/topic-throttle

{
	"version": 1,
	"config": 
	{
		"follower.replication.throttled.replicas": "1:0, 0:2",
		"leader.replication.throttled.replicas": "1:1, 1:2, 0:0, 0:1" 
	}
}
```

可以看到 ZooKeeper 节点内容中的限流副本列表和前面使用 `kafka-config.sh` 脚本时的一样。其实 `kafka-reassign-partitions.sh` 脚本提供的限流功能背后的实现原理就是配置与限流相关的那 4 个参数而己，没有什么太大的差别。**不过使用 `kafka-config.sh` 脚本的方式来实现复制限流的功能比较烦琐，并且在手动配置限流副本列表时也比较容易出错，这里推荐大家使用 `kafka-reassign-partitions.sh` 脚本配合 throttle 参数的方式，方便快捷且不容易出错。** 

### 变更副本因子

---

<p><center><font color="#f08200" size=4><strong>场景需求</strong></font></center></p> 
- 在创建主题时填写了错误的副本因子数而需要修改
- 运行一段时间之后想要通过增加副本因子数来提高容错性和可靠性。 

---

本节中修改副本因子的功能也是通过重分配所使用的 `kafka-reassign-partition.sh` 脚本实现的。 

```json
{ 
	"version": 1, 
	"partitions": [
		{
			"topic": "topic-reassign",
			"partition": 2, 
			"replicas": [2, 1],
			"log_dirs": [
                "any",
                "any"
            ]
		}, 
		{
			"topic": "topic-reassign ",
			"partition": 1,
			"replicas": [1, 0],
			"log_dirs": [
                "any", 
                "any"
            ]
		}, 
		{
			"topic": "topic- reassign",
			"partition": 0,
			"replicas": [0, 2] , 
			"log_dirs ": [
                "any",
                "any"
            ]
		}
	]
}
```

可以观察到 JSON 内容里的 replicas 都是 2 个副本，我们可以自行对每个分区添加一个副本，注意增加副本因子时也要在 log_dirs 中添加一个 “any”，这个 log_dirs 代 表 Kafka 中的日志目录， 对应于 broker 端的 log.dir 或 log.dirs 参数的配置值，如果不需要关注此方面的细节 ，那么可以简单地设置为 “any”。我们将修改后的 JSON 内容保存为新的 add.json 文件 。 比如：

```shell
{
			"topic": "topic-reassign",
			"partition": 0, 
			"replicas": [1, 2, 0],
			"log_dirs": [
                "any",
                "any",
                "any"
            ]
}
```

执行 `kafka-reassign-partition.sh` 脚本（ execute ），详细信息如下： 

```shell
[root@node1 kafka-2.11-2.0.0] bin/kafka-reassign-partitions.sh --zookeeper localhost:2181/kafka --execute --reassignment-json-file add.json

Current partition replica assignment
{ 
	"version": 1, 
	"partitions": [
		{
			"topic": "topic-reassign",
			"partition": 2, 
			"replicas": [2, 1],
			"log_dirs": [
                "any",
                "any"
            ]
		}, 
		{
			"topic": "topic-reassign ",
			"partition": 1,
			"replicas": [1, 0],
			"log_dirs": [
                "any", 
                "any"
            ]
		}, 
		{
			"topic": "topic- reassign",
			"partition": 0,
			"replicas": [0, 2] , 
			"log_dirs ": [
                "any",
                "any"
            ]
		}
	]
}

Save this to use as the --reassignment-json-file option during rollback
Successfully started reassignment of partitions.
```

执行之后再次查看主题 topic-throttle 的详细信息，详细信息如下 ： 

```shell
[root@node1 kafka 2.11-2.0.0] bin/kafka-topics.sh --zookeeper localhost:2181/kafka --describe --topic topic-throttle
Topic: topic-throttle PartitionCount : 3 ReplicationFactor: 3 Configs :
Topic: topic-throttle Partition : 0 Leader: 0 Replicas : 0, 1, 2 Isr : 0, 1, 2
Topic: topic-throttle Partition : 1 Leader: 1 Replicas : 0, 1, 2 Isr : 2, 1, 0
Topic: topic-throttle Partition : 2 Leader: 2 Replicas : 0, 1, 2 Isr : 2, 0, 1
```

**与修改分区数不同的是，副本数还可以减少** 。再次修改 project.json 文件中的内容，内容参考如下 ： 

```shell
{ 
	"version": 1, 
	"partitions": [
		{
			"topic": "topic-reassign",
			"partition": 2, 
			"replicas": [0],
			"log_dirs": [
                "any"
            ]
		}, 
		{
			"topic": "topic-reassign ",
			"partition": 1,
			"replicas": [1],
			"log_dirs": [
                "any"
            ]
		}, 
		{
			"topic": "topic- reassign",
			"partition": 0,
			"replicas": [2] , 
			"log_dirs ": [
                "any"
            ]
		}
	]
}
```

再次执行 kafka-reassign-partition.sh 脚本（ execute ）之后，主题 topic-thro忧le 的详细信息如下： 

```shell
[root@node1 kafka 2.11-2.0.0] bin/kafka-topics.sh --zookeeper localhost:2181/kafka --describe --topic topic-throttle
Topic: topic-throttle PartitionCount : 3 ReplicationFactor: 3 Configs :
Topic: topic-throttle Partition : 0 Leader: 0 Replicas : 2 Isr : 2
Topic: topic-throttle Partition : 1 Leader: 1 Replicas : 1 Isr : 1
Topic: topic-throttle Partition : 2 Leader: 2 Replicas : 0 Isr : 0
```

不过在真实应用 中， 可能面对的是一个包含了几十个 broker 节点的集群 ， 将副本数从 2 修改为 5 ，或者从 4 修改为 3 的时候 ， 如何进行合理的分配是一个关键的问题。 下面演示了**如何通过程序来计算出分配方案** :

```scala
object ComputeReplicaDistribution {
    val partitions = 3
    val replicaFactor = 2
    def main(args : Array[String)) : Unit＝{
        val brokerMetadatas = List(new BrokerMetadata(0, Option("rack1")),
            new BrokerMetadata(1, Option ("rack1")),
            new BrokerMetadata(2, Option ("rack1")))
        val replicaAssignment = AdminUtils.assignReplicasToBrokers(brokerMetadatas, partitions, replicaFactor)
		println(replicaAssignment)
	}
}                          
```

代码中计算的是集群节点为  [0 , 1 , 2］ 、分区数为 3 、副本因子为 2、无机架信息的分配方案 ，程序输出如下 ：

```scala
Map(2 -> ArrayBuffer(0, 2), 1 -> ArrayBuffer(2, 1), 0 -> ArrayBuffer(1, 0))
```

分区 2 对应于[0, 2] ，分区 1 对应于[2 , 1] ，分区 0 对应于[1, 0］。

###  如何选择合适的分区数

---

<p><center><font color="#f08200" size=4><strong>场景需求</strong></font></center></p>
如何选择合适的分区数？这是很多 Kafka 的使用者经常面临的问题 ，毕竟分区影响各方面的性能。 不过对这个问题而言，似乎并没有非常权威的答案。 而且这个问题显然也没有固定的答案，只能从某些角度来做具体的分析，最终还是要根据实际的业务场景、软件条件、硬件条件、负载情况等来做具体的考量。 

---

#### 性能测试工具

在实际生产环境中，我们需要了解一套硬件所对应的性能指标之后才能分配其合适的应用和负荷，所以性能测试工具必不可少。性能测试工具是 Kafka 本身提供的用于生产者性能测试的 kafka-producer-perf-test.sh 和用于消费者性能测试的 kafka-consumer-perf-test.sh。

#### 分区数越多吞吐量就越高

消息中 间件的性能一般是指吞吐量（广义来说还包括延迟）。抛开硬件资源的影响，消息写入的吞吐量还会受到消息大小、消息压缩方式、消息发送方式（同步／异步）、消息确认类型( acks ） 、 副 本因子等参数 的影响， 消息消费 的 吞吐量还会受到应用逻辑处理速度 的影响。 

下面所有的测试除了主题的分区数不同，其余的因素都保持相同 。 首先分别创建分区数为 l 、 20 、
50 、 100 、 200 、 500 、 1000 的主题，对应 的主题名称分别为 topic-1、 topic-20 、 topic-50 、 topic-100 、topic-200 、 topic - 500 、 topic - 1000 ，所有主题的副本因子都设置为 1。

![1570381718751](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1570381718751.png)

**随着分区数的增长，相应的吞吐量也跟着上涨。一旦分区数超过了某个阈值之后，整体的吞吐量是不升反降的。也就是说，并不是分区数越多吞吐量也越大。 这里的分区数临界阈值针对不同的测试环境也会表现出不同的结果 ， 实际应用中可以通过类似的测试案例（比如复制生产流量以便进行测试 回放〉来找到一个合理的临界值区间。** 

![1570381827082](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1570381827082.png)

**在同一套环境下，我们还可以测试一下同时往两个分区数为 200 的主题中发送消息的性能，假设测试结果中两个主题所对应的吞吐量分别为 A 和 B，再测试一下只往一个分区数为 200 的主题中发送消息的性能，假设此次测试结果中得到的吞吐量为 C，会发现 A<C 、 B<C 且 A+B>C 。可以发现 由于共享系统资源的因素， A 和 B 之间会彼此影响。通过 A+B>C 的结果，可知前一图中 topic-200 的那个点位也并没有触及系统资源的瓶颈，发生吞吐量有所下降的结果也并非是系统资源瓶颈造成的** 。 

#### 分区数的上限

```shell
java.io.IOException : Too many open files
```

异常中最关键的信息是“ Too many open flies”，这是一种常见的 Linux 系统错误，通常意味着文件描述符不足，它一般发生在创建线程、创建 Socket、打开文件这些场景下 。 在 Linux系统的默认设置下，这个文件描述符的个数不是很多 ，通过 ulimit 命令可以查看： 

```shell
[root@node1 kafka 2.11-2.0.0] ulimit -n
1024
[root@node1 kafka 2.11-2.0.0] ulimit -Sn
1024
[root@node1 kafka 2.11-2.0.0] ulimit -Hn
4096
```

ulimit  是在系统允许的情况下，提供对特定 shell 可利用的资源的控制。-H 和 -S 选项指定资源的硬限制和软限制 。 硬限制设定之后不能再添加，而软限制则可以增加到硬限制规定的值 。 如果 -H 和 -S 选项都没有指定，则软限制和硬限制同时设定 。 限制值可以是指定资源的数值或 hard 、 soft, unlimited 这些特殊值，其中 hard 代表当前硬限制， soft 代表当前软件限制， unlimited代表不限制 。 如果不指定限制值，则打印指定资源的软限制值，除非指定了  -H 选项 。 硬限制可以在任何时候、任何进程中设置，但硬限制只能由超级用户设置。软限制是内核实际执行的限制，任何进程都可以将软限制设置为任意小于等于硬限制的值 。 

如何避免这种异常情况？对于一个高并发、高性能的应用来说， 1024 或 4096 的文件描述符限制未免太少，可以适当调大这个参数。比如使用 `ulimit -n 65535` 命令将上限提高到65535 ，这样足以应对大多数的应用情况，再高也完全没有必要了。 

```shell
[root@node1 kafka 2.11-2.0.0] ulimit -n 65535
＃可以再次查看相应的软硬限制数
[root@node1 kafka 2.11-2.0.0] ulimit -Hn
65535
[root@node1 kafka 2.11-2.0.0] ulimit -Sn
65535
```

也可以在 `/etc/security/limits.conf` 文件中设置，参考如下： 

```shell
#nofile - max number of open file descriptors
root soft nofile 65535
root hard nofile 65535
```

limits.conf 文件修改之后 需要重启才能生效。 limits.conf 文件与 ulimit 命令的区别在于前者是针对所有用户的，而且在任何 shell 中都是生效的，即与 shell 无关，而后者只是针对特定用户的当前 shell 的设定。在修改最大文件打开数时，最好使用 limits.conf 文件来修改，通过这个文件，可以定义用户、资源类型、软硬限制等。也可以通过在 `/etc/profile` 文件中添加 ulimit 的设置语句来使全局生效 。 

#### 分区数考量因素

如果应用对吞吐量有一定程度上的要求 ， 则建议在投入生产环境之前对同款硬件资源做一个完备的吞吐量相关的测试，以找到合适的分区数阔值区间 。

在创建主题之后，虽然我们还能够增加分区的个数，但**基于 key 计算的主题**需要严谨对待 。当生产者向 Kafka 中写入基于 key 的消息时， Kafka 通过消息的 key 来计算出消息将要写入哪个具体的分区，这样具有相同 key 的数据可以写入同一个分区 。 Kafka 的这一功能对于一部分应用是极为重要的，比如日志压缩（ Log Compaction ）， 再比如对于同一个 key 的所有消息，消费者需要按消息的顺序进行有序的消费，如果分区的数量发生变化，那么有序性就得不到保证 。 在创建主题时，最好能确定好分区数，这样也可以省去后期增加分区所带来的多余操作 。 尤其对于与 key 高关联的应用，在创建主题时可以适当地多创建一些分区，以满足未来的需求。通常情况下，可以根据未来 2 年内的目标吞吐量来设定分区数 。 当然如果应用与 key 弱关联，并且具备便捷的增加分区数的操作接口，那么也可以不用考虑那么长远的目标。

有些应用场景会**要求主题中的消息都能保证顺序性**，这种情况下在创建主题时可以设定分区数为 1 ，通过分区有序性的这一特性来达到主题有序性的目的 。 **提示**：发生重试的话，就无法保证全局有序

**分区数会占用文件描述符**，而一个进程所能支配的文件描述符是有限的，这也是通常所说的文件句柄的开销。虽然我们可以通过修改配置来增加可用文件描述符的个数，但凡事总有一个上限，在选择合适的分区数之前，最好再考量一下当前 Kafka 进程中己经使用的文件描述符的个数 。 

当 broker 发生故障时， leader 副本所属宿主的 broker 节点上的所有分区将暂时处于不可用的状态。分区在进行 leader 角色切换的过程中会变得不可用，不过对于单个分区来说这个过程非常短暂，对用户而言可以忽略不计 。 **如果集群中的某个 broker 节点岩机，那么就会有大量的分区需要同时进行 leader 角色切换，这个切换的过程会耗费一笔可观的时间，并且在这个时间窗口内这些分区也会变得不可用 。分区数越多也会让 Kafka 的正常启动和关闭的耗时变得越长，与此同时 ，主题的分区数越多不仅会增加日志清理的耗时，而且在被删除时也会耗费更多的时间 。**  

**在设定完分区数，或者更确切地说是创建主题之后，还要对其追踪、监控、调优以求更好地利用它** 。一般情况下，根据预估的吞吐量及是否与 key 相关的规则来设定分区数即可，后期可以通过增加分区数、增加 broker 或分区重分配等手段来进行改进。如果一定要给一个准则，则建议将分区数设定为集群中 broker 的倍数 ，不过，如果集群中的 broker 节点数有很多，比如大几十或上百、上千，那么这种准则也不太适用，在选定分区数时进一步可以引入基架等参考因素 。  

### 参考

1. 《深入理解Kafka核心设计与实践原理》

2.  Kafka 官方文档

    
