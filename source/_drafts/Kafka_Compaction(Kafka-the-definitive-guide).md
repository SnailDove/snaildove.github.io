## Compaction 压缩

Normally, Kafka will store messages for a set amount of time and purge messages older than the retention period. However, imagine a case where you use Kafka to store shipping addresses for your customers. In that case, it makes more sense to store the last address for each customer rather than data for just the last week or year. This way, you don’t have to worry about old addresses and you still retain the address for customers who haven’t moved in a while. Another use case can be an application that uses Kafka to store its current state. Every time the state changes, the application writes the new state into Kafka. When recovering from a crash, the application reads those messages from Kafka to recover its latest state. In this case, it only cares about the latest state before the crash, not all the changes that occurred while it was running.

通常，Kafka希望将消息存储一定的时间，并清除早于保留期限的消息。但是，设想一下使用Kafka为客户存储送货地址的情况。在这种情况下，为每个客户存储最后一个地址比仅存储最后一周或上一年更有意义。这样，您仍然必须保留客户地址。 Kafka的另一个用例用于存储其当前状态。每次状态更改时，应用程序都会将新状态写入Kafka。从崩溃中恢复时，Kafka恢复其最新状态。在这种情况下，它只关心崩溃前的最新状态，而不关心运行时发生的所有更改。

Kafka supports such use cases by allowing the retention policy on a topic to be delete, which deletes events older than retention time, to compact, which only stores the most recent value for each key in the topic. Obviously, setting the policy to compact only makes sense on topics for which applications produce events that contain both a key and a value. If the topic contains null keys, compaction will fail.

Kafka支持在主题上使用策略以进行删除（删除比保留时间更久远的事件），以便压缩，压缩仅存储主题中每个键的最新值。显然，将策略设置为紧凑仅对应用程序产生了键值对的主题有效。如果主题包含空键，则压缩将失败。

## How Compaction Works 压缩如何工作

Each log is viewed as split into two portions (see Figure 5-7): 

每个日志都被视为分为两部分（请参见图5-7）：

![1569250115553](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1569250115553.png)

**Clean 干净的记录** 

Messages that have been compacted before. This section contains only one value for each key, which is the latest value at the time of the pervious compaction. 

以前已压缩的消息。此部分的每个键仅包含一个值，该值是前一次压缩时的最新值。

**Dirty 脏记录**

Messages that were written after the last compaction.

上次压缩后写入的消息。

If compaction is enabled when Kafka starts (using the awkwardly named `log.cleaner.enabled` configuration), each broker will start a compaction manager thread and a number of compaction threads. These are responsible for performing the compaction tasks. Each of these threads chooses the partition with the highest ratio of dirty messages to total partition size and cleans this partition.

如果在启动Kafka时启用了压缩（使用命名笨拙的配置项： `log.cleaner.enabled` ），则每个代理（broker）都希望启动一个压缩管理器线程和多个压缩线程。这些负责执行压缩任务。这些线程中的每个线程都会选择脏消息占总分区大小比例最高的分区，并清理该分区。

To compact a partition, the cleaner thread reads the dirty section of the partition and creates an in-memory map. Each map entry is comprised of a 16-byte hash of a message key and the 8-byte offset of the previous message that had this same key. This means each map entry only uses 24 bytes. If we look at a 1 GB segment and assume that each message in the segment takes up 1 KB, the segment will contain 1 million such messages and we will only need a 24 MB map to compact the segment (we may need a lot less—if the keys repeat themselves, we will reuse the same hash entries often and use less memory). This is quite efficient! 

为了压缩分区，清理程序线程读取分区的脏区并创建内存映射表（map）。内存映射表（map）的每一项由消息的 key 的16字节哈希值以及具有相同key的上一条消息的8字节偏移量（offset）。这意味着每项只使用24字节。如果我们看一个1GB 分段以及假设段中的每个消息仅占用 1 KB，分段将包含100万条这样的消息，那么我们将只需要一个 24 MB的内存映射表（map）来压缩该分段（我们可能需要少很多，如果键重复，我们将经常重复使用相同的哈希条目，并使用较少的内存）这非常高效！

When configuring Kafka, the administrator configures how much memory compaction threads can use for this offset map. Even though each thread has its own map, the configuration is for total memory across all threads. If you configured 1 GB for the compaction offset map and you have five cleaner threads, each thread will get 200 MB for its own offset map. Kafka doesn’t require the entire dirty section of the partition to fit into the size allocated for this map, but at least one full segment has to fit. If it doesn’t, Kafka will log an error and the administrator will need to either allocate more memory for the offset maps or use fewer cleaner threads. If only a few segments fit, Kafka will start by compacting the oldest segments that fit into the map. The rest will remain dirty and wait for the next compaction.

配置Kafka时，管理员将配置压缩线程可以使用多少内存用于偏移量映射表。即使每个线程都有自己的映射表，该配置仍适用于所有线程中的总内存。如果为压缩偏移量映射表配置了1 GB，并且有五个清洁线程，则每个线程将为其自己的偏移量映射表获得 200 MB。 Kafka不需要分区的整个脏记录部分去符合分配给映射表的大小，但必须至少包含一个完整的部分。如果不是，则Kafka会日志报错，管理员希望分配更多的内存用于偏移量映射表或者使用更少的清理线程。如果只有几个片段适配，Kafka希望首先压缩符合内存映射表的最旧分段。其余的想要保留脏记录并等待下一次压实。

Once the cleaner thread builds the offset map, it will start reading off the clean segments, starting with the oldest, and check their contents against the offset map. For each message it checks, if the key of the message exists in the offset map. If the key does not exist in the map, the value of the message we’ve just read is still the latest and we copy over the message to a replacement segment. If the key does exist in the map, we omit the message because there is a message with an identical key but newer value later in the partition. Once we’ve copied over all the messages that still contain the latest value for their key, we swap the replacement segment for the original and move on to the next segment. At the end of the process, we are left with one message per key—the one with the latest value. See Figure 5-8.

清理程序线程构建偏移量映射表后，便希望开始读取干净的段，从最旧的段开始，并对照偏移量内存映射表检查其内容。对于每条消息，它都会检查消息键是否在偏移量内存映射表中。如果键在偏移量内存映射表不存在，则我们刚刚读取的消息的值是最新的，然后将消息复制到替换段。如果键确实存在于偏移量内存映射表中，我们将忽略该消息，因为在分区中稍后会有一条消息具有相同的键但值较新。一旦拷贝了所有消息的键的最新值，我们将替换段替换为原始段，然后移至下一个段。在此过程的最后，每个键只剩下一条消息-一条具有最新值的消息。请参阅图5-8。

![1569250398276](C:/Users/ruito/AppData/Roaming/Typora/typora-user-images/1569250398276.png)

## When Are Topics Compacted? 何时压缩主题？

In the same way that the delete policy never deletes the current active segments, the compact policy never compacts the current segment. Messages are eligble for compaction only on inactive segments.

就像删除策略从不删除当前活动段一样，压缩策略也从不压缩当前段。消息仅适用于非活动段上的压缩。

In version 0.10.0 and older, Kafka will start compacting when 50% of the topic contains dirty records. The goal is not to compact too often (since compaction can impact the read/write performance on a topic), but also not leave too many dirty records around (since they consume disk space). Wasting 50% of the disk space used by a topic on dirty records and then compacting them in one go seems like a reasonable trade-off, and it can be tuned by the administrator.

在0.10.0及更低版本中，当50％的主题包含脏记录时，Kafka将开始压缩。目的不是要太频繁地压缩（因为压缩会影响主题的读写性能），而且也不会留下太多脏记录（因为它们消耗磁盘空间）。浪费掉一个主题在脏记录上使用的磁盘空间的50％，然后一次性压缩它们，这似乎是一个合理的权衡，管理员可以对其进行调整。

In future versions, we are planning to add a grace period during which we guarantee that messages will remain uncompacted. This will allow applications that need to see every message that was written to the topic enough time to be sure they indeed saw those messages even if they are lagging a bit. 

在将来的版本中，我们计划增加一个宽限期，在此期间，我们保证消息不会压缩。这将使需要查看写给该主题的每条消息的应用程序有足够的时间来确保即使它们有些滞后，他们也确实看到了这些消息。
