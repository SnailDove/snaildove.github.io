## 零拷贝简述

之前就了解过Linux零拷贝，快速总结摘录一下，以后有机会再深入 Linux 零拷贝的过程。详细请查看文末的参考资料。

所谓的零拷贝（Zero-Copy）是指将数据直接从磁盘文件复制到网卡设备中，而不需要经由应用程序之手 。零拷贝大大提高了应用程序的性能，减少了内核和用户模式之间的上下文切换 。 对 Linux操作系统而言，零拷贝技术依赖于底层的 `sendfile()` 方法实现 。 对应于 Java 语言，`FileChannal.transferTo()` 方法的底层实现就是 `sendfile()` 方法。单纯从概念上理解 “零拷贝”比较抽象，这里简单地介绍一下它 。 考虑这样一种常用的情形 ： 你需要将静态内容(类似图片、文件)展示给用户 。 这个情形就意味着需要先将静态内容从磁盘中复制出来放到一个内存 buf 中，然后将这个 buf 通过套接字（Socket）传输给用户，进而用户获得静态内容 。 这看起来再正常不过了，但实际上这是很低效的流程，我们把上面的这种情形抽象成下面的过程 ：

```c
read(file, tmp_buf, len);
write(socket, tmp_buf, len);
```

首先调用 `read()` 将静态内容(这里假设为文件 A )读取到 tmp_buf， 然后调用 `write()` 将 tmp_buf 写入 Socket，如下图所示 。在这个过程中， 文件 A 经历了 4 次复制的过程： 

1. 调用 `read()` 时，文件 A 中的内容被复制到了内核模式下的 Read Buffer 中。
2. CPU 控制将内核模式数据复制到用户模式下 。
3. 调用 `write()` 时 ，将用户模式下的内容复制到内核模式下的 Socket Buffer 中 。
4. 将内核模式下的 Socket Buffer 的数据复制到网卡（NIC: network interface card）设备中传迭 。 

![](https://www.ibm.com/developerworks/cn/java/j-zerocopy/figure1.gif)

![](https://www.ibm.com/developerworks/cn/java/j-zerocopy/figure2.gif)



如果采用了零拷贝技术，那么应用程序可以直接请求内核把磁盘中的数据传输给 Socket, 如下图所示。 

![](https://www.ibm.com/developerworks/cn/java/j-zerocopy/figure3.gif)

![](https://www.ibm.com/developerworks/cn/java/j-zerocopy/figure4.gif)

所示的 `transferTo()` 方法时的步骤有：

1. `transferTo()` 方法引发 DMA 引擎将文件内容拷贝到一个读取缓冲区。然后由内核将数据拷贝到与输出套接字相关联的内核缓冲区。
2. 数据的第三次复制发生在 DMA 引擎将数据从内核套接字缓冲区传到协议引擎时。

改进的地方：我们将上下文切换的次数从四次减少到了两次，将数据复制的次数从四次减少到了三次（其中只有一次涉及到了 CPU）。但是这个代码尚未达到我们的零拷贝要求。如果底层网络接口卡支持*收集操作* 的话，那么我们就可以进一步减少内核的数据复制。在 Linux 内核 2.4 及后期版本中，套接字缓冲区描述符就做了相应调整，以满足该需求。这种方法不仅可以减少多个上下文切换，还可以消除需要涉及 CPU 的重复的数据拷贝。对于用户方面，用法还是一样的，但是内部操作已经发生了改变：

1. `transferTo()` 方法引发 DMA 引擎将文件内容拷贝到内核缓冲区。
2. 数据未被拷贝到套接字缓冲区。取而代之的是，只有包含关于数据的位置和长度的信息的描述符被追加到了套接字缓冲区。DMA 引擎直接把数据从内核缓冲区传输到协议引擎，从而消除了剩下的最后一次 CPU 拷贝。

展示了结合使用 `transferTo()` 方法和收集操作的数据拷贝：

![](https://www.ibm.com/developerworks/cn/java/j-zerocopy/figure5.gif)

### 性能比较

我们在一个运行 2.6 内核的 Linux 系统上执行了示例程序，并以毫秒为单位分别度量了使用传统方法和 `transferTo()` 方法传输不同大小的文件的运行时间。表 1 展示了度量的结果：

##### 表 1. 性能对比：传统方法与零拷贝

| 文件大小 | 正常文件传输（ms） | transferTo（ms） |
| :------- | :----------------- | :--------------- |
| 7MB      | 156                | 45               |
| 21MB     | 337                | 128              |
| 63MB     | 843                | 387              |
| 98MB     | 1320               | 617              |
| 200MB    | 2124               | 1150             |
| 350MB    | 3631               | 1762             |
| 700MB    | 13498              | 4422             |
| 1GB      | 18399              | 8537             |

如您所见，与传统方法相比，`transferTo()` API 大约减少了 65% 的时间。这就极有可能提高了需要在 I/O 通道间大量拷贝数据的应用程序的性能，如 Web 服务器。

### 参考

1. [走进科学之揭开神秘的零拷贝](https://github.com/javagrowing/JGrowing/blob/master/%E8%AE%A1%E7%AE%97%E6%9C%BA%E5%9F%BA%E7%A1%80/%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F/IO/%E8%B5%B0%E8%BF%9B%E7%A7%91%E5%AD%A6%E4%B9%8B%E6%8F%AD%E5%BC%80%E7%A5%9E%E7%A7%98%E7%9A%84%E9%9B%B6%E6%8B%B7%E8%B4%9D.md)

2. 《深入理解 Kafka核心设计与原理实现》

3. [通过零拷贝实现有效数据传输: Java实例](https://www.ibm.com/developerworks/cn/java/j-zerocopy/)

4. [Zero Copy I: User-Mode Perspective](https://www.linuxjournal.com/article/6345?page=0,1)

5. [翻译：Zero Copy I: User-Mode Perspective](https://leokongwq.github.io/2017/01/12/linux-zero-copy.html)

6. [It's all about buffers: zero-copy, mmap and Java NIO](https://link.jianshu.com/?t=http://xcorpion.tech/2016/09/10/It-s-all-about-buffers-zero-copy-mmap-and-Java-NIO/)
7. [Zero Copy I: User-Mode Perspective](https://link.jianshu.com/?t=http://www.linuxjournal.com/article/6345?page=0,1)
8. [Linux Programmer's Manual  SENDFILE(2)](https://link.jianshu.com/?t=http://man7.org/linux/man-pages/man2/sendfile.2.html)
9. [Linux 中的零拷贝技术，第 1 部分](https://link.jianshu.com/?t=https://www.ibm.com/developerworks/cn/linux/l-cn-zerocopy1/index.html)
10. [Linux 中的零拷贝技术，第 2 部分](https://link.jianshu.com/?t=https://www.ibm.com/developerworks/cn/linux/l-cn-zerocopy2/index.html)
11. [圣思园《精通并发与Netty》](https://link.jianshu.com/?t=https://mp.weixin.qq.com/s?__biz=MzIxOTYzMzExNA==&mid=2247483817&idx=1&sn=e3785b58cb8dc122279e743ad8feba66&chksm=97d9051ca0ae8c0a4312b162e3a8725328245d13308ad4ed3f0ad03a488b91c5e35bda7d7e3b&mpshare=1&scene=1&srcid=09044Yh36rUP5bKPdm0tDTc6&key=4761a0efa8f615feac7473a8a77e4bdcaa1ed3242de44a00287e4030d247f664496bf9011d63d15f1b16d82a921fb27b2c6ad289a74e401c3bc6cd1549c9e104d3be482e30b0fab30c7fec3c6ef3e5f7&ascene=0&uin=NzgxNzc5ODQw&devicetype=iMac+MacBookPro14%2C3+OSX+OSX+10.12.5+build(16F2073)&version=12020810&nettype=WIFI&fontScale=100&pass_ticket=3mPoRIbu%2BxBI0tcVEtfblC8Kiv2hezSrfO9othMf6obZMXtWP9QQetOWmOMJ9w9u)

    

