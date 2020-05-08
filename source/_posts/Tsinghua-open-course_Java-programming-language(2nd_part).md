---
title: Tsinghua-open-course Java programming language(2nd part)
mathjax: true
mathjax2: true
categories: 中文
tags: [java]
date: 2017-10-02
commets: true
copyright: true
toc: true
top: 5
---

本文是清华大学许斌老师的公开课：Java语言程序设计进阶 的课堂笔记，快速复习一下（我不是专门搞java的话），时间有限，因此大量直接截图。许斌老师声明：没有配套讲义，建议参考书籍：周志明《深入理解java虚拟机》。(JUC) java.utile.concurrency 部分参考源码和技术博客。

## 第一章 线程（上）

### 1.0 导学

![1539008813620](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539008813620.png)

![1539008859039](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539008859039.png)

### 1.1 线程的基本概念

![1539009969070](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539009969070.png)

![1539009984077](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539009984077.png)

![1539009143304](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539009143304.png)

![1539009203189](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539009203189.png)

![1539009430772](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539009430772.png)

![1539009574503](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539009574503.png)

![1539009665675](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539009665675.png)

###  1.2 通过Thread类创建线程

![1539009918717](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539009918717.png)

![1539406861072](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539406861072.png)

![1539406869998](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539406869998.png)

![1539406876613](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539406876613.png)

![1539406882916](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539406882916.png)

![1539406887715](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539406887715.png)

![1539406895994](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539406895994.png)

### 1.3 线程的休眠

![1539010059187](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539010059187.png)

![1539010375863](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539010375863.png)

![1539010405934](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539010405934.png)

![1539010618493](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539010618493.png)

![1539011791263](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539011791263.png)

![1539011800469](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539011800469.png)

![1539011806430](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539011806430.png)

**注**：线程休眠的原因就是让其他线程有执行的机会

### 1.4 Thread类详解

![1539012104103](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539012104103.png)

![1539012342296](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539012342296.png)

**注**：线程启动（即调用start方法）并不意味着线程马上运行，线程是否运行取决于线程调度器。

![1539012415734](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539012415734.png)

### 1.5 通过Runnable接口创建线程

![1539012516154](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539012516154.png)

![1539012651662](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539012651662.png)

![1539012759620](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539012759620.png)

![1539012814520](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539012814520.png)

![1539012944363](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539012944363.png)

![1539012958930](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539012958930.png)

![1539013125468](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539013125468.png)

**注**：Runnable接口中我们所实现的run方法就是我们这个线程想要执行的代码

### 1.6 线程内部的数据共享

同样一个线程类，它可以实例化出很多线程。同样一个线程，它们是可以共享它们的代码和数据，那也就是说当我们实现了Runnable接口的这个类，它所实例出来的对象的话，它去构造出的线程，它们之间是可以共享它们的代码和它们之间的一些数据的。

![1539013882402](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539013882402.png)

![1539012958930](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539012958930.png)

![1539014130891](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539014130891.png)

![1539014260700](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539014260700.png)

![1539014850716](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539014850716.png)

![1539014878920](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539014878920.png)

![1539016663199](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539016663199.png)

### 小结

![1539016705556](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539016705556.png)

## 第二章 线程（中）

### 2.0 导学

![1539016795632](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539016795632.png)

![1539016830468](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539016830468.png)

### 2.1 线程同步的思路

![1539017054833](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539017054833.png)

![1539017181751](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539017181751.png)

![1539017224926](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539017224926.png)

![1539017381221](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539017381221.png)

![1539017485915](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539017485915.png)

![1539017568234](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539017568234.png)

![1539017702394](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539017702394.png)

![1539017744493](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539017744493.png)

**注**：那原因就是在于说这两个线程的话，它们是同一优先级，只不过是说这个producer先这个排在前面，所以的话从调度上，往往会调度它这个producer先执行，那它一执行呢就把这个票都生产完了，然后再等待着卖票的程序把它去卖掉，这是一种有意思的这个现象

![1539017987439](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539017987439.png)

### 2.2 线程同步的实现方式—Synchronization

![1539018289399](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539018289399.png)

![1539018392108](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539018392108.png)

![1539018487545](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539018487545.png)

![1539018879503](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539018879503.png)

**注**：把这两行代码变成一个（原子）操作，就是在执行过程中不可能被打散执行

![1539019284302](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539019284302.png)

**注**：用synchronized后面大括号括起来其实是代码，实际上它把它变成一个原子操作，也就是说当我拿到这个对象t的锁的时候，我这里面的这些代码是肯定都会被执行的，不会说我执行某一句以后就被这个打断，然后那个插入别的线程去执行去访问这个对象t，所以这个是synchronized它的很重要的作用。

![1539019370141](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539019370141.png)

![1539019483438](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539019483438.png)

![1539019531529](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539019531529.png)

![1539019552318](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539019552318.png)

![1539019688049](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539019688049.png)

就像刚才我们那个例子：我们在这个售票线程里面，每售出来票的时候，它就会休眠一毫秒，但休眠一毫秒的时候，它不会释放出它所占有的这个ticket对象的锁的，它一直会持有，所以这是一个独特的一个地方。

![1539023144629](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539023144629.png)

### 2.3 线程的等待与唤醒

![1539071230986](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539071230986.png)

![1539071238290](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539071238290.png)

**注**：那现在wait notify notifyAll方法这三个方法都属于object这个类的方法，也就意味着我们java当中所有的类它都有这个三个方法

![1539071537462](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539071537462.png)

**注**：修改之后，相当于票箱大小为1，Tickets.size = 1。Tickets.put() 方法中的 notify() 与 Tickets.sell() 方法中的wait()一一对应，Tickets.put() 方法中的 wait() 与 Tickets.sell() 方法中的notify()一一对应。

![1539071636937](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539071636937.png)

### 2.4 后台进程

![1539072724499](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539072724499.png)

![1539072801199](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539072801199.png)

### 2.5 线程的生命周期与死锁

![1539076855969](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539076855969.png)

![1539076979258](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539076979258.png)

**注**：线程进入就绪状态就是runnable state，即可运行状态，但是并未开始运行，所以不是运行状态（running state），是否运行取决于线程调度器是否调度它。（Runnable state isn't running state）。

![1539077334242](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539077334242.png)

![1539077435366](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539077435366.png)

![1539077461022](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539077461022.png)

![1539077569978](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539077569978.png)

![1539077751809](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539077751809.png)

![1539077820164](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539077820164.png)

![1539077847191](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539077847191.png)

![1539077924190](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539077924190.png)

![1539077981985](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539077981985.png)

### 2.6 线程的调度

![1539073197605](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539073197605.png)

![1539073339084](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539073339084.png)

![1539073553254](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539073553254.png)

**注**：可以通过这个使用这个yield的方法来去稍微改变一下它的这个执行过程，yield方法主要作用是把自己当前运行的线程暂停下来，把线程让给同优先级的线程执行，当如果这时候不存在同优先级的线程，那还是继续执行当前运行的线程。

![1539073606722](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539073606722.png)

![1539075549275](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539075549275.png)

![1539075786770](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539075786770.png)

![1539075932520](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539075932520.png)

![1539075996274](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539075996274.png)

![1539076123946](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539076123946.png)

![1539076577555](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539076577555.png)

**注**：有交错执行的这个过程，重要的原因：线程调用sleep方法，sleep方法是说我自己进入休眠，线程调度器有可能调度低优先级的这个线程，也就是说对高优先级的这个线程，如果要让出自己的执行权限的话，就要调用sleep方法，然后给其它低优先级线程机会。如果高优先级的线程，仅仅只是调用了yield方法，它并不能给我们低优先级线程以执行的机会，它只给了它同优先级的线程以执行的这个机会。

### 小结

![1539076703328](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539076703328.png)

## 第三章 线程（下）线程安全与锁优化

### 3.0 导学

![1539078195419](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539078195419.png)

**注**：它是想描述线程的安全，而最重要的是描述你的程序，甚至你是某个类它的线程安全的特性。

### 3.1 线程安全与线程兼容与对立

![1539092094328](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539092094328.png)

![1539092137232](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539092137232.png)

![1539092210369](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539092210369.png)

![1539093903103](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539093903103.png)

![1539093969693](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539093969693.png)

向大家展示一下，一些java API中类，它在碰到这个线程操作的时候，有可能产生线程出错的这个情况。

![1539094981555](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539094981555.png)

![1539095091501](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539095091501.png)

运行过程当中它不经常出错，但是偶尔也会出错，出现了数组下标越界的错误。最重要的原因：刚才有两个线程Thread remove,Thread print这两个线程都在同时访问一个数据：vector，其中一个线程的操作：删除我们相量中的元素，另外一个线程的操作读取我们相量中的元素。大家看到这其实这两个操作：是有点互逆的 互斥的，那在这个读写向量的过程当中就可能产生错误，那从我们发现了这个运行的结果当中也发现了这一点。

![1539095449832](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539095449832.png)

![1539095474977](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539095474977.png)

![1539095546295](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539095546295.png)

### 3.2 线程的安全实现-互斥同步

![1539099909957](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539099909957.png)

![1539100009472](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539100009472.png)

![1539100178410](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539100178410.png)

![1539100716303](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539100716303.png)

![1539100471798](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539100471798.png)

### 3.3 线程的安全实现-非阻塞同步

![1539101286254](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539101286254.png)

![1539101341800](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539101341800.png)

![1539101432019](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539101432019.png)

![1539101645384](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539101645384.png)

那其实大家也可以这么理解，也就是说我们对于这种线程安全，就是对我们这个访问对象的线程安全的这种控制不是放到我们当前count这个类，它的increment方法来去实现的而是已经放到底下的叫 Atomiclnteger 这个类来实现了，所以你就可以直接去调用它的这个方法来去实现加1的这个功能，那整体上我们这个新的类,class Counter就是这个类通过用 Atomiclnteger 来改进这类,它也是线程安全的,整体上也是线程安全的,只不过说当你写这个类的时候，你不需要考虑自己去加上synchronized这样的同步互斥的这种实现方式,而是通过直接使用了Atomiclnteger这样一个本身就是线程安全的这个类，就能够保证你的整个这个代码达到线程安全的目的。

### 3.4 线程的安全实现-无同步方案

![1539102939632](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539102939632.png)

**注**：Threadlocal是我们java当中的一个类，它是存在于java.lang这个默认这个包当中。

![1539103649773](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539103649773.png)

![1539103656234](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539103656234.png)

这个 SequenceNumber 的实例，通过用 ThreadLocal 的这个方式也能够保证这三个线程来访问同一数据的时候，没有产生错误。这也是为什么说，可以通过这个 ThreadLocal 就来去达到这个同步，就是说安全的这个目的。也不一定非得加个synchronize，因为如果一旦加了synchronize的话，性能可能会受到影响，如果能通过类似 ThreadLocal 这样这种线程的本地存储的方式来达到我们这个对于数据访问安全的控制化，那就能提高这个程序代码的性能。

### 3.5 锁优化

![1539104257575](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539104257575.png)

![1539104378267](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539104378267.png)

![1539104424027](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539104424027.png)

![1539104476304](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539104476304.png)

操作系统的堆栈与数据结构中堆栈概念参考:

1. 什么是堆？什么是栈？他们之间有什么区别和联系？ - 知乎
   https://www.zhihu.com/question/19729973
2. https://jingyan.baidu.com/article/6c67b1d6a09f9a2786bb1e4a.html

![1539105602554](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539105602554.png)

**注**：由于细锁太多，然后不断切换线程的开销反而降低了性能。

![1539105752008](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539105752008.png)

### 小结

![1539105852330](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539105852330.png)

- 线程安全指的是我们的访问对象无论被多少个线程进行访问都能够保证我们这对象访问的正确性，那与之相关的这个概念是线程兼容。
- 线程兼容是指我们的对象本身不是线程安全的，但是通过我们外部的同步控制能够达到线程安全的目的.
- 线程对立指的是我们的访问对象，它本身不是线程安全，那我们外部即使加上了同步的控制也不能保证这个对象的这个正确性。
- 我们还学习实现线程安全的几种方式
  - 首先是互斥同步
  - 其次是非阻塞同步
  - 无同步方案
- 最后我们还学习了锁优化，那锁优化的目标就是在我们不得不给我们的代码加锁的情况下如何去提高锁的效率，进而达到提升整个代码的效率的目标。

## 第四章 网络编程（上）

### 4.0 导学

![1539244505401](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539244505401.png)

![1539244534446](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539244534446.png)

![1539244801576](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539244801576.png)

### 4.1 URL对象

![1539245451379](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539245451379.png)

注：保留端口号是计算机系统进行网络交互需要的端口号，自己编写程序不要去占用这些保留端口号，具体保留端口号对应网络服务google一下。

![1539245767019](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539245767019.png)

![1539245834109](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539245834109.png)

![1539245898509](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539245898509.png)

![1539245956762](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539245956762.png)

![1539245984052](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539245984052.png)

![1539246010514](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539246010514.png)

![1539246052077](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539246052077.png)

### 4.2 URLConnection对象

![1539246211340](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539246211340.png)

![1539246251876](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539246251876.png)

![1539246318190](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539246318190.png)

### 4.3 Get请求与Post请求

![1539246522880](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539246522880.png)

![1539246527495](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539246527495.png)

![1539246570432](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539246570432.png)

![1539246720288](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539246720288.png)

![1539246836812](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539246836812.png)

![1539246936840](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539246936840.png)

### 4.4 Socket通信原理

![1539248246737](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539248246737.png)

![1539248277357](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539248277357.png)

![1539248323241](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539248323241.png)

![1539248356777](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539248356777.png)

![1539248563289](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539248563289.png)

### 4.5 Socket通信实现

![1539249189011](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539249189011.png)

![1539249549584](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539249549584.png)

![1539249542348](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539249542348.png)

accept这个方法属于ServerSocket方法，这种方法我们称之它为阻塞方法，就是说它是在那里运行一直等着有客户端来给它发送Socket连接请求，如果没有客户端给我们的服务端发送这个Socket连接请求accept就一直在那里循环执行一直不返回，一直等到有客户端的Socket发连接请求过来。那我们这服务端的 ServerSocket 这个对象的话它就会accept方法就会返回一个值，返回的是一个Socket对象，而这个Socket对象就是和我们客户端的Socket对象进行对应的。

![1539249811812](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539249811812.png)

![1539249848946](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539249848946.png)

![1539250143290](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539250143290.png)

![1539250150533](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539250150533.png)

![1539250156213](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539250156213.png)

![1539252104203](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539252104203.png)

![1539252110980](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539252110980.png)

![1539252121234](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539252121234.png)

![1539252126376](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539252126376.png)

![1539252147933](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539252147933.png)

**注**：那需要提醒大家注意是我们这个程序非常简单，简单到什么程度呢？就是说聊天的时候，都是你说一句 我说一句如果一个想连续说两句话的话可能现有这个机制还处理不过来，必须是一人一句，当然我们同学可以把这个程序再进一步改进使它更加的丰富。

### 小结

![1539252275033](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539252275033.png)

## 第五章 网络编程（下）

### 5.0 导学

![1539254072310](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539254072310.png)

### 5.1 Socket 多客户端通信实现

![1539254198089](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539254198089.png)

![1539254477911](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539254477911.png)

![1539254535776](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539254535776.png)

![1539254560475](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539254560475.png)

![1539254578716](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539254578716.png)

![1539254484574](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539254484574.png)

**注**：先运行server线程，再运行client。

### 5.2 数据报通信

![1539267932034](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539267932034.png)

![1539267969444](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539267969444.png)

![1539268051255](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539268051255.png)

![1539268477916](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539268477916.png)

![1539268866026](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539268866026.png)

![1539268872283](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539268872283.png)

![1539268971931](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539268971931.png)

one-liners.txt这个是构造了一个文件输入流，因为我们做了一个非常简单的模拟，也就是说把一些股票信息就写到这个文件里面了，写到这个文件里面了以后就是每次有客户端发过来请求，说咨询一下股票的价格的时候，我们就从这个文件里读出某一股票的价格在返还给我们的客户端。

![1539270238283](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539270238283.png)

![1539270248439](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539270248439.png)

![1539270313592](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539270313592.png)

![1539270333292](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539270333292.png)

![1539270367641](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539270367641.png)

那刚才这个程序当中客户端服务端各一个程序，客户端和服务端之间的通讯是通过数据报这个Socket来进行通讯的然后整个过程就非常类似于，我们人类进行平信这个通讯方式，也就是说客户端通过构造一个DatagramPacket这个对象向它写一封信，然后通过DatagramSocket的send方法把它发出去了，服务端收到了这封来信以后，通过这个来信知道了客户端的地址和端口号，然后服务端它自己也写一封信，说白了写信就是构造一个DatagramPacket对象，写好了以后，通过DatagramSocket的这个对象的send方法，把这个信再发出去又发还给客户端，所以这个数据报包总结起来就非常类似于我们人类写平信的这个过程。

### 5.3 使用数据报进行广播通信

![1539271072535](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539271072535.png)

![1539271612788](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539271612788.png)

![1539271626548](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539271626548.png)

![1539271671371](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539271671371.png)

![1539271676934](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539271676934.png)

![1539271843572](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539271843572.png)

![1539271849995](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539271849995.png)

### 5.4 网络聊天程序

![1539279041381](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539279041381.png)

那整个这个布局其实大家采用一个BorderLayout就可以达到你的目标，那就是在BorderLayout的center中间那区域先放一个滚动面板，然后接着再放一个TextArea，然后在它的南部区域，我们先放一个Panel，紧接着再放一个TextView文本输入区域，然后接着再放一个Button而且TextView和Button的话都是按照FlowLayout这种放置规则。

大家需要这个写的事件响应是什么呢？其实首先最重要的是说我们能够接收这个在文本区域，就是最下面这个文本区域这个输入的文本，我们可以给TextView这个组件来注册一个监听器，当我们这个一回车就在TextView里面一输入字符一回车的时候，它产生的是一个ActionEvent，所以我们可以给TextView注册一个EventListener。那TextView旁边的话，是一个按钮发送，其实发送的话它所对应的这个事件处理也是ActionEvent，所以在这个例子当中我们只需要写一个事件处理类然后都分别这个授权来去处理TextView和我们按钮的这么这个事件处理就可以完成获得我们这个文本的这么一个过程以及把它发送的一个过程。那在这里面怎么去获得内容呢？TextView里面有个一个getText的方法，那我们只要在我们的ActionPerform方法里面去通过TextView的getText来获得它的内容然后来决定一个是往外发送同时把它显示到当前我们的这个界面上面。

![1539280107792](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539280107792.png)

![1539280120139](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539280120139.png)

![1539280125626](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539280125626.png)

![1539280132480](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539280132480.png)

![1539280143416](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539280143416.png)

![1539280148860](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539280148860.png)

![1539280153459](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539280153459.png)

![1539280162757](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539280162757.png)

![1539280172673](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539280172673.png)

![1539280181412](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539280181412.png)

![1539280189503](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539280189503.png)

![1539280194806](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539280194806.png)

### 小结

![1539279964634](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539279964634.png)

## 第六章 java虚拟机

### 6.0 导学

![1539106278412](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539106278412.png)

### 6.1 Java虚拟机概念

![1539152742070](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539152742070.png)

![1539152824180](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539152824180.png)

![1539152930663](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539152930663.png)

![1539152984208](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539152984208.png)

![1539153141326](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539153141326.png)

### 6.2 Java虚拟机内存划分

![1539158818149](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539158818149.png)

![1539160258794](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539160258794.png)

![1539160296855](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539160296855.png)

本地方法（native method）它不一定是拿Java语言来编写的方法，Java虚拟机是会运行在不同的操作系统和硬件上面，那在本身Java虚拟机的内部实现的时候，也会有一部分代码是运行的是本地代码非Java这个代码，包括你们自己写程序的时候，也可以比如用C或C++，写一段程序，最后把它嵌入到Java代码当中这也是可以的。这个本地方法栈主要是用来执行本地方法，它同样有可能会抛出异常，那所谓的抛出异常的种类也和虚拟机栈一样，而我们的虚拟机栈它的对应的功能主要是用来执行我们的Java方法。

![1539160957650](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539160957650.png)

**注**：官网G1垃圾回收器介绍[Getting Started with the G1 Garbage Collector](https://www.oracle.com/technetwork/tutorials/tutorials-1876574.html)

![1539161076357](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539161076357.png)

### 6.3 Java虚拟机类加载机制

![1539161252058](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539161252058.png)

比较旧的一些这个编程语言的经验，当我们编译完了以后可能还要做链接然后再执行，Java它有一个这个大的特点就是在程序运行过程当中来进行这个类的加载和连接，这样的话就保证了，它这个一个程序运行的这个流畅和灵活性。

![1539161369363](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539161369363.png)

![1539161595199](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539161595199.png)

![1539161500327](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539161500327.png)

![1539161553038](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539161553038.png)

![1539161763782](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539161763782.png)

![1539161758041](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539161758041.png)

![1539161925586](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539161925586.png)

![1539161967122](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539161967122.png)

### 6.4 判断对象是否存活算法及对象引用

![1539162253019](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539162253019.png)

![1539163215077](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539163215077.png)

![1539162910083](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539162910083.png)

![1539162974202](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539162974202.png)

![1539163015403](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539163015403.png)

![1539163248799](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539163248799.png)

通过sf.get方法可以获取到这个对象，当然当这个对象被标志为需要回收的对象时，它就会返回的是空，所以说软引用主要用户来实现类似缓存的功能，在内存足够的情况下，我们可以直接通过软引用来取值而不需要从繁忙的真实来源去查询数据提升速度，那当内存不足的时候就会自动删除这部分的缓存数据，然后从真正的数据来源当中去查询这些数据。

![1539163344947](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539163344947.png)

![1539163385453](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539163385453.png)

### 6.5 分代垃圾回收

![1539163621558](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539163621558.png)

**注**

1. 将引用对象设置为空，这种方式来去释放内存的话，应该没什么大问题，但如果我们用system gt方法去释放内存的话会大大的影响我们的系统性能。
2. 不可达含义：不用了，没有引用链指向。看英文：unavailable 就明白了，没有引用指向它们而且毕竟不属于空闲区，当然就不可使用。

![1539163888139](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539163888139.png)

![1539164010609](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539164010609.png)

### 6.6 典型的垃圾收集算法

![1539164393543](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539164393543.png)

![1539164434403](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539164434403.png)

![1539164514978](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539164514978.png)

![1539164585155](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539164585155.png)

![1539164642112](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539164642112.png)

![1539164730132](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539164730132.png)

![1539164819538](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539164819538.png)

![1539165861048](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539165861048.png)

### 6.7典型的垃圾收集器

![1539166829998](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539166829998.png)

![1539166851653](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539166851653.png)

![1539166911602](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539166911602.png)

![1539166985009](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539166985009.png)

![1539167023486](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539167023486.png)

**注**：jvm垃圾回收详解参考官网G1垃圾回收器介绍[Getting Started with the G1 Garbage Collector](https://www.oracle.com/technetwork/tutorials/tutorials-1876574.html)

今天介绍这几种垃圾收集器一般说来它会影响到程序执行性能，尤其是当想编一些对效率要求非常高的Java程序的时候，比如说服务器端的Java程序的时候，有时候你会比较顾及Java虚拟机的垃圾回收效率是不是足够那个帮助你的程序的运行，那据我所知国内外都有一些大公司在他们的服务器端性质当中重新改写了一些关于Java虚拟机的里面的垃圾回收的机制。

就举一个例子来说：双十一淘宝它的这个系统肯定就会承受着极大的这个用户购买商品的点击的压力，淘宝实际上它的很多后台系统拿是拿Java写的，所以为了提高这个Java在服务器端的这种工作效率，那我听说他们也是对于一些垃圾回收的这些机制进行了改进。在一些性能要求特别高的情况下的话，可能我们在**服务器端会对这些Java虚拟机以及相应它的一些局部做一些改进**。

## 第七章 深入集合collection

### 7.0 导学

![1539167756540](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539167756540.png)

### 7.1 集合框架与ArrayList

![1539167879951](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539167879951.png)

![1539172675709](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539172675709.png)

![1539172739041](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539172739041.png)

![1539172833305](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539172833305.png)

![1539173020388](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539173020388.png)

![1539173539375](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539173539375.png)

![1539173752342](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539173752342.png)

### 7.2 LinkedList

![1539174547853](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539174547853.png)

![1539174562202](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539174562202.png)

![1539174689397](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539174689397.png)

![1539174707566](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539174707566.png)

![1539175102241](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539175102241.png)

![1539175158234](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539175158234.png)

![1539175293680](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539175293680.png)

![1539175387312](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539175387312.png)

![1539175494926](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539175494926.png)

### 7.3 HashMap与HashTable

![1539175639852](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539175639852.png)

![1539175684847](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539175684847.png)

![1539176066589](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539176066589.png)

![1539176310007](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539176310007.png)

![1539176528691](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539176528691.png)

![1539176623549](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539176623549.png)

可以看到这个HashMap底层数组的长度它总是2的N次方，这就是为了保证数组的使用率最高，尽可能的减少这个碰撞现象的产生，当HashMap中的元素越来越多的时候，这个哈希冲撞的冲突的可能性也就越来越高，因为数组的长度是固定的。那为了提高这查询的效率就要对这个HashMap的数组进行扩容容量变为原来的两倍，这时候，原数组当中的数据必须重新计算它在新数组当中的位置，并且放进去，那这个过程呢就非常耗时。当HashMap中的元素个数超过数组大小（取个名字叫lot fat），就会进行数组的扩容，这个（lot fat）的默认值为0.75。那这个是一个非常消耗性能的操作所以如果我们已经预知这个HashMap当中元素的个数，那么就能够有效的提高HashMap的性能，所以这也是一个HashMap它的一个独特的地方。

![1539177101407](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539177101407.png)

![1539177164867](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539177164867.png)

**注**：关于hashtable与hashmap 建议参考源码解析 [Java7/8 中的 HashMap 和 ConcurrentHashMap 全解析](https://javadoop.com/post/hashmap)

### 7.4 TreeMap与LinkedHashMap

![1539178600822](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539178600822.png)

![1539178636271](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539178636271.png)

![1539178843148](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539178843148.png)

![1539178868002](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539178868002.png)

![1539178948012](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539178948012.png)

![1539179258434](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539179258434.png)

所以说它和之前的那个HashMap的区别就是：HashMap里面的数据结构第一级结构是一个数组，第二级结构是一个单向链表，而我们的LinkedHashMap第二级结构是一个双向链表，所以在往里添加的时候就可以根据你要添加元素的位置来决定是从正向的去检索往里添加，还是反向从队尾开始去检索进行添加。

![1539179432947](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539179432947.png)

![1539179495885](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539179495885.png)

### 7.5 HashSet

![1539179625696](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539179625696.png)

![1539179667322](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539179667322.png)

![1539179718352](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539179718352.png)

### 小结

![1539179788542](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539179788542.png)

1. **常用的一些集合类** 

   首先介绍了List Map Set 这几个接口，那在这几个接口之下又有好多的类是实现了这些接口，比如说我们的HashSet、ArrayList linkedList、HashMap、TreeMap等等这些。

2. **集合类的内部的实现过程**

   为什么要介绍这些呢？是因为要告诉大家说，它的每一个数据结构这些类它到底是怎么实现的，你明白了它的实现原理了以后，你就知道说它的效率和性能到底是怎样的，为什么高为什么低，哪些类到底和线程安全有没有关系，有没有已经实现的线程安全特性，那没有实现的话，你就得自己通过加 synchronize 办法来去实现，所以对于通过了解我们集合类的内部，你就可以很好的去运用这些集合类

3. **各个集合类的适用范围**

   由于这些集合类它本身所具有的特点并不一样，所以当我们在编程序过程中，考虑选择哪个类作为我们的数据结构的时候，你就能够很好的去选择和决定。

## 第八章 反射与代理机制

### 8.0 导学

![1539181467775](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539181467775.png)

### 8.1 Java反射机制

![1539186923792](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539186923792.png)

![1539186930949](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539186930949.png)

![1539186938506](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539186938506.png)

![1539186946955](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539186946955.png)

![1539186953329](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539186953329.png)

![1539187203705](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539187203705.png)

![1539187503835](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539187503835.png)

### 8.2 Java静态代理

![1539190674840](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539190674840.png)

![1539191075024](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539191075024.png)

![1539191329444](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539191329444.png)

![1539191522836](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539191522836.png)

### 8.3 Java动态代理

![1539191759255](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539191759255.png)

![1539225038613](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539225038613.png)

![1539225666987](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539225666987.png)

![1539226052214](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539226052214.png)

这个例子实际上是告诉大家说你可以给一个真实的对象，你给它生成一个动态代理，那生成这个动态代理的话。既然是个代理，你可以在这个真实对象的方法执行之前先做一些预处理，执行之后你还可以做一些后处理，所以你就可以增加一些你想干的这个事情，而通过这个动态代理的话，它的好处就是能够让你更加方便的去实现这些代理的过程。

![1539228376485](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539228376485.png)

![1539228382891](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539228382891.png)

### 8.4 Java 反射扩展-jvm加载类原理

![1539232525052](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539232525052.png)

JAVA虚拟机中类加载的原理，什么叫类加载，想想当编辑完JAVA的源程序以后，一编译会得到是一大堆点class文件，那点class文件就是这些类文件，而这些类文件平常是存在硬盘上，也就存在电脑的文件系统当中那当这些类文件需要执行的时候就需要JAVA虚拟机把它从硬盘上给挪到我们内存当中，那整个挪到JAVA虚拟机内存当中过程实际上就是一个类加载的过程。

![1539233621727](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539233621727.png)

![1539233745371](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539233745371.png)

![1539242915189](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539242915189.png)

1. 第二步要做连接，连接里面也会包括第一部分是验证、注意验证主要验证说，你装载进来的这个点class文件它是不是符合我们JAVA虚拟机对于自解码文件的一个规范，做格式的这种校验，甚至是不是有恶意是不是有危害，这些都是我们验证的过程，那第二个小步骤是准备要把我们这些类它的一些相应的静态的成员做一下内存的分配，那第三小步骤的话是要解析，解析是什么就是把我们很多的这些符号性的引用把它转化成一种直接的引用。
2. 第三个步骤是做类的初始化，比如说我们将类的静态变量给它做赋于正确的初始值，注意这个初始值是指的是程序员在给它定义的这个初始值而不是说默认初始值，默认初始值这个确定是在第二个步骤连接步骤里边，这个准备小步骤里边已经实现了。

![1539242983420](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539242983420.png)

![1539243003067](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539243003067.png)

### 小结

![1539243156735](http://q9kvrafcq.bkt.clouddn.com/gitpages/java/tsinghua_open_course_2/1539243156735.png)

我们今天主要讲了三个方面的内容

1. JAVA的反射机制

   JAVA反射机制为程序员提供了一种直接去获取类以及对象它的方法以及它的成员变量的一种方式，通过反射机制可以去通过一个字符串的名字去创建一个类的对象，并且很灵活的去调动它的所有的方法。

2. JAVA的代理机制

   介绍了静态代理和动态代理，之所以有JAVA代理机制是因为说有些情况下并不想或者不能够去直接访问目标对象而需要中间有一个中介的渠道，那这中介渠道就可以帮助很好的去控制和访问目标对象。并且在访问目标的前和后都可以增加一些预处理或者后处理。介绍了静态代理的方式和动态代理的方式，动态代理方式会给大家很大的一个方便和灵活性

3. 类的加载机制

   把我们所有编译好的点Class文件把它加载到我们的JAVA虚拟机当中来进行运行，那这个类加载过程当中，它实际上对于JAVA是一个动态的过程而且是一个可以从多个源头进行加载的这个过程，理解的加载类的加载过程，对于大家今后编写更加高效有效的JAVA程序会带来很大的帮助。



参考：

1. 周志明《深入理解java虚拟机》
2. [Java7/8 中的 HashMap 和 ConcurrentHashMap 全解析](https://javadoop.com/post/hashmap)
3. [Java Concurrency in Practice](http://jcip.net/)







