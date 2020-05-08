---
title: 翻译 前言 Stream Processing with Apache Spark
date: 2019-07-01
copyright: true
categories: 中文, english
tags: [Improving Deep Neural Networks, deep learning]
mathjax: true
mathjax2: true
toc: true
---
![](https://img1.doubanio.com/view/subject/l/public/s29700009.jpg)

## <p align="right"><font color="#9a161a" >What You Will Learn in This Book</font></p>

在这本书你将学到什么

This book will teach you everything you need to know about stream processing with Apache Flink. It consists of 11 chapters that hopefully tell a coherent story. While some chapters are descriptive and aim to introduce high-level design concepts, others are more hands-on and contain many code examples.

本书将向您介绍使用 Apache Flink 进行流处理时需要了解的所有内容。它由11章组成，希望能够讲述一个连贯的故事。虽然有些章节是描述性的，旨在介绍高阶设计概念，但其他章节则更具实际操作性并包含许多代码示例。

While we intended for the book to be read in chapter order when we were writing it, readers familiar with a chapter’s content might want to skip it. Others more interested in writing Flink code right away might want to read the practical chapters first. In the following, we briefly describe the contents of each chapter, so you can directly jump to those chapters that interest you most.

虽然我们打算在编写本书时以章节顺序阅读本书，但熟悉章节内容的读者可能希望跳过它。其他人对立即编写Flink代码感兴趣可能需要先阅读实用章节。在下文中，我们将简要介绍每章的内容，以便您可以直接跳转到您最感兴趣的章节。

Chapter 1 gives an overview of stateful stream processing, data processing application architectures, application designs, and the benefits of stream processing over traditional approaches. It also gives you a brief look at what it is like to run your first streaming application on a local Flink instance.

**第1章**概述了有状态流处理，数据处理应用程序体系结构，应用程序设计以及流处理相对于传统方法的好处。它还简要介绍了在本地Flink实例上运行第一个流应用程序的情况。

Chapter 2 discusses the fundamental concepts and challenges of stream processing, independent of Flink.

**第2章**讨论了流处理的基本概念和挑战，独立于Flink。

Chapter 3 describes Flink’s system architecture and internals. It discusses distributed architecture, time and state handling in streaming applications, and Flink’s fault-tolerancemechanisms.

**第3章**介绍了Flink的系统架构和内部。它讨论了流应用程序中的分布式体系结构，时间和状态处理，以及Flink的容错机制。

Chapter 4 explains how to set up an environment to develop and debug Flink applications.

**第4章**介绍如何设置开发和调试Flink应用程序的环境。

Chapter 5 introduces you to the basics of the Flink’s DataStream API. You will learn how to implement a DataStream application and which stream transformations, functions, and data types are supported.

**第5章**介绍了Flink的DataStream API的基础知识。您将学习如何实现DataStream应用程序以及支持哪些流转换，函数和数据类型


Chapter 6 discusses the time-based operators of the DataStream API. This includes window operators and timebased joins as well as process functions that provide the most flexibility when dealing with time in streaming applications. 

**第6章**讨论DataStream API的基于时间的运算符。这包括窗口操作符和基于时间的连接以及在流应用程序中处理时间时提供最大灵活性的过程函数。

Chapter 7 explains how to implement stateful functions and discusses everything around this topic, such as the performance, robustness, and evolution of stateful functions. It also shows how to use Flink’s queryable state.

**第7章**解释了如何实现有状态函数并讨论了有关该主题的所有内容，例如状态函数的性能，健壮性和演变。它还显示了如何使用Flink的可查询状态。

Chapter 8 presents Flink’s most commonly used source and sink connectors. It discusses Flink’s approach to end-to-end application consistency and how to implement custom connectors to ingest data from and emit data to external systems.

**第8章**介绍了Flink最常用的源和接收器连接器。它讨论了Flink的端到端应用程序一致性方法，以及如何实现自定义连接器从外部系统中提取数据和向外部系统发送数据。


Chapter 9 discusses how to set up and configure Flink clusters in various environments.

**第9章**讨论如何在各种环境中设置和配置Flink集群。

Chapter 10 covers operation, monitoring, and maintenance of streaming applications that run 24/7.

**第10章**介绍了全天候运行的流应用程序的操作，监视和维护。


Finally, Chapter 11 contains resources you can use to ask questions, attend Flink-related events, and learn how Flink is currently being used.

最后，**第11章**包含可用于提问，参加Flink相关事件以及了解Flink当前如何使用的资源。

## <font color="#9a161a" >Using Code Examples 使用代码示例</font>

Supplemental material (code examples in Java and Scala) is available for download at https://github.com/streaming-with-flink.

补充材料（Java和Scala中的代码示例）可从 https://github.com/streaming-with-flink下载。

This book is here to help you get your job done. In general, if example code is offered with this book, you may use it in your programs and documentation. You do not need to contact us for permission unless you’re reproducing a significant portion of the code. For example, writing a program that uses several chunks of code from this book does not require permission. Selling or distributing a CD-ROM of examples from O’Reilly books does require permission. Answering a question by citing this book and quoting example code does not require permission.

这本书是为了帮助你完成工作。通常，如果本书提供了示例代码，您可以在程序和文档中使用它。除非您复制了大部分代码，否则您无需与我们联系以获得许可。例如，编写使用本书中几个代码块的程序不需要许可。出售或分发O'Reilly书籍中的示例CD-ROM需要获得许可。通过引用本书并引用示例代码来回答问题不需要许可。


Incorporating a significant amount of example code from this book into your product’s documentation does require permission. We appreciate, but do not require, attribution. An attribution usually includes the title, author, publisher, and ISBN. For example: “Stream Processing with Apache Flink by Fabian Hueske and Vasiliki Kalavri (O’Reilly). Copyright 2019 Fabian Hueske and Vasiliki Kalavri, 978-1-491-97429-2.

将本书中的大量示例代码合并到产品文档中需要获得许可。我们感谢，但不要求，归属。归属通常包括标题，作者，出版商和ISBN。例如：“Fabian Hueske和Vasiliki Kalavri（O'Reilly）使用Apache Flink进行流处理。版权所有2019 Fabian Hueske和Vasiliki Kalavri，978-1-491-97429-2。
