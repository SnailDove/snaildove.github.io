---
title: 行编辑器/容错缓冲区-栈的基础应用2
mathjax: true
mathjax2: true
categories: 中文
date: 2013-05-12 22:16:00
tags: [Data Structure, 数据结构]
toc: true
copyright: false
---

## 功能

接收用户的从终端输入程序或数据，并存入用户的数据区。由于用户在终端上输入难免出现差错，因此，若在行编辑程序中，“每接收一个字符即存入用户数据区”显然是不合理的。较好的做法，设立一个缓冲区，用以接收用户输入的每一行字符，然后逐行存入用户数据区。允许用户输入出错，并在发现有误时可以及时更正。例如，当用户发现刚刚键入的一个字符是错的时，补进一个退格符#，以表示前一个字符无效；如果发现当前键入的行内差错较多或难以补救，就可以键入一个退行符号@，以表示当前行中的字符均无效。

## 例子

​         从终端接收这样两行字符：

​                            whi##ilr#e(s#*s)

​                                     outcha@putchar(*s#++)

​          实际有效的是下列两行：

​                            while(*s)

​                                     putchar(*s++)

​         为此，我们可以设立一个缓冲区，结构为栈，每当用户从终端接受了一个字符之后现做如下判别：如果他既不是退格符，也不是退行符，则将该字符压入栈中，如果是退格符，则从栈顶删去一个字符；如果是退行符，则将字符栈清空。

## 源代码

```c
void LineEdit()
{ //利用字符栈s，从终端接收一行并送至调用过程的数据区。算法3.2
	SqStack s;
	char ch,c;
	InitStack(s);
	printf("请输入一个文本文件,^Z结束输入:\n");
	ch=getchar();
	while(ch!=EOF)
	{// EOF为^Z键，全文结束符
		while(ch!=EOF&&ch!='\n')
		{
			switch(ch)
			{
				case '#':Pop(s,c);
						 break; // 仅当栈非空时退栈
				case '@':ClearStack(s);
						 break; // 重置s为空栈
				default :Push(s,ch); // 有效字符进栈
			}
			ch=getchar(); // 从终端接收下一个字符
		}
		StackTraverse(s,copy); // 将从栈底到栈顶的栈内字符传送至文件
		ClearStack(s); // 重置s为空栈
		fputc('\n',fp);
		if(ch!=EOF)
			ch=getchar();
	}
	DestroyStack(s);
}
```

