---
title: 含括号的运算表达式求解-栈的基础应用3
mathjax: true
mathjax2: true
categories: 中文
date: 2013-05-13 22:16:00
tags: [Data Structure, 数据结构]
toc: true
copyright: false
---

本文根据严蔚敏老师数据结构（c语言版） 写的程序 如有需要先去看视频  如有错误不当之处，欢迎指出，以免害人害己。

## 例子

Exp = a * b + ( c – d / e )* f

​    前缀式：+ * a b * - c / d e f

​    中缀式：a * b  + c–d / e * f

​    后缀式：a b *  cd e /-f *  +

### 相同点

数字都是按原式子排列的：因此操作数就按顺序入栈就好了

### 不同点

1:后缀式中运算符的顺序，正好就是求解的顺序

2:每个运算符和它之前出现且紧靠它的2个操作数构成一个最小表达式

关键：就是由原表达式求得后缀式

## 应用步骤

-   Step1： 先设立两个栈，一个运算符栈，另一个后缀式栈。
-   Step2：在表达式前后头尾加入=号，表示运算表达式开始和结束，因此在运算符中，=号优先级最低。
-   Step3：若当前字符是操作数，则直接发送给后缀式栈。符合上面提到的共同点：数字按原表达式从左自右的顺序。
-   Step4：左括号的优先级高于左括号前的运算符，左括号后的运算符优先级高于左括号，这样才能起到隔离的作用，则右括号前的运算符高于右括号，这样才能起到括号隔离内层表达式的作用！
-   Step5：若当前运算符的优先级高于栈顶的运算符，则进运算符栈，否则退出运算符栈的栈顶运算符与从操作数栈栈顶取出的两个操作数运算结果作为新的操作数压入操作数栈，然后再把当前运算符入运算符栈。

## 源代码

```c++
#include<stdio.h> 

#include<malloc.h>//malloc()
#include<process.h>//exit();
// 函数结果状态代码
#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INFEASIBLE -1

#define STACK_INIT_SIZE 100 // 存储空间初始分配量
#define STACKINCREMENT 2 // 存储空间分配增量

typedef int Status; // Status是函数的类型,其值是函数结果状态代码，如OK等
typedef char SElemType;
struct SqStack
{
	SElemType *base; // 在栈构造之前和销毁之后，base的值为NULL
	SElemType *top; // 栈顶指针
	int stacksize; // 当前已分配的存储空间，以元素为单位
};

Status InitStack(SqStack &S)
{ // 构造一个空栈S
	if(!(S.base=(SElemType *)malloc(STACK_INIT_SIZE*sizeof(SElemType))))
		exit(-1); // 存储分配失败
	S.top=S.base;
	S.stacksize=STACK_INIT_SIZE;
	return OK;
}

Status DestroyStack(SqStack &S)
{ // 销毁栈S，S不再存在
	free(S.base);
	S.base=NULL;
	S.top=NULL;
	S.stacksize=0;
	return OK;
}

Status ClearStack(SqStack &S)
{ // 把S置为空栈
	S.top=S.base;
	return OK;
}

Status StackEmpty(SqStack S)
{ // 若栈S为空栈，则返回TRUE，否则返回FALSE
	if(S.top==S.base)
		return TRUE;
	else
		return FALSE;
}

int StackLength(SqStack S)
{ // 返回S的元素个数，即栈的长度
	return S.top-S.base;
}

Status GetTop(SqStack S,SElemType &e)
{ // 若栈不空，则用e返回S的栈顶元素，并返回OK；否则返回ERROR
	if(S.top>S.base)
	{
		e=*(S.top-1);
		return OK;
	}
	else
		return ERROR;
}

Status Push(SqStack &S,SElemType e)
{ // 插入元素e为新的栈顶元素
	if(S.top-S.base>=S.stacksize) // 栈满，追加存储空间
	{
		S.base=(SElemType *)realloc(S.base,(S.stacksize+STACKINCREMENT)*sizeof(SElemType));
		if(!S.base)
			exit(-1); // 存储分配失败
		S.top=S.base+S.stacksize;
		S.stacksize+=STACKINCREMENT;
	}
	*(S.top)++=e;
	return OK;
}

Status Pop(SqStack &S,SElemType &e)
{ // 若栈不空，则删除S的栈顶元素，用e返回其值，并返回OK；否则返回ERROR
	if(S.top==S.base)
		return ERROR;
	e=*--S.top;
	return OK;
}

Status StackTraverse(SqStack S,Status(*visit)(SElemType))
{ // 从栈底到栈顶依次对栈中每个元素调用函数visit()。
	// 一旦visit()失败，则操作失败
	while(S.top>S.base)
		visit(*S.base++);
	printf("\n");
	return OK;
}

SElemType Precede(SElemType a, SElemType b) 
{ //判断运算符优先级
	int i, j;
	char Table[8][8] = {
		{' ','+','-','*','/','(',')','#'},
		{'+','>','>','<','<','<','>','>'},
		{'-','>','>','<','<','<','>','>'},
		{'*','>','>','>','>','<','>','>'},
		{'/','>','>','>','>','<','>','>'},
		{'(','<','<','<','<','<','=',' '},
		{')','>','>','>','>',' ','>','>'},
		{'#','<','<','<','<','<',' ','='}
	};  //优先级表格

	for(i=0; i<8; i++)
		if(Table[0][i]==a)  //寻找运算符a
			break;
	for(j=0; j<8; j++) //寻找运算符
		if(Table[j][0]==b)
			break;
	return Table[j][i];
}

Status In(SElemType c)
{ // 判断c是否为运算符
	switch(c)
	{
		case'+':
		case'-':
		case'*':
		case'/':
		case'(':
		case')':
		case'#':return TRUE;
		default:return FALSE;
	}
}

SElemType Operate(SElemType a,SElemType theta,SElemType b)
{
	SElemType c;
	a=a-48;
	b=b-48;
	switch(theta)
	{
		case'+':c=a+b+48;
				break;
		case'-':c=a-b+48;
				break;
		case'*':c=a*b+48;
				break;
		case'/':c=a/b+48;
	}
	return c;
}

SElemType EvaluateExpression() // 算法3.4
{ // 算术表达式求值的算符优先算法。设OPTR和OPND分别为运算符栈和运算数栈
	SqStack OPTR,OPND;
	SElemType a,b,c,x,theta;
	InitStack(OPTR);
	Push(OPTR,'#');
	InitStack(OPND);
	c=getchar();
	GetTop(OPTR,x);
	while(c!='#'||x!='#')
	{
		if(In(c)) // 是7种运算符之一
			switch(Precede(c,x))
			{
				case'<':Push(OPTR,c); // 栈顶元素优先权低
						c=getchar();
						break;
				case'=':Pop(OPTR,x); // 脱括号并接收下一字符
						c=getchar();
						break;
				case'>':Pop(OPTR,theta); // 退栈并将运算结果入栈
						Pop(OPND,b);
						Pop(OPND,a);
						Push(OPND,Operate(a,theta,b));
						break;
			}
		else if(c>='0'&&c<='9') // c是操作数
		{
			Push(OPND,c);
			c=getchar();
		}
		else // c是非法字符
		{
			printf("非法字符\n");
			exit(-1);
		}
		GetTop(OPTR,x);
	}
	GetTop(OPND,x);
	return x;
}

int main()
{
	printf("请输入算术表达式（中间值及最终结果要在0～9之间），并以#结束\n");
	printf("%c\n",EvaluateExpression());
	return 0;
}
```

