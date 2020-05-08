---
title:  AJAX 
mathjax: true
mathjax2: true
categories: 中文
tags: [javascript]
date: 2016-02-28 20:16:00
commets: true
toc: true
layout: draft
copyright: false
---

## AJAX的介绍和优点

### 什么是AJAX

AJAX = Asynchronous JavaScript and XML,AJAX 是一种用于创建快速动态网页的技术。通过在后台与服务器进行少量数据交换，AJAX 可以使网页实现异步更新。这意味着可以在不重新加载整个网页的情况下，对网页的某部分进行更新。传统的网页（不使用 AJAX）如果需要更新内容，必需重载整个网页面。AJAX 不是新的编程语言，而是一种使用现有标准的新方法。AJAX 是与服务器交换数据并更新部分网页局部数据的艺术，在不重新加载整个页面的情况下(按需取数据)。

### AJAX优缺点 

1. 优点：局部刷新，用户体验好（通过 AJAX，JavaScript 无需等待服务器的响应，而是：
  在等待服务器响应时执行其他脚本
  当响应就绪后对响应进行处理），服务器负载减轻

2. 缺点：移动端不是支持很好，而且创建对象比较麻烦，各浏览器都不一样（总体：2种）

### AJAX应用

大量交互数据的web，提高响应速度与用户体验.有很多使用 AJAX 的应用程序案例：新浪微博、Google 地图、开心网等等。

## AJAX对象
**XMLHttpRequest 是 AJAX 的基础。**

### XMLHttpRequest 对象

所有现代浏览器均支持 XMLHttpRequest 对象（IE5 和 IE6 使用 ActiveXObject）。

XMLHttpRequest 用于在后台与服务器交换数据。这意味着可以在不重新加载整个网页的情况下，对网页的某部分进行更新。

### 创建 XMLHttpRequest 对象

所有现代浏览器（IE7+、Firefox、Chrome、Safari 以及 Opera）均内建 XMLHttpRequest 对象。

### 创建 XMLHttpRequest 对象的语法

```javascript
variable=new XMLHttpRequest();
```

### 老版本的 Internet Explorer （IE5 和 IE6）使用 ActiveX 对象

```javascript
variable=new ActiveXObject("Microsoft.XMLHTTP");
```

为了应对所有的现代浏览器，包括 IE5 和 IE6，请检查浏览器是否支持 XMLHttpRequest 对象。如果支持，则创建 XMLHttpRequest 对象。如果不支持，则创建 ActiveXObject ：

```javascript
var xmlhttp;
if (window.XMLHttpRequest)
{// code for IE7+, Firefox, Chrome, Opera, Safari
    xmlhttp=new XMLHttpRequest();
}
else
{// code for IE6, IE5
    xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
}
```

**XMLHttpRequest 对象用于和服务器交换数据。**

## 向服务器发送请求

如需将请求发送到服务器，我们使用 XMLHttpRequest 对象的 open() 和 send() 方法：

```javascript
xmlhttp.open("GET","test1.txt",true);
xmlhttp.send();
```

| 方法                           | 描述                                       |
| ---------------------------- | ---------------------------------------- |
| open(*method*,*url*,*async*) | 规定请求的类型、URL 以及是否异步处理请求。*method*：请求的类型；GET 或 POST*url*：文件在服务器上的位置*async*：true（异步）或 false（同步） |
| send(*string*)               | 将请求发送到服务器。*string*：仅用于 POST 请求           |

### GET 还是 POST？

与 POST 相比，GET 更简单也更快，并且在大部分情况下都能用。

然而，在以下情况中，请使用 POST 请求：

-   无法使用缓存文件（更新服务器上的文件或数据库）
-   向服务器发送大量数据（POST 没有数据量限制）
-   发送包含未知字符的用户输入时，POST 比 GET 更稳定也更可靠

对象的 open() 和 send() 方法：

| 方法                           | 描述                                       |
| ---------------------------- | ---------------------------------------- |
| open(*method*,*url*,*async*) | 规定请求的类型、URL 以及是否异步处理请求。**method**：请求的类型；GET 或 POST **url**：文件在服务器上的位置，该文件可以是任何类型的文件，比如 .json 和 .xml，或者服务器脚本文件，比如 .asp 和 .php （在传回响应之前，能够在服务器上执行任务）。**async**：true（异步）或 false（同步） |
| send(*string*)               | 将请求发送到服务器。*string*：仅用于 POST 请求           |
#### 例子

```javascript
ajaxObj.open("GET","test1.txt",true);  
ajaxObj.send();
```

### GET or POST？

与 POST 相比，GET 更简单也更快，并且在大部分情况下都能用。 然而，在以下情况中，请使用 POST 请求：无法使用缓存文件（更新服务器上的文件或数据库）向服务器发送大量数据（POST 没有数据量限制）发送包含未知字符的用户输入时，POST 比 GET 更稳定也更可靠。

### GET 请求

一个简单的 GET 请求：

```javascript
xmlhttp.open("GET","demo_get.asp",true);
xmlhttp.send();
```

[亲自试一试](http://www.w3school.com.cn/tiy/t.asp?f=ajax_get)

在上面的例子中，您可能得到的是缓存的结果。

为了避免这种情况，请向 URL 添加一个唯一的 ID：

```javascript
xmlhttp.open("GET","demo_get.asp?t=" + Math.random(),true);
xmlhttp.send();
```

[亲自试一试](http://www.w3school.com.cn/tiy/t.asp?f=ajax_get_unique)

如果您希望通过 GET 方法发送信息，请向 URL 添加信息：

```javascript
xmlhttp.open("GET","demo_get2.asp?fname=Bill&lname=Gates",true);
xmlhttp.send();
```

[亲自试一试](http://www.w3school.com.cn/tiy/t.asp?f=ajax_get2)

### POST 请求

一个简单 POST 请求：

```javascript
xmlhttp.open("POST","demo_post.asp",true);
xmlhttp.send();
```

[亲自试一试](http://www.w3school.com.cn/tiy/t.asp?f=ajax_post)

如果需要像 HTML 表单那样 POST 数据，请使用 setRequestHeader() 来添加 HTTP 头。然后在 send() 方法中规定您希望发送的数据：

```javascript
xmlhttp.open("POST","ajax_test.asp",true);
xmlhttp.setRequestHeader("Content-type","application/x-www-form-urlencoded");
xmlhttp.send("fname=Bill&lname=Gates");
```

[亲自试一试](http://www.w3school.com.cn/tiy/t.asp?f=ajax_post2)

| 方法                                 | 描述                                       |
| ---------------------------------- | ---------------------------------------- |
| setRequestHeader(*header*,*value*) | 向请求添加 HTTP 头。*header*: 规定头的名称*value*: 规定头的值 |

### url - 服务器上的文件

open() 方法的 *url* 参数是服务器上文件的地址：

```javascript
xmlhttp.open("GET","ajax_test.asp",true);
```

该文件可以是任何类型的文件，比如 .txt 和 .xml，或者服务器脚本文件，比如 .asp 和 .php （在传回响应之前，能够在服务器上执行任务）。

### 异步 - True 或 False？

AJAX 指的是异步 JavaScript 和 XML（Asynchronous JavaScript and XML）。

XMLHttpRequest 对象如果要用于 AJAX 的话，其 open() 方法的 async 参数必须设置为 true：

```javascript
xmlhttp.open("GET","ajax_test.asp",true);
```

对于 web 开发人员来说，发送异步请求是一个巨大的进步。很多在服务器执行的任务都相当费时。AJAX 出现之前，这可能会引起应用程序挂起或停止。

通过 AJAX，JavaScript 无需等待服务器的响应，而是：

-   在等待服务器响应时执行其他脚本
-   当响应就绪后对响应进行处理

### Async = true

当使用 async=true 时，请规定在响应处于 onreadystatechange 事件中的就绪状态时执行的函数：

```javascript
xmlhttp.onreadystatechange=function()
  {
  if (xmlhttp.readyState==4 && xmlhttp.status==200)
    {
    document.getElementById("myDiv").innerHTML=xmlhttp.responseText;
    }
  }
xmlhttp.open("GET","test1.txt",true);
xmlhttp.send();
```

[亲自试一试](http://www.w3school.com.cn/tiy/t.asp?f=ajax_async_true)

您将在稍后的章节学习更多有关 onreadystatechange 的内容。

### Async = false

如需使用 async=false，请将 open() 方法中的第三个参数改为 false：

```javascript
xmlhttp.open("GET","test1.txt",false);
```

我们不推荐使用 async=false，但是对于一些小型的请求，也是可以的。

请记住，JavaScript 会等到服务器响应就绪才继续执行。如果服务器繁忙或缓慢，应用程序会挂起或停止。

注释：当您使用 async=false 时，请不要编写 onreadystatechange 函数 - 把代码放到 send() 语句后面即可：

```javascript
xmlhttp.open("GET","test1.txt",false);
xmlhttp.send();
document.getElementById("myDiv").innerHTML=xmlhttp.responseText;
```
[亲自试一试](http://www.w3school.com.cn/tiy/t.asp?f=ajax_async_false)

## AJAX - 服务器响应

-   [XMLHttpRequest 请求](http://www.w3school.com.cn/ajax/ajax_xmlhttprequest_send.asp)
-   [XMLHttpRequest readyState](http://www.w3school.com.cn/ajax/ajax_xmlhttprequest_onreadystatechange.asp)

### 服务器响应

如需获得来自服务器的响应，请使用 XMLHttpRequest 对象的 responseText 或 responseXML 属性。

| 属性           | 描述              |
| ------------ | --------------- |
| responseText | 获得字符串形式的响应数据。   |
| responseXML  | 获得 XML 形式的响应数据。 |

### responseText 属性

如果来自服务器的响应并非 XML，请使用 responseText 属性。

responseText 属性返回字符串形式的响应，因此您可以这样使用：

```
document.getElementById("myDiv").innerHTML=xmlhttp.responseText;
```

[亲自试一试](http://www.w3school.com.cn/tiy/t.asp?f=ajax_async_true)

### responseXML 属性

如果来自服务器的响应是 XML，而且需要作为 XML 对象进行解析，请使用 responseXML 属性：

请求 [books.xml](http://www.w3school.com.cn/example/xmle/books.xml) 文件，并解析响应：

```javascript
xmlDoc=xmlhttp.responseXML;
txt="";
x=xmlDoc.getElementsByTagName("ARTIST");
for (i=0;i<x.length;i++)
{
  txt=txt + x[i].childNodes[0].nodeValue + "<br />";
}
document.getElementById("myDiv").innerHTML=txt;
```

[亲自试一试](http://www.w3school.com.cn/tiy/t.asp?f=ajax_responsexml)

### AJAX - onreadystatechange 事件

-   [XHR 响应](http://www.w3school.com.cn/ajax/ajax_xmlhttprequest_response.asp)
-   [AJAX ASP/PHP](http://www.w3school.com.cn/ajax/ajax_asp_php.asp)

### onreadystatechange 事件

当请求被发送到服务器时，我们需要执行一些基于响应的任务。

每当 readyState 改变时，就会触发 onreadystatechange 事件。

readyState 属性存有 XMLHttpRequest 的状态信息。

下面是 XMLHttpRequest 对象的三个重要的属性：

| 属性                 | 描述                                       |
| ------------------ | ---------------------------------------- |
| onreadystatechange | 存储函数（或函数名），每当 readyState 属性改变时，就会调用该函数。  |
| readyState         | 存有 XMLHttpRequest 的状态。从 0 到 4 发生变化。0: 请求未初始化1: 服务器连接已建立2: 请求已接收3: 请求处理中4: 请求已完成，且响应已就绪 |
| status             | 200: "OK"404: 未找到页面                      |

在 onreadystatechange 事件中，我们规定当服务器响应已做好被处理的准备时所执行的任务。

当 readyState 等于 4 且状态为 200 时，表示响应已就绪：

```javascript
xmlhttp.onreadystatechange=function()
{
  if (xmlhttp.readyState==4 && xmlhttp.status==200)
  {
    document.getElementById("myDiv").innerHTML=xmlhttp.responseText;
  }
}
```

[亲自试一试](http://www.w3school.com.cn/tiy/t.asp?f=ajax_async_true)

注释：onreadystatechange 事件被触发 5 次（0 - 4），对应着 readyState 的每个变化。

### 使用 Callback 函数

callback 函数是一种以参数形式传递给另一个函数的函数。

如果您的网站上存在多个 AJAX 任务，那么您应该为创建 XMLHttpRequest 对象编写一个*标准*的函数，并为每个 AJAX 任务调用该函数。

该函数调用应该包含 URL 以及发生 onreadystatechange 事件时执行的任务（每次调用可能不尽相同）：

```javascript
function myFunction()
{
  loadXMLDoc("ajax_info.txt",function()
  {
  if (xmlhttp.readyState==4 && xmlhttp.status==200)
    {
    document.getElementById("myDiv").innerHTML=xmlhttp.responseText;
    }
  });
}
```

[亲自试一试](http://www.w3school.com.cn/tiy/t.asp?f=ajax_callback)

## AJAX ASP/PHP 请求实例

-   [XHR readyState](http://www.w3school.com.cn/ajax/ajax_xmlhttprequest_onreadystatechange.asp)
-   [AJAX 数据库](http://www.w3school.com.cn/ajax/ajax_database.asp)

**AJAX 用于创造动态性更强的应用程序。**

### AJAX ASP/PHP 实例

下面的例子将为您演示当用户在输入框中键入字符时，网页如何与 web 服务器进行通信：

请在下面的输入框中键入字母（A - Z）：

姓氏：

建议：

[亲自试一下源代码](http://www.w3school.com.cn/tiy/t.asp?f=ajax_suggest)

### 实例解释 - showHint() 函数

当用户在上面的输入框中键入字符时，会执行函数 "showHint()" 。该函数由 "onkeyup" 事件触发：

```javascript
function showHint(str)
{
	var xmlhttp;
	if (str.length==0)
	{
		document.getElementById("txtHint").innerHTML="";
		return;
	}
	if (window.XMLHttpRequest)
	{// code for IE7+, Firefox, Chrome, Opera, Safari
		xmlhttp=new XMLHttpRequest();
	}
	else
	{// code for IE6, IE5
		xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
	}
	xmlhttp.onreadystatechange=function()
	{
		if (xmlhttp.readyState==4 && xmlhttp.status==200)
		{
			document.getElementById("txtHint").innerHTML=xmlhttp.responseText;
		}
	}
	xmlhttp.open("GET","gethint.asp?q="+str,true);
	xmlhttp.send();
}
```

### 源代码解释：

如果输入框为空 (str.length==0)，则该函数清空 txtHint 占位符的内容，并退出函数。

如果输入框不为空，showHint() 函数执行以下任务：

-   创建 XMLHttpRequest 对象
-   当服务器响应就绪时执行函数
-   把请求发送到服务器上的文件
-   请注意我们向 URL 添加了一个参数 q （带有输入框的内容）

## 用XMLHttpRequest对象封装AJAX

```javascript
function Ajax(recvType) {
    //创建ajax对象
    var ajaxObj = new Object();

    //AJAX请求服务器而返回的文件类型
    ajaxObj.recvType = recvType ? recvType.toUpperCase() : 'HTML'; //HTML XML

    //ajax发送的url
    ajaxObj.url = '';

    //ajax发送的字符串
    ajaxObj.sendStr = null;

    //ajax请求成功的回调函数
    ajaxObj.callback = null;

    //声明ajax XMLHttpRequest对象
    ajaxObj.XMLHttpRequest = null;

    //定义产生XMLHttpRequest的方法
    ajaxObj.createXMLHttpRequest = function () {
        var request = false;

        //windows对象中有XMLHttpRequest存在就是非IE，包括（IE7,IE8）
        if (window.XMLHttpRequest) {
            request = new XMLHttpRequest();
            if (request.overrideMimeType) {
                /*
                 如果来自服务器的响应没有 XML mime-type 头部，则一些版本的 Mozilla 浏览器不能正常运行。
                 对于这种情况，httpRequest.overrideMimeType('text/xml'); 语句将覆盖发送给服务器的头部，
                 强制 text/xml 作为 mime-type。
                 */
                request.overrideMimeType("text/xml");
            }
        } else if (window.ActiveXObject) {//windows对象中有ActiveXObject属性存在就是IE
            var browserVersions = ['Microsoft.XMLHTTP', 'MSXML.XMLHTTP', 'Msxml2.XMLHTTP.7.0', 'Msxml2.XMLHTTP.6.0',
                'Msxml2.XMLHTTP.5.0', 'Msxml2.XMLHTTP.4.0', 'MSXML2.XMLHTTP.3.0', 'MSXML2.XMLHTTP'];
            for (var i = 0; i < browserVersions.length; ++i) {
                try {
                    request = new ActiveXObject(browserVersions[i]);
                    if (request) {
                        return request;
                    }
                } catch (e) {
                    request = false;
                }
            }
            return request;
        }
    }

    ajaxObj.XMLHttpRequest = ajaxObj.createXMLHttpRequest();

    //封装ajax底层的GET方法
    ajaxObj.get = function (url, callback) {
        ajaxObj.url = url;
        if (callback != null) {//回调函数不为空时：如何调用
            ajaxObj.XMLHttpRequest.onreadystatechange = ajaxObj.handleCallback();
            ajaxObj.callback = callback;//将GET方法设定的回调方法传递给ajax
        }
        if (window.XMLHttpRequest) {//非IE浏览器,包括（IE7,IE8）
            ajaxObj.XMLHttpRequest.open('GET', ajaxObj.url);//异步
            ajaxObj.XMLHttpRequest.send(null);
        }else{//windows对象中有ActiveXObject属性存在就是IE
            ajaxObj.XMLHttpRequest.open('GET', ajaxObj.url, true);
            ajaxObj.XMLHttpRequest.send();
        }
    }
	//封装ajax底层的POST方法
    ajax.post = function(url, sendStr, callback){
        ajaxObj.url = url;//传递url给ajax对象
        //拼接发送的数据
        if('object'== typeof (sendStr)){//发送的字符串为对象类型
            var str = '';
            for(var key in sendStr){
                str += key + '=' + sendStr[key] + "&";
            }
            ajaxObj.sendStr = str;
        }else{
            ajaxObj.sendStr = sendStr;
        }

        if (callback != null) {//回调函数不为空时：怎么调用
            ajaxObj.XMLHttpRequest.onreadystatechange = ajaxObj.handleCallback();
            ajaxObj.callback = callback;//将GET方法设定的回调方法传递给ajax
        }

        ajaxObj.XMLHttpRequest.open('POST', ajaxObj.url);
        ajaxObj.XMLHttpRequest.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');//必须指定编码
        ajaxObj.XMLHttpRequest.send(ajaxObj.sendStr);
    }

    ajaxObj.handleCallback=function(){
        if(ajaxObj.XMLHttpRequest.readyState == 4){//请求完成才执行
            if(ajaxObj.XMLHttpRequest.status == 200){//请求成功才行
                if(ajaxObj.recvType=="HTML")
                    ajaxObj.callback(ajaxObj.XMLHttpRequest.responseText);
                else if(aj.recvType=="XML")
                    ajaxObj.callback(ajaxObj.XMLHttpRequest.responseXML);
            }
        }
    }
    return ajaxObj;
}
```

## 关于 jQuery 与 AJAX

jQuery 提供多个与 AJAX 有关的方法。

通过 jQuery AJAX 方法，您能够使用 HTTP Get 和 HTTP Post 从远程服务器上请求文本、HTML、XML 或 JSON - 同时您能够把这些外部数据直接载入网页的被选元素中。

**提示：如果没有 jQuery，AJAX 编程还是有些难度的，上面封装AJAX就看得出来。**

编写常规的 AJAX 代码并不容易，因为不同的浏览器对 AJAX 的实现并不相同。这意味着您必须编写额外的代码对浏览器进行测试。不过，jQuery 团队为我们解决了这个难题，我们只需要一行简单的代码，就可以实现 AJAX 功能。

**jQuery get() 和 post() 方法用于通过 HTTP GET 或 POST 请求从服务器请求数据。**

### HTTP 请求：GET vs. POST

两种在客户端和服务器端进行请求-响应的常用方法是：GET 和 POST。

-   *GET* - 从指定的资源请求数据
-   *POST* - 向指定的资源提交要处理的数据

GET 基本上用于从服务器获得（取回）数据。注释：GET 方法可能返回缓存数据。

POST 也可用于从服务器获取数据。不过，POST 方法不会缓存数据，并且常用于连同请求一起发送数据。

如需学习更多有关 GET 和 POST 以及两方法差异的知识，请阅读 [HTTP 方法 - GET 对比 POST](http://www.w3school.com.cn/tags/html_ref_httpmethods.asp)。

### jQuery $.get() 方法

$.get() 方法通过 HTTP GET 请求从服务器上请求数据。语法：

```javascript
$.get(URL,callback);
```

必需的 *URL* 参数规定您希望请求的 URL。

可选的 *callback* 参数是请求成功后所执行的函数名。

下面的例子使用 $.get() 方法从服务器上的一个文件中取回数据：

### 实例

```javascript
$("button").click(function(){
  $.get("demo_test.asp",function(data,status){
    alert("Data: " + data + "\nStatus: " + status);
  });
});
```

[亲自试一试](http://www.w3school.com.cn/tiy/t.asp?f=jquery_ajax_get)

$.get() 的第一个参数是我们希望请求的 URL（"demo_test.asp"）。

第二个参数是回调函数。第一个回调参数存有被请求页面的内容，第二个回调参数存有请求的状态。

提示：这个 ASP 文件 ("demo_test.asp") 类似这样：

```javascript
<%
response.write("This is some text from an external ASP file.")
%>
```

### jQuery $.post() 方法

$.post() 方法通过 HTTP POST 请求从服务器上请求数据。语法：

```javascript
$.post(URL,data,callback);
```

必需的 *URL* 参数规定您希望请求的 URL。

可选的 *data* 参数规定连同请求发送的数据。

可选的 *callback* 参数是请求成功后所执行的函数名。

下面的例子使用 $.post() 连同请求一起发送数据：

### 实例

```javascript
$("button").click(function(){
  $.post("demo_test_post.asp",
  {
    name:"Donald Duck",
    city:"Duckburg"
  },
  function(data,status){
    alert("Data: " + data + "\nStatus: " + status);
  });
});
```

[亲自试一试](http://www.w3school.com.cn/tiy/t.asp?f=jquery_ajax_post)

$.post() 的第一个参数是我们希望请求的 URL ("demo_test_post.asp")。

然后我们连同请求（name 和 city）一起发送数据。

"demo_test_post.asp" 中的 ASP 脚本读取这些参数，对它们进行处理，然后返回结果。

第三个参数是回调函数。第一个回调参数存有被请求页面的内容，而第二个参数存有请求的状态。

提示：这个 ASP 文件 ("demo_test_post.asp") 类似这样：

```javascript
<%
dim fname,city
fname=Request.Form("name")
city=Request.Form("city")
Response.Write("Dear " & fname & ". ")
Response.Write("Hope you live well in " & city & ".")
%>
```

### jQuery AJAX 参考手册

如需完整的 AJAX 方法参考，请访问我们的 [jQuery AJAX 参考手册](http://www.w3school.com.cn/jquery/jquery_ref_ajax.asp)。
