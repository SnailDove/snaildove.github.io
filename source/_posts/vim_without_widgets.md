---
title: vim无插件使用
mathjax: true
mathjax2: true
categories: 中文
date: 2016-01-20 20:16:00
tags: [linux]
toc: true
---

本文根据使用经验，会持续更新。

## vim的四种模式

1.  一般模式：normal模式。可以移动光标，删除字符或整行，也可复制、粘贴文件数据。打开vim就是进入这个模式，3个模式的切换也是在这里中转。
2.  编辑模式：一般模式下按下`i` `I` `o` `O` `a` `A` `r` `R` `s` `S` 任何一个进入该模式。可以编辑文件内容，按Esc回到一般模式。 

    -   `i` `I`是insert（在光标所在字符前和行首）
    -   `o` `O`是open新行（在光标所在行的下面另起一新行和在光标所在行的上面另起一行开始插入
    -   `a` `A`是append（在在光标所在字符后和在光标所在你行的行尾）
    -   `s` `S` 是删除（光标所在的字符并开始插入和光标所在行并开始插入），即substitute替换。
    -   `r` `R`是replace光标所在的字符和变成替换模式
3.  命令行模式：一般模式下按下`:` `/` `？`任何一个进入该模式（下文会介绍这些符号的含义）。可以查找数据操作，读取、保存、大量替换字符、离开vim、显示行号等操作，按Esc回到一般模式。
4.  可视模式：一般模式下按下`v` `V` `ctr+v` 进入可视模式，相当于高亮选取文本后的普通模式，即在该模式下进行任意选择特定区域且被选择的区域高亮显示，`v`选择单位：一个字符；  `V` 又称为可视行模式，选择单位：行；`ctr+v`又称为可视块模式，选择的单位：方块；这三者都有用，详细看下文。

## 移动

normal模式下：

`w` → 到下一个单词的开头 `e` → 到下一个单词的结尾 （单词默认是以空格分隔的）
`W` → 到下一个字符串的开头 `E` → 到下一个字符串的结尾 (字符串指的是数字、字母、下划线组成的字符串)
`B` → 到前一个字符串的首字符上	`b` → "命令则将光标移动到前一个word的首字符上。

>   默认上来说，一个单词由字母，数字和下划线组成
>   如果你认为单词是由blank字符分隔符，那么你需要使用大写的E和W（陈皓: 注）

`0` → 数字零，到行头
`^` → 到本行第一个不是blank字符的位置（所谓blank字符就是空格，tab，换行，回车等）
`$` → 到本行行尾
`g_` → 到本行最后一个不是blank字符的位置
`%` → 到光标所在这对括号的另外一个 
`gg` → 首行
`G` → 最后一行
`h` `j` `k` `l`  (强例推荐使用其移动光标，但不必需) →你也可以使用光标键 (←↓↑→). 注: j 向下伸，k是向上伸

1.  **高频使用场景1**： 修改行中某个变量名 先移把光标移动：`w` and `b` 、`W` and `B` （或者如果本行太长可用下文的搜索功能）到目的单词
2.  **高频使用场景2**： 修改缩进，跳到行头`^`
3.  **高频使用场景3**： 查看函数或类的完整或者变量作用域`%`
4.  **高频使用场景4**： 切分屏幕之后，跳转不同窗口：`ctrl+w+(h or j or k or l)`
5.  **高频使用场景5**： 左下上右移动`（h、j、k、l）`
6.  **高频使用场景6**： 删除到末尾:`d$` 删除到开头: `d^`

### 标记

简记：**标记是为了更好地查找**，normal模式下：

`mx` meaning: mark x, x is the name of mark; 
`'x` meaning: go to the position of x mark

1.  **高频使用场景1：** 在函数中看到调用其他函数，你想去看怎么定义的，你看完之后要回来，那么先标记一下，然后在跳回来。

### 语法相关的跳转

normal模式下：

1.  `gd` 意思： go to definition
2.  先按 `[` 再按 `ctrl+d` 跳转到#define处 
3.  先按 `[` 再按 `ctrl+i` 跳转到函数、变量和#define 

**注意**：语言支持不太良好, 大家可以试试所用的语言

### 快速翻页

normal模式下：

| 伙伴1                    | 伙伴2                |
| ------------------------ | -------------------- |
| `ctr + d`   page down    | `ctr + u`  page up   |
| `ctr + f`   page forward | `ctr + b`  page back |



## 动作操作指令

normal模式下：

| 伙伴1                                      | 伙伴2                                      |
| ---------------------------------------- | ---------------------------------------- |
| `d` **d**elete a character and copy to  clipboard | `D` 从光标所在位置一直**删除**到行尾                   |
| `y` **c**opy to clipboard                | `Y` **复制**一行(=`yy`)                      |
| `s` **s**ubstitue a character            | `S` **替换**光标所在行                          |
| `r` **r**eplace a character              | `R ` *不常用*，表示进入替换模式                      |
| `c` **c**hange a character               | `C` *不常用*，表示**修改**光标所在位置一直到行尾，与`S`呈现效果一样 |
| `p` **p**aste after the cursor           | `P` **黏贴**在光标位置之前（如果是黏贴一整行，则黏贴到上一行）      |
| `u` **u**ndo a operation                 | `U` 一次性**撤销**对一整行的所有操作                   |
| `x` cut a character                      | `X` *不常用*， 向左**剪切**，即退格：删除光标的左边那个字符      |
| `*` **向下搜索当前光标所在的单词**，找到就跳到下一个单词         | `#` **向上搜索当前光标所在的单词**，找到就跳到上一个单词         |
| `/word` **向下全文搜索单词word**，跳到匹配的第一个单词，如果多个，继续向下查找按n键（顺着命令本来方向），向上找按N键。 | `?word` **向上全文搜索单词word**，跳到匹配的第一个单词，如果多个，继续向上查找按n键（顺着命令本来方向），向下找按N键。 |
| `a`  **a**ppend after  the cursor        | `A`是**附加**在光标所在行的行尾）                     |
| `i`  **i**nsert before the cursor        | `I`**插入**在光标所在行的行首                       |
| `o` 在光标所在行的下面另起一新行，open the new world？   | `O`在光标所在行的上面另起一行开始插入                     |
| `v` 进入**v**isual模式，用来选择区域（可跨行），用来配合后续的其他操作（增删改查） | `v` 进入visual行模式，用来选择一些行，用来配合后续的其他操作（增删改查） |
| `f` **f**ind a character after the cursor | `F` 向光标位置之前**查找一个字符**                    |
| `t` **t**ill a character tx和fx相同，区别是跳到字符x前 | `T` Tx 和Fx相同，区别是跳到字符x后                   |

### 单独成型

`.` 重复刚才的操作
`~` 转换大小写

1.  可以对变量首字母改变大小写
2.  可以结合下文提供的命令的选择一个字符串（变量），然后再改变整个字符串（变量）的大小写。比如：宏定义

`=` 自动格式化

1.  对当前行用`==` （连按=两次）, 或对多行用`n==`（n是自然数）表示自动缩进从当前行起的下面n行
2.  或者进入可视行模式选择一些行后再`=`进行格式化，相当于一般IDE里的code format。
3.  使用`gg=G`可对整篇代码进行排版。

### 撤销和恢复

1.  `u` undo撤销上一步的操作，命令可以组合，例如`Nu` N是任意一个整数，表示撤销N步操作，以下类同。
2.  `U`  恢复当前行（即一次撤销对当前行的全部操作）
3.  `ctr+r` control+redo 恢复上一步被撤销的操作
4.  `CTRL-R` 回退前一个命令

### 文本替换

normal 模式下输入替换命令：    `:[range]s/pattern/string/[flags]`

- pattern	就是要被替換掉的字串，可以用 regexp 來表示。
- string      將 pattern 由 string 所取代。
- [range] 有以下一些取值：

| [range] | 含义                                       |
| :-----: | :--------------------------------------- |
|    无    | 默认为光标所在的行                                |
|   `.`   | 光标所在当前的行                                 |
|   `N`   | 第N行                                      |
|   `$`   | 最后一行                                     |
|  `'a`   | 标记a所在的行（之前要使用ma做过标记）                     |
|  `.+1`  | 当前光标所在行的下面一行                             |
|  `$-1`  | 倒数第二行，可以对某一行加减某个数值来确定取得相对的行              |
| `22,33` | 第22～33行                                  |
|  `1,$`  | 第1行 到 最后一行                               |
|  `1,.`  | 第1行 到 当前行                                |
|  `.,$`  | 当前行 到 最后一行                               |
| `'a,'b` | 标记a所在的行 到 标记b所在的行（之前要使用ma和mb做过标记）        |
|   `%`   | 所有行（与 1,$ 等价）                            |
| `?str?` | 从当前位置向上搜索，找到的第一个str所在的行 （其中str可以是任何字符串或者正则表达式） |
| `/str/` | 从当前位置向下搜索，找到的第一个str所在的行（其中str可以是任何字符串或者正则表达式） |
**注意，上面的所有用于range的表示方法都可以通过 +、- 操作来设置相对偏移量。**

- [flags]有以下一些取值：

| flags | 含义                       |
| :---: | ------------------------ |
|  `g`  | 对指定范围内的所有匹配项（global）进行替换 |
|  `c`  | 在替换前请求用户确认（confirm）      |
|  `e`  | 忽略执行过程中的错误               |
|  `i`  | ignore 不分大小写             |
|   无   | 只对指定范围内的第一个匹配项进行替换       |

**注意：上面的所有flags都可以组合起来使用，比如 gc 表示对指定范围内的 所有匹配项进行替换，并且在每一次替换之前都会请用户确认。** 

#### 例子

替换某些行的内容

1.  `:10,20s/from/to/g`  对第10行到第20行的内容进行替换。
2.  ` :1,$s/from/to/g`  对第一行到最后一行的内容进行替换（即全部文本）
3.  `:1,.s/from/to/g`  对第一行到当前行的内容进行替换。
4.  `:.,$s/from/to/g`  对当前行到最后一行的内容进行替换。
5.  ` :'a,'bs/from/to/g` 对标记a和b之间的行（含a和b所在的行）进行替换，其中a和b是之前用m命令所做的标记。

替换所有行的内容：`:%s/from/to/g`

## 动作的重复

normal模式下，任意一个动作都可以重复

注：N是数字

-   数字：`Nyy`从当前行算起向下拷贝N行、`Ndd`从当前行算起向下删除N行、`Ngg`跳到第N行、`dNw`删除从当前光标开始到第N个单词前（不包含空白，即删除N-1个单词)、`yNe`拷贝从当前光标到第N个单词末尾（注意： `yy`=`1yy` `dd`=`1dd`）、`d$`删除到本行末尾
-   重复前一个命令： `N.` （N表示重复的次数）

## 区块选择

注：中括号内容为可选项

normal模式下：`[ctr + ] v + (h or j or k or l)`

1.  **高频使用场景1**: `[ctr + ] v` 选中某些行的行头之后 再按`=` 效果：**代码格式自动调整**
2.  **高频使用场景2**: `[ctr + ] v` 选中某些行的行头之后 再按`I`再按注释的符号（比如：`//`）最后按`ESC` 效果：选中的这些行全部注释了 **多行快速注释**
3.  **高频使用场景3**: `[ctr + ] v` 选中某些行的行头之后 再按`A`再按注释的内容 最后按 `ESC`（比如：`//这是测试代码`） 效果：选中的这些行的行尾全部注释上`//这是测试代码` **多行快速注释**
4.  **高频使用场景4**: `[ctr + ] v` 选中某些行的行头的注释（比如：`//`）之后 再按`d` 最后按`ESC` 效果：选中的这些行全部注释删除了 **多行快速删除注释** 
5.  **高频使用场景5**: `[ctr + ] v` 选中某些区块之后，再按上文动作的按键实现区域操作

## 组合的强大

### 操作光标所在的一个单词

normal模式下：

动作 + 移动 [+重复次数] 
前面已经已经大量使用组合，这里继续：

| 动作操作指令+范围       | 效果                                       |
| --------------- | ---------------------------------------- |
| cw or c1 or c1w | change from current cursor to word end   |
| caw             | change whole word including current cursor |
| dw or d1 or d1w | delete from current cursor to word end   |
| daw             | delete whole word including current cursor |
| yw or y1 or y1w | copy from current cursor to word end     |
| yaw             | copy whole word including current cursor |
| d/word             | delete forward until the former character of the next 'word' |
| d?word             | delete backward until the former character of the last 'word' |

| 动作操作指令+范围       | 效果                                       |
| --------------- | ---------------------------------------- |
| dtc            | delete until before the next 'c' |
| dfc            | delete until after the next 'c' |
| 范围+动作操作指令             | 效果         |
| --------------------- | ---------- |
| `bve` 或 `BvE` + c/d/y | 操作一个变量或字符串 |


上表都是高频使用场景

## 自动补全

在insert模式下直接按： 最常用的补全

```
ctrl + n  
ctrl + p
```

智能补全

```
ctrl + x //进入补全模式
```

-   整行补全 `CTRL-X` `CTRL-L`
-   **根据当前文件里关键字补全** `CTRL-X` `CTRL-N`
-   **根据字典补全** `CTRL-X` `CTRL-K`
-   根据同义词字典补全 `CTRL-X` `CTRL-T`
-   **根据头文件内关键字补全** `CTRL-X` `CTRL-I`
-   根据标签补全 `CTRL-X` `CTRL-]`
-   **补全文件名** `CTRL-X` `CTRL-F`
-   补全宏定义 `CTRL-X` `CTRL-D`
-   补全vim命令 `CTRL-X` `CTRL-V`
-   用户自定义补全方式 `CTRL-X` `CTRL-U`
-   **拼写建议** `CTRL-X` `CTRL-S` //例如：一个英文单词

## 折叠

normal模式下：

```
zo (折+open)
zi (折+indent)
zc (折+close)
```

## 切分屏幕

**切分命令**，normal模式下，输入

1.  `vs`(说明：vertically split 纵向切分屏幕）
2.  `sp`(说明：split 横向切分屏幕，即默认的切分方式）

**屏幕相互跳转**

1.  `ctr + w` 再按 `h`或`j`或`k`或`l`
2.  解释：`h`: left , `j` : down , `k` : up, `l` : right

**调整切分窗口的大小**

1.  `ctrl+w` 在按 `+` 或 `-` 或 `=` ，当然在按 `+` 或 `-` 或 `=` 之前先按一个数字，改变窗口高度，`=` 是均分的意思。。
2.  在normal模式下 输入`：resize -N` 或 `:resize +N` 明确指定窗口减少或增加N行
3.  `ctrl+w` 在按 `<` 或 `>` 或 `=` ，当然在按 `<` 或 `>` 或 `=` 之前先按一个数字，改变窗口宽度，`=` 是均分的意思。
4.  有时候预览大文件，感觉切分的屏幕太小，`ctrl+w` + `T` 移动当前窗口至新的标签页。

## tab窗口

vim 从 vim7 开始加入了多标签切换的功能， 相当于多窗口. 之前的版本虽然也有多文件编辑功能， 但是总之不如这个方便啦。 用法normal模式下：

-   `:tabnew` [++opt选项] ［＋cmd］ 文件 建立对指定文件新的tab
-   `:tabc` 关闭当前的tab or `:q`
-   `:tabo` 关闭**其他**的tab
-   `:tabs` 查看**所有打开**的tab
-   `:tabp` 前一个previous tab window
-   `:tabn` 后一个next tab window

 `gt` , `gT` 可以直接在tab之间切换。 还有很多他命令， :help table 吧。

## 目录

normal模式下：

1.  `:Te` 以tab窗口形式显示当前目录 然后可进行切换目录、打开某个文件
2.  `:!ls` 这种是vim调用shell命令的方式`:!ls + shell_command`,但不是以tab窗口的形式显示当前目录。

## 成对符号的内容操作

以下命令可以对标点内的内容进行操作：

1.  `ci'` `ci" `  `ci(`  `ci[` `ci{` `ci<` 分别change这些配对标点符号中的文本内容 
2.  `di'` `di"` `di(`或`dib` `di[` `di{`或`diB` `di<` 分别删除这些配对标点符号中的文本内容 
3.  `yi'` `yi"` `yi(` `yi[` `yi{` `yi<` 分别复制这些配对标点符号中的文本内容 
4.  `vi'` `vi"` `vi(` `vi[` `vi{` `vi< ` 分别选中这些配对标点符号中的文本内容
5.  `cit` `dit` `yit` `vit` 分别操作一对标签之间的内容，编辑html很好用

**另外如果把上面的 `i` 改成 `a` 可以同时操作配对标点和配对标点内的内容**，举个例子：

比如要操作的文本：111"222"333，将光标移到"222"的任何一个字符处输入命令 

-   di" ,文本会变成： 111""333
-   若输入命令 da" ,文本会变成： 111333

## 剪贴板

### 1. 简单复制和粘贴

vim提供12个剪贴板，它们的名字分别为vim有11个粘贴板，分别是0、1、2、...、9、a、“。如果开启了系统剪贴板，则会另外多出两个+和*。使用:reg命令，可以查看各个粘贴板里的内容。

在vim中简单用`y` 只是复制到 `"` 的粘贴板里，同样用`p` 粘贴的也是这个粘贴板里的内容。

### 2. 复制和粘贴到指定剪贴板

要将vim的内容复制到某个粘贴板，进入正常模式后，选择要复制的内容，然后按 `"Ny` 完成复制，其中N为粘贴板号（注意是按一下双引号然后按粘贴板号最后按y），例如要把内容复制到粘贴板a，选中内容后按"ay就可以了。

要将vim某个粘贴板里的内容粘贴进来，需要退出编辑模式，在正常模式按`"Np`，其中N为粘贴板号。比如，可以按`"5p`将5号粘贴板里的内容粘贴进来，也可以按`"+p`将系统全局粘贴板里的内容粘贴进来。

### 3. 系统剪贴板

查看vim支持的剪切板，normal模式下输入`：reg`

和系统剪贴板的交互又应该怎么用呢？遇到问题一般第一个寻找的是帮助文档，剪切板即是 Clipboard。通过` :h clipboard` 查看帮助

星号*和加号+粘贴板是系统粘贴板。在windows系统下， * 和 + 剪贴板是相同的。对于 X11 系统， * 剪贴板存放选中或者高亮的内容， + 剪贴板存放复制或剪贴的内容。打开clipboard选项，可以访问 + 剪贴板；打开xterm_clipboard，可以访问 * 剪贴板。 * 剪贴板的一个作用是，在vim的一个窗口选中的内容，可以在vim的另一个窗口取出。

**复制到系统剪贴板**

example：

`"*y` `"+y` `"+Nyy`  复制N行到系统剪切板

解释：

| 命令            | 含义                                       |
| ------------- | ---------------------------------------- |
| {Visual}"+y   | copy the selected text into the system clipboard |
| "+y{motion}   | copy the text specified by {motion} into the system clipboard |
| :[range]yank+ | copy the text specified by [range] into the system clipboard |

**剪切到系统剪贴板**

example：

"+dd

**从系统剪贴板粘贴到vim**

normal模式下：

1.  `"*p`
2.  `"+p`
3.  `:put+` 含义： Ex command puts contents of system clipboard on a new line

插入模式下：

`<C-r>+` 含义： From insert mode (or commandline mode)

"+p比 Ctrl-v 命令更好，它可以更快更可靠地处理大块文本的粘贴，也能够避免粘贴大量文本时，发生每行行首的自动缩进累积，因为Ctrl-v是通过系统缓存的stream处理，一行一行地处理粘贴的文本。

## vim编码

Vim 可以很好的编辑各种字符编码的文件，这当然包括UCS-2、UTF-8 等流行的 Unicode 编码方式。

四个字符编码选项，encoding、fileencoding、fileencodings、termencoding (这些选项可能的取值请参考 Vim 在线帮助 :help encoding-names，它们的意义如下:

-   **encoding**: Vim 内部使用的字符编码方式

包括 Vim 的 buffer (缓冲区)、菜单文本、消息文本等。默认是根据你的locale选择.用户手册上建议只在 .vimrc 中改变它的值，事实上似乎也只有在.vimrc 中改变它的值才有意义。你可以用另外一种编码来编辑和保存文件，如你的vim的encoding为utf-8,所编辑的文件采用cp936编码,vim会自动将读入的文件转成utf-8(vim的能读懂的方式），而当你写入文件时,又会自动转回成cp936（文件的保存编码).

-   **fileencoding**: Vim 中当前编辑的文件的字符编码方式

Vim 保存文件时也会将文件保存为这种字符编码方式 (不管是否新文件都如此)。

-   **fileencodings**: Vim会自动探测编码设置项

启动时会按照它所列出的字符编码方式逐一探测即将打开的文件的字符编码方式，并且将 fileencoding 设置为最终探测到的字符编码方式。因此最好将Unicode 编码方式放到这个列表的最前面，将拉丁语系编码方式 latin1 放到最后面。

-   **termencoding**: Vim 所工作的终端 (或者 Windows 的 Console 窗口) 的字符编码方式

如果vim所在的term与vim编码相同，则无需设置。如其不然，你可以用vim的termencoding选项将自动转换成term的编码.这个选项在 Windows 下对我们常用的 GUI 模式的 gVim 无效，而对 Console 模式的Vim 而言就是 Windows 控制台的代码页，并且通常我们不需要改变它。 好了，解释完了这一堆容易让新手犯糊涂的参数，我们来看看 Vim 的多字符编码方式支持是如何工作的。

1.  Vim 启动，根据 .vimrc 中设置的 encoding 的值来设置 buffer、菜单文本、消息文的字符编码方式。
2.  读取需要编辑的文件，根据 fileencodings 中列出的字符编码方式逐一探测该文件编码方式。并设置 fileencoding 为探测到的，看起来是正确的 (注1) 字符编码方式。
3.  对比 fileencoding 和 encoding 的值，若不同则调用 iconv 将文件内容转换为encoding 所描述的字符编码方式，并且把转换后的内容放到为此文件开辟的 buffer 里，此时我们就可以开始编辑这个文件了。注意，完成这一步动作需要调用外部的 iconv.dll(注2)，你需要保证这个文件存在于 $VIMRUNTIME 或者其他列在 PATH 环境变量中的目录里。
4.  编辑完成后保存文件时，再次对比 fileencoding 和 encoding 的值。若不同，再次调用 iconv 将即将保存的 buffer 中的文本转换为 fileencoding 所描述的字符编码方式，并保存到指定的文件中。同样，这需要调用 iconv.dll由于 Unicode 能够包含几乎所有的语言的字符，而且 Unicode 的 UTF-8 编码方式又是非常具有性价比的编码方式 (空间消耗比 UCS-2 小)，因此建议 encoding 的值设置为utf-8。这么做的另一个理由是 encoding 设置为 utf-8 时，Vim 自动探测文件的编码方式会更准确 (或许这个理由才是主要的 ;)。我们在中文 Windows 里编辑的文件，为了兼顾与其他软件的兼容性，文件编码还是设置为 GB2312/GBK 比较合适，因此 fileencoding 建议设置为 chinese (chinese 是个别名，在 Unix 里表示 gb2312，在 Windows 里表示cp936，也就是 GBK 的代码页)。

对于fedora来说，vim的设置一般放在/etc/vimrc文件中，不过，建议不要修改它。可以修改~/.vimrc文件（默认不存在，可以自己新建一个），写入所希望的设置。

我的.vimrc文件如下:

```
:set encoding=utf-8
:set fileencodings=ucs-bom,utf-8,cp936
:set fileencoding=gb2312
:set termencoding=utf-8
```

其中，fileencoding配置可以设置utf-8，但是我的mp3好像不支持utf-8编码，所以干脆，我就设置为gb2312了。现在搞定了，不管是vi中还是mp3上都可以显示无乱码的.txt文件了。

## 个人的配置

本人**无插件使用**过程中的配置很短，写在vim的配置文件.vimrc里， 配置是使用**vim script**进行配置的，它有自己的一套语法，详细请点击[vim Script](https://www.w3cschool.cn/vim/nckx1pu0.html)

```vim
set number;display number
set mouse=a; setting smart mouse
set hlsearch ;high light search
set tabstop=4 ; setting tab width 4 letters
set shiftwidth=4; setting new line incident width
set noexpandtab; tab doesn't expand to space
;set list ;display manipulator, example： \n \t \r ......
set encoding=utf-8
set fileencodings=ucs-bom,utf-8,cp936
set fileencoding=gb2312
set termencoding=utf-8
```

## 前进和后退功能

流行的文本编辑器通常都有前进和后退功能，可以在文件中曾经浏览过的位置之间来回移动（联想到浏览器），在 vim 中使用 `Ctrl-O` 执行后退，使用 `Ctrl-I` 执行前进，相关帮助：  `:help CTRL-O`  `:help CTRL-I`   `:help jump-motions`

## vim比较文件

### 启动方法

首先保证系统中的diff命令是可用的。Vim的diff模式是依赖于diff命令的。

```shell
 vimdiff file1 file2 [file3 [file4]]
```
或者
```shell
vim -d file1 file2 [file3 [file4]]
```
窗口比较局部于当前标签页中。你不能看到某窗口和别的标签页中的窗口的差异。这样，可以同时打开多组比较窗口，每组差异在单独的标签页中。Vim 将为每个文件打开一个窗口，并且就像使用  `-O`  参数一样，使用**垂直分割**。如果你要**水平分割**，加上  `-o`  参数: 
```shell
vimdiff -o file1 file2 [file3 [file4]]
```
如果已在 Vim 中，你可以用三种方式进入比较模式，只介绍一种：
```
:diffs[plit] {filename} 
```
对 {filename} 开一个新窗口。当前的和新开的窗口将设定和"vimdiff" 一样的参数。要垂直分割窗口，在前面加上  `:vertical` 。例如: 
```
:vert diffsplit another_filename
```

### 跳转到差异
有两条命令可用于在跳转到差异文所在的位置:
1. `[c`              反向跳转至上一处更改的开始。计数前缀使之重复执行相应次。
2. `]c`              正向跳转至下一个更改的开始。计数前缀使之重复执行相应次。
    如果不存在光标可以跳转到的更改，将产生错误。

### 合并

比较目的就是合并差异，直接使用以下自带命令或者麻烦的办法：手动从一个窗口拷贝至另一个窗口。

```
:[range]diffg[et] [bufspec]
                用另一个缓冲区来修改当前的缓冲区，消除不同之处。除非只有另外一
                个比较模式下的缓冲区， [bufspec] 必须存在并指定那个缓冲区。
                如果 [bufspec] 指定的是当前缓冲区，则为空动作。[range] 可以参考下面。
:[range]diffpu[t] [bufspec]
                用当前缓冲区来修改另一个缓冲区，消除不同之处。
[count]do       同 ":diffget"，但没有范围。"o" 表示 "obtain" (不能用
                "dg"，因为那可能是 "dgg" 的开始！)。
dp              同 ":diffput"，但没有范围。注意 不适用于可视模式。
                给出的 [count] 用作 ":diffput" 的 [bufspec] 参数。

当没有给定 [range] 时，受影响的仅是当前光标所处位置或其紧上方的差异文本。
当指定 [range] 时，Vim 试图仅改动它指定的行。不过，当有被删除的行时，这不总有效。
参数 [bufspec] 可以是缓冲区的序号，匹配缓冲区名称或缓冲区名称的一部分的模式。
例如:
        :diffget                使用另一个进入比较模式的缓冲区
        :diffget 3              使用 3 号缓冲区
        :diffget v2             使用名字同 "v2" 匹配的缓冲区，并进入比较模式(例如，"file.c.v2")
```
### 更新比较和撤销修改

比较基于缓冲区的内容。因而，如果在载入文件后你做过改动，这些改动也将参加比较。不过，你也许要不时地使用 `:diffupdate[!]`。因为并非所有的改动的结果都能自动更新。包含` !` 时，Vim 会检查文件是否被外部改变而需要重新载入。对每个被改变的文件给出提示。

如果希望撤销修改，可以和平常用vim编辑一样，直接进入normal模式下按` u`但是要注意一定要将光标移动到需要撤销修改的文件窗口中。

### 上下文的展开和查看

比较和合并文件的时候经常需要结合上下文来确定最终要采取的操作。Vimdiff 缺省是会把不同之处上下各 6 行的文本都显示出来以供参考。其他的相同的文本行被自动折叠。如果希望修改缺省的上下文行数，可以这样设置：

 ```
:set diffopt=context:3
 ```

### 多个文件的退出

在比较和合并告一段落之后，可以用下列命令对多个文件同时进行操作。

比如同时退出：`:qa （quit all）`  

如果希望保存全部文件：`:wa （write all）`

或者是两者的合并命令，保存全部文件，然后退出：`:wqa （write, then quit all）`

如果在退出的时候不希望保存任何操作的结果：`:qa! （force to quit all）`

### vimdiff 详细请参考

1. vim下 `:help diff`
2. [vimdiff doc](http://vimcdoc.sourceforge.net/doc/diff.html)

## vim命令行的保存、离开等命令：

1.   `:w ` 将编辑的数据写入硬盘文件中。
2.   `:w!` 若文件属性为“只读”，强制写入该文件。但能否写入还由对该文件的文件权限有关。
3.   `:q`保存后离开。若为“:wq！”则强制保存后离开。
4.   `:w[文件名] ` 将编辑的数据保存为另一个文件。
5.   `:r[文件名] ` 在编辑的数据中读入另一个文件的内容加到光标所在行后面。
6.   `:n1,n2 w[文件名]` 将n1行到n2行的内容保存到另一个文件。
7.   `:!command` 暂时离开vi到命令行模式下执行command的显示结果。
8.   `ZZ` 若文件未改动，则直接离开；若已改动则保存后离开。
9.   `set num/nonum` 显示/取消行号。

## VIM的宏

宏的使用非常强大，前往[vim 中，宏的使用](http://blog.sina.com.cn/s/blog_69e5d8400102w1z1.html)

## 完整版命令
本文只提供个人使用过程中积累的高频场景，完整版请点击[此处](http://q3rrj5fj6.bkt.clouddn.com/gitpage/vim/vim_command.png)，或查阅 vim manual

## 玩游戏来熟能生巧

**用进废退**，所以多用才是王道，这里推荐一个游戏：通过键盘输入控制人物角色冒险的游戏，玩游戏的过程中熟悉VIM命令: [vim-adventures](https://vim-adventures.com/)



## **参考**

1.  [官方文档](https://vim.sourceforge.io/docs.php)
2.  [vim doc](http://vimcdoc.sourceforge.net/doc/help.html) 中文
3.  [freewater 博客](http://www.cnblogs.com/freewater/archive/2011/08/26/2154602.html)
4.  [Thinking In Linux](http://www.linuxsong.org/2010/09/vim-quick-select-copy-delete/)