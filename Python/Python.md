## Python面试题

### 1、赋值、浅拷贝和深拷贝的区别？

**(1) 赋值**

在 Python 中，对象的赋值就是简单的对象引用，如下所示 **a 和 b 是一样的，他们指向同一片内存**，b 不过是 a 的**别名**，是**引用**。

```
a = [1,2,"hello",['python', 'C++']]
b = a
```

**使用 `b is a` 去判断，返回 `True`**，表明他们地址相同，内容相同，也可以**使用` id() `函数查看它们的地址是否相同**。

赋值操作（包括对象作为参数、返回值）**不会开辟新的内存空间**，它只是复制了对象的引用。也就是说除了 b 这个名字之外，没有其他的内存开销。修改了 a，也就影响了 b，同理，修改了 b，也就影响了 a。

**(2) 浅拷贝**

浅拷贝会创建新对象，其内容非原对象本身的引用，而是**原对象内第一层对象的引用**。

浅拷贝有三种形式（上述的列表 a 为例）：

- **切片操作**：b = a[:] 或者 b = [x for x in a]；
- **工厂函数**：b = list(a)；
- **copy 函数**：b = copy.copy(a)；

浅拷贝产生的列表 b 不再是列表 a 了，使用 is 判断可以发现**他们不是同一个对象**，使用 id 查看，他们也不指向同一片内存空间。
但是当使用` id(x) for x in a` 和` id(x) for x in b `来查看 a 和 b 中元素的地址时，可以看到**二者包含的元素的地址是相同的**。

浅拷贝仅仅只拷贝了一层，在列表 a 中有一个嵌套的 list，如果我们修改了它，列表 b 会发生变化。

**(3) 深拷贝**

**深拷贝拷贝了对象的所有元素**，包括多层嵌套的元素。因此，它的时间和空间开销更高。

同样的对列表 a，如果使用 b = copy.deepcopy(a)，再修改列表 b 将不会影响到列表 a，即使嵌套的列表具有更深的层次，也不会产生任何影响，因为深拷贝拷贝出来的对象根本就是一个全新的对象，不再与原来的对象有任何的关联。

**(4) 注意点**

对于非容器类型，如数字、字符，以及其他的“原子”类型，没有拷贝一说，产生的都是原对象的引用。

如果元组变量值包含原子类型对象，即使采用了深拷贝，也只能得到浅拷贝。

### 3、**init** 和__new__的区别？

当我们使用「类名()」创建对象的时候，Python 解释器会帮我们做两件事情：第一件是为对象在内存分配空间，第二件是为对象进行初始化。「分配空间」是__new__ 方法，初始化是__init__方法。

**new** 方法在内部其实做了两件时期：第一件事是为「对象分配空间」，第二件事是「把对象的引用返回给 Python 解释器」。当 Python 的解释器拿到了对象的引用之后，就会把对象的引用传递给 **init** 的第一个参数 self，**init** 拿到对象的引用之后，就可以在方法的内部，针对对象来定义实例属性。

之所以要学习 **new** 方法，就是因为需要对分配空间的方法进行改造，改造的目的就是为了当使用「类名()」创建对象的时候，无论执行多少次，在内存中永远只会创造出一个对象的实例，这样就可以达到单例设计模式的目的。

### 5、创建百万级实例如何节省内存？

==可以定义类的 **slot** 属性，用它来声明实例属性的列表，可以用来减少内存空间的目的。==

**具体解释：**

首先，我们先定义一个普通的 User 类：

```
class User1:
    def __init__(self, id, name, sex, status):
        self.id = id
        self.name = name
        self.sex = sex
        self.status = status
```

然后再定义一个带 **slot** 的类：

```
class User2:
    __slots__ = ['id', 'name', 'sex', 'status']
    def __init__(self, id, name, sex, status):
        self.id = id
        self.name = name
        self.sex = sex
        self.status = status
```

接下来创建两个类的实例：

```
u1 = User1('01', 'rocky', '男', 1)
u2 = User2('02', 'leey', '男', 1)
```

我们已经知道 u1 比 u2 使用的内存多，我们可以这样来想，一定是 u1 比 u2 多了某些属性，我们分别来看一下 u1 和 u2 的属性：

```
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'id', 'name', 'sex', 'status']
['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', 'id', 'name', 'sex', 'status']
```

乍一看好像差别不大，我们下面具体来看一下差别在哪：

```
set(dir(u1)) - set(dir(u2))
```

通过做集合的差集，我们得到 u1 和 u2 在属性上的具体差别：

```
{'__weakref__', '__dict__'}
```

在我们不使用弱引用的时候，**weakref** 并不占用多少内存，那最终这个锅就要 **dict** 来背了。

下面我们来看一下 **dict**:

```
u1.__dict__
```

输出结果如下所示：

```
{'id': '01', 'name': 'rocky', 'sex': '男', 'status': 1}
```

输出一个字典，在它内部我们发现了刚刚在类里定义的属性，这个字典就是为了实例动态绑定属性的一个字典，我们怎么动态绑定呢？比如我们现在没有 u1.level 这个属性，那么我们可以为它动态绑定一个 level 属性，比如 u1.level = 10，然后我们再来考察这个字典：

```
u1.__dict__
```

现在输出的结果为：

```
{'id': '01', 'name': 'rocky', 'sex': '男', 'status': 1, 'level': 10}
```

这样看到 level 进入到这个字典中。

这样一个动态绑定属性的特性，其实是以牺牲内存为代价的，因为这个 **dict** 它本身是占用内存的，接下来我们来验证这件事情：

```
import sys
sys.getsizeof(u1.__dict__)
```

我们用 sys 模块下的 getsizeof 方法，它可以得到一个对象使用的内存：

```
112
```

我们可以看到这个字典占用了 112 的字节。反观 u2，它没有了 **dict** 这个属性，我们想给它添加一个属性，也是被拒绝的。

```
u2.level = 10
```

显示的结果如下所示：

```
AttributeError: 'User2' object has no attribute 'level'
```

### 9、Python 中有日志吗?怎么使用？

Python 中有日志，Python 自带 logging 模块，调用 logging.basicConfig()方法，配置需要的日志等级和相应的参数，Python 解释器会按照配置的参数生成相应的日志。

**补充知识：**

**Python 的标准日志模块**

Python 标准库中提供了 logging 模块供我们使用。在最简单的使用中，默认情况下 logging 将日志打印到屏幕终端，我们可以直接导入 logging 模块，然后调用 debug，info，warn，error 和 critical 等函数来记录日志，默认日志的级别为 warning，级别比 warning 高的日志才会被显示（critical > error > warning > info > debug），「级别」是一个逻辑上的概念，用来区分日志的重要程度。

```
import logging

logging.debug('debug message')
logging.info("info message")
logging.warn('warn message')
logging.error("error message")
logging.critical('critical message')
```

上述代码的执行结果如下所示：

```
WARNING:root:warn message
ERROR:root:error message
CRITICAL:root:critical message
```

我在上面说过，用 print 的话会产生大量的信息，从而很难从中找到真正有用的信息。而 logging 中将日志分成不同的级别以后，我们在大多数时间只保存级别比较高的日志信息，从而提高了日志的性能和分析速度，这样我们就可以很快速的从一个很大的日志文件里找到错误的信息。

**配置日志格式**

我们在用 logging 来记录日志之前，先来进行一些简单的配置：

```
import logging

logging.basicConfig(filename= 'test.log', level= logging.INFO)

logging.debug('debug message')
logging.info("info message")
logging.warn('warn message')
logging.error("error message")
logging.critical('critical message')
```

运行上面的代码以后，会在当前的目录下新建一个 test.log 的文件，这个文件中存储 info 以及 info 以上级别的日志记录。运行一次的结果如下所示：

```
INFO:root:info message
WARNING:root:warn message
ERROR:root:error message
CRITICAL:root:critical message
```

上面的例子中，我是用 basicConfig 对日志进行了简单的配置，其实我们还可以进行更为复杂些的配置，在此之前，我们先来了解一下 logging 中的几个概念：

```
Logger：日志记录器，是应用程序中可以直接使用的接口。
Handler：日志处理器，用以表明将日志保存到什么地方以及保存多久。
Formatter：格式化，用以配置日志的输出格式。
```

上述三者的关系是：一个 Logger 使用一个 Handler，一个 Handler 使用一个 Formatter。那么概念我们知道了，该如何去使用它们呢？我们的 logging 中有很多种方式来配置文件，简单的就用上面所说的 basicConfig，对于比较复杂的我们可以将日志的配置保存在一个配置文件中，然后在主程序中使用 fileConfig 读取配置文件。

基本的知识我们知道了，下面我们来做一个小的题目：日志文件保存所有 debug 及其以上级别的日志，每条日志中要有打印日志的时间，日志的级别和日志的内容。请先自己尝试着思考一下，如果你已经思考完毕请继续向下看：

```
import logging

logging.basicConfig(
   level= logging.DEBUG,
   format = '%(asctime)s : %(levelname)s : %(message)s',
   filename= "test.log"
)

logging.debug('debug message')
logging.info("info message")
logging.warn('warn message')
logging.error("error message")
logging.critical('critical message')
```

上述代码的一次运行结果如下：

```
2018-10-19 22:50:35,225 : DEBUG : debug message
2018-10-19 22:50:35,225 : INFO : info message
2018-10-19 22:50:35,225 : WARNING : warn message
2018-10-19 22:50:35,225 : ERROR : error message
2018-10-19 22:50:35,225 : CRITICAL : critical message
```

我刚刚在上面说过，对于比较复杂的我们可以将日志的配置保存在一个配置文件中，然后在主程序中使用 fileConfig 读取配置文件。下面我们就来看一个典型的日志配置文件（配置文件名为 logging.conf）：

```
[loggers]
keys = root

[handlers]
keys = logfile

[formatters]
keys = generic

[logger_root]
handlers = logfile

[handler_logfile]
class = handlers.TimedRotatingFileHandler
args = ('test.log', 'midnight', 1, 10)
level = DEBUG
formatter = generic

[formatter_generic]
format = %(asctime)s %(levelname)-5.5s [%(name)s:%(lineno)s] %(message)s
```

在上述的日志配置文件中，首先我们在 [loggers] 中声明了一个叫做 root 的日志记录器（logger），在 [handlers] 中声明了一个叫 logfile 的日志处理器（handler），在 [formatters] 中声明了一个名为 generic 的格式化（formatter）。之后在 [logger_root] 中定义 root 这个日志处理器（logger） 所使用的日志处理器（handler） 是哪个，在 [handler_logfile] 中定义了日志处理器（handler） 输出日志的方式、日志文件的切换时间等。最后在 [formatter_generic] 中定义了日志的格式，包括日志的产生时间，级别、文件名以及行号等信息。

有了上述的配置文件以后，我们就可以在主代码中使用 logging.conf 模块的 fileConfig 函数加载日志配置：

```
import logging
import logging.config

logging.config.fileConfig('logging.conf')

logging.debug('debug message')
logging.info("info message")
logging.warn('warn message')
logging.error("error message")
logging.critical('critical message')
```

上述代码的运行一次的结果如下所示：

```
2018-10-19 23:00:02,809 WARNI [root:8] warn message
2018-10-19 23:00:02,809 ERROR [root:9] error message
2018-10-19 23:00:02,809 CRITI [root:10] critical message
```



### 15、关于 Python 程序的运行方面，有什么手段能提升性能？

1、使用多进程，充分利用机器的多核性能

2、对于性能影响较大的部分代码，可以使用 C 或 C++ 编写

3、对于 IO 阻塞造成的性能影响，可以使用 IO 多路复用来解决

4、尽量使用 Python 的内建函数

5、尽量使用局部变量

### 17、os.path和sys.path的区别？

os.path 主要是用于对系统路径文件的操作。

sys.path 主要是对 Python 解释器的系统环境参数的操作（动态的改变 Python 解释器搜索路径）。

os.path是module，包含了各种处理长文件名(路径名)的函数。

sys.path是由目录名构成的列表，Python 从中查找扩展模块( Python 源模块, 编译模块,或者二进制扩展). 启动 Python 时,这个列表从根据内建规则,PYTHONPATH 环境变量的内容, 以及注册表( Windows 系统)等进行初始化。

### 18、4G 内存怎么读取一个 5G 的数据？

方法一：

通过生成器，分多次读取，每次读取数量相对少的数据（比如 500MB）进行处理，处理结束后
在读取后面的 500MB 的数据。

方法二：

可以通过 linux 命令 split 切割成小文件，然后再对数据进行处理，此方法效率比较高。可以按照行
数切割，可以按照文件大小切割。

> 　　在Linux下用split进行文件分割：
> 　模式一：指定分割后文件行数
> 　对与txt文本文件，可以通过指定分割后文件的行数来进行文件分割。
> 　命令：split -l 300 large_file.txt new_file_prefix
> 　模式二：指定分割后文件大小
> 　split -b 10m server.log waynelog

### 19、输入某年某月某日，判断这一天是这一年的第几天？

使用 Python 标准库 datetime

```
import datetime

def dayofyear():
    year = input("请输入年份：")
    month = input("请输入月份：")
    day = input("请输入天：")
    date1 = datetime.date(year=int(year)，month=int(month)，day=int(day))
    date2 = datetime.date(year=int(year)，month=1，day=1)
    return (date1-date2+1).days
```

### 21、Python 中的 os 模块常见方法？

os.remove() 删除文件

os.rename() 重命名文件

os.walk() 生成目录树下的所有文件

os.chdir() 改变目录

os.mkdir/makedirs 创建目录/多层目录

os.rmdir/removedirs 删除目录/多层目录

os.listdir() 列出指定目录的文件

os.getcwd() 取得当前工作目录

os.chmod() 改变目录权限

os.path.basename() 去掉目录路径，返回文件名

os.path.dirname() 去掉文件名，返回目录路径

os.path.join() 将分离的各部分组合成一个路径名

os.path.split() 返回（dirname(),basename())元组

os.path.splitext() 返回(filename,extension)元组

os.path.getatime\ctime\mtime 分别返回最近访问、创建、修改时间

os.path.getsize() 返回文件大小

os.path.exists() 是否存在

os.path.isabs() 是否为绝对路径

os.path.isdir() 是否为目录

os.path.isfile() 是否为文件

### 25、lambda 表达式格式以及应用场景？

lambda函数就是可以接受任意多个参数（包括可选参数）并且返回单个表达式值得函数。

语法：lambda [arg1 [,arg2,.....argn]]:expression

```
def calc(x,y):
    return x*y
```

将上述一般函数改写为匿名函数：

```
lambda x,y:x*y
```

**应用**

(1) lambda函数比较轻便，即用即仍，适合完成只在一处使用的简单功能。

(2) 匿名函数，一般用来给filter，map这样的函数式编程服务

(3) 作为回调函数，传递给某些应用，比如消息处理。

### 26、如何理解 Python 中字符串中的\字符？

1、转义字符

2、路径名中用来连接路径名

3、编写太长代码手动软换行

### 27、常用的 Python 标准库都有哪些？

os 操作系统、time 时间、random 随机、pymysql 连接数据库、threading 线程、multiprocessing
进程、queue 队列

第三方库：
django、flask、requests、virtualenv、selenium、scrapy、xadmin、celery、re、hashlib、md5

常用的科学计算库：Numpy，Pandas、matplotlib

### 28、如何在Python中管理内存？

python中的内存管理由Python私有堆空间管理。所有Python对象和数据结构都位于私有堆中。程序员无权访问此私有堆。python解释器负责处理这个问题。

Python对象的堆空间分配由Python的内存管理器完成。核心API提供了一些程序员编写代码的工具。

Python还有一个内置的垃圾收集器，它可以回收所有未使用的内存，并使其可用于堆空间。

### 29、介绍一下 except 的作用和用法？

except: 捕获所有异常

except:<异常名>: 捕获指定异常

except:<异常名 1, 异常名 2>: 捕获异常 1 或者异常 2

except:<异常名>,<数据>: 捕获指定异常及其附加的数据

except:<异常名 1,异常名 2>:<数据>: 捕获异常名 1 或者异常名 2,及附加的数据

### 30、在 except 中 return 后还会不会执行 finally 中的代码？怎么抛出自定义异常？

会继续处理 finally 中的代码；
用 raise 方法可以抛出自定义异常。

### 31、read、readline 和 readlines 的区别？

read:读取整个文件。

readline：读取下一行，使用生成器方法。

readlines：读取整个文件到一个迭代器以供我们遍历。

### 32、range 和 xrange 的区别？

两者用法相同，不同的是 range 返回的结果是一个列表，而 xrange 的结果是一个生成器，前者是直接开辟一块内存空间来保存列表，后者是边循环边使用，只有使用时才会开辟内存空间，所以当列表
很长时，使用 xrange 性能要比 range 好。（稍作了解即可，xrange 现在不怎么用，但是不排除有人不知道）

### 33、请简述你对 input()函数的理解？

在 Python3 中，input()获取用户输入，不论用户输入的是什么，获取到的都是字符串类型的。

在 Python2 中有 raw_input()和 input(), raw_input()和 Python3 中的 input()作用是一样的，
input()输入的是什么数据类型的，获取到的就是什么数据类型的。

### 34、代码中要修改不可变数据会出现什么问题？抛出什么异常？

代码不会正常运行，抛出 TypeError 异常。

### 35、print 调用 Python 中底层的什么方法？

print 方法默认调用 sys.stdout.write 方法，即往控制台打印字符串。

### 36、Python 的 sys 模块常用方法

sys.argv 命令行参数 List，第一个元素是程序本身路径

sys.modules.keys() 返回所有已经导入的模块列表

sys.exc_info() 获取当前正在处理的异常类,exc_type、exc_value、exc_traceback 当前处理的
异常详细信息

sys.exit(n) 退出程序，正常退出时 exit(0)  sys.hexversion 获取 Python 解释程序的版本值，16 进制格式如：0x020403F0

sys.version 获取 Python 解释程序的版本信息

sys.maxint 最大的 Int 值

sys.maxunicode 最大的 Unicode 值

sys.modules 返回系统导入的模块字段，key 是模块名，value 是模块

sys.path 返回模块的搜索路径，初始化时使用 PYTHONPATH 环境变量的值

sys.platform 返回操作系统平台名称

sys.stdout 标准输出

sys.stdin 标准输入

sys.stderr 错误输出

sys.exc_clear() 用来清除当前线程所出现的当前的或最近的错误信息

sys.exec_prefix 返回平***立的 python 文件安装的位置

sys.byteorder 本地字节规则的指示器，big-endian 平台的值是'big',little-endian 平台的值是
'little'  sys.copyright 记录 python 版权相关的东西

sys.api_version 解释器的 C 的 API 版本

sys.version_info 元组则提供一个更简单的方法来使你的程序具备 Python 版本要求功能

### 37、unittest 是什么？

在 Python 中，unittest 是 Python 中的单元测试框架。它拥有支持共享搭建、自动测试、在测试
中暂停代码、将不同测试迭代成一组等的功能。

### 38、模块和包是什么？

在 Python 中，模块是搭建程序的一种方式。每一个 Python 代码文件都是一个模块，并可以引用
其他的模块，比如对象和属性。

一个包含许多 Python 代码的文件夹是一个包。一个包可以包含模块和子文件夹。

### 39、什么是正则的贪婪匹配？

```
>>>re.search('ab*c', 'abcaxc')
<_sre.SRE_Match object; span=(0, 3), match='abc'>

>>>re.search('ab\D+c', 'abcaxc')
<_sre.SRE_Match object; span=(0, 6), match='abcaxc'>
```

贪婪匹配：正则表达式一般趋向于最大长度匹配，也就是所谓的贪婪匹配。

非贪婪匹配：就是匹配到结果就好，就少的匹配字符。

### 40、常用字符串格式化哪几种？

% 格式化字符串操作符

```
print 'hello %s and %s' % ('df', 'another df')
```

字典形式的字符串格式化方法

```
print 'hello %(first)s and %(second)s' % {'first': 'df', 'second': 'another df'}
```

**字符串格式化（format）**

(1) 使用位置参数

位置参数不受顺序约束，且可以为{}，参数索引从0开始，format里填写{}对应的参数值。

```
>>> msg = "my name is {}, and age is {}"
>>> msg.format("hqs",22)
'my name is hqs, and age is 22'
```

(2) 使用关键字参数

关键字参数值要对得上，可用字典当关键字参数传入值，字典前加**即可

```
>>> hash = {'name':'john' , 'age': 23}
>>> msg = 'my name is {name}, and age is {age}'
>>> msg.format(**hash)
'my name is john,and age is 23'
```

(3) 填充与格式化

:[填充字符][对齐方式 <^>][宽度]

```
>>> '{0:*<10}'.format(10)      # 左对齐
'10********'
```

### 41、面向对象深度优先和广度优先是什么？

在子类继承多个父类时，属性查找方式分深度优先和广度优先两种。

当类是经典类时，多继承情况下，在要查找属性不存在时，会按照深度优先方式查找下去。

当类是新式类时，多继承情况下，在要查找属性不存在时，会按照广度优先方式查找下去。

### 42、“一行代码实现 xx”类题目

**(1) 一行代码实现 1 - 100 的和**

可以利用 sum() 函数。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384287412/7FA2A8FF116DF0F21984BBEDB3DE5A47) 

**(2) 一行代码实现数值交换**

不用二话，直接换。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384311042/0A0689A7AE82C32F5554DA1D96E3CEB8) 

**(3) 一行代码求奇偶数**

使用列表推导式。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384323296/6BF7C878327C7B035055095E84B5CA8C) 

**(4) 一行代码展开列表**

使用列表推导式，稍微复杂一点，注意顺序。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384339662/5189E47789E9BB49691D11860DDDC9D6) 

**(5) 一行代码打乱列表**

用到 random 的 shuffle。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384349124/51254F58911137F7D74BC2A6B68B5038) 

**(6) 一行代码反转字符串**

使用切片。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384358108/2886A87A379169573623217D987C095F) 

**(7) 一行代码查看目录下所有文件**

使用 os 的 listdir。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384377068/4CDD1123065F4F15F15C74492F1F9357) 

**(8) 一行代码去除字符串间的空格**

法 1 replace 函数。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384389780/05505DF5F7B6F7506869F55E34422AC8) 

法 2  join & split 函数。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384402910/61081C7DD90FFDACEF9AF4876D6329E2) 

**(9) 一行代码实现字符串整数列表变成整数列表**

使用 list & map & lambda。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384411453/B7DFD7603B68CA89116607857FE826E0) 

**(10) 一行代码删除列表中重复的值**

使用 list & set。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384422982/CEAA88F36BD7D2B78CD3CCE7D09F846E) 

**(11) 一行代码实现 9 * 9 乘法表

稍稍复杂的列表推导式，耐心点就行，一点点的搞...

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384436364/7ED7D2816054151D0B83778BB2557E63) 

**(12) 一行代码找出两个列表中相同的元素**

使用 set 和 &。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384447174/1D7B140350A883F2B13C6B7A6816EE3A) 

**(13) 一行代码找出两个列表中不同的元素**

使用 set 和 ^。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384457945/05D7BFA63A950DEDCEB0D07F7EFBE247) 

**(14)一行代码合并两个字典**

使用 Update 函数。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384467047/D8A9757BF82401B56766CAB6D670B60D) 

**(15) 一行代码实现字典键从小到大排序**

使用 sort 函数。

![图片说明](https://uploadfiles.nowcoder.com/images/20220721/871523328_1658384475333/8A1FC074215FC8CD3A685173F14EBB81) 

### 43、Python 中类方法、类实例方法、静态方法有何区别？

类方法：是类对象的方法，在定义时需要在上方使用“@ classmethod”进行装饰，形参为 cls，表示类对象，类对象和实例对象都可调用

类实例方法：是类实例化对象的方法，只有实例对象可以调用，形参为 self，指代对象本身

静态方法：是一个任意函数，在其上方使用“[@ staticmethod”进行装饰，可以用对象直接调用，静态方法实际上跟该类没有太大关系]()

### 44、Python 的内存管理机制及调优手段？

**内存管理机制：引用计数、垃圾回收、内存池。**

**引用计数**

引用计数是一种非常高效的内存管理手段， 当一个 Python 对象被引用时其引用计数增加 1， 当其不再被一个变量引用时则计数减 1. 当引用计数等于 0 时对象被删除。

**垃圾回收**

 引用计数

引用计数也是一种垃圾收集机制，而且也是一种最直观，最简单的垃圾收集技术。当 Python 的某
个对象的引用计数降为 0 时，说明没有任何引用指向该对象，该对象就成为要被回收的垃圾了。比如
某个新建对象，它被分配给某个引用，对象的引用计数变为 1。如果引用被删除，对象的引用计数为 0，
那么该对象就可以被垃圾回收。不过如果出现循环引用的话，引用计数机制就不再起有效的作用了

(2)标记清除

如果两个对象的引用计数都为 1，但是仅仅存在他们之间的循环引用，那么这两个对象都是需要被
回收的，也就是说，它们的引用计数虽然表现为非 0，但实际上有效的引用计数为 0。所以先将循环引
用摘掉，就会得出这两个对象的有效计数。

(3) 分代回收

从前面“标记-清除”这样的垃圾收集机制来看，这种垃圾收集机制所带来的额外操作实际上与系统
中总的内存块的数量是相关的，当需要回收的内存块越多时，垃圾检测带来的额外操作就越多，而垃圾
回收带来的额外操作就越少；反之，当需回收的内存块越少时，垃圾检测就将比垃圾回收带来更少的额
外操作。

举个例子：

当某些内存块 M 经过了 3 次垃圾收集的清洗之后还存活时，我们就将内存块 M 划到一个集合 A 中去，而新分配的内存都划分到集合 B 中去。当垃圾收集开始工作时，大多数情况都只对集合 B 进行垃圾回收，而对集合 A 进行垃圾回收要隔相当长一段时间后才进行，这就使得垃圾收集机制需要处理的内存少了，效率自然就提高了。在这个过程中，集合 B 中的某些内存块由于存活时间长而会被转移到集合 A 中，当然，集合 A 中实际上也存在一些垃圾，这些垃圾的回收会因为这种分代的机制而被延迟。

**内存池**

(1) Python 的内存机制呈现金字塔形状，-1，-2 层主要有操作系统进行操作

(2) 第 0 层是 C 中的 malloc，free 等内存分配和释放函数进行操作

(3)第 1 层和第 2 层是内存池，有 Python 的接口函数 PyMem_Malloc 函数实现，当对象小于
256K 时有该层直接分配内存

(4) 第 3 层是最上层，也就是我们对 Python 对象的直接操作

Python 在运行期间会大量地执行 malloc 和 free 的操作，频繁地在用户态和核心态之间进行切换，这将严重影响 Python 的执行效率。为了加速 Python 的执行效率，Python 引入了一个内存池机制，用于管理对小块内存的申请和释放。

Python 内部默认的小块内存与大块内存的分界点定在 256 个字节，当申请的内存小于 256 字节时，PyObject_Malloc 会在内存池中申请内存；当申请的内存大于 256 字节时，PyObject_Malloc 的行为将蜕化为 malloc 的行为。当然，通过修改 Python 源代码，我们可以改变这个默认值，从而改变 Python 的默认内存管理行为。

### 45、内存泄露是什么？如何避免？

由于疏忽或错误造成程序未能释放已经不再使用的内存的情况。

内存泄漏并非指内存在物理上的消失，而是应用程序分配某段内存后，由于设计错误，失去了对该段内存的控制，因而造成了内存的浪费。导致程序运行速度减慢甚至系统崩溃等严重后果。

**del**() 函数的对象间的循环引用是导致内存泄漏的主凶。

不使用一个对象时使用:del object 来删除一个对象的引用计数就可以有效防止内存泄漏问题。

通过 Python 扩展模块 gc 来查看不能回收的对象的详细信息。

可以通过 sys.getrefcount(obj) 来获取对象的引用计数，并根据返回值是否为 0 来判断是否内存
泄漏。

### 46、Python 函数调用的时候参数的传递方式是值传递还是引用传递？

Python 的参数传递有：位置参数、默认参数、可变参数、关键字参数。函数的传值到底是值传递还是引用传递，要分情况：

**不可变参数用值传递**

像整数和字符串这样的不可变对象，是通过拷贝进行传递的，因为你无论如何都不可能在原处改变不可变对象

**可变参数是引用传递的**

比如像列表，字典这样的对象是通过引用传递、和 C 语言里面的用指针传递数组很相似，可变对象能在函数内部改变。

### 47、对缺省参数的理解？

缺省参数指在调用函数的时候没有传入参数的情况下，调用默认的参数，在调用函数的同时赋值时，所传入的参数会替代默认参数。

*args 是不定长参数，他可以表示输入参数是不确定的，可以是任意多个。

**kwargs 是关键字参数，赋值的时候是以键 = 值的方式，参数是可以任意多对在定义函数的时候
不确定会有多少参数会传入时，就可以使用两个参数。

**补充**

***args**

如果你之前学过 C 或者 C++，看到星号的第一反应可能会认为这个与指针相关，然后就开始方了，其实放宽心，Python 中是没有指针这个概念的。

在 Python 中我们使用星号收集位置参数，请看下面的例子：

```
>>> def fun(x,*args):
...    print(x)
...    res = x
...    print(args)
...    for i in args:
...            res += i
...    return res
...
>>> print(fun(1,2,3,4,5,6))
```

上述例子中，函数的参数有两个，但是我们在输出的时候赋给函数的参数个数不仅仅是两个，让我们来运行这个代码，得到如下的结果：

```
1
(2, 3, 4, 5, 6)
21
```

从上面我们可以看出，参数 x 得到的值是 1，参数 args 得到的是一个元组 (2,3,4,5,6) ，由此我们可以得出，如果输入的参数个数不确定，其它的参数全部通过 *args 以元组的方式由 arg 收集起来。

为了更能明显的看出 *args，我们下面用一个简单的函数来表示：

```
>>> def print_args(*args):
...    print(args)
...
```

接下来我传入不同的值，通过参数 *args 得到的结果我们来看一下：

```
>>> print_args(1,2,3)
(1, 2, 3)
>>> print_args('abc','def','ghi')
('abc', 'def', 'ghi')
>>> print_args('abc',['a','b','c'],1,2,3)
('abc', ['a', 'b', 'c'], 1, 2, 3)
```

不管是什么，都可以一股脑的塞进元组里，即使只有一个值，也是用元组收集，所以还记得在元组里一个元素的时候的形式吗？元组中如果只有一个元素，该元素的后面要有一个逗号。

那么如果不给 *args 传值呢？

```
>>> def print_args(*args):
...    print(args)
...
>>> print_args()
()
```

答案就是这时候 *args 收集到的是一个空的元组。

最后提醒一点的是，当使用星号的时候，不一定要把元组参数命名为 args，但这个是 Python 中的一个常见做法。

***\*kwargs**

使用两个星号是收集关键字参数，可以将参数收集到一个字典中，参数的名字是字典的 “键”，对应的参数的值是字典的 “值”。请看下面的例子：

```
>>> def print_kwargs(**kwargs):
...    print(kwargs)
...
>>> print_kwargs(a = 'lee',b = 'sir',c = 'man')
{'a': 'lee', 'b': 'sir', 'c': 'man'}
```

由例子可以看出，在函数内部，kwargs 是一个字典。

看到这的时候，可能聪明的你会想，参数不是具有不确定型吗？如何知道参数到底会用什么样的方式传值？其实这个很好办，把 *args 和 **kwargs 综合起来就好了啊，请看下面的操作：

```
>>> def print_all(x,y,*args,**kwargs):
...    print(x)
...    print(y)
...    print(args)
...    print(kwargs)
...
>>> print_all('lee',1234)
lee
1234
()
{}
>>> print_all('lee',1,2,3,4,5)
lee
1
(2, 3, 4, 5)
{}
>>> print_all('lee',1,2,3,4,5,like = 'python')
lee
1
(2, 3, 4, 5)
{'like': 'python'}
```

如此这般，我们就可以应对各种各样奇葩无聊的参数请求了。当然在这还是要说的是，这里的关键字参数命名不一定要是 kwargs，但这个是通常做法。

### 48、为什么函数名字可以当做参数用？

Python 中一切皆对象，函数名是函数在内存中的空间，也是一个对象。

### 49、Python 中 pass 语句的作用是什么？

在编写代码时只写框架思路，具体实现还未编写就可以用 pass 进行占位，使程序不报错，不会进行任何操作。

### 50、面向对象中super的作用？

super() 函数是用于调用父类(超类)的一个方法。

super 是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。

MRO 就是类的方法解析顺序表, 其实也就是继承父类方法时的顺序表。

作用：

- 根据 mro 的顺序执行方法
- 主动执行 Base 类的方法

### 51、是否使用过functools中的函数？其作用是什么？

Python的functools模块用以为可调用对象（callable objects）定义高阶函数或操作。

简单地说，就是基于已有的函数定义新的函数。

所谓高阶函数，就是以函数作为输入参数，返回也是函数。

### 52、json序列化时，默认遇到中文会转换成unicode，如果想要保留中文怎么办？

```
import json

a = json.dumps({"ddf": "你好"}, ensure_ascii=False)
print(a)
# {"ddf": "你好"}
```

### 53、什么是断言？应用场景？

assert断言——声明其布尔值必须为真判定，发生异常则为假。

```
info = {}
info['name'] = 'egon'
info['age'] = 18

# 用assert取代上述代码：
assert ('name' in info) and ('age' in info)
```

设置一个断言目的就是要求必须实现某个条件。

### 54、有用过with statement吗？它的好处是什么？

with语句的作用是通过某种方式简化异常处理，它是所谓的上下文管理器的一种

用法举例如下：

```
 with open('output.txt', 'w') as f:
        f.write('Hi there!')
```

当你要成对执行两个相关的操作的时候，这样就很方便，以上便是经典例子，with语句会在嵌套的代码执行之后，自动关闭文件。

这种做法的还有另一个优势就是，无论嵌套的代码是以何种方式结束的，它都关闭文件。

如果在嵌套的代码中发生异常，它能够在外部exception handler catch异常前关闭文件。

如果嵌套代码有return/continue/break语句，它同样能够关闭文件。

### 55、简述 Python 在异常处理中，else 和 finally 的作用分别是什么？

如果一个 Try - exception 中，没有发生异常，即 exception 没有执行，那么将会执行 else 语句的内容。反之，如果触发了 Try - exception（异常在 exception 中被定义），那么将会执行exception
中的内容，而不执行 else 中的内容。

如果 try 中的异常没有在 exception 中被指出，那么系统将会抛出 Traceback(默认错误代码）,并且终止程序，接下来的所有代码都不会被执行，但如果有 Finally 关键字，则会在程序抛出 Traceback 之前（程序最后一口气的时候），执行 finally 中的语句。这个方法在某些必须要结束的操作中颇为有用，如释放文件句柄，或释放内存空间等。

### 56、map 函数和 reduce 函数？

(1) 从参数方面来讲：

map()包含两个参数，第一个参数是一个函数，第二个是序列（列表 或元组）。其中，函数（即 map 的第一个参数位置的函数）可以接收一个或多个参数。

reduce()第一个参数是函数，第二个是序列（列表或元组）。但是，其函数必须接收两个参数。

(2) 从对传进去的数值作用来讲：

map()是将传入的函数依次作用到序列的每个元素，每个元素都是独自被函数“作用”一次 。

reduce()是将传人的函数作用在序列的第一个元素得到结果后，把这个结果继续与下一个元素作用（累积计算）。

**补充 Python 特殊函数**

**lambda 函数**

lambda 是一个可以只用一行就能解决问题的函数，让我们先看下面的例子：

```
>>> def add(x):
...     x += 1
...     return x
...
>>> numbers = range(5)
>>> list(numbers)
[0, 1, 2, 3, 4]
>>> new_numbers = []
>>> for i in numbers:
...     new_numbers.append(add(i))
...
>>> new_numbers
[1, 2, 3, 4, 5]
```

在上面的这个例子中，函数 add() 充当了一个中间角色，当然上面的例子也可以如下实现：

```
>>> new_numbers = [i+1 for i in numbers]
>>> new_numbers
[1, 2, 3, 4, 5]
```

首先我要说，上面的列表解析式其实是很好用的，但是我偏偏要用 lambda 这个函数代替 add(x) ：

```
>>> lamb = lambda x: x+1
>>> new_numbers = []
>>> for i in numbers:
...     new_numbers.append(lamb(i))
...
>>> new_numbers
[1, 2, 3, 4, 5]
```

在这里的 lamb 就相当于 add(x) ，lamb = lambda x : x+1 就相当于 add(x) 里的代码块。下面再写几个应用 lambda 的小例子：

```
>>> lamb = lambda x,y : x + y
>>> lamb(1,2)
3
>>> lamb1 = lambda x : x ** 2
>>> lamb1(5)
25
```

由上面的例子我们可以总结一下 lambda 函数的具体使用方法：lambda 后面直接跟变量，变脸后面是冒号，冒号后面是表达式，表达式的计算结果就是本函数的返回值。

在这里有一点需要提醒的是，虽然 lambda 函数可以接收任意多的参数并且返回单个表达式的值，但是 lambda 函数不能包含命令且包含的表达式不能超过一个。如果你需要更多复杂的东西，你应该去定义一个函数。

lambda 作为一个只有一行的函数，在你具体的编程实践中可以选择使用，虽然在性能上没什么提升，但是看着舒服呀。

**map 函数**

我们在上面讲 lambda 的时候用的例子，其实 map 也可以实现，请看下面的操作：

```
>>> numbers = [0,1,2,3,4]
>>> map(add,numbers)
[1, 2, 3, 4, 5]
>>> map(lambda x: x + 1,numbers)
[1, 2, 3, 4, 5]
```

map 是 Python 的一个内置函数，它的基本格式是：map(func, seq)。

func 是一个函数对象，seq 是一个序列对象，在执行的时候，seq 中的每个元素按照从左到右的顺序依次被取出来，塞到 func 函数里面，并将 func 的返回值依次存到一个列表里。

对于 map 要主要理解以下几个点就好了：

1.对可迭代的对象中的每一个元素，依次使用 fun 的方法（其实本质上就是一个 for 循环）。

2.将所有的结果返回一个 map 对象，这个对象是个迭代器。

我们接下来做一个简单的小题目：将两个列表中的对应项加起来，把结果返回在一个列表里，我们用 map 来做，如果你做完了，请往下看：

```
>>> list1 = [1,2,3,4]
>>> list2 = [5,6,7,8]
>>> list(map(lambda x,y: x + y,list1,list2))
[6, 8, 10, 12]
```

你看上面，是不是很简单？其实这个还看不出 map 的方便来，因为用 for 同样也不麻烦，要是你有这样的想法的话，那么请看下面：

```
>>> list1 = [1,2,3,4]
>>> list2 = [5,6,7,8]
>>> list3 = [9,10,11,12]
>>> list(map(lambda x,y,z : x + y + z,list1,list2,list3))
[15, 18, 21, 24]
```

你看三个呢？是不是用 for 的话就稍显麻烦了？那么我们在想如果是 四个，五个乃至更多呢？这就显示出 map 的简洁优雅了，并且 map 还不和 lambda 一样对性能没有什么提高，map 在性能上的优势也是杠杠的。

**filter 函数**

filter 翻译过来的意思是 “过滤器”，在 Python 中，它也确实是起到的是过滤器的作用。这个解释起来略微麻烦，还是直接上代码的好，在代码中体会用法是我在所有的文章里一直在体现的：

```
>>> numbers = range(-4,4)
>>> list(filter(lambda x: x > 0,numbers))
[1, 2, 3]
```

上面的例子其实和下面的代码是等价的：

```
>>> [x for x in numbers if x > 0]
[1, 2, 3]
```

然后我们再来写一个例子体会一下：

```
>>> list(filter(lambda x: x != 'o','Rocky0429'))
['R', 'c', 'k', 'y', '0', '4', '2', '9']
```

**reduce 函数**

我在之前的文章中很多次都说过，我的代码都是用 Python3 版本的。在 Python3 中，reduce 函数被放到 functools 模块里，在 Python2 中还是在全局命名空间。

同样我先用一个例子来跑一下，我们来看看怎么用：

```
>>> reduce(lambda x,y: x+y,[1,2,3,4])
10
```

reduce 函数的第一个参数是一个函数，第二个参数是序列类型的对象，将函数按照从左到右的顺序作用在序列上。如果你还不理解的话，我们下面可以对比一下它和 map 的区别：

```
>>> list1 = [1,2,3,4]
>>> list2 = [5,6,7,8]
>>> list(map(lambda x,y: x + y,list1,list2))
[6, 8, 10, 12]
```

对比上面的两个例子，就知道两者的区别，map 相当于是上下运算的，而 reduce 是从左到右逐个元素进行运算。

### 57、递归函数停止的条件？

递归的终止条件一般定义在递归函数内部，在递归调用前要做一个条件判断，根据判断的结果选择是继续调用自身，还是 return;返回终止递归。

终止的条件：

(1) 判断递归的次数是否达到某一限定值

(2) 判断运算的结果是否达到某个范围等，根据设计的目的来选择

### 58、回调函数，如何通信的？

回调函数是把函数的指针(地址)作为参数传递给另一个函数，将整个函数当作一个对象，赋值给调用的函数。

### 59、  __setattr__，__getattr__，__delattr__函数使用详解？

1.**setattr**(self,name,value)：如果想要给 name 赋值的话，就需要调用这个方法。

2.**getattr**(self,name)：如果 name 被访问且它又不存在，那么这个方法将被调用。

3.**delattr**(self,name)：如果要删除 name 的话，这个方法就要被调用了。

下面我们用例子来演示一下：

```
>>> class Sample:
...    def __getattr__(self,name):
...            print('hello getattr')
...    def __setattr__(self,name,value):
...            print('hello setattr')
...            self.__dict__[name] = value
...
```

上面的例子中类 Sample 只有两个方法，下面让我们实例化一下：

```
>>> s = Sample()
>>> s.x
hello getattr
```

s.x 这个实例属性本来是不存在的，但是由于类中有了 **getattr**(self,name) 方法，当发现属性 x 不存在于对象的 **dict** 中时，就调用了 **getattr**，也就是所谓的「拦截成员」。

```
>>> s.x = 7
hello setattr
```

同理，给对象的属性赋值的时候，调用了 **setattr**(self,name,value) 方法，这个方法中有 self.**dict**[name] = value，通过这个就将属性和数据存到了对象 **dict** 中。如果再调用这个属性的话，会成为下面这样：

```
>>> s.x
7
```

出现这种结果的原因是它已经存在于对象的 **dict** 中了。

看了上面的两个，你是不是觉得上面的方法很有魔力呢？这就是「黑魔法」，但是它具体有什么应用呢？我们接着来看：

```
class Rectangle:
   """
   the width and length of Rectangle
   """

   def __init__(self):
       self.width = 0
       self.length = 0

   def setSize(self,size):
       self.width, self.length = size

   def getSize(self):
       return self.width, self.length

if __name__ == "__main__":
   r = Rectangle()
   r.width = 3
   r.length = 4
   print(r.getSize())
   print(r.setSize((30,40)))
   print(r.width)
   print(r.length)
```

上面是我根据一个很有名的例子改编的，你可以先想一下结果，想完以后可以接着往下看：

```
(3, 4)
30
40
```

这段代码运行的结果如上面所示，作为一个强迫证的码农，对于这种可以改进的程序当然不能容忍。我们在上面介绍的特殊方法，我们一定要学以致用，虽然重新写的不一定比原来的好，但我们还是要尝试去用：

```
class NewRectangle:
   """
   the width and length of Rectangle
   """

   def __init__(self):
       self.width = 0
       self.length = 0

   def __setattr__(self, name, value):
       if name == 'size':
           self.width, self.length = value
       else:
           self.__dict__[name] = value

   def __getattr__(self, name):
       if name == 'size':
           return self.width, self.length
       else:
           return AttributeError

if __name__ == "__main__":
   r = NewRectangle()
   r.width = 3
   r.length = 4
   print(r.size)
   r.size = 30,40
   print(r.width)
   print(r.length)
```

我们来看一下运行的结果：

```
(3, 4)
30
40
```

我们可以看到，除了类的写法变了以外，调用的方式没有变，结果也没有变。

### 60、请描述抽象类和接口类的区别和联系？

**(1) 抽象类**

规定了一系列的方法，并规定了必须由继承类实现的方法。由于有抽象方法的存在，所以抽象类不能实例化。可以将抽象类理解为毛坯房，门窗、墙面的样式由你自己来定，所以抽象类与作为基类的普通类的区别在于约束性更强。

**(2) 接口类**

与抽象类很相似，表现在接口中定义的方法，必须由引用类实现，但他与抽象类的根本区别在于用途：与不同个体间沟通的规则（方法），你要进宿舍需要有钥匙，这个钥匙就是你与宿舍的接口，你的同室也有这个接口，所以他也能进入宿舍，你用手机通话，那么手机就是你与他人交流的接口。

**(3) 区别和关联**

- 接口是抽象类的变体，接口中所有的方法都是抽象的。而抽象类中可以有非抽象方法。抽象类是声明方法的存在而不去实现它的类。
- 接口可以继承，抽象类不行。
- 接口定义方法，没有实现的代码，而抽象类可以实现部分方法。
- 接口中基本数据类型为 static 而抽类象不是。
- 接口可以继承，抽象类不行。
- 可以在一个类中同时实现多个接口。
- 接口的使用方式通过 implements 关键字进行，抽象类则是通过继承 extends 关键字进行。

### 61、请描述方法重载与方法重写？

**(1)方法重载**

是在一个类里面，方法名字相同，而参数不同。返回类型呢？可以相同也可以不同。重载是让类以统一的方式处理不同类型数据的一种手段。

**(2) 方法重写**

子类不想原封不动地继承父类的方法，而是想作一定的修改，这就需要采用方法的重写。方法重写又称方法覆盖。

### 62、什么是 lambda 函数？ 有什么好处？

lambda 函数是一个可以接收任意多个参数(包括可选参数)并且返回单个表达式值的函数

1、lambda 函数比较轻便，即用即仍，很适合需要完成一项功能，但是此功能只在此一处使用，连名字都很随意的情况下

2、匿名函数，一般用来给 filter， map 这样的函数式编程服务

3、作为回调函数，传递给某些应用，比如消息处理

### 63、单例模式的应用场景有哪些？

单例模式应用的场景一般发现在以下条件下：

（1）资源共享的情况下，避免由于资源操作时导致的性能或损耗等。如日志文件，应用配置。

（2）控制资源的情况下，方便资源之间的互相通信。如线程池等。 1.网站的计数器 2.应用配置 3.多线程池 4.数据库配置，数据库连接池 5.应用程序的日志应用....

**补充**

**01.单例设计模式**

「单例设计模式」估计对很多人来说都是一个陌生的概念，其实它就环绕在你的身边。比如我们每天必用的听歌软件，同一时间只能播放一首歌曲，所以对于一个听歌的软件来说，负责音乐播放的对象只有一个；再比如打印机也是同样的道理，同一时间打印机也只能打印一份文件，同理负责打印的对象也只有一个。

结合说的听歌软件和打印机都只有唯一的一个对象，就很好理解「单例设计模式」。

> 单例设计模式确保一个类只有一个实例，并提供一个全局访问点。

「单例」就是单个实例，我们在定义完一个类的之后，一般使用「类名()」的方式创建一个对象，而单例设计模式解决的问题就是无论执行多少遍「类名()」，返回的对象内存地址永远是相同的。

**02.\**new\** 方法**

当我们使用「类名()」创建对象的时候，Python 解释器会帮我们做两件事情：第一件是为对象在内存分配空间，第二件是为对象进行初始化。初始化（**init**）我们已经学过了，那「分配空间」是哪一个方法呢？就是我们这一小节要介绍的 **new** 方法。

那这个 **new** 方法和单例设计模式有什么关系呢？单例设计模式返回的对象内存地址永远是相同的，这就意味着在内存中这个类的对象只能是唯一的一份，为达到这个效果，我们就要了解一下为对象分配空间的 **new** 方法。

明确了这个目的以后，接下来让我们看一下 **new** 方法。**new** 方法在内部其实做了两件时期：第一件事是为「对象分配空间」，第二件事是「把对象的引用返回给 Python 解释器」。当 Python 的解释器拿到了对象的引用之后，就会把对象的引用传递给 **init** 的第一个参数 self，**init** 拿到对象的引用之后，就可以在方法的内部，针对对象来定义实例属性。

这就是 **new** 方法和 **init** 方法的分工。

总结一下就是：之所以要学习 **new** 方法，就是因为需要对分配空间的方法进行改造，改造的目的就是为了当使用「类名()」创建对象的时候，无论执行多少次，在内存中永远只会创造出一个对象的实例，这样就可以达到单例设计模式的目的。

**03.重写 \**new\** 方法**

在这里我用一个 **new** 方法的重写来做一个演练：首先定义一个打印机的类，然后在类里重写一下 **new** 方法。通过对这个方法的重写来强化一下 **new** 方法要做的两件事情：在内存中分配内存空间 & 返回对象的引用。同时验证一下，当我们使用「类名()」创建对象的时候，Python 解释器会自动帮我们调用 **new** 方法。

首先我们先定义一个打印机类 Printer，并创建一个实例：

```
class Printer():
    def __init__(self):
        print("打印机初始化")
# 创建打印机对象
printer = Printer()
```

接下来就是重写 **new** 方法，在此之前，先说一下注意事项，只要⚠️了这几点，重写 **new** 就没什么难度：

重写 **new** 方法一定要返回对象的引用，否则 Python 的解释器得不到分配了空间的对象引用，就不会调用对象的初始化方法；

**new** 是一个静态方法，在调用时需要主动传递 cls 参数。

```
# 重写 __new__ 方法
class Printer():
    def __new__(cls, *args, **kwargs):
        # 可以接收三个参数
        # 三个参数从左到右依次是 class，多值元组参数，多值的字典参数
        print("this is rewrite new")
        instance = super().__new__(cls)
        return instance
    def __init__(self):
        print("打印机初始化")
# 创建打印机对象
player = Printer()
print(player)
```

上述代码对 **new** 方法进行了重写，我们先来看一下输出结果：

```
this is rewrite new
打印机初始化
<__main__.Printer object at 0x10fcd2ba8>
```

上述的结果打印出了 **new** 方法和 **init** 方法里的内容，同时还打印了类的内存地址，顺序正好是我们在之前说过的。**new** 方法里的三行代码正好做了在本小节开头所说的三件事：

- print(this is rewrite new)：证明了创建对象时，**new** 方***被自动调用；
- instance = super().**new**(cls)：为对象分配内存空间（因为 **new** 本身就有为对象分配内存空间的能力，所以在这直接调用父类的方法即可）；
- return instance：返回对象的引用。

**04.设计单例模式**

说了这么多，接下来就让我们用单例模式来设计一个单例类。乍一看单例类看起来比一般的类更唬人，但其实就是差别在一点：单例类在创建对象的时候，无论我们调用多少次创建对象的方法，得到的结果都是内存中唯一的对象。

可能到这有人会有疑惑：怎么知道用这个类创建出来的对象是同一个对象呢？其实非常的简单，我们只需要多调用几次创建对象的方法，然后输出一下方法的返回结果，如果内存地址是相同的，说明多次调用方法返回的结果，本质上还是同一个对象。

```
class Printer():
    pass

printer1 = Printer()
print(printer1)
printer2 = Printer()
print(printer2)
```

上面是一个一般类的多次调用，打印的结果如下所示：

```
<__main__.Printer object at 0x10a940780>
<__main__.Printer object at 0x10a94d3c8>
```

可以看出，一般类中多次调用的内存地址不同（即 printer1 和 printer2 是两个完全不同的对象），而单例设计模式设计的单例类 Printer()，要求是无论调用多少次创建对象的方法，控制台打印出来的内存地址都是相同的。

那么我们该怎么实现呢？其实很简单，就是多加一个「类属性」，用这个类属性来记录「单例对象的引用」。

为什么要这样呢？其实我们一步一步的来想，当我们写完一个类，运行程序的时候，内存中其实是没有这个类创建的对象的，我们必须调用创建对象的方法，内存中才会有第一个对象。在重写 **new** 方法的时候，我们用 instance = super().**new**(cls) ，为对象分配内存空间，同时用 istance 类属性记录父类方法的返回结果，这就是第一个「对象在内存中的返回地址」。当我们再次调用创建对象的方法时，因为第一个对象已经存在了，我们直接把第一个对象的引用做一个返回，而不用再调用 super().**new**(cls) 分配空间这个方法，所以就不会在内存中为这个类的其它对象分配额外的内存空间，而只是把之前记录的第一个对象的引用做一个返回，这样就能做到无论调用多少次创建对象的方法，我们永远得到的是创建的第一个对象的引用。

这个就是使用单例设计模式解决在内存中只创建唯一一个实例的解决办法。下面我就根据上面所说的，来完成单例设计模式。

```
class Printer():
    instance = None
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
    return cls.instance
printer1 = Printer()
print(printer1)
printer2 = Printer()
print(printer2)
```

上述代码很简短，首先给类属性复制为 None，在 **new** 方法内部，如果 instance 为 None，证明第一个对象还没有创建，那么就为第一个对象分配内存空间，如果 instance 不为 None，直接把类属性中保存的第一个对象的引用直接返回，这样在外界无论调用多少次创建对象的方法，得到的对象的内存地址都是相同的。

下面我们运行一下程序，来看一下结果是不是能印证我们的说法：

```
<__main__.Printer object at 0x10f3223c8>
<__main__.Printer object at 0x10f3223c8>
```

上述输出的两个结果可以看出地址完全一样，这说明 printer1 和 printer2 本质上是相同的一个对象。

### 64、什么是闭包？

我们都知道在数学中有闭包的概念，但此处我要说的闭包是计算机编程语言中的概念，它被广泛的使用于函数式编程。

关于闭包的概念，官方的定义颇为严格，也很难理解，在《Python语言及其应用》一书中关于闭包的解释我觉得比较好 -- 闭包是一个可以由另一个函数动态生成的函数，并且可以改变和存储函数外创建的变量的值。乍一看，好像还是比较很难懂，下面我用一个简单的例子来解释一下：

```
>>> a = 1
>>> def fun():
...     print(a)
...
>>> fun()
1
>>> def fun1():
...     b = 1
...
>>> print(b)
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
NameError: name 'b' is not defined
```

毋庸置疑，第一段程序是可以运行成功的，a = 1 定义的变量在函数里可以被调用，但是反过来，第二段程序则出现了报错。

在函数 fun() 里可以直接使用外面的 a = 1，但是在函数 fun1() 外面不能使用它里面所定义的 b = 1，如果我们根据作用域的关系来解释，是没有什么异议的，但是如果在某种特殊情况下，我们必须要在函数外面使用函数里面的变量，该怎么办呢？

我们先来看下面的例子：

```
>>> def fun():
...    a = 1
...    def fun1():
...            return a
...    return fun1
...
>>> f = fun()
>>> print(f())
1
```

如果你看过昨天的文章，你一定觉得的很眼熟，上述的本质就是我们昨天所讲的嵌套函数。

在函数 fun() 里面，有 a = 1 和 函数 fun1() ,它们两个都在函数 fun() 的环境里面，但是它们两个是互不干扰的，所以 a 相对于 fun1() 来说是自由变量，并且在函数 fun1() 中应用了这个自由变量 -- 这个 fun1() 就是我们所定义的闭包。

闭包实际上就是一个函数，但是这个函数要具有 1.定义在另外一个函数里面(嵌套函数)；2.引用其所在环境的自由变量。

上述例子通过闭包在 fun() 执行完毕时，a = 1依然可以在 f() 中，即 fun1() 函数中存在，并没有被收回，所以 print(f()) 才得到了结果。

当我们在某些时候需要对事务做更高层次的抽象，用闭包会相当舒服。比如我们要写一个二元一次函数，如果不使用闭包的话相信你可以轻而易举的写出来，下面让我们来用闭包的方式完成这个一元二次方程：

```
>>> def fun(a,b,c):
...    def para(x):
...            return a*x**2 + b*x + c
...    return para
...
>>> f = fun(1,2,3)
>>> print(f(2))
11
```

上面的函数中，f = fun(1,2,3) 定义了一个一元二次函数的函数对象，x^2 + 2x + 3，如果要计算 x = 2 ，该一元二次函数的值，只需要计算 f(2) 即可，这种写法是不是看起来更简洁一些。

### 65、什么是装饰器？

「装饰器」作为 Python 高级语言特性中的重要部分，是修改函数的一种超级便捷的方式，适当使用能够有效提高代码的可读性和可维护性，非常的便利灵活。

「装饰器」本质上就是一个函数，这个函数的特点是可以接受其它的函数当作它的参数，并将其替换成一个新的函数（即返回给另一个函数）。

可能现在这么看的话有点懵，为了深入理解「装饰器」的原理，我们首先先要搞明白「什么是函数对象」，「什么是嵌套函数」，「什么是闭包」。关于这三个问题我在很久以前的文章中已经写过了，你只需要点击下面的链接去看就好了，这也是面试中常问的知识哦：

[零基础学习 Python 之函数对象](https://gw-c.nowcoder.com/api/sparta/jump/link?link=https%3A%2F%2Fmp.weixin.qq.com%2Fs%3F__biz%3DMzUxMTc3NTI4Ng%3D%3D%26mid%3D2247483765%26idx%3D1%26sn%3D93052f76310ef83795fe93806a32a8ea%26chksm%3Df96fc62cce184f3a13c7da5fa9e7c25c43ed403d3fe6a2cfcebf3b0b6b9b7943a8d6c6d71116%26scene%3D21%23wechat_redirect)

[零基础学习 Python 之嵌套函数](https://gw-c.nowcoder.com/api/sparta/jump/link?link=https%3A%2F%2Fmp.weixin.qq.com%2Fs%3F__biz%3DMzUxMTc3NTI4Ng%3D%3D%26mid%3D2247483774%26idx%3D1%26sn%3D01fe4fe3255b6613ac705b8b0c7b2ec3%26chksm%3Df96fc627ce184f31d005618ffb08dead17221f9a386f55ab85a853a9898769998a482ce879d8%26token%3D456179487%26lang%3Dzh_CN%26scene%3D21%23wechat_redirect)

[零基础学习 Python 之闭包](https://gw-c.nowcoder.com/api/sparta/jump/link?link=https%3A%2F%2Fmp.weixin.qq.com%2Fs%3F__biz%3DMzUxMTc3NTI4Ng%3D%3D%26mid%3D2247483777%26idx%3D1%26sn%3D2cdf9cac826e30815ca1f912c59163b2%26chksm%3Df96fc6d8ce184fcefec7a345ce46825375ea329349d72f17053fb784de433bc052661f857e62%26token%3D456179487%26lang%3Dzh_CN%26scene%3D21%23wechat_redirect)

**装饰器**

搞明白上面的三个问题，其实简单点来说就是告诉你：函数可以赋值给变量，函数可嵌套，函数对象可以作为另一个函数的参数。

首先我们来看一个例子，在这个例子中我们用到了前面列出来的所有知识：

```
def first(fun):
    def second():
        print('start')
        fun()
        print('end')
        print fun.__name__
    return second

def man():
    print('i am a man()')

f = first(man)
f()
```

上述代码的执行结果如下所示：

```
start
i am a man()
end
man
```

上面的程序中，这个就是 first 函数接收了 man 函数作为参数，并将 man 函数以一个新的函数进行替换。看到这你有没有发现，这个和我在文章刚开始时所说的「装饰器」的描述是一样的。既然这样的话，那我们就把上述的代码改造成符合 Python 装饰器的定义和用法的样子，具体如下所示：

```
def first(func):
    def second():
        print('start')
        func()
        print('end')
        print (func.__name__)
    return second

@first
def man():
    print('i am a man()')

man()
```

上面这段代码和之前的代码的作用一模一样。区别在于之前的代码直接“明目张胆”的使用 first 函数去封装 man 函数，而上面这个是用了「语法糖」来封装 man 函数。至于什么是语法糖，不用细去追究，你就知道是类似「@first」这种形式的东西就好了。

在上述代码中「@frist」在 man 函数的上面，表示对 man 函数使用 first 装饰器。「@」 是装饰器的语法，「first」是装饰器的名称。

下面我们再来看一个复杂点的例子，用这个例子我们来更好的理解一下「装饰器」的使用以及它作为 Python 语言高级特性被人津津乐道的部分：

```
def check_admin(username):
    if username != 'admin':
        raise Exception('This user do not have permission')

class Stack:
    def __init__(self):
        self.item = []

    def push(self,username,item):
        check_admin(username=username)
        self.item.append(item)

    def pop(self,username):
        check_admin(username=username)
        if not self.item:
            raise Exception('NO elem in stack')
        return self.item.pop()
```

上述实现了一个特殊的栈，特殊在多了检查当前用户是否为 admin 这步判断，如果当前用户不是 admin，则抛出异常。上面的代码中将检查当前用户的身份写成了一个独立的函数 check_admin，在 push 和 pop 中只需要调用这个函数即可。这种方式增强了代码的可读性，减少了代码冗余，希望大家在编程的时候可以具有这种意识。

下面我们来看看上述代码用装饰器来写成的效果：

```
def check_admin(func):
    def wrapper(*args, **kwargs):
        if kwargs.get('username') != 'admin':
            raise Exception('This user do not have permission')
        return func(*args, **kwargs)
    return wrapper

class Stack:
    def __init__(self):
        self.item = []

    @check_admin
    def push(self,username,item):
        self.item.append(item)

    @check_admin
    def pop(self,username):
        if not self.item:
            raise Exception('NO elem in stack')
        return self.item.pop()
```

PS：可能很多人对 *args 和 **kwargs 不太熟悉，详情请戳下面的链接：

[Python 拓展之 *args & **kwargs](https://gw-c.nowcoder.com/api/sparta/jump/link?link=https%3A%2F%2Fmp.weixin.qq.com%2Fs%3F__biz%3DMzUxMTc3NTI4Ng%3D%3D%26mid%3D2247483762%26idx%3D1%26sn%3D88887b10a1083b2bcf7f6a2fac13037d%26chksm%3Df96fc62bce184f3d7f68fc0ff3d9c1841a078c257e35f01f8afa25aa46f7d3e4d268d592c4f0%26token%3D456179487%26lang%3Dzh_CN%26scene%3D21%23wechat_redirect)

对比一下使用「装饰器」和不使用装饰器的两种写法，乍一看，好像使用「装饰器」以后代码的行数更多了，但是你有没有发现代码看起来好像更容易理解了一些。在没有装饰器的时候，我们先看到的是 check_admin 这个函数，我们得先去想这个函数是干嘛的，然后看到的才是对栈的操作；而使用装饰器的时候，我们上来看到的就是对栈的操作语句，至于 check_admin 完全不会干扰到我们对当前函数的理解，所以使用了装饰器可读性更好了一些。

就和我在之前的文章中所讲的「生成器」那样，虽然 Python 的高级语言特性好用，但也不能乱用。装饰器的语法复杂，通过我们在上面缩写的装饰器就可以看出，它写完以后是很难调试的，并且使用「装饰器」的程序的速度会比不使用装饰器的程序更慢，所以还是要具体场景具体看待。

### 66、函数装饰器有什么作用？

装饰器本质上是一个 Python 函数，它可以在让其他函数在不需要做任何代码的变动的前提下增加额外的功能。装饰器的返回值也是一个函数的对象，它经常用于有切面需求的场景。 比如：插入日志、性能测试、事务处理、缓存、权限的校验等场景 有了装饰器就可以抽离出大量的与函数功能本身无关的雷同代码并发并继续使用。

### 67、生成器、迭代器的区别

迭代器是一个更抽象的概念，任何对象，如果它的类有 next 方法和 iter 方法返回自己本身，对于 string、list、dict、tuple 等这类容器对象，使用 for 循环遍历是很方便的。在后台 for 语句对容器对象调用 iter()函数，iter()是 python 的内置函数。iter()会返回一个定义了 next()方法的迭代器对象，它在容器中逐个访问容器内元素，next()也是 python 的内置函数。在没有后续元素时，next()会抛出一个 StopIteration 异常。

生成器（Generator）是创建迭代器的简单而强大的工具。它们写起来就像是正规的函数，只是在需要返回数据的时候使用 yield 语句。每次 next()被调用时，生成器会返回它脱离的位置（它记忆语句最后一次执行的位置和所有的数据值）

**区别**：生成器能做到迭代器能做的所有事,而且因为自动创建了 iter()和 next()方法,生成器显得特别简洁,而且生成器也是高效的，使用生成器表达式取代列表解析可以同时节省内存。除了创建和保存程序状态的自动方法,当发生器终结时,还会自动抛出 StopIteration 异常。

### 68、多线程交互，访问数据，如果访问到了就不访问了，怎么避免重读？

创建一个已访问数据列表，用于存储已经访问过的数据，并加上互斥锁，在多线程访问数据的时候先查看数据是否已经在已访问的列表中，若已存在就直接跳过。

### 69、Python 中 yield 的用法？

yield 就是保存当前程序执行状态。你用 for 循环的时候，每次取一个元素的时候就会计算一次。用yield 的函数叫 generator，和 iterator 一样，它的好处是不用一次计算所有元素，而是用一次算一次，可以节省很多空间。generator每次计算需要上一次计算结果，所以用 yield，否则一 return，上次计算结果就没了。

**补充**

在 Python 中，定义生成器必须要使用 yield 这个关键词，yield 翻译成中文有「生产」这方面的意思。在 Python 中，它作为一个关键词，是生成器的标志。接下来我们来看一个例子：

```
>>> def f():
...    yield 0
...    yield 1
...    yield 2
...
>>> f
<function f at 0x00000000004EC1E0>
```

上面是写了一个很简单的 f 函数，代码块是 3 个 yield 发起的语句，下面让我们来看看如何使用它：

```
>>> fa = f()
>>> fa
<generator object f at 0x0000000001DF1660>
>>> type(fa)
<class 'generator'>
```

上述操作可以看出，我们调用函数得到了一个生成器（generator）对象。

```
>>> dir(fa)
['__class__', '__del__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__lt__', '__name__', '__ne__',
'__new__', '__next__', '__qualname__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'close', 'gi_code', 'gi_frame', 'gi_running', 'gi_yieldfrom', 'send', 'throw']
```

在上面我们看到了 **iter**() 和 **next**()，虽然我们在函数体内没有显示的写 **iter**() 和 **next**()，仅仅是写了 yield，但它就已经是「迭代器」了。既然如此，那我们就可以进行如下操作：

```
>>> fa = f()
>>> fa.__next__()
0
>>> fa.__next__()
1
>>> fa.__next__()
2
>>> fa.__next__()
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
StopIteration
```

从上面的简单操作可以看出：含有 yield 关键词的函数 f() 是一个生成器对象，这个生成器对象也是迭代器。所以就有了这样的定义：把含有 yield 语句的函数称为生成器，生成器是一种用普通函数语法定义的迭代器。

通过上面的例子可以看出，这个生成器（即迭代器）在定义的过程中并没有昨天讲的迭代器那样写 **iter**()，而是只用了 yield 语句，之后一个普普通通的函数就神奇的成了生成器，同样也具备了迭代器的特性。

yield 语句的作用，就是在调用的时候返回相应的值。下面我来逐行的解释一下上面例子的运行过程：

1.fa = f()：fa 引用生成器对象。

2.fa.**next**()：生成器开始执行，遇到了第一个 yield，然后返回后面的 0，并且挂起（即暂停执行）。

3.fa.**next**()：从上次暂停的位置开始，继续向下执行，遇到第二个 yield，返回后面的值 1，再挂起。

4.fa.**next**()：重复上面的操作。

5.fa.**next**()：从上次暂停的位置开始，继续向下执行，但是后面已经没有 yield 了，所以 **next**() 发生异常。

### 70、谈下 python 的 GIL

GIL 是python的全局解释器锁，同一进程中假如有多个线程运行，一个线程在运行python程序的时候会霸占python解释器（加了一把锁即GIL），使该进程内的其他线程无法运行，等该线程运行完后其他线程才能运行。如果线程运行过程中遇到耗时操作，则解释器锁解开，使其他线程运行。所以在多线程中，线程的运行仍是有先后顺序的，并不是同时进行。

多进程中因为每个进程都能被系统分配资源，相当于每个进程有了一个python解释器，所以多进程可以实现多个进程的同时运行，缺点是进程系统资源开销大

### 71、Python 中的可变对象和不可变对象？

不可变对象，该对象所指向的内存中的值不能被改变。当改变某个变量时候，由于其所指的值不能被改变，相当于把原来的值复制一份后再改变，这会开辟一个新的地址，变量再指向这个新的地址。

可变对象，该对象所指向的内存中的值可以被改变。变量（准确的说是引用）改变后，实际上是其所指的值直接发生改变，并没有发生复制行为，也没有开辟新的出地址，通俗点说就是原地改变。

Python 中，数值类型（int 和 float）、字符串 str、元组 tuple 都是不可变类型。而列表 list、字典 dict、集合 set 是可变类型。

### 72、一句话解释什么样的语言能够用装饰器？

函数可以作为参数传递的语言，可以使用装饰器

### 73、Python 中 is 和==的区别？

is 判断的是 a 对象是否就是 b 对象，是通过 id 来判断的。

==判断的是 a 对象的值是否和 b 对象的值相等，是通过 value 来判断的。

### 74、谈谈你对面向对象的理解？

面向对象是相对于面向过程而言的。

面向过程语言是一种基于功能分析的、以算法为中心的程序设计方法

面向对象是一种基于结构分析的、以数据为中心的程序设计思想。在面向对象语言中有一个有很重要东西，叫做类。面向对象有三大特性：封装、继承、多态。

### 75、Python 里 match 与 search 的区别？

match()函数只检测 RE 是不是在 string 的开始位置匹配

search()会扫描整个 string 查找匹配

也就是说 match()只有在 0 位置匹配成功的话才有返回，如果不是开始位置匹配成功的话，match()就返回 none。

### 76、用 Python 匹配 HTML g tag 的时候，<.*> 和 <.*?> 有什么区别？

<.*>是贪婪匹配，会从第一个“<”开始匹配，直到最后一个“>”中间所有的字符都会匹配到，中间可能会包含“<>”。

<.*?>是非贪婪匹配，从第一个“<”开始往后，遇到第一个“>”结束匹配，这中间的字符串都会匹配到，但是
不会有“<>”。

### 77、Python 中的进程与线程的使用场景？

多进程适合在 CPU 密集型操作(cpu 操作指令比较多，如位数多的浮点运算)。

多线程适合在 IO 密集型操作(读写数据操作较多的，比如爬虫)。

### 78、解释一下并行（parallel）和并发（concurrency）的区别？

并行（parallel）是指同一时刻，两个或两个以上时间同时发生。

并发（parallel）是指同一时间间隔（同一段时间），两个或两个以上时间同时发生。

### 79、如果一个程序需要进行大量的 IO 操作，应当使用并行还是并发？

并发

### 80、如果程序需要进行大量的逻辑运算操作，应当使用并行还是并发？

并行

### 81、在 Python 中可以实现并发的库有哪些？

- 线程
- 进程
- 协程
- threading

### 82、Python 如何处理上传文件？

Python 中使用 GET 方法实现上传文件，下面就是用 Get 上传文件的例子，client 用来发 Get 请求，server 用来收请求。

**请求端代码**

```
import requests #需要安装 requests

with open('test.txt', 'rb') as f:
requests.get('http://服务器 IP 地址:端口', data=f)
```

**服务端代码**

```
var http = require('http');
var fs = require('fs');
var server = http.createServer(function(req, res){

    var recData = "";
    req.on('data', function(data){
    recData += data;
    })
    req.on('end', function(data){
    recData += data;
    fs.writeFile('recData.txt', recData, function(err){
    console.log('file received');
        })
    })
    res.end('hello');
    })
server.listen(端口);
```

### 83、请列举你使用过的 Python 代码检测工具？

- 移动应用自动化测试 Appium
- OpenStack 集成测试 Tempest
- 自动化测试框架 STAF
- 自动化测试平台 TestMaker
- JavaScript 内存泄露检测工具 Leak Finder
- Python 的 Web 应用验收测试 Splinter
- 即插即用设备调试工具 UPnP-Inspector

### 84、python 程序中文输出问题怎么解决？

**方法一**

用encode和decode，如：

```
import os.path
import xlrd,sys

Filename=’/home/tom/Desktop/1234.xls’
if not os.path.isfile(Filename):
    raise NameError,”%s is not a valid filename”%Filename

bk=xlrd.open_workbook(Filename)
shxrange=range(bk.nsheets)
print shxrange

for x in shxrange:
    p=bk.sheets()[x].name.encode(‘utf-8′)
    print p.decode(‘utf-8′)
```

**方法二**

在文件开头加上

```
reload(sys)

sys.setdefaultencoding(‘utf8′)
```

### 85、Python 如何 copy 一个文件？

shutil 模块有一个 copyfile 函数可以实现文件拷贝

### 86、如何用Python删除一个文件？

使用 os.remove(filename) 或者 os.unlink(filename)

### 87、如何用 Python 来发送邮件？

python实现发送和接收邮件功能主要用到poplib和smtplib模块。

poplib用于接收邮件，而smtplib负责发送邮件。

```
#! /usr/bin/env python
#coding=utf-8
import sys
import time
import poplib
import smtplib

# 邮件发送函数
def send_mail():
     try:
        handle = smtplib.SMTP('smtp.126.com',25)
        handle.login('XXXX@126.com','**********')
        msg = 'To: XXXX@qq.com\r\nFrom:XXXX@126.com\r\nSubject:hello\r\n'
        handle.sendmail('XXXX@126.com','XXXX@qq.com',msg)
        handle.close()
        return 1
    except:
        return 0

# 邮件接收函数
def accpet_mail():
    try:
        p=poplib.POP3('pop.126.com')
        p.user('pythontab@126.com')
        p.pass_('**********')
        ret = p.stat() #返回一个元组:(邮件数,邮件尺寸)
       #p.retr('邮件号码')方法返回一个元组:(状态信息,邮件,邮件尺寸)
    except poplib.error_proto,e:
        print "Login failed:",e
        sys.exit(1)

# 运行当前文件时，执行sendmail和accpet_mail函数
if __name__ == "__main__":
    send_mail()
    accpet_mail()
```

### 88、当退出 Python 时，是否释放全部内存？

不是。

循环引用其它对象或引用自全局命名空间的对象的模块，在Python退出时并非完全释放。另外，也不会释放C库保留的内存部分。

### 89、什么是猴子补丁？

在运行期间动态修改一个类或模块。

```
>>> class A:
    def func(self):
        print("Hi")
>>> def monkey(self):
print "Hi, monkey"
>>> m.A.func = monkey
>>> a = m.A()
>>> a.func()
```

运行结果为：

```
Hi, Monkey
```

### 90、Python 函数声明中单独的正斜杠（/）和星号（*）是什么意思

- `/` 规定了在其之前的参数都必须是 **位置参数**，而不能是 **关键字参数**；之后的不管，两种均可；
- `*` 规定了在其之后的参数都必须是 **关键字参数**，而不能是 **位置参数**；之前的不管，两种均可；

```python
def func(a, /, b, *, c):
    print(a, b, c)

func(1, 2, c=3)  # ok
func(1, b=2, c=3)  # ok
func(a=1, 2, 3)  # err
```

## Python操作

### 1. 列表推导式
```
new_list = [表达式 for 元素 in old_list if 可选条件]
例：
o1d_1ist = [1，2，3，4，5，6，7，8，9，10]
new_list = [x**2 for x in old_list if x%2==1]
结果：
[1, 9, 25, 49, 81]
```

## Python数据类型

可变：list、dict、set

不可变：number（int、float）、tuple、str、bool

## Python数据结构

时间复杂度，索引O(1)；切片O(1)；添加O(n)；删除O(n)

空间复杂度，列表O(n)；元组O(n)；字典O(n)；集合O(n)

数组

链表

栈

队列

哈希表

集合

堆：最大堆、最小堆

​		堆是一种完全二叉树，所有节点的值总是不大于或不小于其父节点的值。在一个堆中，根节点是最大（或最小）节点。如果根节点最小，称之为小顶堆（或小根堆），如果根节点最大，称之为大顶堆（或大根堆）。堆的左右孩子没有大小的顺序。

树：

​		真二叉树：所有节点的度都要么为0，要么为2

​		完全二叉树：叶子节点只会出现在最后2层，且最后一层的叶子节点都靠左对齐。（所有节点从上到下从左到右依次排列）

​		满二叉树：所有节点的度要么为0，要么为2，且所有的叶子节点都在最后一层。（除叶节点外每个节点都有左右两个子节点）

​		平衡二叉树：每个节点的左右子树高度差不超过1

​		搜索二叉树：每个节点的非空左子树的所有值都比它的值小，非空右子树的所有值都比它的值大

**生活中的树形结构：**文件夹的管理就是我们生活中最常见的树形结构

图

## 多线程与多进程

**进程是资源分配的最小单位，线程是CPU调度的最小单位**

做个简单的比喻：进程=火车，线程=车厢

- 线程在进程下行进（单纯的车厢无法运行）
- 一个进程可以包含多个线程（一辆火车可以有多个车厢）
- 不同进程间数据很难共享（一辆火车上的乘客很难换到另外一辆火车，比如站点换乘）
- 同一进程下不同线程间数据很易共享（A车厢换到B车厢很容易）
- 进程要比线程消耗更多的计算机资源（采用多列火车相比多个车厢更耗资源）
- 进程间不会相互影响，一个线程挂掉将导致整个进程挂掉（一列火车不会影响到另外一列火车，但是如果一列火车上中间的一节车厢着火了，将影响到所有车厢）
- 进程可以拓展到多机，进程最多适合多核（不同火车可以开在多个轨道上，同一火车的车厢不能在行进的不同的轨道上）
- 进程使用的内存地址可以上锁，即一个线程使用某些共享内存时，其他线程必须等它结束，才能使用这一块内存。（比如火车上的洗手间）－"互斥锁"
- 进程使用的内存地址可以限定使用量（比如火车上的餐厅，最多只允许多少人进入，如果满了需要在门口等，等有人出来了才能进去）－“信号量”

这里有几个知识点要重点记录一下

单个CPU在任一时刻只能执行单个线程，只有多核CPU还能真正做到多个线程同时运行
一个进程包含多个线程，这些线程可以分布在多个CPU上
多核CPU同时运行的线程可以属于单个进程或不同进程
所以，在大多数编程语言中因为切换消耗的资源更少，多线程比多进程效率更高
多进程 vs 多线程
那么是不是意味着python中就只能使用多进程去提高效率，多线程就要被淘汰了呢？

那也不是的。

这里分两种情况来讨论，CPU密集型操作和IO密集型操作。针对前者，大多数时间花在CPU运算上，所以希望CPU利用的越充分越好，这时候使用多进程是合适的，同时运行的进程数和CPU的核数相同；针对后者，大多数时间花在IO交互的等待上，此时一个CPU和多个CPU是没有太大差别的，反而是线程切换比进程切换要轻量得多，这时候使用多线程是合适的。

所以有了结论：

CPU密集型操作使用多进程比较合适，例如海量运算
IO密集型操作使用多线程比较合适，例如爬虫，文件处理，批量ssh操作服务器等等

## Python十大排序

<img src="https://img-blog.csdn.net/20180821151148381?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU3MTQ5Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="img" style="zoom: 33%;" />

[史上最简单十大排序算法（Python实现）_python最简单的算法-CSDN博客](https://blog.csdn.net/weixin_41571493/article/details/81875088)

### 1. 冒泡排序(Bubble Sort) 

—— 交换排序

> - 比较相邻的元素。如果第一个比第二个大，就交换它们两个；
> - 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对，这样在最后的元素应该会是最大的数；
> - 针对所有的元素重复以上的步骤，除了最后一个；
> - 重复步骤1~3，直到排序完成。

冒泡排序对n个数据操作n-1轮，每轮找出一个最大（小）值。

操作只对相邻两个数比较与交换，每轮会将一个最值交换到数据列首（尾），像冒泡一样。

每轮操作O(n)次，共O（n）轮，时间复杂度O(n^2)。

额外空间开销出在交换数据时那一个过渡空间，空间复杂度O(1)

<img src="https://img-blog.csdn.net/2018082119503523?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU3MTQ5Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="img" style="zoom:80%;" />

```python
def BubbleSort(lst):
    n = len(lst)
    if n <= 1:
        return lst
    for i in range(n):
        for j in range(n-i-1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst

x=input("请输入待排序数列：\n")
y=x.split()
arr=[]
for i in  y:
    arr.append(int(i))
arr=BubbleSort(arr)
#print(arr)
print("数列按序排列如下：")
for i in arr:
    print(i,end=' ')
```

### 2. 快速排序(Quick Sort)

—— 交换排序

> - 从数列中挑出一个元素，称为 “基准”（pivot）；
> - 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作；
> - 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。

快速排序基于选择划分，是**简单选择排序的优化**。

每次划分将数据选到基准值两边，循环对两边的数据进行划分，类似于二分法。

算法的整体性能取决于划分的平均程度，即基准值的选择，此处衍生出快速排序的许多优化方案，甚至可以划分为多块。

基准值若能把数据分为平均的两块，划分次数O(logn)，每次划分遍历比较一遍O(n)，时间复杂度O(nlogn)。

额外空间开销出在暂存基准值，O(logn)次划分需要O(logn)个，空间复杂度O(logn)

<img src="https://img-blog.csdn.net/20180821195408748?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU3MTQ5Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="img" style="zoom: 80%;" />

```python
def QuickSort(lst):
    # 此函数完成分区操作
    def partition(arr, left, right):
        key = left  # 划分参考数索引,默认为第一个数为基准数，可优化
        while left < right:
            # 如果列表后边的数,比基准数大或相等,则前移一位直到有比基准数小的数出现
            while left < right and arr[right] >= arr[key]:
                right -= 1
            # 如果列表前边的数,比基准数小或相等,则后移一位直到有比基准数大的数出现
            while left < right and arr[left] <= arr[key]:
                left += 1
            # 此时已找到一个比基准大的书，和一个比基准小的数，将他们互换位置
            (arr[left], arr[right]) = (arr[right], arr[left])
 
        # 当从两边分别逼近，直到两个位置相等时结束，将左边小的同基准进行交换
        (arr[left], arr[key]) = (arr[key], arr[left])
        # 返回目前基准所在位置的索引
        return left
 
    def quicksort(arr, left, right):  
        if left >= right:
            return
        # 从基准开始分区
        mid = partition(arr, left, right)
        # 递归调用
        # print(arr)
        quicksort(arr, left, mid - 1)
        quicksort(arr, mid + 1, right)
 
    # 主函数
    n = len(lst)
    if n <= 1:
        return lst
    quicksort(lst, 0, n - 1)
    return lst
 
print("<<< Quick Sort >>>")
x = input("请输入待排序数列：\n")
y = x.split()
arr = []
for i in y:
    arr.append(int(i))
arr = QuickSort(arr)
# print(arr)
print("数列按序排列如下：")
for i in arr:
    print(i, end=' ')
```

### 3. 简单插入排序(Insert Sort)

—— 插入排序

> - 从第一个元素开始，该元素可以认为已经被排序；
> - 取出下一个元素，在已经排序的元素序列中从后向前扫描；
> - 如果该元素（已排序）大于新元素，将该元素移到下一位置；
> - 重复步骤3，直到找到已排序的元素小于或者等于新元素的位置；
> - 将新元素插入到该位置后；
> - 重复步骤2~5。

简单插入排序同样操作n-1轮，每轮将一个未排序树插入排好序列。

开始时默认第一个数有序，将剩余n-1个数逐个插入。插入操作具体包括：比较确定插入位置，数据移位腾出合适空位

每轮操作O(n)次，共O（n）轮，时间复杂度O(n^2)。

额外空间开销出在数据移位时那一个过渡空间，空间复杂度O(1)。

<img src="https://img-blog.csdn.net/20180821195107945?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU3MTQ5Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="img" style="zoom: 50%;" />

```python
def InsertSort(lst):
    n=len(lst)
    if n<=1:
        return lst
    for i in range(1,n):
        j=i
        target=lst[i]            #每次循环的一个待插入的数
        while j>0 and target<lst[j-1]:       #比较、后移，给target腾位置
            lst[j]=lst[j-1]
            j=j-1
        lst[j]=target            #把target插到空位
    return lst
 
x=input("请输入待排序数列：\n")
y=x.split()
arr=[]
for i in  y:
    arr.append(int(i))
arr=InsertSort(arr)
#print(arr)
print("数列按序排列如下：")
for i in arr:
    print(i,end=' ')
```

### 4. 希尔排序(Shell Sort) 

—— 插入排序

先将整个待排序的记录序列分割成为若干子序列分别进行直接插入排序，具体算法描述：

> - 选择一个增量序列t1，t2，…，tk，其中ti>tj，tk=1；
> - 按增量序列个数k，对序列进行k 趟排序；
> - 每趟排序，根据对应的增量ti，将待排序列分割成若干长度为m 的子序列，分别对各子表进行直接插入排序。仅增量因子为1 时，整个序列作为一个表来处理，表长度即为整个序列的长度。

希尔排序是插入排序的高效实现（大家可以比对一下插入排序和希尔排序的代码），对简单插入排序减少移动次数优化而来。

简单插入排序每次插入都要移动大量数据，前后插入时的许多移动都是重复操作，若一步到位移动效率会高很多。

若序列基本有序，简单插入排序不必做很多移动操作，效率很高。

希尔排序将序列按固定间隔划分为多个子序列，在子序列中简单插入排序，先做远距离移动使序列基本有序；逐渐缩小间隔重复操作，最后间隔为1时即简单插入排序。

希尔排序对序列划分O(n)次，每次简单插入排序O(logn)，时间复杂度O(nlogn)

额外空间开销出在插入过程数据移动需要的一个暂存，空间复杂度O(1)

<img src="https://img-blog.csdn.net/20180821151523537?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU3MTQ5Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="img" style="zoom:80%;" />

---

希尔排序的核心在于间隔序列的设定。既可以提前设定好间隔序列，也可以动态的定义间隔序列。动态定义间隔序列的算法是《算法（第4版）》的合著者Robert Sedgewick提出的。

```python
def ShellSort(lst):
    def shellinsert(arr,d):
        n=len(arr)
        for i in range(d,n):
            j=i-d
            temp=arr[i]             #记录要出入的数
            while(j>=0 and arr[j]>temp):    #从后向前，找打比其小的数的位置
                arr[j+d]=arr[j]                 #向后挪动
                j-=d
            if j!=i-d:
                arr[j+d]=temp
    n=len(lst)
    if n<=1:
        return lst
    d=n//2
    while d>=1:
        shellinsert(lst,d)
        d=d//2
    return lst
 
 
x=input("请输入待排序数列：\n")
y=x.split()
arr=[]
for i in  y:
    arr.append(int(i))
arr=ShellSort(arr)
#print(arr)
print("数列按序排列如下：")
for i in arr:
    print(i,end=' ') 
```

### 5. 简单选择排序(Select Sort) 

—— 选择排序

> - 初始状态：无序区为R[1..n]，有序区为空；
> - 第i趟排序(i=1,2,3…n-1)开始时，当前有序区和无序区分别为R[1..i-1]和R(i..n）。该趟排序从当前无序区中-选出关键字最小的记录 R[k]，将它与无序区的第1个记录R交换，使R[1..i]和R[i+1..n)分别变为记录个数增加1个的新有序区和记录个数减少1个的新无序区；
> - n-1趟结束，数组有序化了。

简单选择排序同样对数据操作n-1轮，每轮找出一个最大（小）值。

操作指选择，即未排序数逐个比较交换，争夺最值位置，每轮将一个未排序位置上的数交换成已排序数，即每轮选一个最值。

每轮操作O(n)次，共O（n）轮，时间复杂度O(n^2)。

额外空间开销出在交换数据时那一个过渡空间，空间复杂度O(1)。

<img src="https://img-blog.csdn.net/20180821195138636?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU3MTQ5Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="img" style="zoom:80%;" />

---

表现最稳定的排序算法之一，因为无论什么数据进去都是O(n2)的时间复杂度，所以用到它的时候，数据规模越小越好。唯一的好处可能就是不占用额外的内存空间了吧。理论上讲，选择排序可能也是平时排序一般人想到的最多的排序方法了吧。

```python
def SelectSort(lst):
    n=len(lst)
    if n<=1:
        return lst
    for i in range(0,n-1):
        minIndex=i
        for j in range(i+1,n):          #比较一遍，记录索引不交换
            if lst[j]<lst[minIndex]:
                minIndex=j
        if minIndex!=i:                     #按索引交换
            (lst[minIndex],lst[i])=(lst[i],lst[minIndex])
    return lst
 
x=input("请输入待排序数列：\n")
y=x.split()
arr=[]
for i in  y:
    arr.append(int(i))
arr=SelectSort(arr)
#print(arr)
print("数列按序排列如下：")
for i in arr:
    print(i,end=' ')
```

### 6. 堆排序(Heap Sort) 

—— 选择排序

堆排序（Heapsort）是指利用堆这种数据结构所设计的一种排序算法。堆积是一个近似完全二叉树的结构，并同时满足堆积的性质：即子结点的键值或索引总是小于（或者大于）它的父节点。

> - 将初始待排序关键字序列(R1,R2….Rn)构建成大顶堆，此堆为初始的无序区；
> - 将堆顶元素R[1]与最后一个元素R[n]交换，此时得到新的无序区(R1,R2,……Rn-1)和新的有序区(Rn),且满足R[1,2…n-1]<=R[n]；
> - 由于交换后新的堆顶R[1]可能违反堆的性质，因此需要对当前无序区(R1,R2,……Rn-1)调整为新堆，然后再次将R[1]与无序区最后一个元素交换，得到新的无序区(R1,R2….Rn-2)和新的有序区(Rn-1,Rn)。不断重复此过程直到有序区的元素个数为n-1，则整个排序过程完成。

堆排序的初始建堆过程比价复杂，对O(n)级别个非叶子节点进行堆调整操作O(logn)，时间复杂度O(nlogn)；之后每一次堆调整操作确定一个数的次序，时间复杂度O(nlogn)。合起来时间复杂度O(nlogn)

额外空间开销出在调整堆过程，根节点下移交换时一个暂存空间，空间复杂度O(1)

<img src="https://img-blog.csdn.net/2018082115050636?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU3MTQ5Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="img" style="zoom: 67%;" />

```python
def  HeapSort(lst):
    def heapadjust(arr,start,end):  #将以start为根节点的堆调整为大顶堆
        temp=arr[start]
        son=2*start+1
        while son<=end:
            if son<end and arr[son]<arr[son+1]:  #找出左右孩子节点较大的
                son+=1
            if temp>=arr[son]:       #判断是否为大顶堆
                break
            arr[start]=arr[son]     #子节点上移
            start=son                     #继续向下比较
            son=2*son+1
        arr[start]=temp             #将原堆顶插入正确位置
#######
    n=len(lst)
    if n<=1:
        return lst
    #建立大顶堆
    root=n//2-1    #最后一个非叶节点（完全二叉树中）
    while(root>=0):
        heapadjust(ls,root,n-1)
        root-=1
    #掐掉堆顶后调整堆
    i=n-1
    while(i>=0):
        (lst[0],lst[i])=(lst[i],lst[0])  #将大顶堆堆顶数放到最后
        heapadjust(lst,0,i-1)    #调整剩余数组成的堆
        i-=1
    return lst
#########
x=input("请输入待排序数列：\n")
y=x.split()
arr=[]
for i in  y:
    arr.append(int(i))
arr=HeapSort(arr)
#print(arr)
print("数列按序排列如下：")
for i in arr:
    print(i,end=' ') 
```

### 7. 二路归并排序(Two-way Merge Sort) 

—— 归并排序

归并排序是建立在归并操作上的一种有效的排序算法。该算法是采用分治法（Divide and Conquer）的一个非常典型的应用。将已有序的子序列合并，得到完全有序的序列；即先使每个子序列有序，再使子序列段间有序。若将两个有序表合并成一个有序表，称为2-路归并。

> - 把长度为n的输入序列分成两个长度为n/2的子序列；
> - 对这两个子序列分别采用归并排序；
> - 将两个排序好的子序列合并成一个最终的排序序列。

<img src="https://img-blog.csdn.net/20180821151049225?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU3MTQ5Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="img" style="zoom:50%;" />

---

归并排序是一种稳定的排序方法。和选择排序一样，归并排序的性能不受输入数据的影响，但表现比选择排序好的多，因为始终都是O(nlogn）的时间复杂度。代价是需要额外的内存空间。

```python
def MergeSort(lst):
    #合并左右子序列函数
    def merge(arr,left,mid,right):
        temp=[]     #中间数组
        i=left          #左段子序列起始
        j=mid+1   #右段子序列起始
        while i<=mid and j<=right:
            if arr[i]<=arr[j]:
                temp.append(arr[i])
                i+=1
            else:
                temp.append(arr[j])
                j+=1
        while i<=mid:
            temp.append(arr[i])
            i+=1
        while j<=right:
            temp.append(arr[j])
            j+=1
        for i in range(left,right+1):    #  !注意这里，不能直接arr=temp,他俩大小都不一定一样
            arr[i]=temp[i-left]
    #递归调用归并排序
    def mSort(arr,left,right):
        if left>=right:
            return
        mid=(left+right)//2
        mSort(arr,left,mid)
        mSort(arr,mid+1,right)
        merge(arr,left,mid,right)
 
    n=len(lst)
    if n<=1:
        return lst
    mSort(lst,0,n-1)
    return lst
 
x=input("请输入待排序数列：\n")
y=x.split()
arr=[]
for i in  y:
    arr.append(int(i))
arr=MergeSort(arr)
#print(arr)
print("数列按序排列如下：")
for i in arr:
    print(i,end=' ') 
```

### 8. 计数排序（Counting Sort） 

—— 线性时间非比较类排序

> - 找出待排序的数组中最大和最小的元素；
> - 统计数组中每个值为i的元素出现的次数，存入数组C的第i项；
> - 对所有的计数累加（从C中的第一个元素开始，每一项和前一项相加）；
> - 反向填充目标数组：将每个元素i放在新数组的第C(i)项，每放一个元素就将C(i)减去1。

计数排序用待排序的数值作为计数数组（列表）的下标，统计每个数值的个数，然后依次输出即可。

计数数组的大小取决于待排数据取值范围，所以对数据有一定要求，否则空间开销无法承受。

计数排序只需遍历一次数据，在计数数组中记录，输出计数数组中有记录的下标，时间复杂度为O(n+k)。

额外空间开销即指计数数组，实际上按数据值分为k类（大小取决于数据取值），空间复杂度O(k)。

<img src="https://img-blog.csdn.net/20180821152639330?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU3MTQ5Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="img" style="zoom:50%;" />

---

计数排序是一个稳定的排序算法。当输入的元素是 n 个 0到 k 之间的整数时，时间复杂度是O(n+k)，空间复杂度也是O(n+k)，其排序速度快于任何比较排序算法。当k不是很大并且序列比较集中时，计数排序是一个很有效的排序算法。

```python
def CountSort(lst):
    n=len(lst)
    num=max(lst)
    count=[0]*(num+1)
    for i in range(0,n):
        count[lst[i]]+=1
    arr=[]
    for i in range(0,num+1):
        for j in range(0,count[i]):
            arr.append(i)
    return arr
 
 
x=input("请输入待排序数列：\n")
y=x.split()
arr=[]
for i in  y:
    arr.append(int(i))
arr=CountSort(arr)
#print(arr)
print("数列按序排列如下：")
for i in arr:
    print(i,end=' ') 
```

### 9. 桶排序（Bucket Sort） 

—— 线性时间非比较类排序

> - 设置一个定量的数组当作空桶；
> - 遍历输入数据，并且把数据一个一个放到对应的桶里去；
> - 对每个不是空的桶进行排序；
> - 从不是空的桶里把排好序的数据拼接起来。 

桶排序实际上是计数排序的推广，但实现上要复杂许多。

桶排序先用一定的函数关系将数据划分到不同有序的区域（桶）内，然后子数据分别在桶内排序，之后顺次输出。

当每一个不同数据分配一个桶时，也就相当于计数排序。

假设n个数据，划分为k个桶，桶内采用快速排序，时间复杂度为O(n)+O(k * n/k*log(n/k))=O(n)+O(n*(log(n)-log(k))),

显然，k越大，时间复杂度越接近O(n)，当然空间复杂度O(n+k)会越大，这是空间与时间的平衡。

![img](https://img-blog.csdn.net/20180821152757838?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU3MTQ5Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

---

桶排序最好情况下使用线性时间O(n)，桶排序的时间复杂度，取决与对各个桶之间数据进行排序的时间复杂度，因为其它部分的时间复杂度都为O(n)。很显然，桶划分的越小，各个桶之间的数据越少，排序所用的时间也会越少。但相应的空间消耗就会增大。

```python
def BucketSort(lst):
    ##############桶内使用快速排序
    def QuickSort(lst):
        def partition(arr,left,right):
            key=left         #划分参考数索引,默认为第一个数，可优化
            while left<right:
                while left<right and arr[right]>=arr[key]:
                    right-=1
                while left<right and arr[left]<=arr[key]:
                    left+=1
                (arr[left],arr[right])=(arr[right],arr[left])
            (arr[left],arr[key])=(arr[key],arr[left])
            return left
 
        def quicksort(arr,left,right):   #递归调用
            if left>=right:
                return
            mid=partition(arr,left,right)
            quicksort(arr,left,mid-1)
            quicksort(arr,mid+1,right)
        #主函数
        n=len(lst)
        if n<=1:
            return lst
        quicksort(lst,0,n-1)
        return lst
    ######################
    n=len(lst)
    big=max(lst)
    num=big//10+1
    bucket=[]
    buckets=[[] for i in range(0,num)]
    for i in lst:
        buckets[i//10].append(i)     #划分桶
    for i in buckets:                       #桶内排序
        bucket=QuickSort(i)
    arr=[]
    for i in buckets:
        if isinstance(i, list):
            for j in i:
                arr.append(j)
        else:
            arr.append(i)
    for i in range(0,n):
        lst[i]=arr[i]
    return lst
    
x=input("请输入待排序数列：\n")
y=x.split()
arr=[]
for i in  y:
    arr.append(int(i))
arr=BucketSort(arr)
#print(arr)
print("数列按序排列如下：")
for i in arr:
    print(i,end=' ') 
```

### 10. 基数排序（Radix Sort） 

—— 线性时间非比较类排序

> - 取得数组中的最大数，并取得位数；
> - arr为原始数组，从最低位开始取每个位组成radix数组；
> - 对radix进行计数排序（利用计数排序适用于小范围数的特点）；

<img src="https://img-blog.csdn.net/20180821153051832?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU3MTQ5Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="img" style="zoom: 50%;" />

---

基数排序基于分别排序，分别收集，所以是稳定的。但基数排序的性能比桶排序要略差，每一次关键字的桶分配都需要O(n)的时间复杂度，而且分配之后得到新的关键字序列又需要O(n)的时间复杂度。假如待排数据可以分为d个关键字，则基数排序的时间复杂度将是O(d*2n) ，当然d要远远小于n，因此基本上还是线性级别的。

基数排序的空间复杂度为O(n+k)，其中k为桶的数量。一般来说n>>k，因此额外空间需要大概n个左右。

```python
import math
def RadixSort(lst):
    def getbit(x,i):       #返回x的第i位（从右向左，个位为0）数值
        y=x//pow(10,i)
        z=y%10
        return z
    def CountSort(lst):
        n=len(lst)
        num=max(lst)
        count=[0]*(num+1)
        for i in range(0,n):
            count[lst[i]]+=1
        arr=[]
        for i in range(0,num+1):
            for j in range(0,count[i]):
                arr.append(i)
        return arr
    Max=max(lst)
    for k in range(0,int(math.log10(Max))+1):             #对k位数排k次,每次按某一位来排
        arr=[[] for i in range(0,10)]
        for i in lst:                 #将ls（待排数列）中每个数按某一位分类（0-9共10类）存到arr[][]二维数组（列表）中
            arr[getbit(i,k)].append(i)
        for i in range(0,10):         #对arr[]中每一类（一个列表）  按计数排序排好
            if len(arr[i])>0:
                arr[i]=CountSort(arr[i])
        j=9
        n=len(lst)
        for i in range(0,n):     #顺序输出arr[][]中数到ls中，即按第k位排好
            while len(arr[j])==0:
                j-=1
            else:
                ls[n-1-i]=arr[j].pop()   
    return lst    
    
x=input("请输入待排序数列：\n")
y=x.split()
arr=[]
for i in  y:
    arr.append(int(i))
arr=RadixSort(arr)
#print(arr)
print("数列按序排列如下：")
for i in arr:
    print(i,end=' ') 
```



## 二叉树遍历总结

这里分别给出了三种二叉树的遍历方法与N叉树的前序遍历，及其时空复杂度
1：递归：直接递归版本、针对不同题目通用递归版本（包括前序、中序、后序）
2：迭代：最常用版本（常用主要包括前序和层序，即【DFS和BFS】）、【前中后】序遍历通用版本（一个栈的空间）、【前中后层】序通用版本（双倍栈（队列）的空间）
3：莫里斯遍历：利用线索二叉树的特性进行遍历
4：N叉树的前序遍历

LeeCode题目链接：

主要参考资料：


```python
Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
```

### 1. 递归

时间复杂度：O(n)，n为节点数，访问每个节点恰好一次。

空间复杂度：O(h)，h为树的高度。最坏情况下需要空间O(n)，平均情况为O(logn)

**递归1：**

二叉树遍历最易理解和实现版本

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        # 前序递归
		return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)
		# 中序递归 
		# return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
		# 后序递归
		# return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]
```

**递归2：**

通用模板，可以适应不同的题目，添加参数、增加返回条件、修改进入递归条件、自定义返回值

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        def dfs(cur):
            if not cur:
                return      
			# 前序递归
			res.append(cur.val)
			dfs(cur.left)
			dfs(cur.right) 
			# 中序递归
			# dfs(cur.left)
			# res.append(cur.val)
			# dfs(cur.right)
			# 后序递归
			# dfs(cur.left)
			# dfs(cur.right)
			# res.append(cur.val)      
		res = []
		dfs(root)
		return res
```





### 2. 迭代

时间复杂度：O(n)，n为节点数，访问每个节点恰好一次。

空间复杂度：O(h)，h为树的高度。取决于树的结构，最坏情况存储整棵树，即O(n)

**迭代1：**

前序遍历最常用模板（后序同样可以用）

```python
class Solution:
	def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []        
        res = []
        stack = [root]
		# 前序迭代模板：最常用的二叉树DFS迭代遍历模板
		while stack:
			cur = stack.pop()
			res.append(cur.val)
			if cur.right:
				stack.append(cur.right)
			if cur.left:
				stack.append(cur.left)
		return res
    
        # 后序迭代，相同模板：将前序迭代进栈顺序稍作修改，最后得到的结果反转
        # while stack:
        #     cur = stack.pop()
        #     if cur.left:
        #         stack.append(cur.left)
        #     if cur.right:
        #         stack.append(cur.right)
        #     res.append(cur.val)
        # return res[::-1]
```

**迭代2：**

层序遍历最常用模板

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        cur, res = [root], []
        while cur:
            lay, layval = [], []
            for node in cur:
                layval.append(node.val)
                if node.left: lay.append(node.left)
                if node.right: lay.append(node.right)
            cur = lay
            res.append(layval)
        return res
```

**迭代3：**

前、中、后序遍历通用模板（只需一个栈的空间）

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]: 
        res = []
        stack = []
        cur = root
        # 中序，模板：先用指针找到每颗子树的最左下角，然后进行进出栈操作
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        return res
    
        # # 前序，相同模板
        # while stack or cur:
        #     while cur:
        #         res.append(cur.val)
        #         stack.append(cur)
        #         cur = cur.left
        #     cur = stack.pop()
        #     cur = cur.right
        # return res

        # # 后序，相同模板
        # while stack or cur:
        #     while cur:
        #         res.append(cur.val)
        #         stack.append(cur)
        #         cur = cur.right
        #     cur = stack.pop()
        #     cur = cur.left
        # return res[::-1]
```

**迭代4：**

标记法迭代（需要双倍的空间来存储访问状态）：
前、中、后、层序通用模板，只需改变进栈顺序或即可实现前后中序遍历，而层序遍历则使用队列先进先出。0表示当前未访问，1表示已访问。               

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = [(0, root)]
        while stack:
            flag, cur = stack.pop()
            if not cur: continue
            if flag == 0:
                # 前序，标记法
                stack.append((0, cur.right))
                stack.append((0, cur.left))
                stack.append((1, cur))
                
                # 后序，标记法
                # stack.append((1, cur))
                # stack.append((0, cur.right))
                # stack.append((0, cur.left))

                # # 中序，标记法
                # stack.append((0, cur.right))
                # stack.append((1, cur))
                # stack.append((0, cur.left))  
            else:
                res.append(cur.val)  
        return res
    
        # 层序，标记法
        # res = []
        # queue = [(0, root)]
        # while queue:
        #     flag, cur = queue.pop(0)  # 注意是队列，先进先出
        #     if not cur: continue
        #     if flag == 0:
                  # 层序遍历这三个的顺序无所谓，因为是队列，只弹出队首元素
        #         queue.append((1, cur))
        #         queue.append((0, cur.left))
        #         queue.append((0, cur.right))
        #     else:
        #         res.append(cur.val)
        # return res
```



#### 3. 莫里斯遍历

时间复杂度：O(n)，n为节点数，看似超过O(n)，有的节点可能要访问两次，实际分析还是O(n)，具体参考大佬博客的分析。

空间复杂度：O(1)，如果在遍历过程中就输出节点值，则只需常数空间就能得到中序遍历结果，空间只需两个指针。如果将结果储存最后输出，则空间复杂度还是O(n)。

PS：莫里斯遍历实际上是在原有二叉树的结构基础上，构造了线索二叉树。线索二叉树定义为：原本为空的右子节点指向了中序遍历顺序之后的那个节点，把所有原本为空的左子节点都指向了中序遍历之前的那个节点

此处只给出中序遍历，前序遍历只需修改输出顺序即可
而后序遍历，由于遍历是从根开始的，而线索二叉树是将为空的左右子节点连接到相应的顺序上，使其能够按照相应准则输出
但是后序遍历的根节点却已经没有额外的空间来标记自己下一个应该访问的节点，
所以这里需要建立一个临时节点dump，令其左孩子是root。并且还需要一个子过程，就是倒序输出某两个节点之间路径上的各个节点。



```python
# 莫里斯遍历，借助线索二叉树中序遍历（附前序遍历）
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        # cur = pre = TreeNode(None)
        cur = root
        
        while cur:
            if not cur.left:
                res.append(cur.val)
                # print(cur.val)
                cur = cur.right
            else:
                pre = cur.left
                while pre.right and pre.right != cur:
                    pre = pre.right
                if not pre.right:
                    # print(cur.val) 这里是前序遍历的代码，前序与中序的唯一差别，只是输出顺序不同
                    pre.right = cur
                    cur = cur.left
                else:
                    pre.right = None
                    res.append(cur.val)
                    # print(cur.val)
                    cur = cur.right
        return res
```



#### 4. N叉树遍历

时间复杂度：O(M)，其中 M 是 N 叉树中的节点个数。每个节点只会入栈和出栈各一次。

空间复杂度：O(M)。在最坏的情况下，这棵 N 叉树只有 2 层，所有第 2 层的节点都是根节点的孩子。

将根节点推出栈后，需要将这些节点都放入栈，共有 M−1个节点，因此栈的大小为 O(M)。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

# N叉树简洁递归
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        if not root: return []
        res = [root.val]
        for node in root.children:
            res.extend(self.preorder(node))
        return res

# N叉树通用递归模板
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        res = []
        def helper(root):
            if not root:
                return
            res.append(root.val)
            for child in root.children:
                helper(child)
        helper(root)
        return res

# N叉树迭代方法

class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        s = [root]
        # s.append(root)
        res = []
        while s:
            node = s.pop()
            res.append(node.val)
            # for child in node.children[::-1]:
            #     s.append(child)
            s.extend(node.children[::-1])
        return res
```



### 树形递归技巧

#### 前言

- 二叉树的三种遍历方式 (前序, 中序, 后序) 代表了三种典型的递归场景;

  ```python
  def dfs(node):
      if node is None: return
      
      ...  # 前序
      dfs(node.left)
      ...  # 中序
      dfs(node.right)
      ...  # 后序
  ```

- 这三种遍历方式的主要区别在于遍历到当前节点时, 已知节点的位置不同, 以根节点为例:

  - 前序遍历到根节点时, 其他节点都未知;
  - 中序遍历到根节点时, 遍历完了整棵左子树;
  - 后序遍历到根节点时, 已经遍历完了所有子节点;

- 可见, 

  - **前序遍历**是**自顶向下**的递归; 如果当前节点需要父节点的信息时, 就用前序遍历; 
  - **后序遍历**是**自底向上**的递归; 如果当前节点需要子节点的信息时, 就用后序遍历;
  - **中序遍历**比较特殊, 可以认为它是二叉树特有的, 比如对一棵多叉树, 中序遍历就无从谈起, 所以中序遍历主要用在那些利用了二叉树特殊性质的情况, 比如**二叉搜索树**;

- 我们使用递归的一个关键, 就是希望将问题分解为子问题后再逐步解决, 

  - 这一点非常契合**后序遍历**的方式, 即从子树的解, 递推全局的解, 所以后序遍历的问题是最多的; 

  - 此时可以把这种在树上进行的递归看作是一种特殊的动态规划, 即**树形 dp**;

    > [递归与动态规划的关系](./从暴力递归到动态规划.md)

- 本篇介绍的技巧, 简单来说就是如何结构化处理这些子问题的解, 这个方法可以用于所有需要**自底向上进行递归**的问题;


#### 后序遍历的递归技巧

> 自底向上的递归技巧, **树形 dp** 等

- 记 `dfs(x)` 模板如下:

  ```python
  from dataclasses import dataclass
  
  @dataclass
  class Info:
      ...  # default
  
  def dfs(x) -> Info:
      if x is None: return Info()
  
      l_info = dfs(x.left)
      r_info = dfs(x.right)
      x_info = get_from(l_info, r_info)
      return x_info
  ```

- 考虑计算出当前节点的答案需要从左右子树获得哪些信息, 并**假设已知**, 记 `l_info` 和 `r_info`; 

- 利用 `l_info` 和 `r_info` 构造出当前节点应该返回的信息 `x_info`;

- 注意空树返回的信息, 即 `Info` 的默认值;

- 为了可读性, 推荐使用**结构体** (python 中推荐使用 `dataclass`) 来保存需要的信息; 

- 进一步的, 最终生成的 `x_info` 不一定都会与 `x` 有关, 此时需要分**与 x 有关的答案**和**与 x 无关的答案**进行讨论, 并取其中的最优解 (详见示例 3 和 4);

##### 示例 1: 二叉树的高度

> [104. 二叉树的最大深度 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

- 二叉树 `x` 的高度等于左右子树高度中的较大值 `+1`;

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        
        def dfs(x):
            if x is None: return 0

            l, r = dfs(x.left), dfs(x.right)
            return max(l, r) + 1
        
        return dfs(root)
```

> 本题比较简单, 故简化了模板;

##### 示例 2: 判断是否为平衡二叉树/搜索二叉树/完全二叉树

> [98. 验证二叉搜索树 - 力扣（LeetCode）](https://leetcode.cn/problems/validate-binary-search-tree/)  
> [110. 平衡二叉树 - 力扣（LeetCode）](https://leetcode-cn.com/problems/balanced-binary-tree/)  
> [958. 二叉树的完全性检验 - 力扣（LeetCode）](https://leetcode.cn/problems/check-completeness-of-a-binary-tree/)  

- 搜索二叉树: 对**每个节点**, 其值大于左子树的**最大值**，小于右子树的**最小值**;

- 平衡二叉树: 对**每个节点**, 其左右子树的**高度**差 `<= 1`;

- 完全二叉树, 这里给出两种判断条件:

  - 条件1:

    - 左右子树都是满二叉树，且高度相同, 即本身是满二叉树；

    - 左右子树都是满二叉树，且左子树的高度+1；

    - 左子树是满二叉树，右子树是完全二叉树，且高度相同；

    - 左子树是完全二叉树，右子树是满二叉树，且左子树的高度+1；

      > **满二叉树**: 左右子树的高度相同, 且都是满二叉树;

  - 条件2, 该方法需要前序遍历进行预处理 (详见下一节: [前序遍历的信息传递](#前序遍历的信息传递)):

    - 记根节点 `id` 为 `1`；若父节点的 `id` 为 `i`，则其左右子节点分别为 `2*i` 和 `2*i+1`；
    - 如果是完全二叉树则有 `max_id == n`，其中 `n` 为总节点数；

- 以下为合并实现

```python
class Solution:
    def isXxxTree(self, root: TreeNode) -> bool:

        from dataclasses import dataclass

        @dataclass
        class Info:
            height: int = 0               # 树的高度
            max_val: int = float('-inf')  # 最大值
            min_val: int = float('inf')   # 最小值
            is_balance: bool = True       # 是否平衡二叉树
            is_search: bool = True        # 是否搜索二叉树
            is_full: bool = True          # 是否满二叉树
            is_complete: bool = True      # 是否完全二叉树
        
        def dfs(x):
            if not x: return Info()

            l, r = dfs(x.left), dfs(x.right)

            # 利用左右子树的info 构建当前节点的info
            height = max(l.height, r.height) + 1
            max_val = max(x.val, l.max_val, r.max_val)
            min_val = min(x.val, l.min_val, r.min_val)
            is_balance = l.is_balance and r.is_balance and abs(l.height - r.height) <= 1
            is_search = l.is_search and r.is_search and x.val > l.max_val and x.val < r.min_val
            is_full = l.is_full and r.is_full and l.height == r.height
            is_complete = is_full \
                or l.is_full and r.is_full and l.height - 1 == r.height \
                or l.is_full and r.is_complete and l.height == r.height \
                or l.is_complete and r.is_full and l.height - 1 == r.height
            
            return Info(height, max_val, min_val, is_balance, is_search, is_full, is_complete)
        
        return dfs(root).xxx  # 根据具体问题确定返回值
```

##### 示例 3: 二叉树中的最大路径和

> [124. 二叉树中的最大路径和 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)

- 根据最大路径和是否经过 `x` 节点分两种情况;
  - 如果不经过 `x` 节点, 那么 `x` 中的最大路径和等于左右子树中最大路径和中的较大值;
  - 如果经过 `x` 节点, 那么 `x` 的最大路径和等于左右子树能提供的最大路径之和, 再加上当前节点的值;
  - 取两者中的较大值;
- 综上, 需要记录的信息包括, 当前节点能提供的最大路径, 当前的最大路径;
  - 所谓 "当前节点能提供的最大路径", 即从该节点向下无回溯遍历能得到的最大值; 假设每个节点能提供的路径都是 1, 那么这个最大路径就是树的高度;
  - 因为节点值存在负数, 这是一个容易出错的点 (详见代码);
- 从本题可以看到, 模板并不能解决问题, 只是减少了问题以外的思考, 即如何将思路转换为代码.

```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:

        from dataclasses import dataclass

        @dataclass
        class Info:
            # 易错点: 因为存在负值, 所以初始化为负无穷 (最小值为 -1000)
            h: int = -1001  # 当前节点能提供的最大路径
            s: int = -1001  # 当前节点下的最大路径
        
        def dfs(x):
            if not x: return Info()

            l, r = dfs(x.left), dfs(x.right)
            # 利用左右子树的信息, 计算当前节点的信息
            h = x.val + max(l.h, r.h, 0)  # 易错点: 如果左右子树能提供的最大路径是一个负数, 则应该直接舍去
            s_through_x = x.val + max(l.h, 0) + max(r.h, 0)  # 易错点: 同上
            # 因为 h 和 s_through_x 都和当前节点有关, 所以必须包含当前节点
            s = max(l.s, r.s, s_through_x)  # 但是最终结果不一定包含 x, 需要取最优解
            return Info(h, s)
        
        return dfs(root).s
```

##### 示例 4: 打家劫舍 III

> [337. 打家劫舍 III - 力扣（LeetCode）](https://leetcode-cn.com/problems/house-robber-iii/)

- 树形 dp，就是否抢劫当前节点分两种情况讨论;

```python
class Solution:
    def rob(self, root: TreeNode) -> int:

        from functools import lru_cache

        @lru_cache
        def dfs(x):
            # 空节点, 不抢
            if not x: return 0
            # 叶节点, 必抢
            if not x.left and not x.right: return x.val

            # 不抢当前节点, 抢左右子节点
            r1 = dfs(x.left) + dfs(x.right)
            # 抢当前节点, 跳过左右子节点, 抢子节点的子节点
            r2 = x.val
            if x.left:  # 非空判断
                r2 += dfs(x.left.left) + dfs(x.left.right)
            if x.right:  # 非空判断
                r2 += dfs(x.right.left) + dfs(x.right.right)
            
            return max(r1, r2)
        
        return dfs(root)
```

#### 前序遍历的信息传递

- 后序遍历中, 父节点获取子节点的信息很自然, 直接在函数体内接收递归的结果即可;

- 前序遍历中, 子节点想要获取父节点的信息就不能这么做, 一般的方法是通过参数传递;

  ```python
  from dataclasses import dataclass
  
  @dataclass
  class Info:
      ...  # default
  
  def dfs(x, f_info) -> Info:
      if x is None: return Info()
  
      f_info = process(f_info)  # 利用父节点的信息进行预处理
  
      l_info = dfs(x.left, f_info)
      r_info = dfs(x.right, f_info)
  
      x_info = get_from(l_info, r_info)
      return x_info
  ```

##### 示例: 判断完全二叉树

> [958. 二叉树的完全性检验 - 力扣（LeetCode）](https://leetcode.cn/problems/check-completeness-of-a-binary-tree/)  

- 思路:
  - 记根节点 `id` 为 `1`；若父节点的 `id` 为 `i`，则其左右子节点分别为 `2*i` 和 `2*i+1`；
  - 如果是完全二叉树则有 `max_id == total_cnt`，其中 `total_cnt` 为总节点数；
- 这里子节点的 id 就需要通过父节点的 id 来计算;

```python
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        
        self.total_cnt = 0
        self.max_id = 0

        def dfs(x, node_id):
            if not x: return

            self.total_cnt += 1
            self.max_id = max(self.max_id, node_id)
            dfs(x.left, node_id * 2)
            dfs(x.right, node_id * 2 + 1)

        dfs(root, 1)
        return self.total_cnt == self.max_id
```

##### 示例: 路径总和

> [112. 路径总和 - 力扣（LeetCode）](https://leetcode.cn/problems/path-sum/)

- 问题简述:
  - 判断树中是否存在 根节点到叶子节点 的路径和等于 targetSum;

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root: return False

        self.ret = False

        def dfs(x, s):
            if self.ret or not x:
                return

            s += x.val
            if x.left is None and x.right is None:
                self.ret = s == targetSum
                return

            dfs(x.left, s)
            dfs(x.right, s)
            s -= x.val  # 回溯
        
        dfs(root, 0)
        return self.ret
```





### 链表常用操作

#### 反转链表

> [206. 反转链表 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-linked-list/)

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        # 顶针写法
        pre, cur = None, head
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt

        return pre
```

#### 找中间节点

> [876. 链表的中间结点 - 力扣（LeetCode）](https://leetcode.cn/problems/middle-of-the-linked-list/)


根据中间节点的定义, 有两种写法; 当链表为奇数个节点时, 两者返回相同; 区别在于当链表中有**偶数个节点**时, 

- 法1 返回**后一个中间节点**, 
- 法2 返回**前一个中间节点**; 

```Python
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        # 法1)
        slow, fast = head, head
        # 法2)
        # slow, fast = head, head.next
        while fast and fast.next:  # 当 fast 不为 None 时, slow 永远不为 None
            slow = slow.next
            fast = fast.next.next
        
        return slow
```

### 滑动窗口模板

#### 模板

```Python
def temp(s: Sequence):

    l = r = 0  # 初始化窗口, [l, r] 闭区间
    while r < len(s):
        ...  # 更新缓存
        while check():  # 检查条件
            ...  # 更新缓存或答案
            l += 1  # 移动左边界, 缩小窗口
        ...  # 更新缓存或答案
        r += 1  # 移动右边界, 扩大窗口
```

- 以上是滑动窗口的基础模板;
- 更新缓存和检查条件的先后顺序要看具体情况, 详见示例;

#### 示例

##### 长度最小的子数组

> [209. 长度最小的子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-size-subarray-sum/)

问题简述

```txt
给定一个含有 n 个正整数的数组和一个正整数 target 。
找出该数组中满足其和 ≥ target 的长度最小的 连续子数组，并返回其长度。
如果不存在符合条件的子数组，返回 0 。
```

```Python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:

        ret = 0
        s = 0  # 记录当前窗口的和
        l, r = 0, 0
        while r < len(nums):
            s += nums[r]  # 更新缓存
            while s >= target:
                if not ret or r - l + 1 < ret:  # 更新答案
                    ret = r - l + 1
                s -= nums[l]
                l += 1
            r += 1
        
        return ret
```

##### 无重复字符的最长子串

> [3. 无重复字符的最长子串 - 力扣（LeetCode）](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        
        used = set()  # 记录已经出现过的字符
        ret = 0  # 记录最大长度

        l = r = 0  # 窗口边界
        while r < len(s):
            while s[r] in used:  # 出现重复字符, 滑动左边界
                # 移除最左边界的字符, 并缩小窗口
                used.remove(s[l])  # 更新缓存
                l += 1
            
            # 更新缓存和答案
            ret = max(ret, r - l + 1)
            used.add(s[r])
            r += 1  # 扩大窗口

        return ret
```


##### 最小覆盖子串

> [76. 最小覆盖子串 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-window-substring/)

```Python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        
        from collections import Counter, defaultdict
        
        l, r = 0, 0

        ret = ''  # 记录答案
        need = Counter(t)  # 需要满足的每种字符数
        book = defaultdict(int)  # 记录出现过的字符数
        
        def check():  # 检验是否满足情况
            return all(book[k] >= need[k] for k in need)
        
        while r < len(s):
            book[s[r]] += 1  # 更新缓存
            while check():  # 条件检查
                if not ret or r - l < len(ret):  # 更新答案
                    ret = s[l: r + 1]
                book[s[l]] -= 1
                l += 1
            r += 1
        
        return ret
```



### 从暴力递归到动态规划

#### 概述

- 从代码角度, 动态规划的实现一般都是通过迭代完成的 (递推公式); 而一般迭代过程都有与之对应的递归写法;

  > 这里默认动态规划都是基于迭代的实现. 实际上, 动态规划只是一种算法, 跟具体实现无关, 递归实现也是动态规划;

- 递归过程一般与人的直接思考过程更接近, 因此如果对递归结构比较熟悉, 就能更容易的写出相应解法;

  > 动态规划的难点是递推过程 (经验和数学能力); 递归的难点是递归结构本身;

- 递归的一个问题是如果存在重复计算, 会大幅降低性能, 此时有两个解决方法:

  1. 转换为迭代结构; 转化过程可以套用固定模板, 详见示例;
  2. 使用记忆化搜索, 即保存中间结果; 如果用 python 书写, 可以使用 `functools.lru_cache` 装饰器, 详见示例;

  - 如何判断是否存在重复计算?

    > 将递归过程展开为树状结构，当发现某些具有相同参数的递归调用出现在多个不同节点时，说明存在重复调用；

    ```text
    以斐波那契数列为例:
            f(5)
        f(4)    f(3)
    f(3) f(2)   ...
    ```

  - 什么情况下可以使用**记忆化搜索**?

    > 这个问题的本质, 实际上是判断问题本身是不是一个动态规划问题; 能用动态规划解决的问题必须具备**无后效性**, 只要具备这个特性, 就可以使用记忆化搜索; 
    >
    > > 无后效性: 某阶段的状态一旦确定, 则此后过程的决策不再受此前各种状态及决策的影响. 

原文总结了 4 种常见的递归过程，基本能覆盖大部分场景：

##### 1. 自底向上的递归/迭代

- 最常见的递归模型, 一般过程:

  1. 确定初始状态 (递归基), 即 $k=0$ 时刻的状态）
  2. 假设已知 $k-1$ 及其之前时刻的状态，推导 $k$ 时刻的状态；

  > 有时候可能是自顶向下, 本质是一样的;

##### 示例 1: 跳台阶 (一维)

> [70. 爬楼梯 - 力扣（LeetCode）](https://leetcode.cn/problems/climbing-stairs/)

- 该模型下的一维问题几乎都是 "**跳台阶**" 问题的变体;

```python
class Solution:
    def climbStairs(self, n: int) -> int:

        from functools import lru_cache

        @lru_cache
        def dfs(i):  # 爬 i 阶存在的方法数
            if i <= 1: return 1
            # 爬 i 阶的方法数 = 爬 i - 1 阶的方法数 + 爬 i - 2 阶的方法数
            return dfs(i - 1) + dfs(i - 2)

        return dfs(n)
```

<details><summary><b>记忆化搜索 (不使用标准库)</b></summary>


```python
class Solution:
    def climbStairs(self, n: int) -> int:

        cache = dict()  # 缓存
        
        def dfs(i):
            if i in cache: return cache[i]  # 搜索"记忆"
            if i <= 1: ret = 1
            else: ret = dfs(i - 1) + dfs(i - 2)
            cache[i] = ret  # "记忆"
            return ret

        return dfs(n)
```

<details><summary><b>迭代写法 (无滚动优化)</b></summary>


```python
class Solution:
    def climbStairs(self, n: int) -> int:
        
        dp = [0] * (n + 1)

        for i in range(n + 1):
            if i <= 1: dp[i] = 1  # if i <= 1: return 1
            else: dp[i] = dp[i - 1] + dp[i - 2]  # dfs(i - 1) + dfs(i - 2)
        
        return dp[-1]
```

##### 示例 2: 解码方法 (一维, "跳台阶" 变体)

> [91. 解码方法 - 力扣（LeetCode）](https://leetcode.cn/problems/decode-ways/)

- 有限制的 "跳台阶" 问题;

```python
class Solution:
    def numDecodings(self, s: str) -> int:

        from functools import lru_cache

        @lru_cache
        def dfs(i):  # s[0: i+1] 的解码方法数
            # 最容易出错的点, 以 0 开头的字符串不存在相应的编码
            if i <= 0: return int(s[0] != '0')

            ret = 0
            if '1' <= s[i] <= '9':  # 如果 s[i] 在 0~9, 存在相应的编码
                ret += dfs(i - 1)  # s[i-1] == 1 和 s[i-2] 的特殊讨论
            if s[i - 1] == '1' or s[i - 1] == '2' and '0' <= s[i] <= '6':
                ret += dfs(i - 2)
            
            return ret
        
        return dfs(len(s) - 1)
```

<details><summary><b>迭代写法</b></summary>


```python
class Solution:
    def numDecodings(self, s: str) -> int:
        
        # if s[0] == '0': return 0

        dp = [0] * (len(s) + 1)
        # dp[-1] = dp[0] = int(s[0] != '0')
        
        # 注意 i 的范围与递归中一致, 这里利用了 python 中 list[-1] 特性, 避免了下标的调整
        for i in range(-1, len(s)):
            # 下面就是把递归中的代码搬过来
            if i <= 0:  # 如果把这一段拿到循环外, 需要调整 i 的遍历范围
                dp[i] = int(s[0] != '0')
                continue
            dp[i] = 0
            if '1' <= s[i] <= '9':
                dp[i] += dp[i - 1]
            if s[i - 1] == '1' or s[i - 1] == '2' and '0' <= s[i] <= '6':
                dp[i] += dp[i - 2]
        
        return dp[len(s) - 1]
```

##### 示例 3: N 个骰子的点数 (一维, "跳台阶" 变体)

> [剑指 Offer 60. n个骰子的点数 - 力扣（LeetCode）](https://leetcode.cn/problems/nge-tou-zi-de-dian-shu-lcof/)

- 进阶版 "跳台阶", 相当于每次能跳的步数更多;

```python
class Solution:
    def dicesProbability(self, n: int) -> List[float]:

        def dfs(k):
            if k == 1:
                return [1] * 7

            dp_pre = dfs(k - 1)
            dp = [0] * (k * 6 + 1)

            for i in range(1 * (n - 1), 6 * (n - 1) + 1):  # n - 1 个骰子的点数范围
                for d in range(1, 7):  # 当前骰子掷出的点数 (跳的台阶数)
                    dp[i + d] += dp_pre[i]

            return dp

        dp = dfs(n)
        return [x / (6 ** n) for x in dp[n:]]
```

<details><summary><b>迭代写法</b></summary>


```python
class Solution:
    def dicesProbability(self, n: int) -> List[float]:

        dp = [1] * 7

        for k in range(2, n + 1):
            dp_pre = dp
            dp = [0] * (k * 6 + 1)
            for i in range(1 * k, 6 * k + 1):  # n 个骰子的点数范围
                for d in range(1, 7):  # 当前骰子掷出的点数
                    if 1 * (k - 1) <= i - d <= 6 * (k - 1):
                        dp[i] += dp_pre[i - d]

        return [x / (6 ** n) for x in dp[n:]]
```




