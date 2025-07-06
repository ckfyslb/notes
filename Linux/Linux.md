## Linux

### 1. 常用指令

```
cd            切换目录  (eg: cd ../.. 表示进入上级目录的上级目录；cd - 表示进入上一次的目录)

ls　　         显示文件或目录
     -l       列出文件详细信息 l(list)   (有些 ll 可以等于 ls -l)
     -a       列出当前目录下所有文件及目录，包括隐藏的 a(all)
     -h       文件大小显示为人看的格式
     
pwd           显示当前目录(print woking directory)     

mkdir         创建目录
     -p       创建多级目录
     
touch         创建空文件

cat [filename]  查看文件内容（-n 显示行号）

head/tail    显示文件头、尾内容，eg：head -n i [filename] 查看文件前i行，默认10行
     
more/less     同：都可以分页显示文本文件内容(全文)，按q退出查看，使用space和b(back)或者page up和page down翻页
	          异：more还可以通过上↑下↓键翻页，翻到文本最后会自动退出
	             less则可以通过上↑下↓键逐行浏览，并且翻到文本最后不会自动退出

cp            拷贝，\cp可以强制覆盖同名文件

mv            移动或重命名

rm            删除文件
     -r       递归删除，可删除子目录及文件
     -f       不询问，强制删除
     
rmdir         删除空目录     

find          在文件系统中搜索某文件
  参数
    -name     指定文件名查找   eg：find . -name filename 在当前目录下递归查找文件
    -user     查找属于指定用户名的所有文件   eg：find /opt -user nobody  在/opt查找属于nobody用户的文件
    -size     按照指定文件大小查找   eg：find / -size +200M 查找大于200M的文件
                                  （+n大于 -n小于 n等于）  （单位k，M，G等）
which         查看命令或可执行文件的位置，eg：which ls

locate        快速定位文件路径。首次使用时先执行 updatedb 创建数据库，再执行 locate filename

grep          在文本文件中查找某个字符串
        -i：忽略大小写进行匹配。
        -v：反向查找，只打印不匹配的行。
        -n：显示匹配行的行号。
        -r：递归查找子目录中的文件。
        -l：只打印匹配的文件名。
        -c：只打印匹配的行数。
             eg：1. grep -r "str" ./filename  在 某文件中 递归查找 某字符串
                 2. cat file | grep "str"
    
echo          1、将字符串输出到终端   echo [options] [stringS]     eg: echo "Hello World"
                 除了字符串，还可以输出变量:
                 name="John"  # 定义变量
                 echo "My name is $name"  # 在变量名前使用"$"符号引用变量
                 也可以使用命令替换:
                 echo "Today is $(date)"  # 可以将date命令输出的时间作为字符串输出
                 输出: Today is xx年 xx月 xx日 星期x xx:xx:xx CST    # $(date)被替换为了date命令的输出结果
              2、将字符串重定向到文件  echo 字符串 > 文件名    eg: echo "Hello World" > hello.txt
                 如果hello.txt不存在，echo就会自动创建；如果存在，echo会覆盖原有内容
                 使用双大于号(>>)会追加到文件末尾  eg: echo "Hello World" >> hello.txt
                 
curl命令：
1）基本使用
curl http://www.linux.com
执行后，www.linux.com 的 html 就会显示在屏幕上了
（Ps：由于安装linux的时候很多时候是没有安装桌面的，也意味着没有浏览器，因此这个方法也经常用于测试一台服务器是否可以到达一个网站）
2）保存访问的网页
2-1）使用 linux 的重定向功能保存
curl http://www.linux.com >> linux.html
2-2）使用 curl 的内置 option:-o(小写) 保存网页
$ curl -o linux.html http://www.linux.com
执行完成后会显示如下界面，显示 100% 则表示保存成功
% Total % Received % Xferd Average Speed Time Time Time Current
Dload Upload Total Spent Left Speed
100 79684 0 79684 0 0 3437k 0 --:--:-- --:--:-- --:--:-- 7781k
2-3）使用 curl 的内置 option:-O(大写) 保存网页中的文件
要注意这里后面的 url 要具体到某个文件，不然抓不下来
curl -O http://www.linux.com/hello.sh

wc            统计文本中行数(-l)、字数(-w)、字符数(-m)
              eg：wc [filename]       输出文件行数、字数、字符数
                  wc -l [filename]    只输出行数
              组合使用：cat [filename] | wc -l

file          查看文件属性

tree          树形结构显示目录，需要安装tree包

ln            创建软链接，eg：ln -s [源文件或目录] [链接名]

mount         挂载
umount        取消挂载
               eg：
                  mkdir /mountdir                # 挂载点必须存在
                  mount /dev/sdb1 /mountdir      # 将外部设备 /dev/sdb1 挂载到 mountdir
                  cp /mnt/hgfs/Ubuntu-Share/uimage /mountdir        # 可以对外部设备进行种种操作
                  umount /mountdir                # 使用完以后，一定要取消挂载
lsblk         查看文件的挂载情况

history       查看执行过的指令

date          显示当前日期
date +%Y
date +%m
date +%d
date "+%Y-%m-%d %H:%M:%S"
date -s "TIME" 设置系统当前时间，eg：date -s "2020-11-03 20:02:10"

cal          显示本月日历（calendar）
cal 2025     显示2025年整年的日历

```

### 2. 系统管理命令

```
ip 用户名          查看用户信息，eg：ip root （显示uid、gid、组）
stat              显示指定文件的详细信息，比ls更详细
who               显示在线登陆用户
whoami            显示当前操作用户（或who am i，更详细，除了用户名还显示登陆时间）
hostname          显示主机名
uname             显示系统信息
top               动态显示当前耗费资源最多进程信息
ps                显示瞬间进程状态
                  eg：ps -aux | grep <进程名>
                      ps -ef | grep <进程名>
du                查看目录大小 du -h /home带有单位显示目录信息
df                查看磁盘大小 df -h 带有单位显示磁盘信息
ifconfig          查看网络情况
ping              测试网络连通
netstat           显示网络状态信息
man               命令不会用了，找男人  如：man ls
clear             清屏
alias             对命令重命名 如：alias showmeit="ps -aux" ，另外解除使用unaliax showmeit
kill (-9) <PID>   杀死进程，-9立即强制终止进程（可以先用ps或top命令查看进程的id，然后再用kill命令杀死进程）
lsof -i:端口号     查看端口占用情况
```

### 3. 打包压缩相关命令

```
gzip、gunzip：只能处理 *.gz 文件

zip、unzip：
	zip -r [file_dir]  递归压缩
	unzip -d [dest_dir] [file_dir]  指定解压后文件的存放目录

tar:                 打包压缩
     -c              创建新的归档文件（打包）
     -x              从归档文件中提取文件（解包）
     -z              gzip压缩文件
     -j              bzip2压缩文件
     -v              显示压缩或解压缩过程 v(view)k
     -f              指定归档文件名
压缩：
tar -cvf /home/abc.tar /home/abc/           压缩成.tar文件; 将/home/下的abc文件打包成/home/下的abc.tar
tar -jcvf /home/abc.tar.bz2 /home/abc/      用 bzip2 压缩 abc 目录
tar -zcvf /home/abc.tar.gz /home/abc/       用 gzip 压缩 abc 目录
tar -zcvf pc.tar.gz /home/111.txt /home/222.txt    用 gzip 压缩多个文件
解压缩：（压缩命令的“c”换成“x”）
tar -xvf pc.tar                  解压.tar文件
tar -jxvf pc.tar.bz2             解压.tar.bz2文件
tar -zxvf pc.tar.gz              解压.tar.gz文件 到当前目录下
tar -zxvf pc.tar.gz -C /home     解压.tar.gz文件 到指定目录下
```

### 4. 关机/重启机器

```
shutdown
     -r          关机重启（reboot）
     -h          关机不重启（halt）
     now         立刻关机
     只输入shutdown=shutdown -h 1（1分钟后关机）
halt             关机
reboot           重启
```

### 5. Linux管道

```
将一个命令的标准输出作为另一个命令的标准输入。也就是把几个命令组合起来使用，后一个命令除以前一个命令的结果。
例：grep -r "close" /home/* | more       在home目录下所有文件中查找，包括close的文件，并分页输出。
```

### 6. Linux软件包管理

```
（1）
dpkg (Debian Package)管理工具，软件包名以.deb后缀。这种方法适合系统不能联网的情况下。

比如安装tree命令的安装包，先将tree.deb传到Linux系统中。再使用如下命令安装。
sudo dpkg -i tree_1.5.3-1_i386.deb         安装软件
sudo dpkg -r tree                          卸载软件

注：将tree.deb传到Linux系统中，有多种方式。VMwareTool，使用挂载方式；使用winSCP工具等；

（2）
APT（Advanced Packaging Tool）高级软件工具。这种方法适合系统能够连接互联网的情况。

依然以tree为例
sudo apt-get install tree                 安装tree
sudo apt-get remove tree                  卸载tree
sudo apt-get update                       更新软件列表，从软件源获得系统上所有包的最新信息，并不会下载或者安装
sudo apt-get upgrade                      把这些包下载和升级到最新版本

将.rpm文件转为.deb文件
.rpm为RedHat使用的软件格式。在Ubuntu下不能直接使用，所以需要转换一下。
sudo alien abc.rpm


centos：
rpm -ivh a.rpm     安装a软件rpm包（也可以一次多个rpm -ivh a.rpm b.rpm c.rpm）
rpm -Uvh a.rpm     升级a软件，如果没安装a则直接安装
rpm -evh a         卸载通过rpm安装的a软件
rpm -qa            显示系统已安装过的全部rpm软件包
rpm -qa | grep a   显示安装的 a 软件的rpm包
rpm -ql a          查询 a 软件的安装路径

   选项	         说明
-a	            显示所有软件包
-f	            显示文件或命令属于哪个软件包
-i, --install	安装一个包
-v	            显示执行过程信息
-vv	            显示执行过程详细信息
-h	            打印 #，显示安装进度
-U, --update	升级一个包
-e, --erase	    卸载一个包
-q, --query	    查询一个包
-V, --verify	校验一个包
-l	            显示软件包的文件列表
-p	            显示指定的软件包信息

-nodeps：不检测依赖性安装。软件安装时会检测依赖性，确定所需的底层软件是否安装，如果没有安装则会报错。如果不管依赖性，想强制安装，则可以使用这个选项。注意，这样不检测依赖性安装的软件基本上是不能使用的，所以不建议这样做。
-replacefiles：替换文件安装。如果要安装软件包，但是包中的部分文件已经存在，那么在正常安装时会报"某个文件已经存在"的错误，从而导致软件无法安装。使用这个选项可以忽略这个报错而覆盖安装。
-replacepkgs：替换软件包安装。如果软件包已经安装，那么此选项可以把软件包重复安装一遍。
-force：强制安装。不管是否已经安装，都重新安装。也就是 -replacefiles 和 -replacepkgs 的综合。
-test：测试安装。不会实际安装，只是检测一下依赖性。
-prefix：指定安装路径。为安装软件指定安装路径，而不使用默认安装路径。
```

### 7. vim使用

```
vim 文件名         打开文件
vim三种模式：命令模式、插入模式、编辑模式。使用 ESC 或 i(insert) 或：来切换模式。

:q                退出
:q!               强制退出
:wq               保存并退出
:set nu           显示行号（或set number）
:set nonu         隐藏行号（或set nonumber）
/关键字            查找关键字，如/apache，在文档中查找apache，按n跳到下一个，shift+n上一个
yy                复制光标所在行
5yy               复制5行
dd                删除光标所在行
5dd               删除5行
p                 粘贴
u                 撤回操作
```

### 8. 用户及用户组管理

```
/etc/passwd    存储用户账号
/etc/group     存储组账号
/etc/shadow    存储用户账号的密码
/etc/gshadow   存储用户组账号的密码
useradd 用户名
userdel 用户名
adduser 用户名
groupadd 组名
groupdel 组名
passwd root     给root设置密码
su root
su - root 
/etc/profile     系统环境变量
bash_profile     用户环境变量
.bashrc              用户环境变量
su user              切换用户，加载配置文件.bashrc
su - user            切换用户，加载配置文件/etc/profile ，加载bash_profile
```

### 9. 更改文件的用户及用户组

```
sudo chown [-R] 用户名[:组名] {File|Directory}
（-R：处理指定目录以及其子目录下的所有文件，即递归）
例如：还以jdk-7u21-linux-i586.tar.gz为例。属于用户hadoop，组hadoop
要想切换此文件所属的用户及组。可以使用命令。
sudo chown root:root jdk-7u21-linux-i586.tar.gz
```

### 10. 控制用户对文件的权限

```
chmod [-cfvR] [ugoa...][[+-=][rwxX] 文件名

u 表示该文件的拥有者，g 表示与该文件的拥有者属于同一个群体(group)者，o 表示其他以外的人，a 表示这三者皆是。
+ 表示增加权限、- 表示取消权限、= 表示唯一设定权限。
r 表示可读取（=4），w 表示可写入（=2），x 表示可执行（=1），X 表示只有当该文件是个子目录或者该文件已经被设定过为可执行。
-c : 若该文件权限确实已经更改，才显示其更改动作
-f : 若该文件权限无法被更改也不要显示错误讯息
-v : 显示权限变更的详细资料
-R : 对目前目录下的所有文件与子目录进行相同的权限变更(即以递归的方式逐个变更)

例：
1. 将文件 file1.txt 设为所有人皆可读取 :
chmod ugo+r file1.txt 或 chmod a+r file1.txt
2. 所有用户设置所有权限
chmod a=rwx file 或 chmod 777 file
```

### 11

1. linux shell 重定向 2>&1

   这里的**&**没有固定的意思

   放在**>**后面的**&**，表示重定向的目标不是一个**文件**，而是一个**文件描述符**，内置的文件描述符如下

   ```
   1 => stdout
   2 => stderr
   0 => stdin
   ```

   换言之 **2>1** 代表将**stderr**重定向到当前路径下文件名为**1**的**regular file**中，而**2>&1**代表将**stderr**重定向到**文件描述符**为**1**的文件(即**/dev/stdout**)中，这个文件就是**stdout**在**file system**中的映射

2. nohup

   nohup是no hang up的缩写，意思是不挂断。
   nohup命令，在默认情况下（非重定向时），会输出一个名叫nohup.out的文件到当前目录下，如果当前目录的nohup.out文件不可写，输出重定向到$HOME/nohup.out文件中。

3. linux 命令末尾加 &

1、&才是后台运行
2、nohup ： 不挂断的运行，注意并没有后台运行的功能，就是指，用nohup运行命令可以使命令永久的执行下去，和用户终端没有关系，例如我们断开SSH连接都不会影响他的运行，注意了nohup没有后台运行的意思；

&符号表示将该命令或脚本放入后台运行。 即&方式启动会有进程号，使用Ctrl+C程序不会中断，但终端关闭后运行会中断。




在命令的末尾加个&符号后，程序可以在后台运行，但是一旦当前终端关闭（即退出当前帐户），该程序就会停止运行。那假如说我们想要退出当前终端，但又想让程序在后台运行，该如何处理呢？

实际上，这种需求在现实中很常见，比如想远程到服务器编译程序，但网络不稳定，一旦掉线就编译就中止，就需要重新开始编译，很浪费时间。

在这种情况下，我们就可以使用nohup命令。nohup就是不挂起的意思( no hang up)。该命令的一般形式为：

```text
nohup ./test &
```



[linux - 一口气搞懂「文件系统」，就靠这 25 张图了 - 个人文章 - SegmentFault 思否](https://segmentfault.com/a/1190000023615225#item-2-7)

[wget命令使用及参数详解-CSDN博客](https://blog.csdn.net/u011598193/article/details/99412491)

[curl 的用法指南 - 阮一峰的网络日志](https://www.ruanyifeng.com/blog/2019/09/curl-reference.html)

[Linux curl命令最全详解-CSDN博客](https://blog.csdn.net/angle_chen123/article/details/120675472)





## Shell

脚本以`#!/bin/bash`开头，告诉系统使用的shell为bash

创建脚本：使用vi或者vim创建扩展名为sh的文件（扩展名不影响脚本执行），如test.sh

执行脚本：

```shell
(1) 使用绝对路径执行(先给脚本文件增加可执行权限)
./test.sh
(2) sh test.sh
(3) source test.sh
```

圆括号()和反引号``可以用来接收语句执行结果

### 变量

```shell
使用变量：
uname="xiaobai"  # 定义变量
echo $uname    # 通过$符号使用变量
echo ${uname}  # 可以加花括号

只读变量：
readonly uname  # uname将不可修改

删除变量
var="abc"
unset var  # 变量被删除后不能再次使用，unset命令不能删除只读变量
```

**特殊变量：** 有一些特殊变量在 Shell 中具有特殊含义，例如 **$0** 表示脚本的名称，**$1**, **$2**, 等表示脚本的参数。**$#**表示传递给脚本的参数数量，**$?** 表示上一个命令的退出状态等。

### 字符串

### 数组









