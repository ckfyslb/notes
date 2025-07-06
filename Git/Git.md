## Git 简介

**一款分布式版本控制系统**

<img src="https://kuangstudy.oss-cn-beijing.aliyuncs.com/bbs/2022/06/01/kuangstudyc7c10da1-a210-4728-8065-0573506ea66b.png" alt="img" style="zoom:80%;" />

### 工作区域

- **工作区**：平时存放项目代码的地方。
- **暂存区(Stage/Index)：**暂存区，用于临时存放你的改动，事实上它只是一个文件，保存即将提交到文件列 表信息
- **版本库：**又称本地仓库，这个不算工作区，而是 Git 的版本库，里面有你提交到所有版本的数据。
- **远程仓库**：托管代码的服务器

### 工作流程







## 安装

### Windows

官网下载：[Git - Downloads (git-scm.com)](https://git-scm.com/downloads)

镜像安装：[CNPM Binaries Mirror (npmmirror.com)](https://registry.npmmirror.com/binary.html?path=git-for-windows/)

全部以默认选项安装

### Linux（Ubuntu）

```shell
$ sudo apt-get install git
```

## 配置

**配置名字邮箱是必须的：**

```shell
$ git config --global user.name "名字"
$ git config --global user.email "email地址"
```

`git config`命令的`--global`参数表示本台机器上所有的 Git 仓库都会使用这个配置，去掉`--global`参数进行配置即可在某个特定的项目中使用其他名字或邮箱。

查看配置信息：

```shell
$ git config --list
```

## 创建版本库

即仓库（repository）。这个目录里面的所有文件都可以被 Git 管理起来，每个文件的修改、删除，Git都能跟踪，以便任何时刻都可以追踪历史，或者在将来某个时刻可以“还原”。

1. 创建一个空目录

   ```shell
   ```

   

2. 在目录下通过`git init`创建仓库，使得该目录成为可以被 Git 管理的仓库



## 常用命令

![img](https://kuangstudy.oss-cn-beijing.aliyuncs.com/bbs/2022/06/01/kuangstudy3c1708a2-4e6e-4433-9c51-6d8b4257ce72.png)

push

pull

commit

add

checkout

fetch/clone



## 在 IDE 上使用 Git





## 参考资料

[Git教程 - 廖雪峰的官方网站 (liaoxuefeng.com)](https://www.liaoxuefeng.com/wiki/896043488029600)

[视频同步笔记：狂神聊Git (qq.com)](https://mp.weixin.qq.com/s/Bf7uVhGiu47uOELjmC5uXQ)

https://blog.csdn.net/qq_56591814/article/details/119841785?spm=1001.2014.3001.5502





## Git基础命令

[视频同步笔记：狂神聊Git (qq.com)](https://mp.weixin.qq.com/s/Bf7uVhGiu47uOELjmC5uXQ)

[Git入门学习-KuangStudy-文章](https://www.kuangstudy.com/bbs/1532247011263672321)

[Git使用教程-KuangStudy-文章](https://www.kuangstudy.com/bbs/1707321856795000833)

[Git命令总结-KuangStudy-文章](https://www.kuangstudy.com/bbs/1557209705922093057)

[Git基本使用-KuangStudy-文章](https://www.kuangstudy.com/bbs/1763817578998136834)

### 1.拉取仓库修改后提交

```shell
拉取仓库
git clone <仓库地址https://******.git>

修改完后
1.写入缓存区（.git目录下的index文件）
git add .
2.将缓存区内容添加到本地仓库
git commit -m '提交描述'
3.将远程仓库拉到本地仓库git版本控制
git pull origin master
4.将本地分支版本上传到远程合并
git push origin master
```

### 2.新建项目上传到仓库

```shell
先创建版本库
git init

修改完后
1.写入缓存区（.git目录下的index文件）
git add .
2.将缓存区内容添加到本地仓库
git commit -m '提交描述'
3.连接远程仓库
git remote add origin <仓库地址https://******.git>
4.将远程仓库拉到本地仓库git版本控制（ push 之前先 pull ，这样不容易冲突。）
git pull origin master
5.将本地分支版本上传到远程合并
git push origin master
```



```shell
git stat  # 查看代码改动情况、提交情况等信息

# git push
git push 命令的基本格式如下：
git push <远程主机名> <本地分支名>:<远程分支名>
例如，将本地的 master 分支推送到远程主机 origin 的 master 分支：
git push origin master
这相当于：
git push origin master:master
如果本地分支名和远程分支名相同，可以省略冒号及其后面的部分。

-u 参数：作用是为当前分支设置上游（upstream）分支
当使用 git push -u 命令时，Git 会为你推送的分支设置一个上游分支，这意味着 Git 会记录你的分支应该推送到远程仓库的哪个分支。这样做的好处是，下次当你再次推送到同一个远程分支时，你可以简单地使用 git push 命令，而不需要指定远程仓库和分支。这是因为 Git 已经“知道”你想要推送到哪里，以及你想要从哪里拉取更新。

例如，如果你执行了 git push -u origin master，Git 不仅会将你的 master 分支推送到远程仓库 origin 的 master 分支，而且还会在 .git/config 文件中记录这个关联。这样，当你在 master 分支上执行 git pull 时，Git 会根据这个配置来确定默认的远程仓库和分支。



# git pull
从远程主机 origin 的 next 分支获取更新，并与本地的 master 分支合并：
git pull origin next:master
如果远程分支（如 next）要与当前分支合并，可以省略冒号后面的部分：
git pull origin next
如果当前分支与远程分支存在追踪关系，可以省略远程分支名：
git pull origin
如果当前分支只有一个追踪分支，连远程主机名都可以省略：
git pull
使用 --rebase 选项进行拉取并采用 rebase 模式：
git pull --rebase origin next:master

git fetch 和 git pull 的区别：
git fetch：从远程获取最新版本到本地，不会自动合并。可以查看更新情况，然后再决定是否合并。
git fetch origin master
git log -p master..origin/master
git merge origin/master
git pull：从远程获取最新版本并合并到本地，相当于 git fetch 和 git merge 的组合。
git pull origin master
注意事项：git pull 命令会自动将远程仓库的更改合并到当前分支中，如果在合并过程中遇到冲突，需要手动解决冲突。在某些场合，Git 会自动在本地分支与远程分支之间建立追踪关系，例如在 git clone 的时候，所有本地分支默认与远程主机的同名分支建立追踪关系。
```



### git本地操作

  在进行任何 Git 操作之前，都要先切换到 Git 仓库目录，也就是切换到项目的文件夹目录下。Git 所有的操作命令开头都要以 git 开头。

首先创建一个新的文件夹并初始化为仓库

mkdir test （创建文件夹test）
cd test （切换到test目录）
git init（初始化为仓库）
git status(查看状态）

  默认就直接在 master 分支，输入后可以看到a.md 文件Untracked files ，就是说 a.md 这个文件还没有被跟踪，没有提交在 git 仓库里呢，可以使用 git add a.md提交的文件。

提交文件git add
  再输入git status，此时提示以下文件 Changes to be committed ， 意思就是 a.md 文件等待被提交，当然你可以使用 git rm --cached 文件名 这个命令去移除这个缓存。或者 git commit -a -m "first commmt"移除名为 "first commmt"的提交。（你妹的，本地也给我删了）
git commit
  接着我们输入 git commit -m "first commit" ，这个命令什么意思呢？ commit 是提交的意思，-m 代表是提交信息，执行了以上命令代表我们已经正式进行了第一次提交。这个时候再输入 git status ，会提示 nothing to commit。

查看提交记录git log
  这个时候我们输入 git log 命令，会看到如下：
git log 命令可以查看所有产生的 commit 记录，所以可以看到已经产生了一条 commit 记录，而提交时候的附带信息叫 ‘first commit’ 。

git commit -am
  看到这里估计很多人会有疑问，我想要提交直接进行 commit 不就行了么，为什么先要再 add一次呢？首先 git add 是先把改动添加到一个「暂存区」，你可以理解成是一个缓存区域，临
时保存你的改动，而 git commit 才是最后真正的提交。这样做的好处就是防止误提交，当然
也有办法把这两步合并成一步，使用git commit -am就可以了。

新建分支git branch a
  branch 即分支的意思，多人协作中很重要。每个人建立自己的分支，互不影响，最后合并。
  执行 git init 初始化git仓库之后会默认生成一个主分支 master（默认分支），也基本是实际开发正式环境下的分支。一般情况下 ，不要在master 分支上直接操作的。
  执行 git branch a 新建了 a 分支（分支 a 跟分支 master 是一模一样的内容）可以输入 git branch 查看下当前分支情况。

切换分支git checkout a

新建并自动切换分支git checkout -b a

**删除分支git branch -d **
  分支新建错了，或者a分支的代码已经顺利合并到master 分支来了，那么a分支没用了，需要删除。如果a分支的代码还没有合并到master，你执行 git branch -d a 是删除不了的，它会智能的提示你a分支还有未合并的代码。

强制删除分支git branch -D

合并分支git merge
  第一步是切换到 master 分支
  第二步执行 git merge a ，合并a分支。但有时候会有冲突合并失败

添加版本标签git tag
  git tag v1.0 就代表我在当前代码状态下新建了一个v1.0的标签，输入 git tag 可以查看历史 tag 记录。执行 **git checkout v1.0 **，这样就顺利的切换到 v1.0 tag的代码状态了。







## GitHub

### GitHub搜索语句

以下是GitHub搜索语句的**系统整理**，涵盖常用语法和实战技巧，助你精准定位开源项目或代码资源：

------

### 🔍 一、基础搜索语法

1. **关键词限定范围**
   - `in:name`：搜索仓库名称（如 `in:name spring boot`）
   - `in:description`：搜索项目描述（如 `in:description "machine learning"`）
   - `in:readme`：搜索README文件（如 `in:readme installation`）
   - `in:file`或`path:`：搜索文件内容或路径（如 `in:file "快速排序" path:/src/`）
2. **常用限定符**
   - `language:`：按编程语言过滤（如 `language:python`）
   - `user:`/`org:`：按用户或组织过滤（如 `user:torvalds`）
   - `filename:`：按文件名搜索（如 `filename:README.md`）

------

### ⚙️ 二、高级过滤条件

| **过滤类型** | **语法示例**                     | **作用**               |
| ------------ | -------------------------------- | ---------------------- |
| 活跃度过滤   | `pushed:>2024-01-01`             | 筛选最近更新的项目     |
| 项目热度过滤 | `stars:>5000 forks:>1000`        | 高热度项目             |
| 时间范围过滤 | `created:2023-01-01..2024-01-01` | 按创建时间筛选         |
| 仓库大小过滤 | `size:>5000`                     | 过滤大型仓库（单位KB） |
| 许可证过滤   | `license:mit`                    | 按开源协议过滤         |

------

### 🔗 三、组合搜索技巧

1. **布尔运算符**

   - `AND`/空格：交集（如 `vue AND stars:>1000`）
   - `OR`：并集（如 `path:*.config OR path:*.ini`）
   - `NOT`/`-`：排除（如 `java -spring`）

2. **多条件组合**
   示例：

   ```
   # 查找2023年后更新、Star>1万、Java编写的商城项目
   商城 in:readme stars:>10000 language:java pushed:>2023-01-01 
   ```

   ```
   # 搜索中文用户创建的Python爬虫项目
   location:china language:python in:description 爬虫 
   ```

------

### 💡 四、实战场景案例

1. **找学习资源**

   - 优质教程合集：`awesome python`（Awesome系列资源）
   - 中文文档项目：`in:readme "中文文档" stars:>500`

2. **找企业级项目**

   ```
   # 搜索大厂维护的Java微服务框架
   org:alibaba in:name microservice language:java pushed:>2024-01-01 
   ```

3. **找配置文件中的敏感信息**

   ```
   # 搜索可能泄露的密钥
   (path:*.yml OR path:*.env) AND (api_key OR password) 
   ```

------

### 🛠️ 五、高级工具与技巧

1. **GitHub高级搜索页面**：
   https://github.com/search/advanced
   可视化生成复杂查询，适合新手。
2. **第三方工具**：
   - [Sourcerer](https://sourcerer.io/)：专注代码搜索的引擎
   - [GitHub Trend](https://github.com/trending)：按语言/时间查看热门项目

------

### 💎 附：搜索避坑指南

- **过滤低质量项目**：
  组合`stars`、`pushed`、`forks`确保项目活跃度。
- **规避法律风险**：
  用`license:`明确协议类型，避免商用侵权。
- **精确匹配短语**：
  用双引号包裹词组（如 `"machine learning"`）。

善用上述语法，可快速从GitHub的**3.3亿+仓库**中锁定目标资源。尝试组合不同条件，逐步缩小范围，比直接关键词搜索效率提升10倍以上 ✅。



### git clone 断点续传

```shell
# 克隆时跳过大文件
GIT_LFS_SKIP_SMUDGE=1 git clone https://github

# 再拉取大文件，同时显示进度条（失败后再执行可以断点续传）
git lfs pull
```

