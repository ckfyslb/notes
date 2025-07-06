### 一、使用镜像部署本地源

1. 下载`iso`文件

   - 在`Linux`文件系统中创建目录，用于存放`iso`文件（名字自定义）
     ```shell
     mkdir /opt/localrepo

   - 下载`CentOS`的`iso`文件，上传到`/opt/localrepo/`

2. 创建挂载目录（名字自定义）

   ```shell
   mkdir /mnt/localrepo
   ```

3. 挂载

   ```shell
   mount -o loop -t iso9660 /opt/localrepo/CentOS-7-x86_64-Everything-2009.iso /mnt/localrepo/
   ```

   - `-o loop`：
     - `-o`：是挂载选项（options），用于指定文件系统的额外参数或行为
     - 使用 loop 模式将一个档案当成硬盘分割挂上系统，用这种方法可以将一般网络上可以找到的 Linux 光 碟 ISO 档在不烧录成光碟的情况下检视其内容
   - `-t iso9660`：
     - `-t`：（--types）限制文件系统类型集合
     - `iso9660`：光盘文件格式

4. 移除或备份原来的 `xx.repo` 文件

   ```shell
   rm -rf /etc/yum.repos.d/*
   ```

   or

   ```shell
   cd /etc/yum.repos.d/
   mv xx.repo xx.repo.back
   ```

5. 编辑新的`repo`文件

   ```shell
   vim /etc/yum.repos.d/local.repo
   ```

   添加以下文本后保存

   ```shell
   [centos]                          # 中括号为固定格式，里面的内容只是一个标识
   name=centos repo                  # yum源的名称，用于描述这个源
   baseurl=file:///mnt/localrepo/    # 本地yum源的位置，指向挂载点目录
   gpgcheck=0                        # 不进行gpg签名检查
   enabled=1                         # 启用该yum源 
   ```

6. 重新加载`yum`

   - 清除缓存

   ```shell
   yum clean all
   ```

   - 生成新的缓存

   ```shell
   yum makecache
   ```

   - 列出当前系统中已配置和启用的`yum`仓库进行查看

   ```shell
   yum repolist
   ```

7. 开机自动挂载

   ```shell
   # 挂载命令添加到/etc/rc.d/rc.local
   sed -i '$a\mount -o loop -t iso9660 /opt/localrepo/CentOS-7-x86_64-Everything-2009.iso /mnt/localrepo/' /etc/rc.d/rc.local
   
   # rc.local添加可执行权限
   chmod +x /etc/rc.d/rc.local
   ```

   - `sed -i`：就地修改文件，将命令`'mount -o loop -t iso9660 /opt/localrepo/CentOS-7-x86_64-Everything-2009.iso /mnt/localrepo/'`添加到`/etc/rc.d/rc.local`文件末尾
   - `$a\`：在文件的最后一行之后追加一行新内容
     - `$`：行尾定位符，表示文件的最后一行
     - `a`：在`sed`命令中表示追加命令（append），表示在匹配的行之后插入新内容
     - `\`：用于每行末尾续

8. 重启机器测试

   ```shell
   shutdown -r now
   ```

   

### 二、下载所有网络源的rpm包作为本地源

使用 `reposync` 下载所有`yum`源的包

```shell
# 1. 安装，reposync 来自 yum-utils
yum -y install yum-utils

# 2. 创建本地存放目录
mkdir myrepo

# 3. 查看网络yum源仓库标识
$ yum repolist
$ Loading mirror speeds from cached hostfile
     * base: mirrors.aliyun.com
     * epel: repo.jing.rocks
     * extras: mirrors.aliyun.com
     * updates: mirrors.aliyun.com

# 4. 根据每个据仓库标识下载安装包到本地目录
reposync -r base -p /path/to/myrepo/
reposync -r epel -p /path/to/myrepo/
reposync -r extras -p /path/to/myrepo/
reposync -r updates -p /path/to/myrepo/
```

-  `-r`：指定已经本地已经配置的 `yum` 仓库的 `repo` 源的名称
-  `-p`：指定下载的路径



### 三、扩展仓库

下载的安装包必须构建 `repodata` 才能使用

1. 安装`createrepo`
   ```shell
   # yum命令安装
   yum isntall createrepo
   ```

   or

   ```shell
   # 下载rpm包安装
   rpm -ivh createrepo-x.x.x-xx.elx.noarch.rpm
   ```

2. 使用`createrepo`生成`repodata`数据
   ```shell
   # 生成base的repodata数据
   createrepo /path/to/myrepo/base
   ```

3. 编辑`repo`文件
   ```shell
   vim /etc/yum.repos.d/local.repo
   
   [base]                               # 中括号为固定格式，里面的内容只是一个标识
   name=base local repo                 # yum源的名称，用于描述这个源
   baseurl=file:///path/to/myrepo/base  # 指定baseurl的路径，本机使用file指定即可
   gpgcheck=0                           # 不进行gpg签名检查
   enabled=1                            # 启用该yum源 
   # 添加epel、extras、updates同理
   ```

4. 重新加载`yum`

   ```shell
   # 清除缓存
   yum clean all
   
   # 生成新的缓存
   yum makecache
   ```


5. `base、epel、extras、updates`文件夹添加新的`rpm`包后更新

   ```shell
   createrepo --update /path/to/myrepo/base
   ```

   



> 参考：
>
> 1. [centos7配置离线yum源_centos 离线yum源-CSDN博客](https://blog.csdn.net/sinat_32724581/article/details/110119231?spm=1001.2014.3001.5506)
> 2. [Centos7配置本地Yum源以及网络YUM源（保姆级）_centos7yum源-CSDN博客](https://blog.csdn.net/2302_77228926/article/details/139559880?spm=1001.2014.3001.5506)
> 3. [下载整个Yum源的所有安装包到本地指定目录_yum下载到指定目录-CSDN博客](https://blog.csdn.net/qq_44895681/article/details/127616357?spm=1001.2014.3001.5506)
> 4. [离线YUM仓库搭建-CSDN博客](https://blog.csdn.net/weixin_41934601/article/details/105619688)
