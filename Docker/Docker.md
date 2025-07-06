## Docker介绍

### 1. 什么是 Docker 容器？

Docker 容器 在应用程序层创建抽象并**将应用程序及其所有依赖项打包在一起**。这使我们能够快速可靠地部署应用程序。容器不需要我们安装不同的操作系统。相反，它们**使用底层系统的 CPU 和内存来执行任务**。这意味着任何容器化应用程序都可以在任何平台上运行，而不管底层操作系统如何。我们也可以将容器视为 Docker 镜像的运行时实例。

### 2. 什么是 DockerFile？

Dockerfile 是一个文本文件，里面包含我们需要运行以构建 Docker 镜像的所有命令。Docker 使用 Dockerfile 中的指令自动构建镜像。我们可以docker build用来创建按顺序执行多个命令行指令的自动构建。

### 3. 如何从 Docker 镜像创建 Docker 容器？

为了从镜像创建容器，我们从 Docker 存储库中提取我们想要的镜像并创建一个容器。我们可以使用以下命令：

>  $ docker run -it -d <image_name>

### 4. Docker Compose 可以使用 JSON 代替 YAML 吗？

是的，我们可以对Docker Compose文件使用 JSON 文件而不是YAML

> $ docker-compose -f docker-compose.json up

### 5. 什么是Docker Swarm？

Docker Swarm 是一个容器编排工具，它允许我们跨不同主机管理多个容器。使用 Swarm，我们可以将多个 Docker 主机变成单个主机，以便于监控和管理。

### 6. 如果你想使用一个基础镜像并对其进行修改，你怎么做？

可以使用以下 Docker 命令将镜像从 Docker Hub 拉到本地系统上：

> $ docker pull <image_name>

### 7. 如何启动、停止和终止容器？

要启动 Docker 容器，请使用以下命令：

> $ docker start <container_id>

要停止 Docker 容器，请使用以下命令：

> $ docker stop <container_id>

要终止 Docker 容器，请使用以下命令：

> $ docker kill <container_id>

### 8. Docker 运行在哪些平台上？

Docker 在以下 Linux 发行版上运行：

- CentOS 6+
- Gentoo
- ArchLinux
- CRUX 3.0+
- openSUSE 12.3+
- RHEL 6.5+
- Fedora 19/20+
- Ubuntu 12.04、13.04

Docker 还可以通过以下云服务在生产中使用：

- 微软Azure
- 谷歌计算引擎
- 亚马逊 AWS EC2
- 亚马逊 AWS ECS
- 机架空间

> 提示：我们始终建议您在面试之前进行一些公司研究，要为这个特定问题做准备，请了解公司如何使用 Docker 并在您的答案中包含他们使用的平台。

### 9. 解释 Docker 组件

三个架构组件包括: Docker 客户端、主机和注册表。

Docker 客户端：该组件执行构建和运行操作以与 Docker 主机通信。

Docker 主机：该组件包含 Docker 守护程序、Docker 镜像和 Docker 容器。守护进程建立到 Docker Registry 的连接。

Docker Registry：该组件存储 Docker 镜像。它可以是公共注册表，例如 Docker Hub 或 Docker Cloud，也可以是私有注册表。

### 10. 虚拟化和容器化有什么区别？

**虚拟化**

虚拟化帮助我们在单个物理服务器上运行和托管多个操作系统。在虚拟化中，管理程序为客户操作系统提供了一个虚拟机。VM 形成了硬件层的抽象，因此主机上的每个 VM 都可以充当物理机。

**容器化**

容器化为我们提供了一个独立的环境来运行我们的应用程序。我们可以在单个服务器或 VM 上使用相同的操作系统部署多个应用程序。容器构成了应用层的抽象，所以每个容器代表一个不同的应用。

### 11. 管理程序的功能是什么？

管理程序或虚拟机监视器是帮助我们创建和运行虚拟机的软件。它使我们能够使用单个主机来支持多个来宾虚拟机。它通过划分主机的系统资源并将它们分配给已安装的来宾环境来实现这一点。可以在单个主机操作系统上安装多个操作系统。有两种类型的管理程序：

- Native：本机管理程序或裸机管理程序，直接在底层主机系统上运行。它使我们可以直接访问主机系统的硬件，并且不需要基本服务器操作系统。

- 托管：托管管理程序使用底层主机操作系统。

### 12. 如何构建Dockerfile？

为了使用我们概述的规范创建镜像，我们需要构建一个 Dockerfile。要构建 Dockerfile，我们可以使用以下docker build命令：

> $ docker build

### 13. 使用什么命令将新镜像推送到 Docker Registry？

可以使用以下docker push命令：

> $ docker push myorg/img

### 14.什么是Docker引擎？

Docker Engine 是一种开源容器化技术，我们**可以使用它来构建和容器化我们的应用程序**。Docker Engine 由以下组件支持：

- Docker 引擎 REST API
- Docker 命令行界面 (CLI)
- Docker 守护进程

### 15. 如何访问正在运行的容器？

使用以下命令：

> $ docker exec -it <container_id> bash

### 16.如何列出所有正在运行的容器？

要列出所有正在运行的容器，我们可以使用以下命令：

> $ docker ps

### 17. 描述 Docker 容器的生命周期。

Docker 容器经历以下阶段：

- 创建容器
- 运行容器
- 暂停容器（可选）
- 取消暂停容器（可选）
- 启动容器
- 停止容器
- 重启容器
- 杀死容器
- 销毁容器

### 18. 什么是Docker对象标签？

Docker 对象标签是存储为字符串的键值对。它们使我们能够将元数据添加到 Docker 对象，例如容器、网络、本地守护进程、图像、Swarm 节点和服务。

### 19. 使用Docker Compose时如何保证容器1先于容器2运行？

Docker Compose 在继续下一个容器之前不会等待容器准备就绪。为了控制我们的执行顺序，我们可以使用“取决于”条件，depends_on。这是在 docker-compose.yml 文件中使用的示例：

```yaml
version: "2.4"
services:
 backend:
   build: .
   depends_on:
     - db
 db:
   image: postgres
```

该docker-compose up命令将按照我们指定的依赖顺序启动和运行服务。

### 20.docker create命令有什么作用？

该docker create命令在指定映像上创建可写容器层，并准备该映像以运行指定命令。



## 1. 配置

修改 `/etc/docker/daemon.json`

```json
{
    "registry-mirrors": [
  		"https://docker.1ms.run",
  		"https://docker.xuanyuan.me"
  	],  # 设置国内镜像源
    "data-root": "/mnt/docker-root"  # 设置根目录，默认为/var/lib/docker
}
```

然后执行：

```shell
systemctl daemon-reload   # 重新载入配置文件，让设置生效
systemctl restart docker  # 立刻关闭后启动，即执行 stop 再 start
docker info  # 查看更新的配置信息
```



## 2. 常用命令

### 1）docker 服务

```shell
# 启动
systemctl start docker

# 守护进程重启
systemctl daemon-reload

# 重启docker服务
systemctl restart docker
或
service docker restart

# 关闭docker
systemctl stop docker
或
service docker stop
```

### 2）镜像

```shell
# 查找镜像
docker search [imageName]

# search失败，指定国内镜像源
docker search register.liberx.info/[imageName]

# 拉取镜像(默认下载最新版本的镜像)
 docker pull [imageName]:[指定版本]
 
# 列出所有下载到本地的镜像，包括镜像名称、标签、镜像ID、创建时间和大小
docker images

# 删除镜像
docker rmi [imageName]:[指定版本]

# 删除所有镜像
docker rmi $(docker images -q)

# 从Dockerfile创建镜像
docker build -t [imageName]:[指定版本] .
# --progress=plain: 打印详细输出
```

### 3）容器

```shell
# 显示正在运行的容器
docker ps

# 只显示容器 ID
docker ps -q

# 显示所有容器，包括停止的容器
docker ps -a　
docker ps -a | grep xxx  # 过滤查看

# 查看所有退出的容器id列表
docker ps -a|grep Exited|awk '{print $1}'

# 运行容器
docker start [容器id]

# 删除容器
docker rm [-f] [容器id/名字]

# 停止容器
docker stop [容器id]

# 停止所有容器
docker stop $(docker ps -a -q)

# 删除所有容器
docker rm $(docker ps -a -q)

# 创建并启动一个新的容器
docker run [OPTIONS] [imageName]
# [OPTIONS]参数：
# -d: 后台运行容器并返回容器 ID。
# -it: 交互式运行容器，分配一个伪终端。
# --name: 给容器指定一个名称。
# -p: 端口映射，格式为 host_port:container_port。
# -v: 挂载卷，格式为 host_dir:container_dir。
# --rm: 容器停止后自动删除容器。
# --env 或 -e: 设置环境变量。
# --network: 指定容器的网络模式。
# --restart: 容器的重启策略（如 no、on-failure、always、unless-stopped）。
# -u: 指定用户。


# 进入容器
# 知道容器的ID或名称后，使用docker exec命令进入容器
docker exec -it <container_id_or_name> /bin/bash
# 输入exit 或 使用Ctrl+D 退出
# -it：用于打开一个交互式终端
# <container_id_or_name>：要进入的容器的ID或名称
# /bin/bash：想要在容器中运行的Shell
```

> **docker run 与 docker start 的区别：**
>
> docker run 只在第一次运行时使用，将镜像放到容器中，以后再次启动这个容器时，只需要使用命令docker start即可。
>
> docker run 相当于执行了两步操作：
> 将镜像放入容器中（docker create），然后将容器启动，使之变成运行时容器（docker start）。而 docker
> start 的作用是，重新启动已存在的镜像。也就是说，如果使用这个命令，我们必须事先知道这个容器的ID，或者这个容器的名字，我们可以使用 docker ps 找到这个容器的信息。




## 3. 打包镜像到无网络环境中安装

- **第一步：使用 `Docker CLI` 拉取所需的镜像**

  ```shell
  docker pull <镜像名称>:<标签>
  ```

  例如，拉取最新的 `Nginx` 镜像，可以运行：

  ```shell
  docker pull nginx:latest
  ```

- **第二步：将 `Docker` 镜像保存为 `.tar` 文件**

  拉取镜像后使用 `docker save` 命令将其保存为 `.tar` 文件

  ```shell
  docker save -o <镜像名称>.tar <镜像名称>:<标签>
  ```

  以 `Nginx` 为例：

  ```shell
  docker save -o nginx.tar nginx:latest
  ```

  此命令将在当前目录中创建一个名为 `nginx.tar` 的文件，其中包含 `Docker` 镜像及其元数据

- **第三步：传输 `.tar` 文件**

  将 `.tar` 文件传输到目标机器或环境中

- **第四步：从 `.tar` 文件加载 `Docker` 镜像**

  在目标机器上，使用 `docker load` 命令从 `.tar` 文件中导入 `Docker` 镜像

  ```shell
  docker load -i <镜像名称>.tar
  ```

以 `Nginx` 为例：

  ```shell
docker load -i nginx.tar
  ```

  	`Docker` 现在将从 `.tar` 文件中加载镜像，并使其在本地可用

​	  使用 `docker images` 查看所有镜像进行验证：

  ```shell
docker images
  ```

  

## Docker Compose ls

**1、列出所有运行的 Docker Compose 项目**

```
docker compose ls
```

显示当前正在运行的所有项目及其状态。

**2、列出所有项目（包括未启动的）**

```
docker compose ls --all
```

显示包括未启动的项目在内的所有 Compose 项目。

**3、仅显示项目名称**

```
docker compose ls --quiet
```

只返回项目名称，而不包括详细信息。

**4、按状态过滤项目**

```
docker compose ls --filter "status=running"
```

仅显示状态为 running 的项目。







> 参考：
>
> 1. [Docker/DockerHub 国内镜像源/加速列表（3月19日更新-长期维护）-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/2485043)
> 2. [如何下载一个docker镜像 | PingCode智库](https://docs.pingcode.com/baike/3477239)
> 3. [Docker Search失败，但是Pull成功的解决方法（环境：腾讯云服务器CentOS7系统安装Docker）_docker search 超时-CSDN博客](https://blog.csdn.net/weixin_45391996/article/details/143703963)
> 4. [手动下载和导入Docker镜像：全面指南_手动下载docker镜像-CSDN博客](https://blog.csdn.net/ab13631152127/article/details/142955901)
> 5. [linux如何进入某个docker目录 | PingCode智库](https://docs.pingcode.com/baike/3822722)
> 6. [Docker 教程——理解 Docker 镜像和容器的存储路径](https://www.freecodecamp.org/chinese/news/where-are-docker-images-stored-docker-container-paths-explained/)
> 7. [Docker常用命令及docker run 和 docker start区别_docker start -it-CSDN博客](https://blog.csdn.net/weixin_44722978/article/details/89704085)
> 8. [Docker常用命令大全（非常详细）零基础入门到精通，收藏这一篇就够了-CSDN博客](https://blog.csdn.net/Python_0011/article/details/140313812)





