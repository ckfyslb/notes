1. 安装`devtoolset`：

   ```shell
   yum -y install centos-release-scl
   yum -y install devtoolset-11
   ```
   
   `devtoolset`后的数字对应`gcc/g++`版本
   
   - **离线安装：**
   
     - 使用`repotrack`下载 `devtoolset`全量依赖包：
   
     ```shell
     # 安装yum-utils
     $ yum -y install yum-utils
     
     # 下载 devtoolset 全量依赖包
     $ repotrack -p /path/to/save/dependency_packages devtoolset-11
     # -p：指定下载目录
     ```
   
     - 将`dependency_packages`所有包转移到离线设备，然后使用`rpm`进行安装
   
     ```shell
     rpm -Uvh dependency_packages/* --force --nodeps
     
     # --force：强制安装
     # --nodeps：安装时不检查依赖关系
     ```

2. 启用 `devtoolset-11` 软件集和

   - 暂时生效

     ```shell
     scl enable devtoolset-11 bash
     ```

   - 长期生效

     ```shell
     # 写入配置文件
     echo "source /opt/rh/devtoolset-11/enable" >>/etc/profile
     ```


3. ==避坑：==
   下载新版`gcc/g++`直接替换`/usr/bin`下的`gcc/g++`
   这样虽然编译器是新版的，但是相应的头文件和库文件还都是原先老版本的



> 参考：
>
> 1. [Centos7下升级gcc/g++版本（简单 + 避坑）_devtoolset-11-gcc-CSDN博客](https://blog.csdn.net/whc18858/article/details/135484071)
>
> 2. [yum 下载全量依赖 rpm 包及离线安装（终极解决方案）_rpm -uvh --force --nodeps-CSDN博客](https://blog.csdn.net/jlh21/article/details/104775084)
>
> 3. [使用repotrack下载指定rpm包及其全量依赖包-CSDN博客](https://blog.csdn.net/weixin_42216109/article/details/127216195)