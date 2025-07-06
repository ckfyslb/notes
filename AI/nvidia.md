1. ```shell
   fuser -v /dev/nvidia*
   
   查找使用`GPU`的进程
   ```

2. ```shell
   watch -n 1 -d nvidia-smi
   
   实时监测`GPU`，每个1秒刷新一次，使用 -d 会标出变化的地方
   ```

3. ```shell
   nvidia-smi -l 1
   
   每隔1秒打印一下`GPU`的状态
   ```

4. ```shell
   nvidia-smi -pm 1    # 0为关闭，1为打开
   
   持久模式`Persistence-M` (需要 `root`)。
   - 除非使用 `-i` 参数指定单个 `GPU`，否则将影响所有 `GPU`。
   - 启用持久性模式后，即使没有活动的客户端，`NVIDIA`驱动程序也会保持加载状态。这样可以最大程度地减少与运行依赖的应用程序 (例如 `CUDA`程序) 相关的驱动程序加载延迟。持久模式够让 `GPU` 更快响应任务，待机功耗增加。
   - 持久模式不会在重新启动后持续存在。每次重新启动后，持久性模式将默认为禁用。
   ```

5. ```shell
   nvidia-smi -L
   
   列出所有可用的 `NVIDIA` 设备信息。
   ```

6. 置 TORCH_CUDA_ARCH_LIST 环境变量

[如何设置 TORCH_CUDA_ARCH_LIST 环境变量以优化 PyTorch 性能-CSDN博客](https://blog.csdn.net/GHY2016/article/details/143635720)

7. lspci | grep NVIDIA
   查看显卡物理连接
8. ls -l /dev/nvidia*
   查看显卡驱动文件
