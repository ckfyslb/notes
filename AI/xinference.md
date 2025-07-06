### 1. python环境

#### 1）安装

```shell
# vLLM 引擎
pip install "xinference[vllm]"
```

#### 2）使用

**启动本地服务**

```shell
# 使用-H或--host
xinference-local -H 0.0.0.0    
```

- 默认端口`--port 9997`

- 启动服务后:
  - 访问 http://127.0.0.1:9997/ui 可以使用 `UI`
  - 访问 http://127.0.0.1:9997/docs 可以查看 `API` 文档

> 默认情况下，`Xinference` 会使用 `<HOME>/.xinference` 作为主目录来存储一些必要的信息，比如日志文件和模型文件，其中 `<HOME>` 就是当前用户的主目录。
>
> 可以通过配置环境变量 `XINFERENCE_HOME` 修改主目录， 比如：
>
> ```shell
> XINFERENCE_HOME=/tmp/xinference xinference-local --host 0.0.0.0 --port 9997
> ```

**加载模型**

- 注册

```shell
xinference register --model-type <model_type> --file model.json --persist

# model_type： LLM、embedding、rerank
```

- 启动

```shell
# llm
xinference launch --model-engine vllm -n qwen-ds --model_uid qwen-ds -s 32 -f pytorch --gpu_memory_utilization 0.95 --max_model_len 30000 --max_num_seqs 1024

# embedding
xinference launch -n qwen2-7b-embedding --model_uid qwen2-7b-embedding --model_type embedding 
```

- 免注册直接启动

```shell
xinference launch --model_path <model_file_path> --model-engine <engine> -n qwen1.5-chat
```

**查看启动的模型**

```shell
xinferece list
```



### 2. docker

#### 1）拉取镜像

```shell
docker pull xprobe/xinference
```

#### 2）使用镜像

```shell
docker run -e XINFERENCE_MODEL_SRC=modelscope -p 9997:9997 --gpus all xprobe/xinference:v<your_version> xinference-local -H 0.0.0.0 --log-level debug
```

> - `--gpus` 必须指定，镜像必须运行在有 `GPU` 的机器上，否则会出现错误。
> - `-H 0.0.0.0` 也必须指定，否则在容器外无法连接到 `Xinference` 服务。
> - 可以指定多个 `-e` 选项赋值多个环境变量。

#### 3）挂载模型目录

默认情况下，镜像中不包含任何模型文件，使用过程中会在容器内下载模型。
如果需要使用已经下载好的模型，需要将宿主机的目录挂载到容器内。
这种情况下，需要在运行容器时指定本地卷，并且为 `Xinference` 配置环境变量。

```shell
# 格式
docker run \
-v </your/home/path>/.xinference:/root/.xinference \
-v </your/home/path>/.cache/huggingface:/root/.cache/huggingface \
-v </your/home/path>/.cache/modelscope:/root/.cache/modelscope \
-v </on/your/host>:</on/the/container> \ 
-e XINFERENCE_HOME=</on/the/container> \ 
-p 9997:9997 \
--gpus all \
xprobe/xinference:v<your_version> \
xinference-local -H 0.0.0.0

# 使用
docker run -v /root/.xinference:/root/.xinference -e XINFERENCE_HOME=/root/.xinference -p 9997:9997 --gpus all xprobe/xinference:latest xinference-local -H 0.0.0.0
```

上述命令的原理是将主机上指定的目录挂载到容器中，并设置 `XINFERENCE_HOME` 环境变量指向容器内的该目录。这样，所有下载的模型文件将存储在主机上指定的目录中，下次运行容器时，可以直接使用现有的模型，无需重复下载。




```
docker run -v /root/.xinference:/root/.xinference -v /home/qwen/DeepSeek-R1-Distill-Qwen-32B:/root/qwen-ds -v /home/cai/embedding_models/gte_Qwen2-7B-instruct:/root/qwen2-7b-embedding -e XINFERENCE_HOME=/root/.xinference -p 9997:9997 --gpus all xprobe/xinference:latest xinference-local -H 0.0.0.0 xinference launch --model_path /root/qwen-ds --model-engine vllm -n qwen-ds --model_uid qwen-ds -s 32 -f pytorch --gpu_memory_utilization 0.95 --max_model_len 30000 --max_num_seqs 1024 && xinference launch --model_path /root/qwen2-7b-embedding -n qwen2-7b-embedding --model_uid qwen2-7b-embedding --model_type embedding && tail -f /dev/null
```