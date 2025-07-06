# 使用UV管理Python项目虚拟环境

## 💿安装

### 1. 安装命令

```shell
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# pip 安装
pip install uv
```

### 2. 添加环境变量

```shell
# 打开环境变量文件，比如/etc/profile、/etc/profile.d/、~/.bash_profile、~/.bashrc、/etc/bashrc
vim /root/.bashrc

# 将以下语句加入.bashrc
export UV_PATH="/root/.local/bin/uv"

# 保存并退出，重新加载环境变量
source /root/.bashrc
```

### 3. 验证

```shell
uv --version
```



## 📁 项目结构

```text
install_python/├── demo_project/          # Python 3.9 项目│   ├── .python-version    # 指定Python 3.9│   ├── pyproject.toml     # 项目配置和依赖│   ├── .venv/            # 虚拟环境│   └── main.py           # 数据处理脚本└── python_312_proj/      # Python 3.12 项目    ├── .python-version    # 指定Python 3.12    ├── pyproject.toml     # 项目配置和依赖    ├── .venv/            # 虚拟环境    └── main.py           # FastAPI Web应用
```

复制代码

## 🚀 UV 项目配置方法

### 方法一：创建新项目

```bash
# 创建Python 3.9项目mkdir my_project_39cd my_project_39uv init --python 3.9
# 创建Python 3.12项目mkdir my_project_312cd my_project_312uv init --python 3.12
```

复制代码

### 方法二：在现有项目中配置

```bash
# 进入项目目录cd existing_project
# 指定Python版本echo "3.9" > .python-version
# 初始化项目配置uv init
# 添加依赖uv add requests pandas numpy
```

复制代码

### 方法三：使用 uv run 直接运行

```bash
# 无需激活虚拟环境，直接运行uv run python script.py
# 运行特定Python版本uv run --python 3.12 python script.py
# 安装并运行uv run --with requests python -c "import requests; print('Hello')"
```

复制代码

## 📦 依赖管理

### 添加依赖

```bash
# 添加生产依赖uv add requests pandas
# 添加开发依赖uv add --dev pytest black
# 添加指定版本uv add "requests>=2.25.0"
# 添加可选依赖uv add --optional redis
```

复制代码

### 移除依赖

```bash
# 移除依赖uv remove requests
# 移除开发依赖uv remove --dev pytest
```

复制代码

### 同步环境

```bash
# 同步到最新版本uv sync
# 同步到锁文件版本uv sync --frozen
```

复制代码

## 🔄 项目切换

### 自动切换 Python 版本

```bash
# 进入项目目录，自动使用指定Python版本cd demo_projectuv run python --version  # Python 3.9.23
cd ../python_312_projuv run python --version  # Python 3.12.11
```

复制代码

### 虚拟环境管理

```bash
# 激活虚拟环境source .venv/bin/activate
# 或使用uv run（推荐）uv run python script.py
# 查看已安装包uv pip list
```

复制代码

## 🛠️ 常用命令

### 项目初始化

```bash
uv init [--python VERSION] [--name PROJECT_NAME]
```

复制代码

### 依赖管理

```bash
uv add PACKAGE [VERSION]     # 添加依赖uv remove PACKAGE           # 移除依赖uv sync                     # 同步环境uv lock                     # 更新锁文件
```

复制代码

### 运行命令

```bash
uv run COMMAND              # 在项目环境中运行命令uv run --python VERSION     # 指定Python版本运行uv run --with PACKAGE       # 临时安装包并运行
```

复制代码

### 工具管理

```bash
uv tool install black       # 安装工具uv tool run black .         # 运行工具
```

复制代码

## 📋 最佳实践

### 1. 项目隔离

- 每个项目使用独立的虚拟环境
- 使用 `.python-version` 指定 Python 版本
- 使用 `pyproject.toml` 管理依赖

### 2. 依赖管理

- 使用 `uv add` 而不是手动编辑 `pyproject.toml`
- 定期运行 `uv sync` 更新环境
- 使用锁文件确保环境一致性

### 3. 开发工作流

- 使用 `uv run` 而不是激活虚拟环境
- 使用 `uv tool` 管理开发工具
- 在 CI/CD 中使用 `uv sync --frozen`

### 4. 性能优化

- UV 比 pip 快 10-100 倍
- 使用缓存加速重复安装
- 并行下载和安装包

## 🔧 配置文件示例

### pyproject.toml

```csharp
[project]name = "my-project"version = "0.1.0"description = "项目描述"requires-python = ">=3.9"dependencies = [    "requests>=2.25.0",    "pandas>=1.3.0",]
[project.optional-dependencies]dev = [    "pytest>=6.0",    "black>=21.0",]test = [    "pytest-cov>=2.0",]
```

复制代码

### .python-version

```text
3.9
```

复制代码

## 🎯 项目示例对比

## 💡 提示

- UV 会自动管理虚拟环境，无需手动创建
- 使用 `uv run` 可以避免激活/退出虚拟环境的麻烦
- 每个项目的依赖完全隔离，不会相互影响
- UV 的锁文件确保在不同机器上环境一致











> [UV Python项目环境配置指南_Python_虚实的星空_InfoQ写作社区](https://xie.infoq.cn/article/1854608ba3e4cb67b0b6e7083)
>
> [UV虚拟环境的使用教程 - luckAI - 博客园](https://www.cnblogs.com/luckAI/p/18919512)
>
> https://www.cnblogs.com/wang_yb/p/18635441