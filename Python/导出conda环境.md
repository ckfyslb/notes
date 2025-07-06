导出conda环境：

```shell
conda env export -n your_env_name --no-builds | grep -v "^prefix: " > environment.yml
```





conda env export -n qwen --no-builds | grep -v "^prefix: " > environment.yml