## Jupyter Lab

> 脚本快速实践开发工具，基础镜像说明见[这里](https://jupyter-docker-stacks.readthedocs.io/en/latest/)


```shell
# 打包
docker build --no-cache=true -t howie6879/jupyter-lab-for-python37:v3.1.4 -f Dockerfile . 
# 启动
cd {your_pylab_path}
docker run --name jupyter_pylab -it -d --restart=always -p 0.0.0.0:8765:8888 -e SHELL="/bin/zsh" -v "`pwd`:/project-dir" howie6879/jupyter-lab-for-python37:v3.1.4 --allow-root --no-browser --port=8888
```

