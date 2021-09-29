FROM python:3.7-slim
# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#run-jupyterlab

# change china source
# RUN sed -i 's#http://deb.debian.org#https://mirrors.163.com#g' /etc/apt/sources.list

# COPY ./sources.list /etc/apt/sources.list

# set proxy
RUN export http_proxy=http://127.0.0.1:1086
RUN export https_proxy=http://127.0.0.1:1086

# refresh SSL certificates
RUN apt-get update && apt-get install -y  vim git curl wget ca-certificates && update-ca-certificates --fresh

# install nodejs for jupyterextension install support
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - && apt-get install -y nodejs

# Swap dash for normal bash to get more full fledged terminal session in Jupyter
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install Jupyter Lab
COPY ./requirements.txt .
RUN pip3 install pip -U
RUN pip3 config set global.index-url https://pypi.douban.com/simple/

RUN pip3 install -r requirements.txt

# custom extensions
# RUN pip3 install jupyterlab_code_formatter

# Get our custom extensions for jupyterlab.
RUN jupyter labextension install @krassowski/jupyterlab_go_to_definition --no-build && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build && \
    jupyter lab build --dev-build=False

COPY ./jupyter_server_config.py /root/.jupyter/jupyter_server_config.py

# Hardcoded /project-dir because Python config can't be injected into Dockerfile
WORKDIR /project-dir

COPY start.sh /
ENTRYPOINT [ "/start.sh" ]