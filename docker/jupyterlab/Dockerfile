FROM python:3.7-slim
# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#run-jupyterlab

# change china source
RUN sed -i 's#http://deb.debian.org#https://mirrors.163.com#g' /etc/apt/sources.list

# COPY ./sources.list /etc/apt/sources.list

# refresh SSL certificates
RUN apt-get update && apt-get install -y vim git curl wget ca-certificates zsh autojump && update-ca-certificates --fresh

# set proxy
RUN export http_proxy=http://127.0.0.1:1087
RUN export https_proxy=http://127.0.0.1:1087

# install nodejs for jupyterextension install support
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - && apt-get install -y nodejs

# Swap dash for normal bash to get more full fledged terminal session in Jupyter
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install Jupyter Lab
COPY ./requirements.txt .
RUN pip3 install pip -U
RUN pip3 config set global.index-url https://pypi.douban.com/simple/

# Install basic lib
RUN pip3 install -r requirements.txt

# Get our custom extensions for jupyterlab.
RUN jupyter labextension install @krassowski/jupyterlab_go_to_definition --no-build && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build && \
    jupyter lab build --dev-build=False

# Custom extensions
RUN pip3 install ipydrawio[all]==1.1.3
RUN pip3 install jupyterlab_code_formatter jupyterlab-unfold jupyterlab-theme-solarized-dark theme-darcula jupyterlab-language-pack-zh-CN

# Modify jupyter config 
COPY ./config/jupyter_server_config.py /root/.jupyter/jupyter_server_config.py
RUN mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/notebook-extension
COPY ./config/notebook-extension/tracker.jupyterlab-settings /root/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings
RUN mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/translation-extension
COPY ./config/translation-extension/plugin.jupyterlab-settings /root/.jupyter/lab/user-settings/@jupyterlab/translation-extension/plugin.jupyterlab-settings
RUN mkdir -p /root/.jupyter/lab/user-settings/@ryantam626/jupyterlab_code_formatter
COPY ./config/jupyterlab_code_formatter/settings.jupyterlab-settings /root/.jupyter/lab/user-settings/@ryantam626/jupyterlab_code_formatter
RUN mkdir /root/.jupyter/lab/user-settings/@jupyterlab/extensionmanager-extension
COPY ./config/extensionmanager-extension/plugin.jupyterlab-settings /root/.jupyter/lab/user-settings/@jupyterlab/extensionmanager-extension


# ZSH
RUN wget --no-check-certificate https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O - | sh
RUN chsh -s /bin/zsh
RUN echo '. /usr/share/autojump/autojump.sh' >> ~/.zshrc
RUN source ~/.zshrc

# Hardcoded /project-dir because Python config can't be injected into Dockerfile
WORKDIR /project-dir

COPY start.sh /
RUN chmod +x /start.sh

ENTRYPOINT [ "/start.sh" ]