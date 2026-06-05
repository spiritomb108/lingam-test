FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

ENV TZ=Asia/Tokyo

# 1. Yarnのリポジトリ鍵エラーを回避
RUN rm -f /etc/apt/sources.list.d/yarn.list

# 2. システムライブラリのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev sqlite3 default-jre-headless \
    chromium fonts-noto-color-emoji graphviz \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. uv を root でインストールして全ユーザーが使える場所に配置
RUN curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# 4. PATHの設定
ENV PATH="/usr/local/bin:/home/vscode/.local/bin:/home/vscode/.npm-global/bin:$PATH"

USER vscode
WORKDIR /workspace