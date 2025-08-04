# 基础镜像：Ubuntu 22.04
FROM ubuntu:22.04

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone

# 切换源 + 安装依赖，包括 git、vim、ll（通过 alias 实现）
RUN sed -i \
      -e 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
      -e 's|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
      /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      libstdc++6 \
      libz1 \
      ffmpeg \
      libsm6 libxext6 libxrender-dev \
      git \
      vim \
      coreutils \
      bash \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 设置 ll 为 ls -l 的别名（可选）
RUN echo "alias ll='ls -alF'" >> /etc/bash.bashrc

WORKDIR /app

# 复制可执行文件和配置
COPY dist/analysis_data /app/analysis_data
COPY config /app/config
COPY .env /app/.env

EXPOSE 40317

ENTRYPOINT ["./analysis_data"]