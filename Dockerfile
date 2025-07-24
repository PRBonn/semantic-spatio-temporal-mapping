FROM ubuntu:24.04
RUN userdel -r ubuntu

RUN apt-get update && apt-get upgrade -y \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --no-install-recommends -y \
  git \
  sudo \
  curl \
  wget \
  build-essential \
  software-properties-common \
  ninja-build \
  libgmp-dev \
  libmpfr-dev \
  libmpc-dev \
  libisl-dev \
  zlib1g-dev \
  file \
  cmake \
  libeigen3-dev \
  libtbb-dev \
  pybind11-dev \
  libopencv-dev \
  libceres-dev \
  && rm -rf /var/lib/apt/lists/*

# Python 3.10.9 as version to use
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    uuid-dev \
    && rm -rf /var/lib/apt/lists/*
RUN wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz && \
    tar xzf Python-3.10.9.tgz && \
    cd Python-3.10.9 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    sudo make altinstall && \
    cd .. && \
    rm -rf Python-3.10.9 Python-3.10.9.tgz
RUN sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 && \
    sudo update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.10 1

ARG UID=1000
ARG GID=1000
# not an arg, because otherwise its a pain in the ass
ENV UNAME=dev

# Add normal sudo-user to container, passwordless, taken from nacho's ros in docker
RUN addgroup --gid $GID $UNAME \
  && adduser --disabled-password --gecos '' --uid $UID --gid $GID $UNAME \
  && adduser $UNAME sudo \
  && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
  && sed -i 's/required/sufficient/' /etc/pam.d/chsh \
  && touch /home/$UNAME/.sudo_as_admin_successful

WORKDIR /home/${UNAME}/work
ENV HOME=/home/${UNAME}
USER ${UNAME}
ENV PATH="${PATH}:${HOME}/.local/bin"

ENV SHELL=/usr/bin/bash
SHELL ["/bin/bash", "-lc"]
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
CMD ["/bin/bash", "-i"]
