FROM ubuntu:20.04
MAINTAINER Derek Pisner
ENV DEBIAN_FRONTEND noninteractive
ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8"

RUN apt-get update -y \
    && apt-get upgrade -y \
    # Install system dependencies.
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libsqlite3-dev \
        libreadline-dev \
        libffi-dev \
        curl \
        libbz2-dev \
	    liblzma-dev \
        cmake \
        wget \
        bzip2 \
        ca-certificates \
        libxtst6 \
        libgtk2.0-bin \
        libxft2 \
        libxmu-dev \
        libgl1-mesa-glx \
        libpng-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        gnupg \
        libgomp1 \
        libmpich-dev \
        mpich \
        g++ \
        zip \
        unzip \
        libglu1 \
        libfreetype6-dev \
        pkg-config \
        libgsl0-dev \
        openssl \
        gsl-bin \
        libglu1-mesa-dev \
        libglib2.0-0 \
        libglw1-mesa \
        libxkbcommon-x11-0 \
        gcc-multilib \
        apt-transport-https \
        debian-archive-keyring \
        dirmngr \
        git \
    && cd /tmp \
    && curl -O https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tar.xz \
    && tar -xf Python-3.8.2.tar.xz \
    && cd Python-3.8.2 \
    && ./configure --enable-optimizations --enable-shared --prefix=/usr/local --enable-shared LDFLAGS="-Wl,-rpath /usr/local/lib" --with-ensurepip=install --enable-loadable-sqlite-extensions \
    && make -j 8 \
    && make altinstall

RUN export GIT_SSL_NO_VERIFY=1 \
    && curl https://bootstrap.pypa.io/get-pip.py | python3.8 \
    && pip install --upgrade pip \
    && pip install Cython setuptools ipython \
    && pip install certifi -U --ignore-installed \
    && pip install --upgrade pyopenssl \
    && git config --global url."https://".insteadOf git:// \
    && git config --global http.postBuffer 1048576000 \
    && git clone -b main https://github.com/dPys/ForecastIntensity /home/ForecastIntensity \
    && cd /home/ForecastIntensity \
    && pip install -r requirements.txt \
    && find /home/ForecastIntensity/forecastintensity -type f -iname "*.py" -exec chmod 777 {} \; \
    && mkdir /working \
    && chmod -R 777 /working \
    && apt-get clean -y && apt-get autoclean -y && apt-get autoremove -y \
    && apt-get purge -y --auto-remove \
        wget \
        cmake \
        gcc \
        curl \
        openssl \
        build-essential \
        ca-certificates \
        libc6-dev \
        gnupg \
        g++ \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 8080 80 443 445 139 22

#ENTRYPOINT ["/usr/bin/python3", "/home/ForecastIntensity/forecastintensity/carbon_model.py"]
