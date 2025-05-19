FROM jupyter/base-notebook:latest

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    vim \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install additional Python packages
RUN pip install --no-cache-dir \
    jupyterhub \
    notebook \
    matplotlib \
    pandas \
    scikit-learn \
    numpy

# Create directory for JupyterHub configuration
RUN mkdir -p /srv/jupyterhub

# Copy config file
COPY jupyterhub_config.py /srv/jupyterhub/jupyterhub_config.py

# Rename the jovyan user to adi
RUN usermod -l adi jovyan && \
    groupmod -n adi users && \
    usermod -d /home/adi -m adi && \
    echo "adi:password" | chpasswd

# Fix permissions
RUN chown -R adi:adi /srv/jupyterhub /home/adi

WORKDIR /srv/jupyterhub

EXPOSE 8000

# Switch to adi user
USER adi

CMD ["jupyterhub", "-f", "/srv/jupyterhub/jupyterhub_config.py"]