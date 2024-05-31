FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory and ensure PATH includes user's local bin
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Install micromamba
RUN apt-get update && apt-get install -y wget bzip2 \
    && wget -qO-  https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \
    && touch /root/.bashrc \
    && ./bin/micromamba shell init -s bash -p /opt/conda  \
    && grep -v '[ -z "\$PS1" ] && return' /root/.bashrc  > /opt/conda/bashrc   # this line has been modified \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Copy the current directory contents into the container at /home/user/app with the new user ownership
COPY --chown=user:user . /home/user/app

# Make the setup script executable
RUN chmod +x install.sh

# Add SSH key secret and perform necessary setup as non-root user
RUN --mount=type=secret,id=GithubToken \
    mkdir -p /home/user/.ssh && \
    cat /run/secrets/GithubToken > /home/user/.ssh/id_rsa && \
    chmod 600 /home/user/.ssh/id_rsa && \
    ssh-keyscan github.com >> /home/user/.ssh/known_hosts

# Run the install script
RUN ./install.sh

# Expose the necessary port for Gradio
EXPOSE 7860

# Command to run your Gradio app
CMD ["sh", "-c", "source /opt/conda/etc/profile.d/conda.sh && micromamba activate xprize_pipeline && python app.py"]
