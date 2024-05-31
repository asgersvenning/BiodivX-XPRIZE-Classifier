# Stage 1: Bring in the micromamba image to copy files from it
FROM mambaorg/micromamba:1.5.8 as micromamba

# Stage 2: The main image we are going to add micromamba to
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install necessary packages including openssh-client, lsb-release, and other dependencies
RUN apt-get update && apt-get install -y wget bzip2 curl ca-certificates openssh-client lsb-release git dos2unix openssl libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

USER root

# Set environment variables for micromamba
ARG MAMBA_USER=user
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

# Copy micromamba files from the micromamba stage
COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

# Initialize user accounts and set up the root prefix
RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

# Print openssl version
RUN openssl version

# Add the encrypted SSH key to the container
COPY id_rsa.enc /home/user/.ssh/id_rsa.enc

# Add the secret passphrase for decryption
RUN --mount=type=secret,id=EncryptionPassphrase \
    mkdir -p /home/user/.ssh && \
    cat /run/secrets/EncryptionPassphrase | tr -d '\r' > /home/user/.ssh/passphrase

# Add SSH key secret and perform necessary setup as root user
RUN openssl enc -aes-256-cbc -d -in /home/user/.ssh/id_rsa.enc -out /home/user/.ssh/id_rsa -pass file:/home/user/.ssh/passphrase && \
    chmod 600 /home/user/.ssh/id_rsa

# Generate the public key
RUN ssh-keygen -y -f /home/user/.ssh/id_rsa > /home/user/.ssh/id_rsa.pub && \
    chmod 644 /home/user/.ssh/id_rsa.pub

# Add GitHub to known hosts
RUN ssh-keyscan github.com >> /home/user/.ssh/known_hosts && \
    chown -R user:user /home/user/.ssh

# Create SSH config
RUN echo "Host *\n    StrictHostKeyChecking no\n" > /home/user/.ssh/config && \
    chmod 600 /home/user/.ssh/config

# Debug: Load the expected SHA256SUM of the SSH key
ARG GithubTokenSHA256SUM

# Debug: Print the hashes of the SSH key to verify its integrity
RUN sha256sum /home/user/.ssh/id_rsa

# Debug: Check the hashes match the expected value
RUN test "$(sha256sum /home/user/.ssh/id_rsa | cut -d ' ' -f 1)" = "$GithubTokenSHA256SUM" && echo "Secrets match" || echo "Secrets do not match"

# Switch to the "user" user
USER $MAMBA_USER

# Use the micromamba shell
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

# Pre-create the environment to speed up the build and improve caching
RUN micromamba create --name xprize_localizer python=3.11.5 -y -c conda-forge && \
    micromamba install --yes -n xprize_localizer -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.1 && \
    micromamba install --yes -n xprize_localizer -c conda-forge gradio && \
    micromamba clean --all --yes

# Set home to the user's home directory and ensure PATH includes user's local bin
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at /home/user/app with the new user ownership
COPY --chown=user:user . /home/user/app

# Make the setup script executable
RUN chmod +x install.sh

# Check if the SSH key is working
RUN git ls-remote git@github.com:github/gitignore.git > /dev/null 2>&1 || exit 1

# Run the install script
RUN ./install.sh

# Expose the necessary port for Gradio
EXPOSE 7860

# Command to run your Gradio app
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["bash", "-c", "eval \"$(micromamba shell hook --shell bash)\" && micromamba activate xprize_localizer && python app.py"]