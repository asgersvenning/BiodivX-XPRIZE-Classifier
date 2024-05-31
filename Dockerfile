# Use the official Hugging Face Spaces Gradio base image
FROM huggingface/transformers-gradio:latest

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory
WORKDIR /app

# Make the setup script executable
RUN chmod +x install.sh

# Add SSH key secret
RUN --mount=type=secret,id=GithubToken \
    mkdir -p /root/.ssh && \
    cat /run/secrets/GithubToken > /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# Run the install script
RUN ./install.sh

# Expose the necessary port for Gradio
EXPOSE 7860

# Command to run your Gradio app
CMD ["sh", "-c", "source /opt/conda/etc/profile.d/conda.sh && micromamba activate xprize_pipeline && python app.py"]
