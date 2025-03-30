
FROM ollama/ollama:latest

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0
ENV OLLAMA_MODELS=/root/.ollama/models

# Create directories if needed
RUN mkdir -p /root/.ollama

# Start Ollama service by default
CMD ["ollama", "serve"]