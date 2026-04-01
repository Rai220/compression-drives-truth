FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir numpy scipy sympy tqdm

# Copy project files
COPY training_torch/ training_torch/
COPY training/ training/
COPY data/generate_math.py data/
COPY data/generate_paired_test.py data/
COPY scripts/reproduce_minimal.sh scripts/

RUN chmod +x scripts/reproduce_minimal.sh

ENTRYPOINT ["bash", "scripts/reproduce_minimal.sh"]
