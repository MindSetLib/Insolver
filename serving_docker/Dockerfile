FROM python:3.8

# Setup build arguments
ARG INSOLVER_VER=0.4.11
ARG PORT=5000

RUN pip install --no-cache-dir insolver==$INSOLVER_VER

# Install JRE for H2O framework support
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean

RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

COPY insolver_glm_h2o_1610467176142 ./
COPY transforms.pkl ./

EXPOSE $PORT
ENTRYPOINT insolver_serving -model insolver_glm_h2o_1610467176142 -transforms transforms.pkl -service fastapi -ip 0.0.0.0 -port 5000
