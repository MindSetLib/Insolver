FROM python:3.8

# Setup build arguments
ARG INSOLVER_VER=0.4.11
ARG MODEL
ARG TRANSFORMS

RUN pip install --no-cache-dir insolver==$INSOLVER_VER

# Install JRE for H2O framework support
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean

RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

COPY $MODEL ./insolver.model
COPY $TRANSFORMS ./transforms.pkl

EXPOSE 5000
ENTRYPOINT ["insolver_serving", "-model", "insolver.model", "-transforms", "transforms.pkl", "-ip", "0.0.0.0", "-port", "5000"]
CMD ["-service", "fastapi"]
