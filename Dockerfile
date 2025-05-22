FROM continuumio/miniconda3 

WORKDIR /app 

COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy && \
    rm -f /tmp/environment.yml

COPY . .

CMD ["python", "./src/test.py"]
