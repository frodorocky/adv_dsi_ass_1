FROM jupyter/scipy-notebook:python-3.9.13
RUN conda install lightgbm
RUN conda install lime
RUN conda install hyperopt
RUN conda install graphviz
RUN conda install awswrangler
ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/work"
WORKDIR /home/jovyan/work
