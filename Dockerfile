FROM continuumio/miniconda:latest

WORKDIR /home/vb/PycharmProjects/creditCardApproval

COPY environment.yml ./
COPY src ./
COPY datasets ./
COPY LICENSE ./
COPY README.md ./

COPY boot.sh ./

RUN chmod +x boot.sh

RUN conda env create -f environment.yml

RUN echo "source activate creditCard" > ~/.bashrc
ENV PATH /home/vb/anaconda3/envs/creditCard/bin:$PATH

EXPOSE 9999

ENTRYPOINT ["./boot.sh"]
