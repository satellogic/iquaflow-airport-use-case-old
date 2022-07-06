FROM nvidia/cuda:9.0-devel-ubuntu16.04

ENV CONDA_DIR $HOME/miniconda3
# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH
# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# make conda activate command available from /bin/bash --interative shells
#RUN conda init bash
#RUN echo "source activate airport-env" &gt; ~/.bashrc
#ENV PATH /opt/conda/envs/airport-env/bin:$PATH

RUN apt update 
RUN apt install wget 
RUN apt -y install git
RUN apt install libglib2.0-0 -y 
RUN apt install libgl1-mesa-glx -y 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
RUN chmod 775 Miniconda3-latest-Linux-x86_64.sh 
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR 
RUN rm Miniconda3-latest-Linux-x86_64.sh 
RUN export PATH="$HOME/miniconda3/bin:$PATH" 
RUN conda create -n airport-env python=3.6 -q -y 
RUN conda install -n airport-env pytorch=0.4.0 cuda90 -c pytorch -y 
RUN conda install -n airport-env torchvision -c soumith -y

WORKDIR /airport
COPY ./requirements.txt ./requirements.txt
COPY ./lib ./lib
#COPY ./ ./
# git clone https://deploy-token-31:paXXCtbTEaeJirhz1hhw@publicgitlab.satellogic.com/iqf/iq-sateairports.git && \
RUN  conda run -n airport-env pip install -r requirements.txt
RUN  conda run -n airport-env pip install Pillow==6.2.2
RUN  conda run -n airport-env pip install numpy==1.16.1
RUN  conda run -n airport-env pip install scipy==1.1.0
RUN  conda run -n airport-env pip install termcolor
RUN cd lib && conda run -n airport-env sh make.sh




### Install IQF in the project

RUN  conda run -n airport-env pip install notebook
RUN  conda run -n airport-env pip install jupyterlab
RUN conda run -n airport-env pip install git+https://github.com/satellogic/iquaflow.git
CMD ["/bin/bash", "-c", "source activate airport-env && /bin/bash"]
