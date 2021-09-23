FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel
RUN apt-get update && apt-get install -y libaio-dev
COPY external/apex /workspace/external/apex
COPY requirements.txt /workspace/requirements.txt
RUN pip install -U pip && pip install -r /workspace/requirements.txt
RUN cd /workspace/external/apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
ENV VIRTUAL_ENV=/workspace/venv
RUN python -m venv --system-site-packages $VIRTUAL_ENV
ENV PATH=$VIRTUAL_ENV/bin:/workspace/external/deepspeed/bin:$PATH
ENV PYTHONPATH=/workspace:/workspace/external/deepspeed
RUN touch /workspace/VERSION /workspace/VERSION.out
COPY . /workspace
RUN rm /workspace/VERSION /workspace/VERSION.out
COPY VERSION.out /workspace/VERSION