FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
RUN apt-get update && apt-get install -y libaio-dev

ENV VIRTUAL_ENV=/workspace/venv
RUN python -m venv --system-site-packages $VIRTUAL_ENV

COPY external/apex /workspace/external/apex
RUN cd /workspace/external/apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

COPY requirements.txt /workspace/requirements.txt
RUN pip install -U pip && pip install -r /workspace/requirements.txt

COPY external/deepspeed/setup.py /workspace/external/deepspeed/setup.py
COPY external/deepspeed/README.md /workspace/external/deepspeed/README.md
COPY external/deepspeed/version.txt /workspace/external/deepspeed/version.txt
RUN mkdir -p /workspace/external/deepspeed/deepspeed /workspace/external/deepspeed/bin
COPY external/deepspeed/csrc /workspace/external/deepspeed/csrc
COPY external/deepspeed/op_builder /workspace/external/deepspeed/op_builder
COPY external/deepspeed/requirements /workspace/external/deepspeed/requirements

ENV PATH=$VIRTUAL_ENV/bin:/workspace/external/deepspeed/bin:$PATH
ENV PYTHONPATH=/workspace:/workspace/external/deepspeed

RUN cd /workspace/external/deepspeed && DS_BUILD_OPS=1 python setup.py build_ext
RUN cd /workspace/external/deepspeed/build/lib.linux-x86_64-3.7 && find . -name '*.so' -exec install -Dm644 {} /workspace/external/deepspeed/{} \;

COPY . /workspace
RUN mv /workspace/.dockerversion /workspace/VERSION
