Afforable deep learning through resilient preemptible instances.

v0.1 - 01/20/22

# Summary of Bamboo
Bamboo is a system for running large scale DNNs using **pipeline parallelism**
affordably, reliably, and efficiently on spot instances.
It is built on top of [DeepSpeed](https://github.com/microsoft/DeepSpeed).
It uses redundant computation in the pipeline by taking advantage of
pipeline bubbles to enable low-pause recovery from failures.


## Setup

Ensure you have the following requirements:

- Python 3.7
- PyTorch 1.10.0

Documentation has the following requirements:

- TeX Live
- Biber

First, create the virtual environment:

    python -m venv --system-site-packages venv
    source venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt

For the documentation you may want to create a `~/.latexmkrc` file containing
the following (this example uses Evince):

    $pdf_previewer = 'start evince';

## Running

Start all commands with the following:

    python -m project_pactum

For the documentation, go to the directory of whichever document you want to
build and run the following:

    latexmk -pvc

This command will recompile the LaTeX file as many times as needed and open it
in your preferred PDF viewer. For modifications keep this command running, and
the document recompiles automatically.
