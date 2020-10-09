# Project Pactum

TODO.

## Setup

Ensure you have the following requirements:

- Python 3.8
- TensorFlow 2.3

Documentation has the following requirements:

- TeX Live
- Biber

First, create the virtual environment:

    python -m venv --system-site-packages venv
    source venv/bin/activate
    pip install -U pip

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
