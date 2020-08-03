# Generic_GNN
A generic GNN framework to enable rapid research and development with DGL.
## Description
Meant for general purpose graph learning tasks, specifically those which require large graphs and assays.
## Code Style
Documentation follows [Google's Python style guidelines](https://google.github.io/styleguide/pyguide.html), with TODOs and supporting comments scattered throughout on an as needed basis. Entry point for the entire codebase is in main.py. 3 pipelines are available to run, as described in main.py documentation. Configuration files are stored as json files in config directory.
Command line arguments for runs:
```shell
python -c path/to/config/files/file.json -d bool_for_CUDA
```
## Directory Tree
Stored in directory.txt
```
C:.
│   data.zip
│   directory.txt
│   main.py
│   README.md
│   requirements.txt
│
├───config
│       primary_config.json
│       
├───data
│       edgelist.csv
│       embeddings_086.json
│       quinary_labels.csv
│       
├───logs
├───nn
│   │   DGL_layers.py
│   │   DGL_models.py
│   │   __init__.py
│
├───ops
│   │   benchmark.py
│   │   train.py
│   │   tune.py
│   │   __init__.py
│
├───read
│   │   preprocessing.py
│   │   __init__.py
│
└───utils
    │   helper.py
    │   holdout.py
    │   logger.py
    │   __init__.py
```

# Getting Started
## Setup
To be set up.

# Acknowledgements
To my mentors and my friends whom have taught and inspired me all these years.
