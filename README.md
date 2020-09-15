# Generic_GNN
A generic GNN framework to enable rapid research and development with DGL.
## Description
Meant for general purpose graph learning tasks, specifically those which require large graphs and assays.
## Code Style
Documentation follows [Google's Python style guidelines](https://google.github.io/styleguide/pyguide.html), with TODOs and supporting comments scattered throughout on an as needed basis. Entry point for the entire codebase is in main.py. 3 pipelines are available to run, as described in main.py documentation. Configuration files are stored as json files in config directory.

## Directory Tree
Stored in directory.txt (use ```Get-ChildItem | tree /F > foo.txt``` in PowerShell to create your own)

# Getting Started
## Conda Env Setup
First you'll want to create a new conda (or pip) env with Python 3.7
```shell
conda create -n env_name python=3.7 anaconda
source activate env_name
```

Before cloning into this repository:
```shell
git clone https://github.com/flawnson/Generic_GNN.git
OR
pip install git+https://github.com/flawnson/Generic_GNN.git
```

Then you can run setup.py
```shell
python setup.py
```

and install depedencies in the requirements.txt
```shell
pip install -r requirements.txt
```

Then you'll need to create an empty directory for model outputs (including saved models)
```shell
cd Generic_GNN && mkdir outputs
```

Finally you can run a demo version of the pipeline (default configs in configs directory)
```shell
python -c path/to/config/files/file.json -s path/to/schema/files/file.json
```

You can see the logged results using TensorBoard
```shell
tensorboard --logdir=logs/GAT_tuning/tune_model
```

None of the above will work without the correct data files, all of which are not publically available as of currently.

## Docker Container Setup
Change directories into the one with `Dockerfile` and run (add tags as necessary):
```shell
docker build .
```

Copy the container ID and run the container by executing:
```shell
docker run -p port_number:8000 container_id
```

Be sure to stop (or kill if necessary) the container when not in use.

# Acknowledgements
To my mentors and my friends whom have taught and inspired me all these years.
Thanks to PyTorch Geometric for providing me with the background to use DGL, and DGL for providing me with a kick ass library.
