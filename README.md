# Neuron Detection

A Python package and command-line tool for robust neuron detection and pre- and postprocessing of imaging data.

The project is designed to be **cross-platform (Linux & Windows)** and suitable for **HPC environments**. Dependencies are managed via **conda**, and the tool is exposed through a convenient CLI.

It is effectively a wrapper around CaImAn <https://github.com/flatironinstitute/CaImAn>, providing a realization of the neuron detection suitable for the kind of data that I have been working with, with minimal need for finetuning and the ability to provide the few open parameters via provided functions and CLI.

I don't claim any rights to CaImAn (but am deeply grateful to all the work that people have been putting into developing it!)

---

## Features

- Command-line interface for running neuron detection pipelines
- Modular Python package (`neuron_detection`)
- Conda-managed dependencies for reproducibility
- Works on Linux, Windows, and HPC systems

---

## Requirements

- **Conda** (Miniconda or Miniforge recommended)
- Python **≥ 3.10**
- Internet access for initial environment creation

> ⚠️ Windows does **not** come with Python or conda preinstalled.

---

## Installing Conda

### Linux

Install **Miniconda** or **Miniforge**:

- <https://docs.conda.io/en/latest/miniconda.html>
- <https://github.com/conda-forge/miniforge>

Verify installation:

```bash
conda --version
```

### Windows

1. Install **Miniconda** or **Miniforge**:
2. Open Anaconda Prompt (recommended)
3. Verify:
```
conda --version
```

## Installation
### 1) Create the conda environment
From the project root directory:

```bash
conda env create -f environment.yml
```

Activate the environment:
```
conda activate neuron-detection
caimanmanager install
```
> On Windows, run these commands in **Anaconda Prompt**.

For further installation instructions, refer to the official CaImAn repository: <https://github.com/flatironinstitute/CaImAn>

### 2) Install the package

Install the package using the Python interpreter from the activated environment:
```
python -m pip install .
```

For development (editable install):
```
python -m pip install -e .
```

### 3) Verify installation
```bash
neuron-detection --help
```

Optional sanity check:
```
python -c "import neuron_detection; print(neuron_detection.__file__)"
```

## Usage
Basic command-line usage
```
neuron-detection \
  --source path/to/data \
  --param1 value1 \
  --param2 value2
```
```--source``` should always be provided, while all other arguments are optional. 
For a list of possible arguments, refer to ```neuron-detection --help```


Path examples:

- **Linux**: ```/path/to/data```
- **Windows**: ```C:\path\to\data``` or ```path\to\data```

Paths are handled internally in a platform-independent way.

### Running without activating the environment (optional)

If you prefer not to activate the environment manually:
```
conda run -n neuron-detection neuron-detection --help
```

This is especially useful for scripts and batch jobs.

## HPC Usage (Linux)

For batch systems such as **Slurm**, explicitly activate the environment in your job script:
```
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neuron-detection

neuron-detection --source ... --param1 ...
```

Do **not** rely on shell startup files being sourced automatically.

## Troubleshooting
### ```ModuleNotFoundError``` despite package appearing in ```conda list```

Most common causes:
* The conda environment is not actually active
* The package was installed with a different Python interpreter

Check:
```
which python
python -c "import sys; print(sys.executable)"
```

Reinstall if necessary:
```
python -m pip install -e .
```

### Command exists but uses the wrong Python

Check where the CLI comes from:
```
which neuron-detection   # Linux
where neuron-detection   # Windows
```

Reinstall the package **inside the active environment**.

### Windows: command not found

Ensure you are using:
* **Anaconda Prompt**, or
* PowerShell with ```conda init powershell``` applied

## Uninstalling

Remove the conda environment entirely:
```
conda env remove -n neuron-detection
```

## Development

* Packaging is defined via ```pyproject.toml```
* CLI entry point: ```neuron-detection```
* Python entry point: ```neuron_detection.run_detection:main```
* Tested on Linux and Windows (not yet)

## License

not yet...

## Support

When reporting issues, please include:
* operating system
* output of ```conda info```
* output of ```which python``` / ```where python```
* the exact command used