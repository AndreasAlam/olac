# Online Learning at Cost
Classifying dynamic unbalanced data

## Installation
For convenience the repo contains a `Makefile`  that facilitates:
* make venv - Virtual environment build & installation
* make data - Data download
* make requirements - Update requirements.txt

`make venv` assumes `homebrew` is installed and either `zsh` or `bash` is the default shell ($SHELL).
If the above conditions are not met or if you want to use a different environment/package-manager please see the dependencies
and manual installation section.
Pyenv will automatically 'activate' the local environment when entering the directory.

### Manual installation
#### Dependencies
1. Python >= 3.6.5
2. pip >= 9.03
3. Library requirements `pip install -r requirements.txt`

#### Optional:
1. [pyenv](https://github.com/pyenv/pyenv) - Simple Python version management
2. [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) - pyenv plugin
3. Python 3.6.5 - `pyenv install 3.6.5`
Set the environment
4. `pyenv virtualenv 3.6.5 olac_base `
5. `pyenv local olac_base`

## Data


See Docs/data_exploration for more detailed information

## Authors
* John Paton [paton.john@kpmg.nl]
* Ralph Urlus [urlus.ralph@kpmg.nl]
* Bram Schermer [schermer.bram@kpmg.nl]
* Susanne Groothuis [groothuis.susanne@kpmg.nl]
