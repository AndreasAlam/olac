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
Pyenv will automatically 'activate' the local environment when entering the directory. Moreover, the makefile install ipython kernel based on the local virtualenv.

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
6. `pip install ipykernel`
7. `ipython kernel install --user --name=olac_base`

## Data
See Docs/datasets for the documentation and exploration of the datasets.

#### [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
##### CCFD0
The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

#### [Synthetic Financial Datasets For Fraud Detection - Kaggle](https://www.kaggle.com/ntnu-testimon/paysim1)
##### SMMT0
We present a synthetic dataset generated using the simulator called PaySim as an approach to such a problem. PaySim uses aggregated data from the private dataset to generate a synthetic dataset that resembles the normal operation of transactions and injects malicious behaviour to later evaluate the performance of fraud detection methods.

#### [German Credit Risk - Kaggle](https://www.kaggle.com/uciml/german-credit)
##### CCFD1

The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by Prof. Hofmann. In this dataset, each entry represents a person who takes a credit by a bank. Each person is classified as good or bad credit risks according to the set of attributes. The link to the original dataset can be found below.

#### [PKDD99 Berka](http://lisp.vse.cz/pkdd99/berka.htm)
##### CCFD2
Data from a real Czech bank from 1999. The data contains bank transactions, account info, and loan records released for PKDD'99 Discovery Challenge.


#### Kagle API
'To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account)
and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location ~/.kaggle/kaggle.json.'

See the [kagle api](https://github.com/Kaggle/kaggle-api) for more details.

## Authors
* John Paton [paton.john@kpmg.nl]
* Ralph Urlus [urlus.ralph@kpmg.nl]
* Bram Schermer [schermer.bram@kpmg.nl]
* Susanne Groothuis [groothuis.susanne@kpmg.nl]
