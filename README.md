# Online Learning at Cost
Data that changes over time can be an issue in regards to classification tasks in machine learning. Especially if new characteristics emerge within the same class. An example of this is machine learning applied to fraud detection in financial institutions. New kinds of fraud appear over time, as new ways to ‘cheat the system’ are invented, especially if current ways are being successfully detected or stopped.

A problem for ML is that the flexibility of most algorithms is not strong enough to keep up with these new types of the target-class appearing over time. Retraining is the a common way of dealing with these changes. However, successfully retraining your model to detect new types of your target-class highly depends on these new types of being labelled. Retrieving new labels is an expensive exercise as cases need to be investigated by the human in the loop. In addition, there is a risk of of introducing bias in your model by only investigating the most likely cases produced by the model.

In this project we are investigating retraining strategies for models in production models in a cost-effective way, applied to toy-datasets, with a basis in fraud detection as a use case. We will do this by comparing ’traditional’ deep learning vs Online learning models. The goal of the project is to research optimal settings for labelling new data and providing feedback to a trained model, provided that we are trying balance the cost of obtaining new labels, with the cost of model decay over time.

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
