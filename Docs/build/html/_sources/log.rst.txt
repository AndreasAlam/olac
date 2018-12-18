##################
Decisions and logs
##################

:latest update: 2018-07-27
:Version: 0.2

a. Model comparisson
====================
:code:`sbe`
In order to make a fair comparisson between the attempted strategies, decisions on a baseline have to be made. One of them is deciding which online vs non-online (from now on refered to as offline) models to compare.

There are several options available, from regression to deeplearning. Since the start of the project a focus has been put on deeplearning. Thus, we decided to compare two deeplearning models.

Mostly for ease, we use the :code:`sklearn.models.MLPClassifier` function, which has the option to use :code:`.partial_fit` method, which is used to update the model. In combination with setting the :code:`batch_size` to 1 we get an online learning algorithm.
We can use the same hyperparameter settings for both models so to hopefully eliminate other possible influences on the performance other than the training stategies.

