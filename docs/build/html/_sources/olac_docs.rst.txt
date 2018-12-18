##############################
OLAC - Online Learning at Cost
##############################

:status: WIP
:date: 2018-06-22
:Version: 0.2

:authors: Susanne Groothuis, John Paton, Bram Schermer, Ralph Urlus


a. Identify the problem & business case
=======================================

Data that changes over time can be an issue in regards to classification tasks in machine learning. Especially if new characteristics emerge within the same class. An example of this is machine learning applied to fraud detection in financial institutions. New kinds of fraud appear over time, as new ways to ‘cheat the system’ are invented, especially if current ways are being successfully detected or stopped. A problem for ML is that the flexibility of most algorithms is not strong enough to keep up with these new types of fraud appearing over time. Retraining is the a common way of dealing with these changes. However, successfully retraining your model to detect new types of fraud highly depends on these new types of fraud being labelled. Retrieving new labels is an expensive exercise as cases need to be investigated by employees. In addition, there is a risk of of introducing bias in your model by only investigating the high-risk cases produced by the model. 
In this project we are investigating retraining strategies for models in production models in a cost-effective way, applied to financial fraud detection as a use case. We will do this by comparing ’traditional’ deep learning vs Online learning models. The goal of the project is to research optimal settings for labelling new data and providing feedback to a trained model, provided that we are trying balance the cost of obtaining new labels, with the cost of model decay over time.


b. Describe the data-driven artifact
====================================

We will produce a Proof of Concept demonstrating our progress towards an automated solution for preventing model decay on dynamic data. In particular we expect to produce a series of demonstrative Jupyter Notebooks with explorations of the results, as well as code snippets that could be compiled into a software package (or incorporated into existing packages) at a later date, e.g. as part of a follow-up project.

c. Data
=======

i. Process that produced the data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The data we want to use will be a simulated set that represents the use cases described in the problem. The precise characteristics will still have to be defined, but overall they will represent data that is:

    #. dynamic in nature (new data comes in over time)
    #. Introduces new features resulting in the same outcome (e.g. new ways of committing fraud, but the algorithm still needs to detect it as fraud)

The data will either be completely simulated, or based existing data where different portions are introduced over time.

ii. Benefits of using this (type of) data (structured, unstructured, simulated)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The benefits of using simulated data is that we can control when new features are introduced, and monitor exactly how the algorithms react to these changes. This will allow us to create a good baseline and see how different retraining or online learning strategies affect the performance of our model.

iii. Describe the datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^

For description of the datasets we will potentially use for de simulated data described in point c.i. See our github: https://github.com/RUrlus/olac

iv. Data Quality
^^^^^^^^^^^^^^^^

Data quality issues will be addressed in an ad-hoc manner as they arise during the initial data explorations. Furthermore, most of the data sets that we will use for data simulation are from kaggle which are per definition pre-cleaned.
    
v. Selections on data
^^^^^^^^^^^^^^^^^^^^^

To simulate data of a dynamic nature while trying to retain a realistic dataset, we may make use of data masking to distort the statistics of our train and test set. Specifically, one option is to cluster a static dataset and then return the rows in a streaming fashion, sampling from the different clusters with dynamic probability that varies in time. This will ensure that future data does not necessarily resemble past data, while still utilizing available real(istic) static data. 

d. Project steps
================

i. Schematic overview
^^^^^^^^^^^^^^^^^^^^^
    #. Define the use case and get an overview of what we want to present in the end by asking the following things to ourselves:
        #. What does the real data look like?
        #. What are the real problems with dynamic datasets and online learning?
        #. What would a preliminary demo design look like?
    #. Research what the general shape of the data should be and how it is mapped to the real life problems.
    #. Model the case where the problems (degradation of the model) actually occurs 
    #. Create the toy data we need for the problem, the data will be created in the order that it starts simplistic and made more realistic on the fly. 

    #. In parallel part of the team shall start with the setup for:
        #. The online learning algorithms
        #. Retraining the algorithms
        #. Learning at cost case

    #. We will make use of deep learning algorithms to classify the data points. Thus for example in case of fraud the labels are is a certain transaction fraudulent or not. Because of the fact that the dataset is dynamic we will either need to retrain or perform online learning. Retraining or performing online learning have  each their own costs. Also the fact that data has to be newly labeled will also have a price. 

    #. Merge the two parts of the algorithm so that we can compare the whole result and find if there are improvements.
    #. Create the final pitch of our results including live demo

ii. Planned interactions with the sponsor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Erik Postma: Weekly update calls and periodic face-to-face meetings for more in-depth reviews and discussions.
- Max Baak: Sporadic input and validation.



iii. Potential issues
^^^^^^^^^^^^^^^^^^^^^

**Scope creep**
    - The problem space is big enough that it could fill multiple PhDs so we should be careful to prevent digging in to much in a particular step given the allotted time.

Preventative measure
    - By strongly scoping the problem space beforehand and solidifying this in a set of assumptions we can prevent creep.

**Circular dependencies** 
    - The way the subsections of the problem are intertwined could cause circular dependencies in the work, where the project members are dependent on work of others.

Preventative measure:
    - This will be addressed by creating a minimal viable pipeline between the various elements which will reduces the chance of blockages.

**Highly abstract project**
    - Although not inherently an issue it would be preferable if we can strongly link it to practical use cases.

e. Models and algorithms
========================

i. Data transformations
^^^^^^^^^^^^^^^^^^^^^^^

The biggest transformation that we will have to make is to process our data line for line like a stream, although the data is provided as a batch. We will also likely use cluster-based masking to warp the statistics of the dataset through time, so that the training data is displaying substantially different statistics from the stream of new data points


ii. Model/algorithm selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will be comparing two classes of algorithms: batch-trained vs online. We expect that online algorithms will be more flexible for statistically dynamic data, but they will likely bring their own challenges too (discovering this is part of the project). Traditional batch-trained models (classification or outlier detection) are more familiar territory, but often make the assumption that the dataset is stationary. 
We will also be using traditional optimization methods to try to balance the cost of acquiring new labelled samples with maintaining good model performance.

iii. Computation
^^^^^^^^^^^^^^^^

Since we have experience with Docker it seems feasible (given time/demand) to dockerize whatever model architecture we come up with. By deploying the model as a service behind a load balancer we could run multiple instances in parallel, all sharing the same set of learned parameters and updating them as needed. Realistically, this will probably be out of scope for this project. 

iv. Performance measurement
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The goal of our model is to find an optimization between the costs of training, retraining, labeling and the performance of the classification of Fraud or not. The performance of our model will be measured in these two quantities: cost and how well the labeling of the model performs. This will be done by using the metrics of precision and the cost function.
The specific function of the cost is part of the research but we estimate that it will consist of 3 terms: Cost of retraining or additive training, cost of obtaining new labels, cost of bad performance of the model. 
 
v. Testing & validation
^^^^^^^^^^^^^^^^^^^^^^^

To test our setup we will need some sort of model experiment scenario consisting of the following phases:
    #. (Optional) Initial training phase on “historical” data
    #. Performance monitoring phase on data that happens "now"
    #. Adjusting the model phase, retraining or online training
    #. Optimizing phase we optimize the best settings for the lowest cost and the highest performance of the model


vi. Feature engineering & selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The features we use will be dummy features. We will start to extract a subselection of the features from real fraud cases. We will start with the feature that fraud occurs in different clusters, we will add also in the time dependency. So in plain words, x and y component and a time component. 

f. 
===

g. 
===

h. Ethics and legal
===================
We are going to design a toy problem so that will result in no issues with ethics and legal. No personal data will be used in the analysis so there are no issues with that. 

i. 
===

j. Contributions and learning goals of team members
===================================================

**Susanne:**
    Contributions: Programming experience in Python, machine learning and project management.
    Learning goals: Deep Learning techniques, online learning and visualization.

**John:**
    Contributions: technical/programming experience in Python, solid math background. 
    Learning goals: Practical implementation of deep learning models, learning about new machine learning techniques
**Ralph:**
    Contributions: Programming experience in Python and Nim, know some things about maths and statistics.
    Learning goals: gain/improve knowledge about optimization methodologies, (extended) Kalman filters, Bayesian modelling, streaming/online learning.
**Bram:**
    Contributions: Programming experience in Python and C++, Experience with Monte-Carlo simulations, background in physics. 
    Learning goals: Learn how to create and implement a deep learning network. 
