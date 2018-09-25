Fraud, for example credit card fraud, is a serious problem for financial companies. They suffer huge losses each year. State of the art fraud detection makes use of AI but keeping the model up-to-date in a dynamic environment can be challenging, new kinds of fraud are invented each day. The main problem  of most algorithms is that they are not flexible enough and/or require human intervention to keep up with these new kinds of fraud. To keep current one has to retrain the algorithm again and again. Re-training can be extremely time consuming and thus expensive. We will make a self-learning algorithm that makes use of online-learning, the algorithms 're-train' on each observation. Online learning typically is the right choice when dealing with streaming data that is non-stationary or quite heterogeneous.
The goal of the project is to research the optimal strategy for labelling new data and providing feedback to a trained model, provided there are limited resources to label new data. This strategy should explore as well as exploit to ensure that the model does not become biased and is aware of new trends which need to be researched and labelled to provide feedback to the model. The question is thus what the optimal balance would be between exploitation and exploration and the optimal exploration strategies, e.g. random, furthest first and entropy gain, such that the model stays relevant, and the case worker's time is optimally allocated.

## Describe the data-driven artifact
We will produce a Proof of Concept demonstrating our progress towards an automated solution for preventing model decay on dynamic data. In particular we expect to produce a series of demonstrative Jupyter Notebooks with explorations of the results, as well as code snippets that could be compiled into a software package (or incorporated into existing packages) at a later date, e.g. as part of a follow-up project.

## Data
1. Process that produced the data
The data we want to use will be a simulated set that represents the use cases described in the problem. We cope with two types of data we might use in this project. Our main type of data will be simulated data. This data is simulated according to the following determined characteristics:
Has to be dynamic in nature (new data comes in over time)
Data has only x and y coordinates as characteristics
Data with new ‘characteristics’ will appear over time. In the toy problem, which we will mainly use, will this mean that there will appear a new cluster of data with a new center in the xy space. 
Noise will be simulated in the way that the clusters have a certain width. The density of the cluster will generally be given by a gaussian distribution.
(optional) we will sample from a existing dataset as input for the algorithm. This will only be done if there is time left.
So in other words the data will either be completely simulated, or based on existing data where portions of this data will be introduced over time. Because of the fact that it is a simulation the data will not have missing values or significant error that we have to fix or cope with. 

Benefits of using this (type of) data (structured, unstructured, simulated)
The benefits of using simulated data is that we can control when new features are introduced, and monitor exactly how the algorithms react to these changes. This will allow us to create a good baseline and see how different retraining or online learning strategies affect the performance of our model.

Describe the datasets
For description of the datasets we will potentially use for de simulated data described in point c.i. See our github: https://github.com/RUrlus/olac

Data Quality
Data quality issues will be addressed in an ad-hoc manner as they arise during the initial data explorations. We are using most of the time our own simulated data or simulated data from kaggle (last is the optional part). 
	
Selections on data
To simulate data of a dynamic nature while trying to retain a realistic dataset, we may make use of data masking to distort the statistics of our train and test set. Specifically, one option is to cluster a static dataset and then return the rows in a streaming fashion, sampling from the different clusters with dynamic probability that varies in time. This will ensure that future data does not necessarily resemble past data, while still utilizing available real(istic) static data. 
For the data model see the input of a single line from the data generators below. This is the data we use for our toy problem. The data is streaming so will come in one by one over time for 100s to 1000 records per second. The data is fed into the algorithm directly. The generated data is not by default not saved but generated for each simulation again because it has to be streaming.
