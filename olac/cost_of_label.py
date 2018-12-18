import numpy as np
import pandas as pd


def cost_of_label(data, decision, data_type='array', salary=-1.5):
    """
    This function gives back the cost of obtaining a label from the incoming data. It will check if the classification
    of the points one wants to investigate is correct and assesses a profit or loss on each result.

    a data point has the format [x,y,label] where label is always the last entry in the list

    Parameters
    ----------
    data : can be either a data point or a array of data points.
    decision : bool or array of bools, investigate a point True or False
    data_type : how does the data come in, either 'array' or 'point' like.
    salary : How much it costs to investigate a point

    Returns
    -------
    costs for obtaining a certain label of a data point or labels of an array of data points

    """
    if data_type == 'point':
        cost = cost_investigation(data, decision, salary)
        return cost

    elif data_type == 'array':
        print('Labeler:\t evaluate the array of data in queue')
        # array like data input
        df = pd.DataFrame(data)
        df.columns = ['data']
        df['decision'] = decision
        df_invest = df[df['decision'] == 1]
        if len(df_invest) == 0:
            print('Labeler:\t no investigation done')
            df['cost'] = 0
        else:
            df['cost'] = df_invest.apply(lambda x: cost_investigation(x['data'],
                                                                      x['decision'],
                                                                      salary),
                                         axis=1)
            df['cost'].fillna(0, inplace=True)
        print('Labeler:\t finished evaluate the array')

        return np.array(df['cost'])


def cost_investigation_eval(data_point, decision, salary=-1.0, fraud_label=1):
    """
    Function that looks at the data is there fraud or not compared to what the model predicts.
    We want to obtain the labels, did we make a correct decission or not and the cost consequences are made. This
    function can be used for the evaluation of the model
    Parameters
    ----------
    data_point : series or array like object that has as last entry the true label
    decision : array or bool if we have investigated the record or not
    salary : how much does a standard investigation cost, default = -1
    fraud_label : default we have determined fraud at 1

    Returns
    -------
    cost of investigating for a label

    """
    profit = 0
    loss = 0
    correct_label = data_point.true_label
    if (decision == 1) & (correct_label == fraud_label):
        # we found a fraud label so we investigated in the right way! We made money!
        profit = 4
        loss = 0
    elif (decision == 1) & (correct_label != fraud_label):
        # we did investigate a none fraud case so we lost only the salary
        profit = 0
        loss = 0
    elif (decision == 0) & (correct_label == fraud_label):
        # we missed some fraud this is bad!!!
        profit = 0
        loss = -2
        salary = 0
    elif (decision == 0) & (correct_label != fraud_label):
        # we did not investigate no fraud
        profit = 0
        loss = 0
        salary = 0
    cost = salary + profit + loss
    return cost


def cost_investigation(data_point, decision, salary, fraud_label=1):
    """
    When labeling one wants to keep track of their costs during the labeling in time because of a certain budget
    Parameters
    ----------
    data_point : series or array like object that has as last entry the true label
    decision : array or bool if we have investigated the record or not
    salary : how much does a standard investigation cost, default = -1
    fraud_label : default we have determined fraud at 1

    Returns
    -------
    Cost of label in the loop
    """
    correct_label = data_point.true_label
    if (decision == 1) & (correct_label == fraud_label):
        # we found a fraud label so we investigated in the right way! We made money!
        profit = 3
        loss = 0
    else:
        # we investigated on non-fraud
        profit = 0
        loss = 0
    cost = salary + profit + loss
    return cost


def simple_decider(self, points, prob):
    """
    Simplest labeler available. It will generate a decision randomly whether to investigate or not.
    Parameters
    ----------
    self :
    points : data points from queue
    prob : probability that a point is investigated

    Returns
    -------

    """
    n = len(points)
    return np.random.binomial(1, prob, size=n)
